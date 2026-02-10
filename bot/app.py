import os
import tempfile
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📘")
st.title("PDF Q&A Chatbot")
st.caption("Answers strictly from the uploaded document.")

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

INDEX_NAME = "pdf-qa-production"
EMBEDDING_DIM = 384

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    st.error("Missing API keys.")
    st.stop()

# --------------------------------------------------
# INIT PINECONE
# --------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if st.button("Clear Chat"):
        st.session_state.messages = []


if not uploaded_pdf:
    st.info("Upload a PDF to begin.")
    st.stop()

# --------------------------------------------------
# SAVE FILE
# --------------------------------------------------
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_pdf.read())
    pdf_path = tmp.name

file_hash = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()
namespace = file_hash

# --------------------------------------------------
# BUILD / LOAD VECTORSTORE
# --------------------------------------------------
@st.cache_resource
def load_vectorstore(pdf_path, namespace):

    existing = index.describe_index_stats().get("namespaces", {})

    if namespace in existing:
        return PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=namespace
        )

    docs = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )

vectorstore = load_vectorstore(pdf_path, namespace)

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.0,
        max_new_tokens=300,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY
    )
)

# --------------------------------------------------
# STRICT PROMPT
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict document extractor.\n"
     "Use ONLY text from the context.\n"
     "Return ONLY the exact relevant sentences.\n"
     "Do not add explanations.\n"
     "If the answer is not explicitly written, reply exactly:\n"
     "'I cannot find this information in the document.'"
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
])

# --------------------------------------------------
# ANSWER FUNCTION
# --------------------------------------------------
def answer_question(question):

    results = vectorstore.similarity_search_with_score(question, k=3)

    # Filter by similarity threshold
    filtered = [(doc, score) for doc, score in results if score < 0.6]

    if not filtered:
        return "I cannot find this information in the document."

    # Sort by similarity (lower is better)
    filtered = sorted(filtered, key=lambda x: x[1])

    # Use top 2 strong chunks max
    strong_docs = [doc for doc, score in filtered[:2]]

    context = "\n---\n".join([doc.page_content for doc in strong_docs])

    try:
        response = llm.invoke(
            prompt.format(context=context, question=question)
        )
    except Exception:
        return "Model error. Please try again."

    answer = response.content.strip()

    if "cannot find" in answer.lower():
        return "I cannot find this information in the document."

    pages = sorted({doc.metadata.get("page", 0) + 1 for doc in strong_docs})[:3]

    return answer + f"\n\n📄 Source: Page(s) {', '.join(map(str, pages))}"

# --------------------------------------------------
# STABLE CHAT LOOP
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
prompt_input = st.chat_input("Ask a question about the PDF...")

if prompt_input:

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt_input
    })

    # Generate assistant response
    answer = answer_question(prompt_input)

    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Immediately render the latest messages
    with st.chat_message("assistant"):
        st.markdown(answer)
