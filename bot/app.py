import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone

# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_INDEX_NAME = "mycode"

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    st.error("API keys not found. Check your .env file.")
    st.stop()

# --------------------------------------------------
# NAMESPACE
# --------------------------------------------------
uploaded_pdf = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    key="pdf_uploader"
)

if not uploaded_pdf:
    st.info("Please upload a PDF to continue.")
    st.stop()

pdf_name = uploaded_pdf.name.replace(" ", "_").replace(".", "_").lower()
pdf_name = uploaded_pdf.name.replace(" ", "_").replace(".", "_").lower()

if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = pdf_name

# If user uploads a new PDF
if st.session_state.active_pdf != pdf_name:
    st.session_state.active_pdf = pdf_name
    st.session_state.messages = []          # clear chat
    st.cache_resource.clear()               # clear vectorstore cache

NAMESPACE = pdf_name

# --------------------------------------------------
# PINECONE INIT + DUPLICATION CHECK
# --------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

stats = index.describe_index_stats()
namespaces = stats.get("namespaces", {})

SKIP_EMBEDDING = (
    NAMESPACE in namespaces
    and namespaces[NAMESPACE].get("vector_count", 0) > 0
)

if SKIP_EMBEDDING:
    st.info("Using existing vectors from Pinecone")
else:
    st.warning("Embedding PDF for the first time")

# --------------------------------------------------
# LOAD VECTORSTORE (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_vectorstore(pdf_name, uploaded_pdf):
    _ = pdf_name  # ensures cache invalidation per PDF

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not SKIP_EMBEDDING:
        vectorstore = PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            namespace=NAMESPACE
        )
    else:
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=NAMESPACE
        )

    return vectorstore



vectorstore = load_vectorstore(pdf_name, uploaded_pdf)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# --------------------------------------------------
# CONTEXT FORMATTER
# --------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs[:3])

# --------------------------------------------------
# LLM
# --------------------------------------------------
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    temperature=0,
    max_new_tokens=100,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)

llm = ChatHuggingFace(llm=endpoint)

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a factual question answering system.\n"
            "- Use ONLY the given context.\n"
            "- Answer in EXACTLY ONE sentence.\n"
            "- If not found, say:\n"
            "  Answer not found in the PDF."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ]
)

# --------------------------------------------------
# RAG CHAIN
# --------------------------------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------
# OUTPUT GUARD
# --------------------------------------------------
def enforce_output(text: str) -> str:
    text = text.strip()
    if "Answer not found" in text:
        return "Answer not found in the PDF."
    return text.split(".")[0].strip() + "."

# --------------------------------------------------
# CHAT UI (FIXED)
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1️. Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2️. Take new input
query = st.chat_input("Ask a question from the PDF...")

# 3️. Handle new message ONLY once
if query:
    # store user message
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    # generate assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Searching PDF..."):
            response = rag_chain.invoke(query)
            final_answer = enforce_output(response)
            st.markdown(final_answer)

    # store assistant message ONCE
    st.session_state.messages.append(
        {"role": "assistant", "content": final_answer}
    )

