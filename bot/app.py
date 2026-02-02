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
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone

# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_INDEX_NAME = "bot768"

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    st.error("API keys not found. Check your .env file.")
    st.stop()

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if not uploaded_pdf:
    st.info("Please upload a PDF to continue.")
    st.stop()

pdf_name = uploaded_pdf.name.replace(" ", "_").replace(".", "_").lower()
NAMESPACE = pdf_name

# --------------------------------------------------
# SESSION RESET ON NEW PDF
# --------------------------------------------------
if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = pdf_name

if st.session_state.active_pdf != pdf_name:
    st.session_state.active_pdf = pdf_name
    st.session_state.messages = []
    st.cache_resource.clear()

# --------------------------------------------------
# PINECONE INIT
# --------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

stats = index.describe_index_stats()
namespaces = stats.get("namespaces", {})

SKIP_EMBEDDING = (
    NAMESPACE in namespaces
    and namespaces[NAMESPACE].get("vector_count", 0) > 0
)

# --------------------------------------------------
# VECTORSTORE (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_vectorstore(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
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

vectorstore = load_vectorstore(uploaded_pdf)

# --------------------------------------------------
# CONTEXT FORMATTER
# --------------------------------------------------
def format_docs(docs):
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)

# --------------------------------------------------
# LLM
# --------------------------------------------------
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.0,
    max_new_tokens=200,
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
            "You are a strict extractive question-answering system.\n\n"
            "RULES:\n"
            "1. Use ONLY the provided context.\n"
            "2. Do NOT use prior knowledge.\n"
            "3. Do NOT infer or assume missing information.\n"
            "4. If the answer is not explicitly stated, output exactly:\n"
            "Answer not found in the context.\n"
            "5. One or two sentences maximum.\n"
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ]
)

rag_chain = prompt | llm | StrOutputParser()

# --------------------------------------------------
# CHAT UI
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question from the PDF...")

if query:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching PDF..."):
            docs = vectorstore.similarity_search(query, k=4)
            context = format_docs(docs)

            if not context.strip():
                response = "Answer not found in the context."
            else:
                response = rag_chain.invoke(
                    {
                        "context": context,
                        "question": query
                    }
                )

            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
