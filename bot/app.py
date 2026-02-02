# import os
# import tempfile
# import streamlit as st
# from dotenv import load_dotenv

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import (
#     HuggingFaceEmbeddings,
#     HuggingFaceEndpoint,
#     ChatHuggingFace
# )
# from langchain_pinecone import PineconeVectorStore
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from pinecone import Pinecone

# # --------------------------------------------------
# # LOAD ENV
# # --------------------------------------------------
# load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# PINECONE_INDEX_NAME = "bot768"

# if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
#     st.error("API keys not found. Check your .env file.")
#     st.stop()

# # --------------------------------------------------
# # FILE UPLOAD
# # --------------------------------------------------
# uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

# if not uploaded_pdf:
#     st.info("Please upload a PDF to continue.")
#     st.stop()

# pdf_name = uploaded_pdf.name.replace(" ", "_").replace(".", "_").lower()

# if "active_pdf" not in st.session_state:
#     st.session_state.active_pdf = pdf_name

# if st.session_state.active_pdf != pdf_name:
#     st.session_state.active_pdf = pdf_name
#     st.session_state.messages = []
#     st.cache_resource.clear()

# NAMESPACE = pdf_name

# # --------------------------------------------------
# # PINECONE INIT
# # --------------------------------------------------
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(PINECONE_INDEX_NAME)

# stats = index.describe_index_stats()
# namespaces = stats.get("namespaces", {})

# SKIP_EMBEDDING = (
#     NAMESPACE in namespaces
#     and namespaces[NAMESPACE].get("vector_count", 0) > 0
# )

# # --------------------------------------------------
# # VECTORSTORE (CACHED)
# # --------------------------------------------------
# @st.cache_resource
# def load_vectorstore(pdf_name, uploaded_pdf):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(uploaded_pdf.read())
#         pdf_path = tmp.name

#     loader = PyPDFLoader(pdf_path)
#     documents = loader.load()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1200,
#         chunk_overlap=200
#     )
#     docs = splitter.split_documents(documents)

#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-mpnet-base-v2"
#     )

#     if not SKIP_EMBEDDING:
#         vectorstore = PineconeVectorStore.from_documents(
#             documents=docs,
#             embedding=embeddings,
#             index_name=PINECONE_INDEX_NAME,
#             namespace=NAMESPACE
#         )
#     else:
#         vectorstore = PineconeVectorStore.from_existing_index(
#             index_name=PINECONE_INDEX_NAME,
#             embedding=embeddings,
#             namespace=NAMESPACE
#         )

#     return vectorstore

# vectorstore = load_vectorstore(pdf_name, uploaded_pdf)

# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 6,
#         "fetch_k": 20,
#         "lambda_mult": 0.6
#     }
# )

# # --------------------------------------------------
# # CONTEXT FORMATTER
# # --------------------------------------------------
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # --------------------------------------------------
# # LLM
# # --------------------------------------------------
# endpoint = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#     task="conversational",
#     temperature=0.2,
#     max_new_tokens=450,
#     huggingfacehub_api_token=HUGGINGFACE_API_KEY
# )

# llm = ChatHuggingFace(llm=endpoint)

# # --------------------------------------------------
# # PROMPT
# # --------------------------------------------------
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a strict literature question-answering system.\n\n"

#             "RULES:\n"
#             "1. Answer ONLY what the question asks.\n"
#             "2. Use ONLY the provided context.\n"
#             "3. Do NOT mix events from different chapters or time periods.\n"
#             "4. If the context shows an early assumption that is later disproved, "
#             "state only what is true at that point in the story.\n"
#             "5. If the text gives a definite answer, give it directly. Do NOT hedge.\n"
#             "6. Do NOT include later events unless the question explicitly asks for them.\n"
#             "7. Do NOT add interpretation, analysis, or personal commentary.\n"
#             "8. If the answer is not present anywhere in the context, say exactly:\n"
#             "   Answer not found in the PDF.\n\n"

#             "STYLE:\n"
#             "- Be concise and factual.\n"
#             "- Prefer exact statements from the text over summaries.\n"
#             "- One to three sentences maximum.\n"
#         ),
#         (
#             "human",
#             "Context:\n{context}\n\nQuestion:\n{question}"
#         )
#     ]
# )


# # --------------------------------------------------
# # RAG CHAIN
# # --------------------------------------------------
# rag_chain = (
#     {
#         "context": retriever | format_docs,
#         "question": RunnablePassthrough()
#     }
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # --------------------------------------------------
# # CHAT UI
# # --------------------------------------------------
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# query = st.chat_input("Ask a question from the PDF...")

# if query:
#     st.session_state.messages.append({"role": "user", "content": query})

#     with st.chat_message("user"):
#         st.markdown(query)

#     with st.chat_message("assistant"):
#         with st.spinner("Searching PDF..."):
#             response = rag_chain.invoke(query)
#             st.markdown(response)

#     st.session_state.messages.append({"role": "assistant", "content": response})










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
# PAGE CONFIG (UI)
# --------------------------------------------------
st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📘",
    layout="wide"
)

st.title("📘 PDF Literature Question–Answer Bot")
st.caption("Ask questions strictly from the uploaded PDF")

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
# SIDEBAR (UPLOAD + CONTROLS)
# --------------------------------------------------
with st.sidebar:
    st.header("📄 PDF Settings")

    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Only answers from the PDF are allowed.")

if not uploaded_pdf:
    st.info("Upload a PDF from the sidebar to begin.")
    st.stop()

# --------------------------------------------------
# PDF SESSION HANDLING
# --------------------------------------------------
pdf_name = uploaded_pdf.name.replace(" ", "_").replace(".", "_").lower()

if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = pdf_name

if st.session_state.active_pdf != pdf_name:
    st.session_state.active_pdf = pdf_name
    st.session_state.messages = []
    st.cache_resource.clear()

NAMESPACE = pdf_name

st.success(f"Active PDF: **{uploaded_pdf.name}**")

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
def load_vectorstore(pdf_name, uploaded_pdf):
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

vectorstore = load_vectorstore(pdf_name, uploaded_pdf)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20,
        "lambda_mult": 0.6
    }
)

# --------------------------------------------------
# CONTEXT FORMATTER
# --------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --------------------------------------------------
# LLM
# --------------------------------------------------
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    temperature=0.2,
    max_new_tokens=450,
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
            "You are a strict literature question-answering system.\n\n"
            "Use ONLY the provided context.\n"
            "If the answer is missing, say:\n"
            "Answer not found in the context.\n"
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
# CHAT UI
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("💬 Ask a question")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Type your question here...")

if query:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF..."):
            response = rag_chain.invoke(query)
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
