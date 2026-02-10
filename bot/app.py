import os
import tempfile
import hashlib
import streamlit as st
from dotenv import load_dotenv
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📘",
    layout="wide"
)

st.title("📘 PDF Q&A Chatbot")
st.caption("Ask questions and get accurate answers directly from your uploaded document.")

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

INDEX_NAME = "pdf-qa-production"
EMBEDDING_DIM = 384

# Improved configuration
CHUNK_SIZE = 800  # Larger chunks for better context
CHUNK_OVERLAP = 200  # More overlap for continuity
TOP_K_RETRIEVAL = 5  # Retrieve more candidates
TOP_K_CONTEXT = 3  # Use best 3 for context
SIMILARITY_THRESHOLD = 0.7  # Minimum relevance score

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    st.error("❌ Missing API keys. Please check your .env file.")
    st.stop()

# --------------------------------------------------
# INIT PINECONE
# --------------------------------------------------
@st.cache_resource
def init_pinecone():
    """Initialize Pinecone connection and index"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    
    return pc, pc.Index(INDEX_NAME)

pc, index = init_pinecone()

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------
@st.cache_resource
def get_embeddings():
    """Initialize embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

embeddings = get_embeddings()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    uploaded_pdf = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Upload a PDF file to ask questions about"
    )
    
    st.divider()
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        retrieval_k = st.slider(
            "Number of chunks to retrieve",
            min_value=3,
            max_value=10,
            value=TOP_K_RETRIEVAL,
            help="More chunks = more context but slower"
        )
        
        context_k = st.slider(
            "Chunks to use for answer",
            min_value=2,
            max_value=5,
            value=TOP_K_CONTEXT,
            help="Chunks sent to the LLM"
        )
        
        temperature = st.slider(
            "Response creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="0 = factual, 1 = creative"
        )
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Display document info
    if uploaded_pdf:
        st.success(f"✅ Loaded: {uploaded_pdf.name}")
        st.caption(f"Size: {uploaded_pdf.size / 1024:.1f} KB")

if not uploaded_pdf:
    st.info("👈 Upload a PDF document from the sidebar to begin.")
    st.stop()

# --------------------------------------------------
# SAVE PDF & GENERATE HASH
# --------------------------------------------------
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_pdf.read())
    pdf_path = tmp.name

file_hash = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()
namespace = file_hash

# --------------------------------------------------
# BUILD / LOAD VECTORSTORE WITH PROGRESS
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vectorstore(_pdf_path: str, _namespace: str) -> PineconeVectorStore:
    """Load or create vector store with improved chunking"""
    
    existing = index.describe_index_stats().get("namespaces", {})
    
    # If already indexed, load from Pinecone
    if _namespace in existing:
        return PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=_namespace,
        )
    
    # Otherwise, process the PDF
    with st.spinner("📄 Processing PDF..."):
        docs = PyPDFLoader(_pdf_path).load()
        
        # Enhanced text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )
        
        chunks = splitter.split_documents(docs)
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        st.info(f"Created {len(chunks)} chunks from {len(docs)} pages")
        
        return PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=_namespace,
        )

vectorstore = load_vectorstore(pdf_path, namespace)

# --------------------------------------------------
# LLM WITH BETTER CONFIG
# --------------------------------------------------
@st.cache_resource
def get_llm(_temperature: float = 0.1):
    """Initialize LLM with configurable temperature"""
    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=_temperature,
            max_new_tokens=400,
            top_p=0.95,
            repetition_penalty=1.1,
            huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        )
    )

llm = get_llm(temperature)

# --------------------------------------------------
# ENHANCED PROMPT TEMPLATE
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert document analyst that extracts precise information from documents.

STRICT RULES:
1. Answer ONLY using information explicitly stated in the provided context
2. Quote or closely paraphrase exact text from the context
3. If the answer spans multiple sentences, include all relevant parts
4. If information is partially available, provide what exists and note what's missing
5. If the answer is not in the context, respond EXACTLY: "I cannot find this information in the document."
6. Do not add interpretation, assumptions, or external knowledge
7. Be concise but complete - include all relevant details from the context

Context passages are provided below with page numbers."""
    ),
    (
        "human",
        """Context from the document:
{context}

Question: {question}

Provide a direct answer based solely on the context above:"""
    ),
])

# --------------------------------------------------
# IMPROVED ANSWER FUNCTION
# --------------------------------------------------
def answer_question(question: str, k_retrieval: int = 5, k_context: int = 3) -> str:
    """
    Generate answer with improved retrieval and context assembly
    """
    
    if not question.strip():
        return "Please ask a valid question."
    
    try:
        # Retrieve relevant chunks with scores
        results = vectorstore.similarity_search_with_score(question, k=k_retrieval)
        
        if not results:
            return "I cannot find this information in the document."
        
        # Filter by similarity threshold (lower score = more similar)
        # Note: Pinecone cosine distance ranges from 0 (identical) to 2 (opposite)
        filtered_results = [(doc, score) for doc, score in results if score < SIMILARITY_THRESHOLD]
        
        if not filtered_results:
            return "I cannot find sufficiently relevant information in the document to answer this question."
        
        # Sort by similarity (lower score = better)
        sorted_results = sorted(filtered_results, key=lambda x: x[1])
        
        # Take top k_context chunks
        top_docs = [doc for doc, score in sorted_results[:k_context]]
        
        # Build context with page numbers
        context_parts = []
        for doc in top_docs:
            page_num = doc.metadata.get("page", 0) + 1
            content = doc.page_content.strip()
            context_parts.append(f"[Page {page_num}]\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer
        response = llm.invoke(
            prompt.format(context=context, question=question)
        )
        
        answer = response.content.strip()
        
        # Check if model couldn't find answer
        if any(phrase in answer.lower() for phrase in [
            "cannot find",
            "not mentioned",
            "does not contain",
            "no information"
        ]):
            return "I cannot find this information in the document."
        
        # Add source citations
        pages = sorted(set(doc.metadata.get("page", 0) + 1 for doc in top_docs))
        page_str = ", ".join(map(str, pages[:5]))  # Limit to 5 pages
        
        # Calculate average relevance score
        avg_score = sum(score for _, score in sorted_results[:k_context]) / k_context
        confidence = "High" if avg_score < 0.3 else "Medium" if avg_score < 0.5 else "Moderate"
        
        return f"{answer}\n\n📄 **Source:** Page(s) {page_str}\n🎯 **Confidence:** {confidence}"
    
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "An error occurred while processing your question. Please try rephrasing or try again."

# --------------------------------------------------
# CHAT INTERFACE WITH IMPROVEMENTS
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask a question about the PDF...", key="user_input")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate and display answer
    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            answer = answer_question(
                user_input,
                k_retrieval=retrieval_k,
                k_context=context_k
            )
        st.markdown(answer)
    
    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})

# --------------------------------------------------
# FOOTER WITH TIPS
# --------------------------------------------------
with st.expander("💡 Tips for Better Results"):
    st.markdown("""
    - **Be specific**: Ask clear, focused questions
    - **Use keywords**: Include terms likely to appear in the document
    - **Break it down**: Split complex questions into simpler ones
    - **Check sources**: Review the cited page numbers for context
    - **Rephrase**: If you don't get a good answer, try asking differently
    
    **Example questions:**
    - "What is the main conclusion of this document?"
    - "What methodology was used in the study?"
    - "What are the key findings on page 5?"
    """)

# Clean up temp file on session end
if st.session_state.get("cleanup_needed", False):
    try:
        os.unlink(pdf_path)
    except:
        pass
