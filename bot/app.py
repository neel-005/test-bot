import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Page Config ---
st.set_page_config(page_title="PDF Q&A Bot (Hugging Face)", page_icon="🤗")
st.title("🤗 PDF Q&A Chatbot (Hugging Face + Pinecone)")

# --- Sidebar for API Keys ---
with st.sidebar:
    st.header("Configuration")
    hf_token = st.text_input("Hugging Face Token", type="password", help="Get it from huggingface.co/settings/tokens")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password")
    pinecone_environment = st.text_input("Pinecone Environment", placeholder="e.g., us-east-1", value="us-east-1")
    
    st.divider()
    st.markdown("### Models Used")
    st.markdown("- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`")
    st.markdown("- **LLM:** `mistralai/Mistral-7B-Instruct-v0.2`")
    
    if st.button("Clear Vector Store"):
        if pinecone_api_key:
            try:
                pc = Pinecone(api_key=pinecone_api_key)
                pc.Index("pdf-qa-hf").delete(delete_all=True)
                st.success("Vector store cleared!")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Enter Pinecone Key first.")

# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, hf_token, pinecone_key, pinecone_env):
    # 1. Initialize Embeddings (Local Sentence Transformers)
    # Model dimension is 384
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Initialize Pinecone
    pc = Pinecone(api_key=pinecone_key)
    index_name = "pdf-qa-hf"
    
    # 3. Create Index if not exists (Dimension MUST match embedding model: 384)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384, 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=pinecone_env)
        )
    
    # 4. Upsert to Pinecone
    PineconeVectorStore.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=pinecone_key
    )
    
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_key
    )
    return vectorstore

def get_conversation_chain(vectorstore, hf_token):
    # 1. Initialize LLM via Hugging Face Inference API
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=hf_token
    )
    
    # 2. Strict Prompt for PDF Only Answers
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context from the PDF to answer the question. "
        "If you don't know the answer based on the context, say 'I don't have enough information in the document to answer that'. "
        "Do not use your own general knowledge."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return chain

# --- Main App Logic ---

def main():
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed" not in st.session_state:
        st.session_state.processed = False

    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=['pdf'])

    if st.button("Submit & Process"):
        if not hf_token or not pinecone_api_key:
            st.error("Please provide Hugging Face Token and Pinecone Key in the sidebar.")
        elif not pdf_docs:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing PDFs (Embeddings & Pinecone)..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    
                    st.session_state.vectorstore = get_vectorstore(
                        text_chunks, 
                        hf_token, 
                        pinecone_api_key, 
                        pinecone_environment
                    )
                    
                    st.session_state.processed = True
                    st.success("PDF processed! Ready to chat.")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.processed and st.session_state.vectorstore:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about the PDF..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chain = get_conversation_chain(st.session_state.vectorstore, hf_token)
                        response = chain.invoke({"input": prompt})
                        answer = response["answer"]
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        st.info("Tip: Hugging Face Inference API might be busy. Try again in a moment.")

    elif not st.session_state.processed:
        st.info("👈 Upload a PDF and click 'Submit & Process' to start.")

if __name__ == '__main__':
    main()
