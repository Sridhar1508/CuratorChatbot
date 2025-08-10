import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()  
import tempfile

# Page Config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("Curator Chatbot")
st.markdown("Upload a file and ask questions from it!")


# Set Groq API Key & Endpoint
os.environ["OPENAI_API_KEY"] = "API_KEY_XXXXXXXXXXXXXXXXXX"
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# Upload Section
uploaded_file = st.file_uploader("📤 Upload your File (PDF)", type=["pdf"], accept_multiple_files=True)

# Define LLM using Groq
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
    )

llm = get_llm()

if uploaded_file:
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Save to temp file
    for file in uploaded_file:
        st.write(f"Processing file: {file.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        # Load and chunk the PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)
        
    # Create vectorstore from all chunks
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Set up retriever and RAG chain
    retriever = vectorstore.as_retriever(k=3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Query input
    query = st.text_input("🔍 Ask a question :")

    if st.button("Get Answer") and query:
        with st.spinner("Thinking..."):
            result = qa_chain(query)
            st.success(result['result'])

            

else:
    st.info("👆 Please upload a PDF file to get started.")
