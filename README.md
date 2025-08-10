# 🤖 Curator Chatbot

Curator Chatbot is a **Streamlit-based RAG (Retrieval-Augmented Generation) chatbot** that allows users to upload PDF files and ask questions based on their content. It leverages **LangChain**, **FAISS**, and **Groq's LLM API** to deliver accurate, context-aware answers from uploaded documents.

---

## 🚀 Features

- 📤 Upload multiple PDF files
- 🔍 Ask questions based on document content
- 🧠 Uses Groq's LLaMA 3.3 70B model for fast and intelligent responses
- 🧱 Embedding via HuggingFace (`all-MiniLM-L6-v2`)
- 📚 Document chunking and vector search with FAISS
- 🛠️ Built with LangChain and Streamlit

---

## 🧰 Technologies Used

| Tool/Library         | Purpose                                      |
|----------------------|----------------------------------------------|
| Streamlit            | UI and interaction                          |
| LangChain            | RAG pipeline and document handling          |
| FAISS                | Vector similarity search                    |
| HuggingFace Embeddings | Text embedding model                     |
| Groq API             | LLM backend (LLaMA 3.3 70B)                 |
| dotenv               | Environment variable management             |
| PyPDFLoader          | PDF parsing and text extraction             |

---

## 📦 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Set up environment variables**

    Create a .env file in the root directory and add your Groq API key:

    ```env
    OPENAI_API_KEY=your_groq_api_key
    OPENAI_API_BASE=https://api.groq.com/openai/v1
