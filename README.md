# Chat with PDF using Azure OpenAI 💬📄

Interact with your PDF documents using Azure OpenAI's GPT-4 and text embeddings. Just upload a PDF, ask a question, and get smart, context-aware answers. Built with LangChain, FAISS, and Streamlit.

---

## 🚀 Features

- 📄 Upload and process one or more PDF files
- 🔍 Intelligent context retrieval using Azure OpenAI embeddings
- 🧠 GPT-4-based answer generation via LangChain QA chain
- ⚡ Fast document search powered by FAISS
- 🖥️ Clean Streamlit UI with sidebar upload and question input

---

## 🧠 How It Works

1. PDF text is extracted and split into overlapping chunks
2. Embeddings are generated using Azure's `text-embedding-3-small`
3. Chunks are indexed using FAISS
4. User asks a question → similar chunks are retrieved
5. GPT-4 answers the question using only the relevant context

---

## 💡 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Azure OpenAI (GPT-4 + Embeddings)
- **Orchestration**: LangChain
- **Vector Storage**: FAISS
- **PDF Parsing**: PyPDF2

---
