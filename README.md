# RAG-MultiFile-QA
![GitHub Repo stars](https://img.shields.io/github/stars/Uni-Creator/RAG-MultiFile-QA?style=social)  ![GitHub forks](https://img.shields.io/github/forks/Uni-Creator/RAG-MultiFile-QA?style=social)

📚 **Multi-File Retrieval-Augmented Generation (RAG) Q&A System**

This project is a **Streamlit-based Q&A application** that allows users to upload multiple document types (**PDF, DOCX, TXT, CSV**) and ask questions about their content using **retrieval-augmented generation (RAG)**.

## 🔹 Features
- Upload and process multiple files at once.
- Supports **PDF, DOCX, TXT, and CSV** formats.
- Uses **Hugging Face Embeddings** and **FAISS vector search** for document retrieval.
- Integrates **Hugging Face Inference API** for generating responses.
- Maintains **chat history** for seamless user experience.
- **Clear all** button to reset uploaded files and chat history.

## 🛠️ Tech Stack
- **Python**
- **Streamlit** (Frontend UI)
- **Langchain** (Document Processing & Retrieval)
- **Hugging Face Inference API** (LLM-based Answer Generation)
- **FAISS** (Vector Store for Efficient Retrieval)
- **PyPDFLoader, TextLoader, CSVLoader** (File Parsing)

## 🚀 How to Run
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/RAG-MultiFile-QA.git
   cd RAG-MultiFile-QA
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set your **Hugging Face API Key** as an environment variable:
   ```sh
   export HUGGINGFACE_API_KEY="your_api_key"
   ```
4. Run the app:
   ```sh
   streamlit run main.py
   ```

## 📌 Notes
- Ensure your **Hugging Face API Key** is correctly set.
- The system works best with **structured documents** containing well-defined sections and tables.
- **FAISS indexing** helps in faster search and retrieval from large documents.

## 📜 License
This project is **open-source** and available under the **MIT License**.

