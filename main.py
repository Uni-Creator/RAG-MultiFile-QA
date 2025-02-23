# Import Langchain dependencies
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Other imports
import streamlit as st
import tempfile
import os

# Load API key from environment variable
import os
api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    st.error("API key not found. Please set the 'HUGGINGFACE_API_KEY' environment variable.")
    st.stop()

# Setup LLM
llm = HuggingFaceEndpoint(
    repo_id="facebook/opt-1.3b",
    task="text-generation",
    token=api_key,
    temperature=0.5
)

# Streamlit UI
st.title("Ask RAG - Multi-file Support")

# Upload multiple files
uploaded_files = st.file_uploader("Upload files (PDF, DOCX, TXT, CSV)", type=["pdf", "docx", "txt", "csv"], accept_multiple_files=True)

# Function to load and process multiple files
@st.cache_resource
def load_files(files):
    if not files:
        return None  # No files uploaded
    
    loaders = []
    temp_files = []

    # Process each file
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[-1]) as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name
            temp_files.append(temp_path)  # Store paths to delete later
        
        # Detect file type and use appropriate loader
        if file.name.endswith(".pdf"):
            loaders.append(PyPDFLoader(temp_path))
        elif file.name.endswith(".txt"):
            loaders.append(TextLoader(temp_path))
        elif file.name.endswith(".docx"):
            loaders.append(UnstructuredWordDocumentLoader(temp_path))
        elif file.name.endswith(".csv"):
            loaders.append(CSVLoader(temp_path))

    # Create vector database from all loaders
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
    ).from_loaders(loaders)

    return index, temp_files

# Load files only if any files are uploaded
if uploaded_files:
    index, temp_files = load_files(uploaded_files)
else:
    index, temp_files = None, []

# Initialize Q&A chain if the index is created
if index:
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
        input_key="question"
    )

    # Setup session state messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # User input
    prompt = st.chat_input("Enter your prompt")

    if prompt:
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process prompt with LLM
        response = chain.run(prompt)

        # Display assistant response
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
else:
    st.warning("Please upload files to start querying.")

# Button to clear all messages and reset file upload
if st.button("Clear All"):
    st.session_state.messages = []
    uploaded_files = None  # Reset file uploader

    # Delete temporary files
    for file_path in temp_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    # Reload the app
    st.rerun()
