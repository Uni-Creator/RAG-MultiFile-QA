# Import Langchain dependencies
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Other imports
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load API key
api_key = None

if "HUGGINGFACE_HUB_TOKEN" in st.secrets:
    api_key = st.secrets["HUGGINGFACE_HUB_TOKEN"]
else:
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Setup LLM using ChatHuggingFace wrapper
llm_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=api_key,
    task="conversational",
    max_new_tokens=512,
    temperature=0.5,
)

llm = ChatHuggingFace(llm=llm_endpoint)

# Streamlit UI
st.title("Ask RAG - Multi-file Support")

# Upload multiple files
uploaded_files = st.file_uploader(
    "Upload files (PDF, DOCX, TXT, CSV)",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True
)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to load and process multiple files
@st.cache_resource(show_spinner=True)
def load_files(files):
    if not files:
        return None, []

    documents = []
    temp_files = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[-1]) as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name
            temp_files.append(temp_path)

        try:
            if file.name.endswith(".pdf"):
                documents.extend(PyPDFLoader(temp_path).load())
            elif file.name.endswith(".txt"):
                documents.extend(TextLoader(temp_path).load())
            elif file.name.endswith(".docx"):
                documents.extend(Docx2txtLoader(temp_path).load())
            elif file.name.endswith(".csv"):
                documents.extend(CSVLoader(temp_path).load())
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")
            continue

    if not documents:
        return None, temp_files

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    # Create embeddings and vectorstore``
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore, temp_files

# Load files
if uploaded_files:
    result = load_files(uploaded_files)
    if result and result[0] is not None:
        vectorstore, temp_files = result
    else:
        vectorstore, temp_files = None, result[1]
else:
    vectorstore, temp_files = None, []

# Initialize Q&A chain if vectorstore is ready
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ChatPromptTemplate works better with ChatHuggingFace
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question using only the context below.

Context: {context}

Question: {input}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt_template
        | llm
    )

    # Setup session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # User input
    prompt = st.chat_input("Enter your prompt")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Invoke chain — ChatHuggingFace returns AIMessage with .content
        response = chain.invoke(prompt)
        output_text = response.content if hasattr(response, 'content') else str(response)

        st.chat_message("assistant").markdown(output_text)
        st.session_state.messages.append({"role": "assistant", "content": output_text})

else:
    st.warning("Please upload files to start querying.")

# Clear button
if st.button("Clear All"):
    st.session_state.messages = []

    for file_path in temp_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    st.rerun()