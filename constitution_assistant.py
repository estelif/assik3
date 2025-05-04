import os
import tempfile
import streamlit as st
from document_loader import get_constitution_documents
from vector_db_manager import VectorDBManager
from qa_system import QASystem
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.schema import Document
from typing import List

st.set_page_config(page_title="🇰🇿 AI Assistant for the Constitution of Kazakhstan", layout="wide")

CONSTITUTION_URL = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"
VECTOR_DB_DIR = "./chroma_db"
OLLAMA_MODEL = "llama2"

vector_db_manager = VectorDBManager(VECTOR_DB_DIR)
qa_system = QASystem(OLLAMA_MODEL)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_files(uploaded_files) -> List[Document]:
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.name.endswith('.docx'):
            loader = Docx2txtLoader(temp_file_path)
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(temp_file_path)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            continue

        documents.extend(loader.load())
        os.unlink(temp_file_path)
    return documents

def main():
    st.title("🇰🇿 AI Assistant for the Constitution of Kazakhstan")
    st.markdown("""
    This assistant can answer questions about the Constitution of Kazakhstan and any additional documents you provide.
    """)

    # Sidebar for file uploads
    with st.sidebar:
        st.header("Document Management")
        uploaded_files = st.file_uploader(
            "Upload additional documents (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if st.button("Load Constitution"):
            with st.spinner("Loading the Constitution of Kazakhstan..."):
                try:
                    constitution_docs = get_constitution_documents(CONSTITUTION_URL)
                    
                    if constitution_docs:
                        st.session_state.vector_db = vector_db_manager.initialize_db(constitution_docs)
                        st.success(f"Loaded {len(constitution_docs)} constitutional articles!")
                    else:
                        fallback_docs = [
                            Document(
                                page_content="The Constitution of Kazakhstan establishes Kazakhstan as a democratic, secular, legal and social state.",
                                metadata={"source": "fallback"}
                            )
                        ]
                        st.session_state.vector_db = vector_db_manager.initialize_db(fallback_docs)
                        st.warning("Using basic constitutional knowledge. For full text, please upload documents.")
                        
                except Exception as e:
                    st.error(f"Error loading constitution: {str(e)}")
                    st.session_state.vector_db = vector_db_manager.initialize_db([])
        
        if uploaded_files and st.button("Process Uploaded Files"):
            with st.spinner("Processing files..."):
                uploaded_docs = process_files(uploaded_files)
                if st.session_state.vector_db:
                    st.session_state.vector_db = vector_db_manager.add_documents(
                        st.session_state.vector_db, uploaded_docs
                    )
                else:
                    st.session_state.vector_db = vector_db_manager.initialize_db(uploaded_docs)
                st.success(f"Processed {len(uploaded_docs)} document(s) successfully!")

    # Main chat area
    st.header("Ask about the Constitution")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the Constitution"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            if st.session_state.vector_db is None:
                st.warning("Please load the Constitution or upload documents first.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        qa_chain = qa_system.get_qa_chain(st.session_state.vector_db)
                        result = qa_chain({"query": prompt})
                        response = result["result"]
                        
                        # Display sources if available
                        if "source_documents" in result:
                            sources = list(set(
                                doc.metadata.get("source", "Unknown") 
                                for doc in result["source_documents"]
                            ))
                            response += f"\n\nSources: {', '.join(sources)}"
                        
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()