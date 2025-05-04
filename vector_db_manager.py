from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import os

class VectorDBManager:
    def __init__(self, db_dir: str = "./chroma_db"):
        self.db_dir = db_dir
        self.embedding_model = "sentence-transformers/all-mpnet-base-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        
    def initialize_db(self, documents: List[Document]):
        """Initialize a new vector database"""
        texts = self.text_splitter.split_documents(documents)
        vector_db = Chroma.from_documents(
            texts,
            self.embeddings,
            persist_directory=self.db_dir
        )
        return vector_db
        
    def add_documents(self, vector_db: Chroma, documents: List[Document]):
        """Add documents to an existing vector database"""
        texts = self.text_splitter.split_documents(documents)
        vector_db.add_documents(texts)
        return vector_db
        
    def load_existing_db(self):
        """Load an existing vector database"""
        if os.path.exists(self.db_dir):
            return Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
        return None