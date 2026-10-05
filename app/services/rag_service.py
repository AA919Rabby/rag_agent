import os
import shutil
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings


class RAGService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance.initialize_rag()
        return cls._instance

    def initialize_rag(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GEMINI_API_KEY
        )
        self.persist_directory = "./vector_db"
        self.data_directory = "./data"
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        self.vector_store = self._load_and_vectorize_documents()

    def _load_and_vectorize_documents(self):
        all_docs = []
        if not os.path.exists(self.data_directory):
            return None

        for file in os.listdir(self.data_directory):
            file_path = os.path.join(self.data_directory, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                all_docs.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
                all_docs.extend(loader.load())

        if all_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(all_docs)
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )
            return vector_store

        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return None

    async def save_and_process_file(self, file: UploadFile):
        file_path = os.path.join(self.data_directory, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        self.vector_store = self._load_and_vectorize_documents()
        return file.filename


    def query_vector_store(self, question: str):
        if not self.vector_store:
             return "No documents uploaded yet. Please upload a PDF first.", []

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})


        relevant_docs = retriever.invoke(question)

        context = "\n".join([doc.page_content for doc in relevant_docs])
        sources = [doc.metadata.get("source", "Unknown") for doc in relevant_docs]
        return context, sources



rag_service = RAGService()
