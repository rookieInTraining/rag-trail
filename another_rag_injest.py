import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from another_rag_trial import Models

load_dotenv()

models = Models()
embeddings = models.embeddins_ollama
llm = models.model_llama

data_folder = "./data"
chunk_size = 1000
chunk_overlap = 50
check_interval = 10

vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db",
)

def ingest_file(file_path):
    if not file_path.lower().endswith('.pdf'):
        print(f"Skipping non-PDF files: {file_path}")
        return
    
    print(f"Starting to ingest file: {file_path}")
    loader = PyPDFLoader(file_path)
    loaded_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", " ", ""]
    )

    documents = text_splitter.split_documents(loaded_docs)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    print(f"Adding {len(documents)} documents tothe vector store")
    vector_store.add_documents(documents=documents, ids=uuids)
    print(f"Finished ingesting file: {file_path}")

def main_loop():
    while True:
        for filename in os.listdir(data_folder):
            if not filename.startswith("_"):
                file_path = os.path.join(data_folder, filename)
                ingest_file(file_path)
                new_filename = "_" + filename
                new_file_path = os.path.join(data_folder, new_filename)
                os.rename(file_path, new_file_path)
        time.sleep(check_interval)

if __name__ == "__main__":
    main_loop()