import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def run_ingestion():
    # 1. Configuration from .env
    docs_folder = os.getenv("DOCS_PATH", "./docs")
    db_path = os.getenv("VECTOR_DB_PATH", "faiss_index")
    model_name = os.getenv("EMBEDDING_MODEL")

    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)
        print(f"📁 Created '{docs_folder}' folder. Put your PDFs there and re-run.")
        return

    # 2. Load Documents
    print(f"📖 Reading PDFs from {docs_folder}...")
    loader = DirectoryLoader(docs_folder, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("❌ No PDFs found in the folder!")
        return

    # 3. Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    
    # 4. Create Embeddings with task_type (2026 best practice)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name,
        task_type="RETRIEVAL_DOCUMENT"
    )
    
    # 5. Build and Save Vector DB
    print(f"⏳ Encoding chunks with {model_name}...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(db_path)
    print(f"✅ Success! Vector DB saved to {db_path}")

if __name__ == "__main__":
    run_ingestion()