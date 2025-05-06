import os
from glob import glob
from dotenv import load_dotenv

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def load_pdfs(pdf_folder: str):
    """
    LOAD: Carrega cada PDF em pdf_folder via UnstructuredPDFLoader.load()
    """
    pdf_paths = glob(os.path.join(pdf_folder, "*.pdf"))
    all_docs = []
    for path in pdf_paths:
        loader = UnstructuredPDFLoader(path)
        docs = loader.load()  # retorna lista de Document
        print(f"✅ Loaded {len(docs)} docs from {os.path.basename(path)}")
        all_docs.extend(docs)
    return all_docs

def split_documents(docs, chunk_size: int = 1000, overlap: int = 200):
    """
    SPLIT: divide os Documents em pedaços de até chunk_size tokens,
    com overlap, para melhorar a recuperação.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  Split into {len(chunks)} chunks")
    return chunks

def build_and_save_faiss(chunks, index_dir: str = "faiss_index"):
    """
    EMBED + STORE: gera embeddings e salva o índice FAISS em disco.
    """
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    faiss_index = FAISS.from_documents(chunks, embedder)
    faiss_index.save_local(index_dir)
    print(f"💾 FAISS index written to ./{index_dir}/")

if __name__ == "__main__":
    load_dotenv()  # carrega OPENAI_API_KEY do .env

    # Como estamos fora da pasta selected_paper, ajustamos o caminho:
    PDF_FOLDER = os.path.join("selected_paper")
    INDEX_DIR  = os.path.join("selected_paper", "faiss_index")

    docs   = load_pdfs(PDF_FOLDER)
    chunks = split_documents(docs)
    build_and_save_faiss(chunks, INDEX_DIR)
