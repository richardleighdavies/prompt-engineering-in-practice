import os
from glob import glob
from dotenv import load_dotenv

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def load_pdfs(pdf_folder: str):
    """
    LOAD: Load each PDF in pdf_folder using UnstructuredPDFLoader.load()
    """
    pdf_paths = glob(os.path.join(pdf_folder, "*.pdf"))
    all_docs = []
    for path in pdf_paths:
        loader = UnstructuredPDFLoader(path)
        docs = loader.load()  # returns a list of Document objects
        print(f"✅ Loaded {len(docs)} docs from {os.path.basename(path)}")
        all_docs.extend(docs)
    return all_docs


def split_documents(docs, chunk_size: int = 2500, overlap: int = 250):
    """
    SPLIT: Break Documents into chunks of up to chunk_size tokens,
    with overlap to improve retrieval accuracy.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  Split into {len(chunks)} chunks")
    return chunks


def build_and_save_chroma(chunks):
    """
    EMBED + STORE: Generate embeddings and save the Chroma index to disk.
    """   
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    persist_dir = os.path.join("vectorstore") 
   
    chroma_index = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persist_dir
    )

    chroma_index.persist()

    print(f"💾 Chroma index written to ./{persist_dir}/")


if __name__ == "__main__":
    load_dotenv()  
   
    PAPERS = os.path.join("selected_paper")    

    docs   = load_pdfs(PAPERS)
    chunks = split_documents(docs)
    build_and_save_chroma(chunks)
