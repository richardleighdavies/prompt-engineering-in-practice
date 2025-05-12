import os
from glob import glob
from dotenv import load_dotenv

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def load_pdfs(pdf_folder: str):
    """
    LOAD: Load all PDF files in the given folder using UnstructuredPDFLoader.
    Adds metadata such as source filename and page number.
    """
    pdf_paths = glob(os.path.join(pdf_folder, "*.pdf"))
    all_docs = []
    for path in pdf_paths:
        loader = UnstructuredPDFLoader(path)
        docs = loader.load()

        # Add useful metadata to each document
        for doc in docs:
            doc.metadata["source_file"] = os.path.basename(path)
            doc.metadata["file_path"] = path

            # Optional heuristic to infer the title from the first line
            if "title" not in doc.metadata or not doc.metadata["title"]:
                first_line = doc.page_content.strip().split("\n")[0]
                if 10 < len(first_line) < 120:
                    doc.metadata["title"] = first_line

        print(f"Loaded {len(docs)} documents from {os.path.basename(path)}")
        all_docs.extend(docs)
    return all_docs

def split_documents(docs, chunk_size: int = 2500, overlap: int = 250):
    """
    SPLIT: Break the documents into smaller chunks to prepare them for vectorization.
    Metadata from each document is preserved in each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def build_and_save_chroma(chunks):
    """
    EMBED + STORE: Generate vector embeddings from the chunks
    and persist them to disk using Chroma.
    """
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    persist_dir = os.path.join("vectorstore")

    chroma_index = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persist_dir
    )
    chroma_index.persist()
    print(f"Chroma index saved to ./{persist_dir}/")

if __name__ == "__main__":
    load_dotenv()

    pdf_folder = os.path.join("selected_paper")
    docs = load_pdfs(pdf_folder)
    chunks = split_documents(docs)
    build_and_save_chroma(chunks)