import os
from glob import glob
from dotenv import load_dotenv

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def infer_title_from_text(text: str, fallback_filename: str) -> str:
    """
    Try to extract a meaningful title from the first lines of the document.
    If not possible, fall back to the filename (without extension).
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    candidate_lines = [
        line for line in lines[:5]
        if 15 < len(line) < 120 and not any(char.isdigit() for char in line[:10])
    ]
    if candidate_lines:
        return candidate_lines[0]
    return os.path.splitext(fallback_filename)[0]


def load_pdfs(pdf_folder: str):
    """
    LOAD: Load all PDF files in the given folder using UnstructuredPDFLoader.
    Adds metadata such as source filename and inferred title.
    """
    pdf_paths = glob(os.path.join(pdf_folder, "*.pdf"))
    all_docs = []

    for path in pdf_paths:
        loader = UnstructuredPDFLoader(path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = os.path.basename(path)
            doc.metadata["file_path"] = path
            doc.metadata["title"] = infer_title_from_text(doc.page_content, doc.metadata["source_file"])

        print(f"Loaded {len(docs)} documents from: {os.path.basename(path)}")
        all_docs.extend(docs)

    return all_docs


def split_documents(docs, chunk_size: int = 1000, overlap: int = 250):
    """
    SPLIT: Break documents into semantic chunks.
    Metadata is preserved for use in RAG or metadata-aware prompts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks\n")

   
    for i, chunk in enumerate(chunks, 1):
        print(f"=== Chunk {i} ===")
        print("Content preview:")
        print(chunk.page_content[:500])
        print("\nMetadata:")
        for key, value in chunk.metadata.items():
            print(f"- {key}: {value}")
        print("\n" + "=" * 80 + "\n")

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
    print(f"Saved {len(chunks)} chunks to Chroma DB at ./{persist_dir}/")


if __name__ == "__main__":
    load_dotenv()

    print("Loading PDF documents...")
    pdf_folder = os.path.join("selected_paper")
    docs = load_pdfs(pdf_folder)

    print("Splitting documents into chunks...")
    chunks = split_documents(docs)

    print("Building and saving vector index...")
    build_and_save_chroma(chunks)