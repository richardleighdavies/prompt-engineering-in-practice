import os
import json

from dotenv import load_dotenv
load_dotenv()

from langchain.schema import Document, Generation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.prompts import PromptTemplate


# ─── STEP 1: LOAD ──────────────────────────────────────────────────────────────
def load_menu_items(path: str) -> list[dict]:
    """
    • Load the raw menu JSON from the given path.
    • Return a list of dicts (no Document conversion here).
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items from {path}")
    return items


# ─── STEP 2: SPLIT ─────────────────────────────────────────────────────────────
def split_documents(
    items: list[dict],
    chunk_size: int = 500,
    overlap: int = 50
) -> list[Document]:
    """
    • Convert each dict into a Document.
    • Optionally split Documents longer than chunk_size into smaller chunks.
    """
    # 2.1) Convert each JSON item into a Document
    docs = [
        Document(page_content=f"{item['name']}: {item['description']}")
        for item in items
    ]
    print(f"Converted {len(docs)} items into Document objects")

    # 2.2) Split Documents if they exceed chunk_size
    print(f"Splitting {len(docs)} documents into chunks of size {chunk_size} with overlap {overlap}...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(docs)


# ─── STEP 3: EMBED & STORE ─────────────────────────────────────────────────────
def build_vectorstore(
    splits: list[Document],
    embedding_model: str = "text-embedding-ada-002"
) -> InMemoryVectorStore:
    """
    • Initialize the embedding model.
    • Create an in-memory vector store.
    • Add (embed & index) the split documents.
    """
    embedder = OpenAIEmbeddings(model=embedding_model)
    vs = InMemoryVectorStore(embedder)
    vs.add_documents(splits)
    return vs


# ─── STEP 4: RETRIEVE ──────────────────────────────────────────────────────────
def retrieve_documents(
    vectorstore: InMemoryVectorStore,
    question: str,
    k: int = 3
) -> list[Document]:
    """
    • Create a retriever and fetch the top-k relevant chunks for the question.
    • Uses .invoke() instead of the deprecated get_relevant_documents().
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    # new API: use .invoke() to retrieve documents
    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} chunks for question: {question!r}")
    return docs


# ─── STEP 5: GENERATE ──────────────────────────────────────────────────────────
def generate_answer(
    llm: ChatOpenAI,
    context_chunks: list[Document],
    question: str
) -> str:
    """
    • Build a prompt with PromptTemplate.
    • Chain it with the LLM using the new RunnableSequence API (prompt | llm).
    • Use .invoke() to run, then extract .content from the Generation.
    """
    prompt_template = """\
You are a helpful barista assistant. Use ONLY the information in the context to answer the question.
If you don't know, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    # Join retrieved chunks into one context string
    context = "\n\n".join(doc.page_content for doc in context_chunks)

    # Create PromptTemplate and compose with LLM
    prompt = PromptTemplate.from_template(prompt_template)
    pipeline = prompt | llm

    # Run the pipeline via .invoke() — returns a Generation
    generation = pipeline.invoke({"context": context, "question": question})

    # If you get a list of generations, grab the first one:
    if isinstance(generation, list):
        generation = generation[0]

    # Extract just the text content
    return generation.content


# ─── MAIN ENTRYPOINT ──────────────────────────────────────────────────────────
def main():
    # 0) VERIFY OPENAI API KEY
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY in your .env file")

    # 1) LOAD raw JSON items
    items = load_menu_items("data/menu.json")

    # 2) CONVERT to Documents + SPLIT into chunks
    splits = split_documents(items, chunk_size=500, overlap=50)

    # 3) EMBED & STORE in vector store
    vectorstore = build_vectorstore(splits, embedding_model="text-embedding-ada-002")

    # 4) PREPARE LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

    # 5) TEST with example questions
    example_questions = [
        "What dairy-free drinks do you offer?",
        "Tell me the ingredients of your Mocha.",
        "Which drinks contain chocolate?"
    ]

    for question in example_questions:
        print("\n" + "=" * 60)
        print(f"Question: {question}\n")

        # RETRIEVE relevant chunks
        top_chunks = retrieve_documents(vectorstore, question, k=3)
        print("Retrieved chunks:")
        for chunk in top_chunks:
            print(" -", chunk.page_content)

        # GENERATE an answer
        answer = generate_answer(llm, top_chunks, question)
        print("\nAnswer:")
        print(answer)
        print("=" * 60)


if __name__ == "__main__":
    main()
