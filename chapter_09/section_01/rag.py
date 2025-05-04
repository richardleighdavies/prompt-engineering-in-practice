import os
import json

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import InMemoryVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import LLMChain

# ─── STEP 1: LOAD ──────────────────────────────────────────────────────────────
def load_menu_as_docs(path: str) -> list[Document]:
    """
    • Loads the menu JSON
    • Returns a list of Documents, each with "Name: Description" as page_content
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return [Document(page_content=f"{i['name']}: {i['description']}") for i in items]


# ─── STEP 2: SPLIT ─────────────────────────────────────────────────────────────
def split_documents(
    docs: list[Document],
    chunk_size: int = 500,
    overlap: int = 50
) -> list[Document]:
    """
    • Breaks each Document into smaller chunks
    • Uses recursive splitting to respect sentence/paragraph boundaries
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)


# ─── STEP 3: EMBED & STORE ─────────────────────────────────────────────────────
def build_vectorstore(
    splits: list[Document],
    embedding_model: str = "text-embedding-ada-002"
) -> InMemoryVectorStore:
    """
    • Initializes embeddings
    • Creates an in-memory vector store
    • Adds (embeds & indexes) the split documents
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
    • Uses the vectorstore retriever to fetch top-k relevant chunks for the question
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(question)


# ─── STEP 5: GENERATE ──────────────────────────────────────────────────────────
def generate_answer(
    llm: ChatOpenAI,
    context_chunks: list[Document],
    question: str
) -> str:
    """
    • Builds and runs an LLMChain with a fixed prompt template
    • {context} and {question} are injected into the prompt
    """
    default_prompt = """
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

    # Create PromptTemplate and LLMChain
    prompt = PromptTemplate.from_template(default_prompt)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain
    return chain.run({"context": context, "question": question})


# ─── MAIN ENTRYPOINT ──────────────────────────────────────────────────────────
def main():
    # 0) VERIFY OPENAI KEY
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

    # 1) LOAD
    docs = load_menu_as_docs("data/menu.json")

    # 2) SPLIT
    splits = split_documents(docs, chunk_size=500, overlap=50)

    # 3) EMBED & STORE
    vectorstore = build_vectorstore(splits, embedding_model="text-embedding-ada-002")

    # 4) Prepare LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

    # 5) RUN & GENERATE for example questions
    example_questions = [
        "What dairy-free drinks do you offer?",
        "Tell me the ingredients of your Mocha.",
        "Which drinks contain chocolate?"
    ]

    for q in example_questions:
        print("\n" + "="*60)
        print(f"Question: {q}\n")

        # RETRIEVE
        top_chunks = retrieve_documents(vectorstore, q, k=3)
        print("Retrieved chunks:")
        for doc in top_chunks:
            print(" -", doc.page_content)

        # GENERATE
        answer = generate_answer(llm, top_chunks, q)
        print("\nAnswer:")
        print(answer)
        print("="*60)

if __name__ == "__main__":
    main()
