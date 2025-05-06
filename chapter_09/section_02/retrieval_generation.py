# retrieval_generation.py

import os
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

def load_faiss(index_dir: str = "faiss_index"):
    """
    RETRIEVE: Load the FAISS index from disk using the same embedding model.
    """
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    faiss_index = FAISS.load_local(
        index_dir,
        embedder,
        allow_dangerous_deserialization=True
    )
    print(f"🔍 Loaded FAISS index from ./{index_dir}/")
    return faiss_index

def retrieve_documents(faiss_index, query: str, k: int = 5):
    """
    RETRIEVE: Fetch the top‐k most relevant chunks for the given query.
    """
    retriever = faiss_index.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)  # new API
    print(f"📄 Retrieved {len(docs)} chunks for query: {query!r}")
    return docs

def generate_answer(docs, question: str, temperature: float = 0.0):
    """
    GENERATE: Build a prompt chain, call the LLM, and return the answer.
    """
    # 1) Assemble context text
    context = "\n\n".join(doc.page_content for doc in docs)

    # 2) Create a PromptTemplate
    template = """\
You are a science co-pilot. Use ONLY the context below to answer the question.
If the answer is not contained in the context, respond "I don't know."

Context:
{context}

Question:
{question}

Answer:"""
    prompt = PromptTemplate.from_template(template)

    # 3) Initialize the Chat model
    llm = ChatOpenAI(model_name="gpt-4", temperature=temperature)

    # 4) Compose the pipeline via the '|' operator
    pipeline = prompt | llm

    # 5) Invoke with inputs as a dict
    generation = pipeline.invoke({"context": context, "question": question})

    # 6) If you get a list, pick the first
    if isinstance(generation, list):
        generation = generation[0]

    # 7) Extract the content
    return generation.content.strip()

if __name__ == "__main__":
    load_dotenv()  # loads OPENAI_API_KEY

    INDEX_DIR = "faiss_index"
    faiss_index = load_faiss(INDEX_DIR)

    # Interactive query loop
    while True:
        query = input("\n❓ Enter your question (or 'exit' to quit):\n> ")
        if query.lower() in ("exit", "quit"):
            break
        docs   = retrieve_documents(faiss_index, query, k=5)
        answer = generate_answer(docs, query)
        print("\n🤖 Answer:\n", answer)
