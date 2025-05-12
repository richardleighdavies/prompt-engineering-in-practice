import os
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


def load_chroma():
    """
    RETRIEVE: Load the Chroma vector database from disk using the specified embedding model.
    """
    persist_dir = "vectorstore"
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

    db_client = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedder
    )
    print(f"Loaded Chroma vector database from: {persist_dir}")
    return db_client


def retrieve_documents(vectorstore, query: str, k: int = 5):
    """
    RETRIEVE: Fetch the top-k most relevant document chunks for the given query.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} chunks for query: {query!r}")
    return docs


def generate_answer(docs, question: str, temperature: float = 0.0):
    """
    GENERATE: Use a prompt template and a language model to answer a question based on the retrieved documents.
    """
    context = "\n\n".join(doc.page_content for doc in docs)

    template = """\
You are a science co-pilot. Use ONLY the context below to answer the question.
If the answer is not contained in the context, respond "I don't know."

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4", temperature=temperature)

    pipeline = prompt | llm
    generation = pipeline.invoke({"context": context, "question": question})

    if isinstance(generation, list):
        generation = generation[0]

    return generation.content.strip()


if __name__ == "__main__":
    load_dotenv()
    chroma_db = load_chroma()

    while True:
        query = input("\nEnter your question (or type 'exit' to quit):\n> ")
        if query.lower() in ("exit", "quit"):
            break

        docs = retrieve_documents(chroma_db, query, k=5)
        answer = generate_answer(docs, query)

        print("\nAnswer:\n", answer)
