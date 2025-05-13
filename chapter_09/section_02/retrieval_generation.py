import os
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


def load_vector_store():
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
    top_docs = retriever.invoke(query)

    for i, doc in enumerate(top_docs, 1):
        print(f"Chunk {i}:")
        print("Content:")
        print(doc.page_content[:500])
        print("\nMetadata:")
        for key, value in doc.metadata.items():
            print(f"- {key}: {value}")
        print("\n" + "=" * 80 + "\n")

    return top_docs


def format_metadata(doc):
    """
    Format metadata from a single Document object as readable text.
    """
    lines = []
    meta = doc.metadata
    if meta.get("title"):
        lines.append(f"Title: {meta['title']}")
    if meta.get("authors"):
        lines.append(f"Authors: {meta['authors']}")
    if meta.get("page") is not None:
        lines.append(f"Page: {meta['page']}")
    if meta.get("source_file"):
        lines.append(f"File: {meta['source_file']}")
    return "\n".join(lines)


def build_context(docs):
    """
    Build the context string with metadata + chunk content to send to the LLM.
    """
    context_chunks = []
    for doc in docs:
        metadata = format_metadata(doc)
        content = doc.page_content.strip()
        block = f"{metadata}\nContent:\n{content}"
        context_chunks.append(block)
    return "\n\n".join(context_chunks)


def build_prompt_template():
    """
    Define the instruction that guides the LLM's behavior.
    """
    template = """
Answer the user's question using ONLY the context below.
If the answer is not in the context, respond with "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate.from_template(template)


def call_llm(prompt, context, question, temperature=0.0):
    """
    Send the prompt, context, and question to the language model and return the result.
    """
    llm = ChatOpenAI(model_name="gpt-4", temperature=temperature)
    pipeline = prompt | llm
    response = pipeline.invoke({"context": context, "question": question})

    if isinstance(response, list):
        response = response[0]
    return response.content.strip()


def generate_answer(context, question: str, temperature: float = 0.0):
    """
    Orchestrates the steps to generate an answer: context building, prompt formatting, and LLM call.
    """
    context = build_context(context)
    prompt = build_prompt_template()
    return call_llm(prompt, context, question, temperature)


if __name__ == "__main__":
    load_dotenv()
    vectore_store = load_vector_store()

    while True:
        query = input("\nEnter your question (or type 'exit' to quit):\n> ")
        if query.lower() in ("exit", "quit"):
            break

        context = retrieve_documents(vectore_store, query, k=5)
        answer = generate_answer(context, query)

        print("\nAnswer:\n", answer)
