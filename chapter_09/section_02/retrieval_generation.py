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
    Print content and metadata for inspection.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} chunks for query: {query!r}\n")

    for i, doc in enumerate(docs, 1):
        print(f"Chunk {i}:")
        print("Content:")
        print(doc.page_content[:500])  # print first 500 characters to avoid overload
        print("\nMetadata:")
        for key, value in doc.metadata.items():
            print(f"- {key}: {value}")
        print("\n" + "=" * 80 + "\n")

    return docs

def generate_answer(docs, question: str, temperature: float = 0.0):
    """
    GENERATE: Use a prompt template and a language model to answer a question based on the retrieved documents.
    Include metadata to provide additional context to the answer.
    """
    context = "\n\n".join(doc.page_content for doc in docs)

    template = """
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

    answer_text = generation.content.strip()

    # Extract metadata context
    sources = set()
    for doc in docs:
        title = doc.metadata.get("title")
        source_file = doc.metadata.get("source_file")

        if title and source_file:
            source_entry = f"{title} (file: {source_file})"
        elif title:
            source_entry = title
        elif source_file:
            source_entry = f"File: {source_file}"
        else:
            continue

        sources.add(source_entry)

    if sources:
        source_note = "\n\nContextual sources:\n" + "\n".join(f"• {s}" for s in sorted(sources))
    else:
        source_note = ""

    return answer_text + source_note

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