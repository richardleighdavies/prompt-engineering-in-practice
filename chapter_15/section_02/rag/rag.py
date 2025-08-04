import os
import json

from dotenv import load_dotenv
load_dotenv()

from langchain.schema import Document, Generation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.prompts import PromptTemplate


def load_menu_items(path: str) -> list[dict]:
    """
    • Load the raw menu JSON from the given path.
    • Return a list of dicts (no Document conversion here).
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items from {path}")
    return items

def split_documents(
    items: list[dict],
    chunk_size: int = 500,
    overlap: int = 50
) -> list[Document]:
    """
    • Convert each dict into a Document.
    • Optionally split Documents longer than chunk_size into smaller chunks.
    """   
    docs = [
        Document(page_content=f"{item['name']}: {item['description']}")
        for item in items
    ]
    print(f"Converted {len(docs)} items into Document objects")   
    print(f"Splitting {len(docs)} documents into chunks of size {chunk_size} with overlap {overlap}...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(docs)

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
   
    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} chunks for question: {question!r}")
    return docs

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
    context = "\n\n".join(doc.page_content for doc in context_chunks)    
    prompt = PromptTemplate.from_template(prompt_template)
    pipeline = prompt | llm  
    generation = pipeline.invoke({"context": context, "question": question})   
    if isinstance(generation, list):
        generation = generation[0]

    return generation.content

class RAGAgent:
    def __init__(self, document_path: str):
        items = load_menu_items(document_path)
        splits = split_documents(items)
        self.vectorstore = build_vectorstore(splits)
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Please set OPENAI_API_KEY in your .env file") 

    def retrieve(self, query: str):
        return retrieve_documents(self.vectorstore, query)

    def generate(self, query: str, context_chunks):
        return generate_answer(self.llm, context_chunks, query)

def main():    
    agent = RAGAgent("data/menu.json")

    queries = [
        "What dairy-free drinks do you offer?",
        "Tell me the ingredients of your Mocha.",
        "Which drinks contain chocolate?"
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}\n")
      
        top_chunks = agent.retrieve(query)

        print("Retrieved chunks:")
        for chunk in top_chunks:
            print(" -", chunk.page_content)
        
        answer = agent.generate(query, top_chunks)
        print("\nAnswer:")
        print(answer)
        print("=" * 60)

if __name__ == "__main__":
    main()
