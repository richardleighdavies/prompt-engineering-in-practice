# Barista Assistant with LangChain

This project implements a simple barista assistant powered by LangChain and OpenAI embeddings. It allows you to load a menu of drinks and snacks, embed their descriptions into a vector store, retrieve relevant items based on a user question, and generate natural language answers using a chat-based LLM.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Project Structure](#project-structure)
7. [Data Preparation](#data-preparation)
8. [Usage](#usage)

   - [Running the Main Script](#running-the-main-script)
   - [Example Queries](#example-queries)

9. [Detailed Workflow](#detailed-workflow)

   - [STEP 1: Load Menu Items](#step-1-load-menu-items)
   - [STEP 2: Split Documents](#step-2-split-documents)
   - [STEP 3: Embed & Store](#step-3-embed--store)
   - [STEP 4: Retrieve](#step-4-retrieve)
   - [STEP 5: Generate Answer](#step-5-generate-answer)

10. [Environment & Virtual Environments](#environment--virtual-environments)
11. [Dependencies](#dependencies)
12. [Customization](#customization)
13. [Troubleshooting](#troubleshooting)
14. [License](#license)

---

## Project Overview

The Barista Assistant project demonstrates how to build a retrieval-augmented question-answering system using:

- **LangChain** for document splitting, vector store management, and prompt composition.
- **OpenAI Embeddings** (`text-embedding-ada-002`) for encoding text into vectors.
- **ChatOpenAI** (`gpt-3.5-turbo`) for generating natural language responses.

Given a static JSON menu, the assistant can answer questions such as “What dairy-free drinks do you offer?” by retrieving relevant menu items and composing concise responses.

## Features

- Load menu items from a JSON file.
- Split long descriptions into manageable chunks with overlap.
- Embed text chunks into an in-memory FAISS vector store.
- Retrieve top-𝑘 relevant chunks for a given question.
- Generate answers using a chat-based LLM, constrained to the retrieved context.
- Zero-configuration beyond installing dependencies and setting an API key.

## Prerequisites

- Python 3.8 or higher
- A valid OpenAI API key
- Internet access for API calls to OpenAI

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/barista-assistant.git
   cd barista-assistant
   ```

2. **(Optional) Create a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .\.venv\\Scripts\\activate  # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Configuration

1. **Create a `.env` file** in the project root:

   ```ini
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **Ensure** that the `data/menu.json` file is present. It contains an array of menu item objects with `id`, `name`, and `description` fields.

## Project Structure

```
barista-assistant/
├── data/
│   └── menu.json          # Sample menu items
├── src/
│   ├── main.py            # Entry point with load, split, embed, retrieve, answer steps
│   └── utils.py           # (Optional) Helper modules if refactored
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
└── README.md              # You are here
```

## Data Preparation

The only data source is `data/menu.json`. It should follow this schema:

```json
[
  {"id": 1, "name": "Cappuccino", "description": "Espresso, steamed milk and creamy foam."},
  {"id": 2, "name": "Mocha", "description": "Espresso, hot chocolate and whipped cream."},
  ...
]
```

No further preprocessing is required; the code will load and convert each entry into a `langchain.schema.Document` automatically.

## Usage

### Running the Main Script

In the project root, execute:

```bash
python src/main.py
```

This script:

1. Verifies the OpenAI API key.
2. Loads and prints the number of menu items.
3. Splits documents into chunks (default chunk size 500 characters, 50 overlap).
4. Builds an in-memory FAISS vector store and embeds each chunk.
5. Sets up a zero-temperature `gpt-3.5-turbo` chat model.
6. Runs example queries to demonstrate retrieval and answer generation.

### Example Queries

The script includes three sample questions:

- "What dairy-free drinks do you offer?"
- "Tell me the ingredients of your Mocha."
- "Which drinks contain chocolate?"

You can modify or extend `example_questions` in `main.py` to test other queries.

## Detailed Workflow

### STEP 1: Load Menu Items

```python
def load_menu_items(path: str) -> list[dict]:
    # Reads data/menu.json and returns a list of dicts
```

- Uses Python’s built-in `json` module.
- Prints the count of loaded items.

### STEP 2: Split Documents

```python
def split_documents(items: list[dict], chunk_size: int, overlap: int) -> list[Document]:
    # Converts each item to a Document, then uses RecursiveCharacterTextSplitter
```

- Converts each menu item into a `Document` with combined `name` and `description`.
- Splits any document exceeding `chunk_size` into overlapping chunks of `chunk_size` with `chunk_overlap`.

### STEP 3: Embed & Store

```python
def build_vectorstore(splits: list[Document], embedding_model: str) -> InMemoryVectorStore:
    # Initializes OpenAIEmbeddings and InMemoryVectorStore, then adds documents
```

- Uses `text-embedding-ada-002` by default.
- Stores embeddings in an in-memory FAISS index via `langchain-core.vectorstores.InMemoryVectorStore`.

### STEP 4: Retrieve

```python
def retrieve_documents(vectorstore: InMemoryVectorStore, question: str, k: int) -> list[Document]:
    # Retrieves top-k relevant document chunks for the question
```

- Converts the vector store into a retriever with `search_kwargs={"k": k}`.
- Calls `.invoke(question)` on the retriever to return the top-k matches.

### STEP 5: Generate Answer

```python
def generate_answer(llm: ChatOpenAI, context_chunks: list[Document], question: str) -> str:
    # Builds a PromptTemplate, composes with ChatOpenAI, and invokes
```

- Constructs a prompt requiring the model to use **only** provided context.
- Joins chunk texts into a single context string.
- Executes the prompt–LLM pipeline with `.invoke()`.
- Returns the `.content` of the generation.

## Environment & Virtual Environments

- It is recommended to isolate dependencies in a Python virtual environment.
- Use `python -m venv .venv` or tools like `pipenv` or `poetry` if preferred.

## Dependencies

The following Python packages are required (see `requirements.txt`):

```
openai
python-dotenv
tiktoken
langchain>=0.0.331
langchain-openai
langchain-core
faiss-cpu
```

Install them via:

```bash
pip install -r requirements.txt
```

## Customization

- **Chunk Size & Overlap:** Adjust in `split_documents()` to optimize retrieval granularity.
- **Embedding Model:** Swap out `text-embedding-ada-002` for newer models if available.
- **LLM Parameters:** Modify `ChatOpenAI(model_name, temperature, ...)` to control creativity.
- **Vector Store Backend:** Replace `InMemoryVectorStore` with a persistent store (e.g., FAISS on disk, Redis).

## Troubleshooting

- **API Key Errors:** Ensure `OPENAI_API_KEY` is set in your environment or `.env`.
- **Missing Data:** Verify `data/menu.json` exists and is valid JSON.
- **Slow Embeddings:** Check network connectivity and OpenAI rate limits.
- **FAISS Install Issues:** On some platforms, `faiss-cpu` may require extra system libraries.

## License

This project is released under the MIT License. See `LICENSE` for details.
