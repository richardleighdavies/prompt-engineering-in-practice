import pickle
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from rag import RAGAgent

# Carrega os goldens previamente salvos
with open("goldens.pkl", "rb") as f:
    goldens = pickle.load(f)

agent = RAGAgent("data/menu.json")

metrics = [
    ContextualRelevancyMetric(),
    ContextualRecallMetric(),
    ContextualPrecisionMetric(),
]

retriever_test_cases = []
for golden in goldens:
    retrieved_docs = agent.retrieve(golden.input)

    # ⬇️ Transforma os Document em lista de strings
    retrieval_context_strs = [doc.page_content for doc in retrieved_docs]

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=golden.expected_output,
        expected_output=golden.expected_output,
        retrieval_context=retrieval_context_strs  # <- corrigido aqui
    )
    retriever_test_cases.append(test_case)

evaluate(retriever_test_cases, metrics)

# deepeval test run test_retriever.py