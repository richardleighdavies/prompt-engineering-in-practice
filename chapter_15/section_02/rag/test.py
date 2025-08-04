# test_generator.py

from rag import RAGAgent
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import pickle

with open("goldens.pkl", "rb") as f:
    goldens = pickle.load(f)

agent = RAGAgent("data/menu.json")

metrics = [
    GEval(
        name="Answer Correctness",
        criteria="Evaluate if the actual output is correct and complete from the input and context.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT
        ]
    )
]

test_cases = []
for golden in goldens:
    retrieved_docs = agent.retrieve(golden.input)
    answer = agent.generate(golden.input, retrieved_docs)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=answer,
        expected_output=golden.expected_output,
        retrieval_context=retrieved_docs
    )
    test_cases.append(test_case)

from deepeval.evaluate import evaluate
evaluate(test_cases, metrics)
