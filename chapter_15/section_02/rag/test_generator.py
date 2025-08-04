# test_generator.py

import pickle
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.evaluate import evaluate
from deepeval.metrics import GEval
from rag import RAGAgent  


with open("goldens.pkl", "rb") as f:
    goldens = pickle.load(f)

agent = RAGAgent("data/menu.json")

metrics = [
    GEval(
        name="Answer Correctness",
        criteria="Evaluate if the actual output is correct and complete from the input and retrieved context.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT
        ]
    )
]

generator_test_cases = []
for golden in goldens:
    retrieved_docs = agent.retrieve(golden.input)
    generated_answer = agent.generate(golden.input, retrieved_docs)
   
    retrieval_context_strs = [doc.page_content for doc in retrieved_docs]

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=generated_answer,
        expected_output=golden.expected_output,
        retrieval_context=retrieval_context_strs
    )
    generator_test_cases.append(test_case)

evaluate(generator_test_cases, metrics)

# deepeval test run test_generator.py