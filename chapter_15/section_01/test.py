import os
import json
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import evaluate
from bug_triage import BugTriageAssistant

# python test.py

# Initialize the BugTriageAssistant
triager = BugTriageAssistant()

# Sample bug reports for testing
bug_reports = [
    {
        "input": "The application crashes when clicking the 'Save' button after entering data.",
        "expected_summary_criteria": "Is the summary concise and focused on the core issue?",
        "expected_triage_criteria": "Does the triage data accurately reflect the bug's priority, component, issue type, and suggested next step?"
    },
    {
        "input": "Users report a delay in loading the dashboard after login.",
        "expected_summary_criteria": "Is the summary concise and focused on the core issue?",
        "expected_triage_criteria": "Does the triage data accurately reflect the bug's priority, component, issue type, and suggested next step?"
    }
]

# Define metrics
summary_metric = GEval(
    name="Summary Conciseness",
    criteria="Assess whether the summary is concise and focused only on the essential points of the bug report.",
    threshold=0.9,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)

triage_metric = GEval(
    name="Triage Data Accuracy",
    criteria="Assess whether the triage data accurately reflects the bug's priority, component, issue type, and suggested next step.",
    threshold=0.9,
     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)

# Create test cases
summary_test_cases = []
triage_test_cases = []

for report in bug_reports:
    summary = triager.summarize_bug(report["input"])
    triage_data = triager.classify_bug(report["input"])

    summary_test_case = LLMTestCase(
        input=report["input"],
        actual_output=summary
    )
    triage_test_case = LLMTestCase(
        input=report["input"],
        actual_output=json.dumps(triage_data)
    )

    summary_test_cases.append(summary_test_case)
    triage_test_cases.append(triage_test_case)

# Run evaluations
print("Evaluating summaries...")
evaluate(test_cases=summary_test_cases, metrics=[summary_metric])

print("\nEvaluating triage data...")
evaluate(test_cases=triage_test_cases, metrics=[triage_metric])