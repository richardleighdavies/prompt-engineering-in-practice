import os
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate
from app import MeetingSummarizer 

DOCUMENTS_PATH = "path/to/documents/folder" 
THRESHOLD = 0.9

summary_concision = GEval(
    name="Summary Concision",
    criteria=(
        "Assess whether the summary is concise and focused only on the essential points of the meeting. "
        "It should avoid repetition, irrelevant details, and unnecessary elaboration."
    ),
    threshold=THRESHOLD,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

action_item_check = GEval(
    name="Action Item Accuracy",
    criteria=(
        "Are the action items accurate, complete, and clearly reflect the key tasks or follow-ups mentioned in the meeting?"
    ),
    threshold=THRESHOLD,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

transcripts = []
for filename in os.listdir(DOCUMENTS_PATH):
    if filename.endswith(".txt"):
        with open(os.path.join(DOCUMENTS_PATH, filename), "r", encoding="utf-8") as file:
            transcripts.append(file.read().strip())

summarizer = MeetingSummarizer()

summary_test_cases = []
action_item_test_cases = []

for transcript in transcripts:
    summary, action_items = summarizer.summarize(transcript)

    summary_test_cases.append(
        LLMTestCase(
            input=transcript,
            actual_output=summary
        )
    )

    action_item_test_cases.append(
        LLMTestCase(
            input=transcript,
            actual_output=str(action_items)
        )
    )


print("Running evaluation for summaries...")
evaluate(test_cases=summary_test_cases, metrics=[summary_concision])

print("\nRunning evaluation for action items...")
evaluate(test_cases=action_item_test_cases, metrics=[action_item_check])
