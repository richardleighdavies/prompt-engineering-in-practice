import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import Any

load_dotenv()

class BugTriageAssistant:
    def __init__(
        self, 
        model: str = "gpt-4", 
    ):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))       

        self.summary_prompt = """
        You are a senior software engineer helping triage incoming bug reports.
        Read the following bug report and summarize the core issue in 1-3 sentences.
        Avoid unnecessary details. Respond with only the plain text summary — no explanations or formatting.
        """

        self.triage_prompt = """
        You are an AI assistant for bug triage. Your task is to extract structured triage data from the following bug report.

        Return only a valid JSON object with the following fields:
        {
          "priority": "high | medium | low",
          "component": "affected module or feature",
          "issue_type": "bug | feature_request | question",
          "suggested_next_step": "brief suggestion or comment on next step"
        }

        Base your response strictly on what is mentioned in the input. If unsure, respond with "unknown" for that field. 
        Respond ONLY with a valid JSON object — no explanations or formatting.
        """

    def summarize_bug(self, bug_report: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.summary_prompt},
                    {"role": "user", "content": bug_report}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating bug summary: {e}")
            return f"Error: Could not generate summary: {e}"

    def classify_bug(self, bug_report: str) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.triage_prompt},
                    {"role": "user", "content": bug_report}
                ]
            )
            triage_result = response.choices[0].message.content.strip()
            try:
                return json.loads(triage_result)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON returned", "raw_output": triage_result}
        except Exception as e:
            print(f"Error classifying bug: {e}")
            return {"error": f"API call failed: {e}", "raw_output": ""}

    def triage(self, bug_report: str) -> tuple[str, dict[str, Any]]:
        summary = self.summarize_bug(bug_report)
        triage_data = self.classify_bug(bug_report)
        return summary, triage_data


# RUNNING THE BUG TRIAGE
if __name__ == "__main__":
    triager = BugTriageAssistant()

    current_dir = os.path.dirname(__file__)
    bug_report_path = os.path.join(current_dir, "bug_reports", "bug_report.txt")
    
    if not os.path.exists(bug_report_path):
        print(f"Erro: O arquivo {bug_report_path} não foi encontrado.")
        exit(1)

    with open(bug_report_path, "r") as file:
        bug_report = file.read().strip()    

    summary, triage_info = triager.triage(bug_report)
    print("Summary:")
    print(summary)
    print("\nTriage Info:")
    print(json.dumps(triage_info, indent=2))
