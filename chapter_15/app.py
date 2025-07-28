import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import Any

load_dotenv()

class MeetingSummarizer:
    def __init__(
        self, 
        model: str = "gpt-4", 
        system_prompt: str = "",
    ):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))       
        self.summary_system_prompt = """
            You are an AI assistant summarizing meeting transcripts. Provide a clear and 
            concise summary of the following conversation, avoiding interpretation and 
            unnecessary details. Focus on the main discussion points only. Do not include 
            any action items. Respond with only the summary as plain text — no headings, 
            formatting, or explanations.
        """       
        self.action_item_prompt = """
            Extract all action items from the following meeting transcript. Identify individual 
            and team-wide action items in the following format:

            {
              "individual_actions": {
                "Alice": ["Task 1", "Task 2"],
                "Bob": ["Task 1"]
              },
              "team_actions": ["Task 1", "Task 2"],
              "entities": ["Alice", "Bob"]
            }

            Only include what is explicitly mentioned. Do not infer. You must respond strictly in 
            valid JSON format — no extra text or commentary.
        """   

    def get_summary(self, transcript: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.summary_system_prompt},
                    {"role": "user", "content": transcript}
                ]
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error: Could not generate summary due to API issue: {e}"

    def get_action_items(self, transcript: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.action_item_prompt},
                    {"role": "user", "content": transcript}
                ]
            )
            action_items = response.choices[0].message.content.strip()
            try:
                return json.loads(action_items)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON returned from model", "raw_output": action_items}
        except Exception as e:
            print(f"Error generating action items: {e}")
            return {"error": f"API call failed: {e}", "raw_output": ""}

    def summarize(self, transcript: str) -> tuple[str, dict[str, Any]]:
        summary = self.get_summary(transcript)
        action_items = self.get_action_items(transcript)
        return summary, action_items


# RUNNING THE SUMMARIZER
if __name__ == "__main__":
    summarizer = MeetingSummarizer()

    with open("meeting_transcript.txt", "r") as file:
        transcript = file.read().strip()

    summary, action_items = summarizer.summarize(transcript)
    print(summary)
    print("JSON:")
    print(json.dumps(action_items, indent=2))
