#!/usr/bin/python3
""" """

# Standard Library
import json

from pathlib import Path

# Third Party
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

# Local
import schemas

from services.language_model.service import LanguageModelService

relative_path = Path(__file__).resolve().parent

config = dotenv_values(".env")


def load_markdown_file(file_path: Path) -> str:
    with open(file_path, "r") as file:
        content = file.read()
    return content


def main():

    language_model = ChatOpenAI(
        api_key=config["OPENAI_API_KEY"],
        model=config["OPENAI_MODEL_NAME"],
        temperature=config["OPENAI_TEMPERATURE"],
    )

    language_model_service = LanguageModelService(language_model)

    messages: list[dict[str, str]] = []

    markdown_file_path = relative_path / "data/input_02.md"

    markdown_file_content = load_markdown_file(markdown_file_path)

    user_message = {"role": "user", "content": markdown_file_content}

    messages.append(user_message)

    print(f"User:\n\n{user_message['content']}", end="\n\n")

    agent_message = language_model_service.generate_structured_text_two_steps(
        messages=messages,
        system_prompt_template_path_01=relative_path / "prompt_templates/system_prompt_template_01.md",
        system_prompt_template_path_02=relative_path / "prompt_templates/system_prompt_template_02.md",
        output_schema=schemas.StructuredOutputSchema,
    )

    print(f"Agent:\n\n{json.dumps(agent_message, indent=4)}", end="\n\n")


if __name__ == "__main__":
    main()
