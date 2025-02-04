#!/usr/bin/python3
""" """

# Standard Library
from pathlib import Path

# Third Party
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

# Local
from services.language_model.service import LanguageModelService

relative_path = Path(__file__).resolve().parent

config = dotenv_values(".env")


def main():

    language_model = ChatOpenAI(
        api_key=config["OPENAI_API_KEY"],
        model=config["OPENAI_MODEL_NAME"],
        temperature=config["OPENAI_TEMPERATURE"],
    )

    language_model_service = LanguageModelService(language_model)

    messages: list[dict[str, str]] = []

    user_message: str = input("User Message: ")

    messages.append({"role": "user", "content": user_message})

    agent_message: str = language_model_service.generate_unstructured_text(
        messages=messages,
        system_prompt_template_path=relative_path / "prompt_templates/system_prompt_template.md",
    )

    print(f"\nAgent Message: {agent_message}", end="\n\n")

    messages.append({"role": "assistant", "content": agent_message})

    user_message: str = input("User Message: ")

    messages.append({"role": "user", "content": user_message})

    agent_message: str = language_model_service.generate_unstructured_text(
        messages=messages,
        system_prompt_template_path=relative_path / "prompt_templates/system_prompt_template.md",
    )

    print(f"\nAgent Message: {agent_message}", end="\n\n")


if __name__ == "__main__":
    main()
