#!/usr/bin/python3

# Standard Library Imports
from pathlib import Path

# Third Party Imports
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate

relative_path = Path(__file__).resolve().parent

config = dotenv_values(".env")

language_model = ChatOpenAI(
    api_key=config["OPENAI_API_KEY"],
    model="gpt-4o",
    temperature=0.7,
)


def generate_text(
    system_prompt_template_path: Path,
    user_message: dict[str, str],
    system_prompt_template_input_variables: dict[str, str] = {},
) -> dict[str, str]:

    def get_system_prompt():
        system_prompt_template = SystemMessagePromptTemplate.from_template_file(
            template_file=system_prompt_template_path,
            input_variables=list(system_prompt_template_input_variables.keys()),
        )
        return system_prompt_template.format(**system_prompt_template_input_variables).content

    def create_messages(system_prompt, user_message):
        messages = [{"role": "system", "content": system_prompt}]
        messages.append(user_message)
        return messages

    def generate_response(language_model, messages):
        response = language_model.invoke(messages)
        return {"role": "agent", "content": response.content}

    system_prompt = get_system_prompt()

    messages = create_messages(system_prompt, user_message)

    response = generate_response(language_model, messages)

    return response


def main():

    user_message = {"role": "user", "content": input("User Message: ")}
    print(end="\n\n")

    agent_message = generate_text(
        system_prompt_template_path=relative_path / "prompt_templates/system_prompt_template.md",
        user_message=user_message,
    )

    print(f"Agent Message: {agent_message['content']}", end="\n\n")


if __name__ == "__main__":
    main()
