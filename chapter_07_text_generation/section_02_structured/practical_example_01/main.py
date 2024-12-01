#!/usr/bin/python3

# Standard Library Imports
from pathlib import Path

# Third Party Imports
import pydantic

from dotenv import dotenv_values
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

# Local Imports
import schemas

relative_path = Path(__file__).resolve().parent

config = dotenv_values(".env")

language_model_unstructured = ChatOpenAI(
    api_key=config["OPENAI_API_KEY"],
    model="gpt-4o",
    temperature=0.0,
)


def generate_text(
    system_prompt_template_path: Path,
    user_message: dict[str, str],
    system_prompt_template_input_variables: dict[str, str] = {},
    is_structured_output: bool = False,
    output_schema: pydantic.BaseModel = None,
) -> dict[str, str]:

    def setup_language_model(output_schema, output_format_instructions=""):
        if is_structured_output:
            pydantic_output_parser = PydanticOutputParser(pydantic_object=output_schema)
            output_format_instructions = pydantic_output_parser.get_format_instructions()
            language_model = language_model_unstructured.with_structured_output(schema=output_schema, method="json_mode")
        else:
            language_model = language_model_unstructured

        return language_model, output_format_instructions

    def get_system_prompt(output_format_instructions):
        system_prompt_template = SystemMessagePromptTemplate.from_template_file(
            template_file=system_prompt_template_path,
            partial_variables={"output_format_instructions": output_format_instructions},
            input_variables=list(system_prompt_template_input_variables.keys()),
        )
        return system_prompt_template.format(**system_prompt_template_input_variables).content

    def create_messages(system_prompt, user_message):
        messages = [{"role": "system", "content": system_prompt}]
        messages.append(user_message)
        return messages

    def generate_response(language_model, messages):
        response = language_model.invoke(messages)
        content = response.model_dump() if is_structured_output else response.content
        return {"role": "agent", "content": content}

    language_model, output_format_instructions = setup_language_model(output_schema)

    system_prompt = get_system_prompt(output_format_instructions)

    messages = create_messages(system_prompt, user_message)

    response = generate_response(language_model, messages)

    return response


def main():

    user_message = {"role": "user", "content": input("User Message: ")}
    print(end="\n\n")

    agent_message = generate_text(
        system_prompt_template_path=relative_path / "prompt_templates/system_prompt_template.md",
        user_message=user_message,
        is_structured_output=True,
        output_schema=schemas.StructuredOutputSchema,
    )

    print(f"Agent Message: {agent_message['content']}", end="\n\n")


if __name__ == "__main__":
    main()
