#!/usr/bin/python3
"""Script to generate text using a language model."""

# Standard Library Packages
from pathlib import Path

# Third Party Packages
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load configuration from .env file
config = dotenv_values(".env")

# Initialize the language model
language_model = ChatOpenAI(
    api_key=config['OPENAI_API_KEY'],
    model="gpt-4o",
    temperature=0.7,
)


def generate_text(
    system_prompt_template_path: Path,
    user_prompt_template_path: Path,
    input_variables: dict[str, str],
) -> str:
    """Generates text based on the provided prompt templates and input variables."""

    # Load and format the system prompt template
    system_prompt_template = SystemMessagePromptTemplate.from_template_file(
        template_file=system_prompt_template_path,
        input_variables=[]
    )
    system_prompt = system_prompt_template.format()

    # Load and format the user prompt template
    user_prompt_template = HumanMessagePromptTemplate.from_template_file(
        user_prompt_template_path,
        input_variables=list(input_variables.keys())
    )
    user_prompt = user_prompt_template.format(**input_variables)

    # Prepare the messages for the language model
    messages = [
        {"role": "system", "content": system_prompt.content},
        {"role": "user", "content": user_prompt.content},
    ]

    # Invoke the language model and return the response
    response = language_model.invoke(messages)

    return response.content


input_variables = {
    "topic": "REST API Authentication Strategies",
    "level": "Intermediate",
    "audience": "Backend Development Professionals",
    "version": "Version 2.0",
    "language": "Python",
    "context": "Emphasis on JWT Implementation and Security Best Practices for Robust API Authentication"
}

# Generate text using the specified prompt templates and input variables
response = generate_text(
    system_prompt_template_path=Path("./prompt_templates/system_prompt_template.md"),
    user_prompt_template_path=Path("./prompt_templates/user_prompt_template.md"),
    input_variables=input_variables,
)

# Print the generated response
print(f"Response: {response}")
