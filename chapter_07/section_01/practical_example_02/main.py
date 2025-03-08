#!/usr/bin/python3
"""
A script that demonstrates structured output generation from language models using LangChain.
This example shows a two-step approach to generate structured responses from markdown input,
with validation through Pydantic schemas.
"""

# Standard Library
import json  # For formatting structured output
from pathlib import Path  # For cross-platform file path handling

# Third Party
from langchain_openai import ChatOpenAI  # LangChain's wrapper for OpenAI's chat models
from pydantic_settings import BaseSettings, SettingsConfigDict  # For type-safe configuration

# Local
import schemas  # Contains Pydantic models for structured output validation
from services.language_model.service import LanguageModelService  # Our abstraction for LLM interactions

# Get the directory where this script is located to find relative paths
relative_path = Path(__file__).resolve().parent


class Config(BaseSettings):
    """
    Configuration class that loads environment variables from .env file.
    Provides type-safe access to configuration values needed for the application.
    """
    # Tell Pydantic to load from .env file
    model_config = SettingsConfigDict(env_file=".env")

    # Required API key for OpenAI
    OPENAI_API_KEY: str
    
    # Model name with default value - can be overridden in .env
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    
    # Temperature controls randomness (0.0 = deterministic, 1.0 = creative)
    OPENAI_TEMPERATURE: float = 0.0


def load_file(file_path: Path) -> str:
    """
    Helper function to load content from a file.
    """
    with open(file_path, "r") as file:
        content = file.read()
    return content


def main():
    """
    Main function that initializes the language model, creates a service,
    and processes markdown input to generate structured JSON output.
    """
    # ----- Setup -----
    
    # Load configuration from environment variables
    config = Config()

    # Initialize the OpenAI chat model with our configuration
    language_model = ChatOpenAI(
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_MODEL_NAME,
        temperature=config.OPENAI_TEMPERATURE,
    )

    # Create our service that will handle interactions with the language model
    language_model_service = LanguageModelService(language_model)

    # ----- Input Processing -----
    
    # Initialize an empty list to store conversation messages
    messages: list[dict[str, str]] = []

    # Define the path to our markdown input file
    markdown_file_path = relative_path / "data/input_02.md"

    # Load the content of the markdown file
    markdown_file_content = load_file(markdown_file_path)

    # Format the markdown content as a user message
    user_message = {"role": "user", "content": markdown_file_content}
    messages.append(user_message)  # Add to conversation history

    # Display the user message for debugging/visibility
    print(f"User:\n\n{user_message['content']}", end="\n\n")

    # ----- Generate Structured Response -----
    
    # Generate a structured response using our two-step approach
    # This method uses two different system prompts and validates the output against a schema
    agent_message = language_model_service.generate_structured_text_two_steps(
        messages=messages,
        system_prompt_template_path_01=relative_path / "prompt_templates/system_prompt_template_01.md",
        system_prompt_template_path_02=relative_path / "prompt_templates/system_prompt_template_02.md",
        output_schema=schemas.StructuredOutputSchema,
    )

    # Display the structured response as formatted JSON
    print(f"Agent:\n\n{json.dumps(agent_message, indent=4)}", end="\n\n")


if __name__ == "__main__":
    main()
