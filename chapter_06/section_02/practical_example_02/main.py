#!/usr/bin/python3
"""
A simple application demonstrating how to interact with OpenAI's language models
using LangChain. This example shows how to configure the model, create a service
for handling structured interactions, and process user input/output with schema validation.
"""

# Standard Library
from pathlib import Path  # For cross-platform file path handling

# Third Party
from langchain_openai import ChatOpenAI  # LangChain's wrapper for OpenAI's chat models
from pydantic_settings import BaseSettings, SettingsConfigDict  # For type-safe configuration

# Local
import schemas  # Import schemas for structured output validation

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
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Temperature set to 0.0 for deterministic outputs (important for structured data)
    OPENAI_TEMPERATURE: float = 0.0


def main():
    """
    Main function that initializes the language model, creates a service,
    and handles the conversation flow with the user, generating structured output.
    """

    # ----- Setup -----

    # Load configuration from environment variables
    config = Config()

    # Initialize the OpenAI chat model with our configuration
    language_model = ChatOpenAI(
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_MODEL,
        temperature=config.OPENAI_TEMPERATURE,
    )

    # Create our service that will handle interactions with the language model
    language_model_service = LanguageModelService(language_model)

    # ----- User Interaction -----

    # Initialize an empty list to store conversation messages
    messages = []

    # Get input from the user and format it as a message
    user_message = {"role": "user", "content": input("User Message: ")}
    messages.append(user_message)  # Add to conversation history
    print(end="\n\n")  # Add spacing for readability

    # ----- Generate Structured Response -----

    # Generate a response using our language model service with schema validation
    # The system prompt template is loaded from an external file for better organization
    # The output_schema parameter ensures the response conforms to our defined structure
    agent_message = language_model_service.generate_structured_text(
        messages=messages,
        system_prompt_template_path=relative_path / "prompt_templates/system_prompt_template.md",
        output_schema=schemas.StructuredOutputSchema,  # Schema for validating the response structure
    )

    # Display the agent's response to the user
    print(f"Agent Message: {agent_message['content']}", end="\n\n")


if __name__ == "__main__":
    main()
