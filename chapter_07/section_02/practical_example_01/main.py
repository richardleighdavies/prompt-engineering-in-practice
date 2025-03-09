#!/usr/bin/python3
"""
A document analysis system that processes text documents using multiple concurrent
language model tasks. This application demonstrates how to use asyncio with LangChain
to perform parallel analysis of different aspects of a document and then synthesize
the results into a comprehensive assessment.
"""

# Standard Library
import asyncio  # For concurrent processing

from pathlib import Path  # For cross-platform file path handling

# Third Party
from langchain_core.prompts import HumanMessagePromptTemplate  # For templating prompts
from langchain_openai import ChatOpenAI  # LangChain's wrapper for OpenAI's chat models
from pydantic_settings import BaseSettings, SettingsConfigDict  # For type-safe configuration

# Local
from services.language_model.service import LanguageModelService  # Our abstraction for LLM interactions

# Get the directory where this script is located to find relative paths
relative_path = Path(__file__).resolve().parent


class Config(BaseSettings):
    """Configuration settings for the Document Analysis System."""

    model_config = SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY: str  # Required API key for OpenAI
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"  # Model name with default value
    OPENAI_TEMPERATURE: float = 0.7  # Temperature set to 0.7 for deterministic outputs


async def load_file(file_path: Path) -> str:
    """Load the file content from the given path."""
    with open(file_path, "r") as file:
        return file.read()


async def main():
    """
    Main async function that orchestrates the document analysis workflow:
    1. Loads the document
    2. Initializes the language model
    3. Creates parallel analysis tasks
    4. Synthesizes the results into a comprehensive assessment
    """

    # ----- Document Loading -----
    document_content = await load_file(relative_path / "data/input_01.md")

    print("# Document Content", end="\n\n")
    print(document_content, end="\n\n")

    # ----- Setup -----
    config = Config()

    # Initialize the OpenAI chat model with our configuration
    language_model = ChatOpenAI(
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_MODEL_NAME,
        temperature=config.OPENAI_TEMPERATURE,
    )

    # Create our service that will handle async interactions with the language model
    language_model_service = LanguageModelService(language_model)

    # ----- Initial Message Preparation -----
    messages = []
    user_message = {"role": "user", "content": document_content}
    messages.append(user_message)

    # ----- Parallel Analysis Tasks -----
    print("Creating analysis tasks...", end="\n\n")

    # Create three concurrent tasks for different aspects of document analysis
    methodology_task = asyncio.create_task(
        language_model_service.generate_unstructured_text(
            messages=messages,
            system_prompt_template_path=relative_path / "prompt_templates/methodology_analysis_system_prompt_template.md",
        )
    )

    results_task = asyncio.create_task(
        language_model_service.generate_unstructured_text(
            messages=messages,
            system_prompt_template_path=relative_path / "prompt_templates/results_analysis_system_prompt_template.md",
        )
    )

    implications_task = asyncio.create_task(
        language_model_service.generate_unstructured_text(
            messages=messages,
            system_prompt_template_path=relative_path / "prompt_templates/implications_analysis_system_prompt_template.md",
        )
    )

    print("Waiting for analysis tasks to complete...", end="\n\n")

    # Wait for all analysis tasks to complete concurrently
    methodology_output, results_output, implications_output = await asyncio.gather(
        methodology_task, results_task, implications_task
    )

    print("Analysis tasks completed.", end="\n\n")

    print("# Methodology Output", end="\n\n")
    print(f"{methodology_output['content']}")

    print("# Results Output", end="\n\n")
    print(f"{results_output['content']}")

    print("# Implications Output", end="\n\n")
    print(f"{implications_output['content']}")

    # ----- Synthesis of Analysis Results -----
    # Load the template for the synthesis prompt
    user_prompt_template_path = relative_path / "prompt_templates/synthesis_user_prompt_template.md"
    user_prompt_template = HumanMessagePromptTemplate.from_template_file(
        template_file=user_prompt_template_path,
        input_variables=["methodology_output", "results_output", "implications_output"],
    )

    # Format the template with the results from our analysis tasks
    user_prompt = user_prompt_template.format(
        methodology_output=methodology_output["content"],
        results_output=results_output["content"],
        implications_output=implications_output["content"],
    )

    # Prepare messages for the synthesis request
    messages = []

    user_message = {"role": "user", "content": user_prompt.content}
    messages.append(user_message)

    # Generate the final synthesis of all analyses
    synthesis_response = await language_model_service.generate_unstructured_text(
        messages=messages,
        system_prompt_template_path=relative_path / "prompt_templates/synthesis_system_prompt_template.md",
    )

    # ----- Output Results -----
    print("# Synthesis Output", end="\n\n")
    print(f"{synthesis_response['content']}")


if __name__ == "__main__":
    asyncio.run(main())  # Run the async main function
