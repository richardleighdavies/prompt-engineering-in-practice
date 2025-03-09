#!/usr/bin/python3
"""
Product Review Synthesizer

This application demonstrates an advanced implementation of parallel prompt chaining,
processing multiple data streams simultaneously to generate comprehensive product assessments.
The system analyzes product reviews through three parallel streams (feature analysis,
sentiment analysis, and market comparison) and synthesizes the results into a coherent
final assessment.
"""

# Standard Library
import asyncio  # For asynchronous programming
import json

from pathlib import Path  # For cross-platform file path handling

# Third Party
from langchain_core.prompts import HumanMessagePromptTemplate  # For templating user prompts
from langchain_openai import ChatOpenAI  # LangChain's wrapper for OpenAI's chat models
from pydantic_settings import BaseSettings, SettingsConfigDict  # For type-safe configuration

# Local
import schemas  # Pydantic models for structured data validation
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


async def load_file(file_path: Path) -> str:
    """Load the file content from the given path."""
    with open(file_path, "r") as file:
        return file.read()


async def main():
    """
    Main async function that orchestrates the product review analysis workflow:
    1. Loads the product review
    2. Initializes the language model
    3. Creates parallel analysis tasks for features, sentiment, and market comparison
    4. Synthesizes the results into a comprehensive product assessment
    """

    # ----- Review Loading -----
    # Load the product review text from file
    review_text = await load_file(relative_path / "data/input_01.md")

    # Display the review text for reference
    print("# Product Review", end="\n\n")
    print("-" * 40, end="\n\n")
    print(review_text, end="\n\n")
    print("-" * 40, end="\n\n")

    # ----- Setup -----
    # Load configuration from environment variables
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
    # Initialize an empty list to store conversation messages
    messages = []

    # Format the review text as a user message
    user_message = {"role": "user", "content": review_text}
    messages.append(user_message)  # Add to conversation history

    # ----- Parallel Analysis Tasks -----
    print("Creating analysis tasks...", end="\n\n")

    # Create three concurrent tasks for different aspects of product review analysis
    feature_task = asyncio.create_task(
        language_model_service.generate_unstructured_text(
            messages=messages,
            system_prompt_template_path=relative_path / "prompt_templates/feature_analysis_system_prompt_template.md",
        )
    )

    sentiment_task = asyncio.create_task(
        language_model_service.generate_unstructured_text(
            messages=messages,
            system_prompt_template_path=relative_path / "prompt_templates/sentiment_analysis_system_prompt_template.md",
        )
    )

    market_task = asyncio.create_task(
        language_model_service.generate_unstructured_text(
            messages=messages,
            system_prompt_template_path=relative_path / "prompt_templates/market_comparison_system_prompt_template.md",
        )
    )

    print("Waiting for analysis tasks to complete...", end="\n\n")

    # Wait for all analysis tasks to complete concurrently
    feature_output, sentiment_output, market_output = await asyncio.gather(feature_task, sentiment_task, market_task)

    print("Analysis tasks completed.", end="\n\n")

    # ----- Synthesis of Analysis Results -----
    # Load the template for the synthesis prompt
    user_prompt_template = HumanMessagePromptTemplate.from_template_file(
        template_file=relative_path / "prompt_templates/user_prompt_template.md",
        input_variables=["review_text", "feature_result", "sentiment_result", "market_result"],
    )

    # Format the template with the results from our analysis tasks
    user_prompt = user_prompt_template.format(
        review_text=review_text,
        feature_result=feature_output["content"],
        sentiment_result=sentiment_output["content"],
        market_result=market_output["content"],
    )

    # Prepare messages for the synthesis request
    messages = []

    user_message = {"role": "user", "content": user_prompt.content}
    messages.append(user_message)  # Add to conversation history

    # Generate the final synthesis of all analyses
    final_output = await language_model_service.generate_structured_text(
        messages=messages,
        system_prompt_template_path=relative_path / "prompt_templates/synthesis_system_prompt_template.md",
        output_schema=schemas.OutputSchema,
    )

    # ----- Output Results -----
    # Display the comprehensive analysis results in a structured format
    print("-" * 40, end="\n\n")
    print("Product Review Analysis Results", end="\n\n")
    print("-" * 40, end="\n\n")

    print(f"# Product Name\n\n{final_output['content']['product_name']}", end="\n\n")
    print(f"# Summary\n\n{final_output['content']['summary']}", end="\n\n")
    
    print(f"# Key Points\n\n{json.dumps(final_output['content']['key_points'], indent=4)}", end="\n\n")
    print(f"# Market Analysis\n\n{json.dumps(final_output['content']['key_points'], indent=4)}", end="\n\n")
    
    # Conclusions
    print(f"# Recommendation\n\n{final_output['content']['recommendation']}", end="\n\n")
    print(f"# Overall Rating\n\n{final_output['content']['overall_rating']}", end="\n\n")
    print(f"# Rating Category\n\n{final_output['content']['rating_category']}", end="\n\n")


if __name__ == "__main__":
    asyncio.run(main())
