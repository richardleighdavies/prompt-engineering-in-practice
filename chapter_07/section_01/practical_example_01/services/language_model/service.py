""" """

# Standard Library
from pathlib import Path

# Third Party
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI


class LanguageModelService:

    def __init__(self, language_model: ChatOpenAI):
        self.language_model = language_model

    def generate_unstructured_text(
        self,
        messages: list[dict[str, str]],
        system_prompt_template_path: Path,
    ) -> dict[str, str]:

        system_prompt = self._get_system_prompt(system_prompt_template_path)

        language_model_input = self._get_language_model_input(messages, system_prompt)

        response: dict[str, str] = self.language_model.invoke(language_model_input)

        return {"role": "assistant", "content": response.content}

    def generate_structured_text(
        self,
        messages: list[dict[str, str]],
        system_prompt_template_path: Path,
        output_schema: PydanticOutputParser,
    ) -> dict[str, str]:

        structured_language_model, output_format_instructions = self._get_structured_language_model(output_schema)

        system_prompt = self._get_system_prompt(system_prompt_template_path, output_format_instructions)

        language_model_input = self._get_language_model_input(messages, system_prompt)

        response: dict[str, str] = structured_language_model.invoke(language_model_input)

        structured_response: dict[str, str] = response.model_dump()

        return {"role": "assistant", "content": structured_response}

    def _get_system_prompt(
        self,
        system_prompt_template_path: Path,
        output_format_instructions: str = None,
    ) -> str:

        partial_variables = self._get_partial_variables(output_format_instructions)

        system_prompt_template = SystemMessagePromptTemplate.from_template_file(
            template_file=system_prompt_template_path,
            partial_variables=partial_variables,
            input_variables=[],
        )

        system_prompt = system_prompt_template.format().content

        return system_prompt

    def _get_partial_variables(self, output_format_instructions: str) -> dict[str, str]:

        if output_format_instructions:
            partial_variables = {"output_format_instructions": output_format_instructions}
        else:
            partial_variables = {}

        return partial_variables

    def _get_language_model_input(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
    ) -> list[dict[str, str]]:

        language_model_input = []

        language_model_input.append({"role": "system", "content": system_prompt})

        language_model_input.extend(messages)

        return language_model_input

    def _get_structured_language_model(
        self,
        output_schema: PydanticOutputParser,
    ) -> tuple[ChatOpenAI, str]:

        pydantic_output_parser = PydanticOutputParser(pydantic_object=output_schema)

        structured_language_model = self.language_model.with_structured_output(schema=output_schema, method="json_mode")

        output_format_instructions = pydantic_output_parser.get_format_instructions()

        return structured_language_model, output_format_instructions
