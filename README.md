# Prompt Engineering in Practice - Code Examples

This repository contains the practical code examples from the book "Prompt Engineering in Practice" by Richard Davies, published by Manning Publications.

## About the Book

"Prompt Engineering in Practice" teaches you how to write, refine, organize, and optimize AI prompts that generate relevant and useful text and images. The book covers essential techniques for working with models like ChatGPT, Stable Diffusion, and Gemini, focusing on:

- Designing prompts that generate accurate and readable responses from LLMs
- Mitigating hallucinations in LLM output
- Domain-aware content generation using RAG (Retrieval Augmented Generation)
- Understanding how AI model design affects your prompts
- Evaluating, optimizing, and organizing your prompts

## Project Structure

```
.
├── chapter_07/
│   ├── section_01_text_generation/
│   ├── section_02_prompt_chaining/
│   └── section_03_prompt_routing/
├── LICENSE
├── pyproject.toml
└── README.md
```

## Features

- Text generation with customizable system prompts
- Structured output handling using Pydantic schemas
- Prompt chaining for complex conversational flows
- Dynamic prompt routing based on user input
- Integration with OpenAI's GPT models

## Requirements

- Python 3.x
- OpenAI API key
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd <repository_name>
   ```

2. Navigate to the desired example directory and set up your `.env` file:
   ```sh
   cd chapter_07/section_01_text_generation/practical_example_01
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

3. Run the installation command to create a virtual environment and install dependencies:
   ```sh
   make install
   ```

## Usage

Execute the example:
```sh
make
```

### Notes
- **Self-Contained Examples:** Each section is an independent project. Repeat the steps above for each example you wish to run.
- **Environment Isolation:** `make install` ensures each example has its own isolated environment.

## License

This project is licensed under the MIT License - see the 

LICENSE

 file for details.

## Author

Richard Davies - Chief Technology Officer (CTO) at Vera, specializing in Applied Artificial Intelligence.

## Contributing

This is a companion repository for the book "Prompt Engineering in Practice". While we appreciate your interest, we are not accepting contributions at this time.