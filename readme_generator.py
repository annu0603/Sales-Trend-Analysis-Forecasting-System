import os
from init_models import init_models

PROMPT = """You are an expert technical writer.
Read the source code below and write a complete, professional README.md for this GitHub repository.

Repository : {repo_name}
GitHub URL : https://github.com/{repo_name}

File structure:
{structure}

Source code:
{context}

Write the README.md with ALL of these sections:

# <Project Title>

## Description
What the project does and why it exists.

## Tech Stack
Languages, frameworks, and libraries used.

## Project Structure
Explain the folder and file layout.

## Installation
Step-by-step setup instructions.

## Usage
How to run the project. Include code examples.

## Configuration
Any .env variables or config files needed.

## Contributing
How others can contribute.

## License
MIT License.

Rules:
- Use proper Markdown (headers, bullet points, fenced code blocks)
- Use ```bash for terminal commands, ```python for Python code
- Be specific to this actual code — do NOT write generic content
- If a section truly does not apply, write N/A
- Start directly with # ProjectTitle — no preamble

Write the README now:
"""


class ReadmeGenerator:

    def __init__(self):
        self.models = init_models()

    def _clean(self, text):
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("#"):
                return "\n".join(lines[i:]).strip()
        return text.strip()

    def generate(self, repo_name, structure, context):
        prompt = PROMPT.format(
            repo_name = repo_name,
            structure = structure,
            context   = context
        )
        print("  Calling Gemini...")
        raw = self.models.gemini_model("gemini-2.5-flash", prompt)
        print("  Done.")
        return self._clean(raw)

    def save(self, text, path="README.md"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  Saved: {path}")
