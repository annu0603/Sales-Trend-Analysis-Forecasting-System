import os
from dotenv import load_dotenv
from rag_chunker      import RAGChunker
from readme_generator import ReadmeGenerator

load_dotenv()

repo_name = os.getenv("GITHUB_REPOSITORY", "your-username/your-repo")

print(f"Generating README for: {repo_name}")

chunker   = RAGChunker()
files     = chunker.read_repo(repo_dir=".")

if not files:
    print("No code files found.")
    exit(1)

structure = chunker.get_structure(files)
context   = chunker.build_context(files, max_chunks=40)

gen  = ReadmeGenerator()
text = gen.generate(repo_name, structure, context)
gen.save(text, path="README.md")

print("README.md updated successfully.")
