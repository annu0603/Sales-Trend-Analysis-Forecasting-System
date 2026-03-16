import os

CHUNK_SIZE    = 3000
CHUNK_OVERLAP = 200

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
    ".html", ".css", ".sh", ".yaml", ".yml", ".toml", ".json", ".md"
}

SKIP_FOLDERS = {
    "node_modules", ".git", "__pycache__", ".venv",
    "venv", "dist", "build", ".idea", ".vscode", ".github"
}

PRIORITY_FILES = {
    "main.py", "app.py", "index.js", "index.ts", "server.py",
    "manage.py", "setup.py", "requirements.txt", "package.json",
    "dockerfile", "docker-compose.yml", "config.py"
}


class RAGChunker:

    def _lang(self, path):
        ext_map = {
            ".py": "Python",   ".js": "JavaScript", ".ts": "TypeScript",
            ".java": "Java",   ".go": "Go",          ".rs": "Rust",
            ".rb": "Ruby",     ".cs": "C#",          ".cpp": "C++",
            ".c": "C",         ".html": "HTML",      ".css": "CSS",
            ".sh": "Shell",    ".json": "JSON",      ".yaml": "YAML",
            ".yml": "YAML",    ".md": "Markdown",    ".toml": "TOML"
        }
        _, ext = os.path.splitext(path)
        return ext_map.get(ext.lower(), "Unknown")

    def _split(self, text):
        chunks, start = [], 0
        while start < len(text):
            end = start + CHUNK_SIZE
            if end < len(text):
                nl = text.rfind("\n", start, end)
                if nl > start:
                    end = nl
            chunks.append(text[start:end].strip())
            start = end - CHUNK_OVERLAP
        return [c for c in chunks if c]

    def read_repo(self, repo_dir="."):
        files = {}
        for root, dirs, filenames in os.walk(repo_dir):
            dirs[:] = [
                d for d in dirs
                if d not in SKIP_FOLDERS and not d.startswith(".")
            ]
            for fname in filenames:
                _, ext = os.path.splitext(fname)
                if ext.lower() not in CODE_EXTENSIONS:
                    continue
                full_path = os.path.join(root, fname)
                rel_path  = os.path.relpath(full_path, repo_dir)
                try:
                    with open(full_path, encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if not content.strip():
                        continue
                    if len(content) > 80_000:
                        content = content[:80_000] + "\n...[truncated]"
                    files[rel_path] = content
                except Exception:
                    pass
        return files

    def build_context(self, files, max_chunks=40):
        priority, others = [], []
        for path, content in files.items():
            lang   = self._lang(path)
            pieces = self._split(content)
            for i, piece in enumerate(pieces):
                chunk = (
                    f"### File: {path} [{lang}]"
                    f" (chunk {i+1}/{len(pieces)})\n{piece}\n"
                )
                fname = os.path.basename(path).lower()
                (priority if fname in PRIORITY_FILES else others).append(chunk)

        selected = (priority + others)[:max_chunks]
        return "\n".join(selected)

    def get_structure(self, files):
        return "\n".join(sorted(files.keys()))
