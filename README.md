# flexible-llm

**Personal memory and context management for LLMs.**

Create isolated knowledge directories, attach them to any LLM (Anthropic Claude, OpenAI GPT, or local models via Ollama/LM Studio), and query your accumulated knowledge like a second brain.

## Features

- 🔮 **Memory Directories** — Git-tracked, structured knowledge containers with manifest metadata
- 🧠 **RAG-Style Queries** — Retrieve-relevant-then-generate instead of dumping all files
- 🤖 **Multi-Provider LLM Routing** — Claude, GPT, or local models via a unified interface
- 🌐 **Web UI** — Full dashboard with chat, memory browser, and configuration
- 💬 **CLI** — Fast command-line interface for scriptability and workflows
- 📦 **Portable** — All data stored as plain Markdown/JSON files, fully Git-versioned
- ⚙️ **Per-Memory LLM Profiles** — Different LLMs per memory context (e.g., claude for research, llama for brainstorming)

## Install

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Initialize project structure
flexible-llm init

# Create a memory directory
flexible-llm memory create research --description "Research notes and papers"

# Add files
flexible-llm memory add notes.md paper.md --to research

# Query the LLM with your memory context
flexible-llm query research "What did I learn about transformers?"

# Use a specific provider
flexible-llm query research "Summarize this" --provider local --stream
```

## Commands

### Init

```bash
flexible-llm init                          # Create config files and memory root
```

### Memory

```bash
flexible-llm memory create <name> [flags]   # Create a memory directory
flexible-llm memory list                     # List all memories
flexible-llm memory add <files> --to <name>  # Add file(s) to memory
flexible-llm memory files <name>             # List files in memory
flexible-llm memory search <name> <query>   # Search across memory files
flexible-llm memory delete <name>            # Delete a memory (irreversible)
```

### LLM

```bash
flexible-llm llm list                          # Show available providers
flexible-llm llm set <provider> [model]        # Set global default LLM
flexible-llm llm set <provider> [model] --memory <name>  # Per-memory override
```

### Query

```bash
flexible-llm query <memory> <prompt>           # Chat with memory context
flexible-llm query <memory> <prompt> --stream  # Stream the response
flexible-llm query <memory> <prompt> --provider local  # Override provider
```

### Chat (conversational mode)

```bash
flexible-llm chat "Your prompt" --memory research  # Session-based chat with memory
```

### Web Server

```bash
python app/server.py          # Launch web UI at http://localhost:8000
```

### Session Management

```bash
flexible-llm session create <memory> <name>  # Start a named session
flexible-llm session list <memory>             # List sessions
flexive-llm session chat <memory> <session>    # Chat within a session
flexible-llm session show <memory> <session>   # View session history
flexible-llm session delete <memory> <name>    # Delete a session
```

### Export / Import

```bash
flexible-llm export <memory>                     # Zip-export a memory directory
flexible-llm import <file.zip>                   # Import a memory from zip
```

## Directory Structure

After `flexible-llm init`, your repo looks like:

```
flexible_llm/
├── README.md
├── pyproject.toml
├── requirements.txt
├── config.json                # Global config (gitignored)
├── .env                       # API keys (gitignored)
├── memories/                  # Memory root (tracked by git)
│   ├── my-project/
│   │   ├── manifest.json      # Memory metadata (name, description, tags, default_llm)
│   │   ├── context/           # Knowledge files (.md, .txt, .json, .pdf)
│   │   ├── conversation/      # Stored conversations per subtopic
│   │   ├── sessions/          # Multi-turn chat sessions
│   │   └── .llmrc             # Optional: per-memory LLM override
│   └── another-mem/
├── src/
│   ├── memory/
│   │   └── manager.py         # Core memory CRUD + RAG retrieval
│   ├── llm/
│   │   ├── base.py            # LLM provider abstraction
│   │   ├── anthropic.py       # Claude provider
│   │   ├── openai.py          # GPT provider
│   │   └── local.py           # Ollama / LM Studio provider
│   ├── rag/                   # RAG / retrieval module
│   │   ├── vector.py          # Simple TF-IDF vectorization
│   │   └── retriever.py       # Retrieve relevant chunks from memory
│   ├── session/               # Session management
│   └── export.py              # Export / import
├── app/
│   ├── cli.py                 # CLI commands
│   └── server.py              # FastAPI web server
└── templates/
    └── index.html             # Web UI template
```

## Configuration

### Global Config (`config.json` or `.env`)

```json
{
  "default_llm": {
    "provider": "anthropic",
    "model": "claude-3.5-sonnet-20241022"
  },
  "api_keys": {
    "ANTHROPIC_API_KEY": "sk-ant-...",
    "OPENAI_API_KEY": "sk-..."
  },
  "local_llm": {
    "base_url": "http://localhost:11434/v1",
    "model": "llama3.3"
  }
}
```

### Memory LLM Override (`.llmrc`)

Place a `.llmrc` file inside any memory directory to override its default LLM:

```json
{
  "provider": "local",
  "model": "mistral:7b"
}
```

## RAG Retrieval

The `src/rag/` module implements lightweight retrieval-augmented generation:

```python
from src.rag.retriever import Retriever

# Retrieve the 3 most relevant chunks from a memory for a query
retriever = Retriever("path/to/memories/my-project")
chunks = retriever.retrieve("transformer architectures", top_k=3)
for c in chunks:
    print(c.score, c.content[:200])
```

This enables efficient queries even on large memories by only injecting the most relevant context into the LLM prompt. 🚀

## Why This Exists

LLMs forget. Context windows have limits. You need a reliable, versioned way to accumulate knowledge, organize it by project/topic, and hand the right pieces to the right model on demand.

flexible-llm turns your project codebase into a **structured second brain** — all stored as plain files you can version with git, diff, and manage independently of the LLM.

## License

MIT