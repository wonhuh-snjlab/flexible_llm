"""
Manage memory directories: create, delete, list, add files, query.
All memories are Git-managed directories with JSON manifests.
"""

import datetime
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

MANIFEST_FILENAME = "manifest.json"

TEMPLATE_SCHEMES = {
    "default": {
        "folders": ["context", "conversation", "sessions"],
        "files": [],
        "description": "General-purpose memory for any knowledge",
    },
}


@dataclass
class Manifest:
    id: str = ""
    name: str = ""
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    tags: list = field(default_factory=list)
    schema_version: int = 1
    default_llm: Optional[dict] = None
    folder_names: list = field(default_factory=list)
    file_hints: list = field(default_factory=list)


def _coerce_config_dir(config_dir):
    """Convert config_dir arg to Path, resolving defaults."""
    if config_dir is None:
        return get_or_create_config_dir()
    if isinstance(config_dir, str):
        return Path(config_dir)
    return config_dir


def _load_json(path: Path) -> Optional[dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_or_create_config_dir() -> Path:
    """Get the directory where the config lives (repo root).
    Walks up the tree to find the repo root (has .git dir)."""
    current = Path(__file__).resolve().parent.parent  # up from memory/ to repo root
    while current != current.parent:
        if (current / ".git").is_dir():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


def get_memory_root(config_dir: Optional[Union[str, Path]] = None) -> Path:
    config = load_global_config(config_dir) or {}
    mem_dir = config.get("memory_root", "memories")
    cd = _coerce_config_dir(config_dir)
    return cd / mem_dir


def load_global_config(config_dir: Optional[Union[str, Path]] = None) -> Optional[dict]:
    cd = _coerce_config_dir(config_dir)
    config_path = cd / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    default_config = {
        "schema_version": 1,
        "default_llm": {"provider": "anthropic", "model": "claude-3.5-sonnet-20241022"},
        "memory_root": "memories",
        "api_keys": {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""},
        "local_llm": {"base_url": "http://localhost:11434/v1", "model": "llama3.3"},
    }
    if config_path.parent.exists():
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
    return default_config


def _manifest_path(memory_dir: Path) -> Path:
    return memory_dir / MANIFEST_FILENAME


def create_memory(
    name: str,
    description: str = "",
    tags: Optional[list] = None,
    config_dir: Optional[Union[str, Path]] = None,
    use_template: str = "default",
) -> Path:
    """Create a new memory directory with a manifest file."""
    cd = _coerce_config_dir(config_dir)
    memory_root = get_memory_root(cd)
    memory_root.mkdir(parents=True, exist_ok=True)

    existing = list_memories(cd)
    if any(m["name"] == name for m in existing):
        raise ValueError(f'Memory "{name}" already exists')

    template = TEMPLATE_SCHEMES.get(use_template, TEMPLATE_SCHEMES["default"])

    now = datetime.datetime.utcnow().isoformat()
    mem_id = str(uuid.uuid4())[:8]

    memory_dir = memory_root / name
    memory_dir.mkdir(parents=True, exist_ok=True)

    manifest = Manifest(
        id=mem_id,
        name=name,
        description=description,
        created_at=now,
        updated_at=now,
        tags=tags or [],
        folder_names=template["folders"],
    )

    _save_json(_manifest_path(memory_dir), asdict(manifest))

    for folder in template["folders"]:
        (memory_dir / folder).mkdir(exist_ok=True)

    return memory_dir


def delete_memory(name: str, config_dir: Optional[Union[str, Path]] = None) -> bool:
    """Delete a memory directory and all its contents."""
    cd = _coerce_config_dir(config_dir)
    memory_root = get_memory_root(cd)
    memory_dir = memory_root / name
    if not memory_dir.exists():
        raise FileNotFoundError(f'Memory "{name}" not found')
    shutil.rmtree(memory_dir)
    return True


def list_memories(config_dir: Optional[Union[str, Path]] = None) -> list:
    """List all memory directories with their manifest info."""
    cd = _coerce_config_dir(config_dir)
    memory_root = get_memory_root(cd)
    memories = []

    if not memory_root.exists():
        return memories

    for entry in memory_root.iterdir():
        if not entry.is_dir():
            continue
        manifest_path = _manifest_path(entry)
        data = _load_json(manifest_path)
        if data:
            memories.append(data)

    memories.sort(key=lambda x: x.get("created_at", ""))
    return memories


def get_memory(name: str, config_dir: Optional[Union[str, Path]] = None) -> Optional[dict]:
    """Get manifest for a single memory by name."""
    cd = _coerce_config_dir(config_dir)
    memories = list_memories(cd)
    for m in memories:
        if m["name"] == name:
            data = m.copy()
            memory_root = get_memory_root(cd)
            context_dir = memory_root / name / "context"
            if context_dir.exists():
                data["files"] = [
                    str(p.relative_to(memory_root)) for p in context_dir.iterdir() if p.is_file()
                ]
            return data
    return None


def update_manifest_field(
    name: str, field_name: str, value, config_dir: Optional[Union[str, Path]] = None
) -> dict:
    """Update a specific field in a memory's manifest."""
    cd = _coerce_config_dir(config_dir)
    memory_root = get_memory_root(cd)
    memory_dir = memory_root / name
    manifest_path = _manifest_path(memory_dir)
    data = _load_json(manifest_path)
    if not data:
        raise FileNotFoundError(f'Memory "{name}" not found')
    data["updated_at"] = datetime.datetime.utcnow().isoformat()
    data[field_name] = value
    _save_json(manifest_path, data)
    return data


def add_file_to_memory(
    file_path: str, memory_name: str, subfolder: str = "context", config_dir: Optional[Union[str, Path]] = None
) -> str:
    """Add a file to a memory directory."""
    cd = _coerce_config_dir(config_dir)
    memory_root = get_memory_root(cd)
    memory_dir = memory_root / memory_name
    manifest = _load_json(_manifest_path(memory_dir))
    if not manifest:
        raise FileNotFoundError(f'Memory "{memory_name}" not found')

    target_dir = memory_dir / subfolder
    target_dir.mkdir(exist_ok=True)

    src = Path(file_path)
    dest = target_dir / src.name
    shutil.copy2(str(src), str(dest))

    update_manifest_field(
        memory_name,
        "updated_at",
        datetime.datetime.utcnow().isoformat(),
        cd,
    )

    return str(dest)


def list_files(
    memory_name: str, subfolder: Optional[str] = None, config_dir: Optional[Union[str, Path]] = None
) -> list:
    """List all files in a memory directory, optionally under one subfolder."""
    cd = _coerce_config_dir(config_dir)
    memory_root = get_memory_root(cd)
    memory_dir = memory_root / memory_name
    manifest = _load_json(_manifest_path(memory_dir))
    if not manifest:
        raise FileNotFoundError(f'Memory "{memory_name}" not found')

    if subfolder:
        search_dir = memory_dir / subfolder
    else:
        search_dir = memory_dir

    if not search_dir.exists():
        return []

    files = []
    for f in search_dir.rglob("*"):
        if f.is_file():
            files.append(str(f.relative_to(memory_dir)))
    return files


def search_context(
    memory_name: str, query: str, config_dir: Optional[Union[str, Path]] = None
) -> list:
    """Search for query string in all .md/.txt files in a memory's context folder."""
    cd = _coerce_config_dir(config_dir)
    memory_root = get_memory_root(cd)
    context_dir = memory_root / memory_name / "context"

    if not context_dir.exists():
        return []

    results = []
    query_lower = query.lower()
    for f in context_dir.rglob("*"):
        if f.is_file() and f.suffix in (".md", ".txt", ".json"):
            try:
                content = f.read_text(encoding="utf-8")
                if query_lower in content.lower():
                    results.append({
                        "file": str(f.relative_to(memory_dir)),
                        "content": content[:500],
                    })
            except (UnicodeDecodeError, OSError):
                continue
    return results


class TemporalMemory:
    """Track time-series memories (conversation turns, event logs)."""

    def __init__(
        self, memory_name: str, config_dir: Optional[Union[str, Path]] = None, subfolder: str = "sessions"
    ):
        cd = _coerce_config_dir(config_dir)
        self.memory_root = get_memory_root(cd) / memory_name
        self.subfolder = subfolder
        self.dir = self.memory_root / subfolder
        self.dir.mkdir(parents=True, exist_ok=True)

    def append(self, turn_id: str, role: str, content: str) -> None:
        # All turns for a session go into one file
        session_file = self.dir / "session.json"
        data = _load_json(session_file) or {"id": session_file.stem, "turns": []}
        data["turns"].append({"role": role, "content": content, "id": turn_id})
        _save_json(session_file, data)

    def get_all(self) -> list:
        sessions = []
        for f in sorted(self.dir.rglob("*.json")):
            data = _load_json(f)
            if data:
                sessions.append(data)
        return sessions

    def read_last(self, memory_name: str, max_tokens: int = 10000) -> list:
        """Read the last N conversation turns as a list of {role, content} dicts."""
        sessions_dir = self.memory_root / "sessions"
        if not sessions_dir.exists():
            return []

        files = sorted(sessions_dir.rglob("*.json"))
        if not files:
            return []

        last_file = files[-1]
        data = _load_json(last_file)
        if not data or "turns" not in data:
            return []

        turns = data["turns"]
        total_len = sum(len(t["content"]) for t in turns)
        if total_len > max_tokens * 4:
            while total_len > max_tokens * 4 and len(turns) > 0:
                total_len -= len(turns[0]["content"])
                turns = turns[1:]

        return turns