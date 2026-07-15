"""
flexible-llm — Memory/context directory system with pluggable LLM providers

Usage:
    flexible-llm init                # Initialize config files
    flexible-llm memory create <name> # Create a memory directory
    flexible-llm memory list          # List all memories
    flexible-llm memory add <files>   # Add files to a memory
    flexible-llm query <memory>       # Ask the LLM your memories
    flexible-llm llm list             # List available LLMs
    flexible-llm llm set <provider>   # Set default LLM
"""

import os
import json
from pathlib import Path

import click

from memory.manager import (
    create_memory,
    delete_memory,
    list_memories,
    get_memory,
    add_file_to_memory,
    list_files,
    search_context,
    load_global_config,
    get_or_create_config_dir,
    get_memory_root,
)
from llm_router import LLMRouter


def _ensure_config(ctx):
    """Make sure config.json exists."""
    config_dir = get_or_create_config_dir()
    config_path = config_dir / "config.json"
    if not config_path.exists():
        default_config = {
            "schema_version": 1,
            "default_llm": {"provider": "anthropic", "model": "claude-3.5-sonnet-20241022"},
            "memory_root": "memories",
            "api_keys": {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""},
            "local_llm": {"base_url": "http://localhost:11434/v1", "model": "llama3.3"},
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        click.echo("Created config.json 📝")


@click.group()
def cli():
    """flexible-llm: Memory directories with pluggable LLMs."""
    pass


@cli.command()
def init():
    """Initialize the project with config templates."""
    config_dir = get_or_create_config_dir()
    _ensure_config(None)
    # Create memories dir
    mem_root = get_memory_root(config_dir)
    mem_root.mkdir(parents=True, exist_ok=True)
    # Create .env.example if not present
    env_exc = config_dir / ".env.example"
    if not env_exc.exists():
        with open(env_exc, "w") as f:
            f.write("# API Keys for flexible-llm\nANTHROPIC_API_KEY=\nOPENAI_API_KEY=\n")
        click.echo("Created .env.example 📝")
    click.echo("Ready! You can now:")
    click.echo(f"  memories dir: {mem_root}")
    click.echo(f"  config file: {config_dir / 'config.json'}")
    click.echo()
    click.echo('  flexible-llm memory create <name>')


@cli.group()
def memory():
    """Manage memory directories."""
    pass


@memory.command()
@click.argument("name")
@click.option("--description", "-d", default="", help="Description of this memory")
@click.option("--tags", "-t", default="", help="Comma-separated tags")
def create(name, description, tags):
    """Create a new memory directory."""
    _ensure_config(None)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    try:
        path = create_memory(
            name,
            description=description,
            tags=tag_list,
        )
        click.echo(f"Created memory: {name} 📁")
        click.echo(f"  path:   {path}")
        click.echo(f"  tags:   {tag_list}")
        click.echo(f"  folders: context/, conversation/, sessions/")
    except ValueError as e:
        raise click.ClickException(str(e))


@memory.command("list")
def list_cmd():
    """List all memory directories."""
    _ensure_config(None)
    memories = list_memories()
    if not memories:
        click.echo("No memories found. Create one with:")
        click.echo('  flexible-llm memory create <name>')
        return

    click.echo(f"Found {len(memories)} memory(s):\n")
    for m in memories:
        mem_root = get_memory_root(get_or_create_config_dir())
        mem_dir = mem_root / m["name"]
        folder_names = m.get("folder_names", [])
        click.echo(f"  📁 {m['name']}")
        click.echo(f"     {m.get('description', '')}")
        click.echo(f"     tags: {', '.join(m.get('tags', [])) or 'none'}")
        click.echo(f"     folders: {', '.join(folder_names)}")
        click.echo()


@memory.command("add")
@click.argument("file_path", nargs=-1)
@click.option("--to", "-t", "memory_name", required=True, help="Memory to add files to")
@click.option("--subfolder", "-s", default="context", help="Subfolder within memory")
def add(file_path, memory_name, subfolder):
    """Add file(s) to a memory directory."""
    _ensure_config(None)
    if not file_path:
        raise click.ClickException("No files specified")

    config_dir = get_or_create_config_dir()
    mem_data = get_memory(memory_name, config_dir)
    if not mem_data:
        raise click.ClickException(f'Memory "{memory_name}" not found')

    for fp in file_path:
        src = Path(fp)
        if not src.exists():
            raise click.ClickException(f'File not found: {fp}')
        try:
            dest = add_file_to_memory(str(src), memory_name, subfolder=subfolder)
            click.echo(f"  ➕ {src.name} → {memory_name}/{subfolder}/")
        except Exception as e:
            click.echo(f"  ✖ {src.name}: {e}")


@memory.command("files")
@click.argument("memory_name")
@click.option("--subfolder", "-s", default=None, help="List files in specific subfolder")
def files(memory_name, subfolder):
    """List files in a memory directory."""
    _ensure_config(None)
    try:
        files_list = list_files(memory_name, subfolder=subfolder)
        if not files_list:
            click.echo(f"No files found in {memory_name}/")
            return
        click.echo(f"Files in {memory_name}/:\n")
        for f in files_list:
            click.echo(f"  📄 {f}")
    except FileNotFoundError as e:
        raise click.ClickException(str(e))


@memory.command("search")
@click.argument("memory_name")
@click.argument("query")
def search(memory_name, query):
    """Search for text in a memory's files."""
    _ensure_config(None)
    try:
        results = search_context(memory_name, query)
        if not results:
            click.echo(f'No matches for "{query}" in {memory_name}/')
            return
        click.echo(f'Found "{query}" in {len(results)} file(s):\n')
        for r in results:
            click.echo(f"  📄 {r['file']}")
            ctx = r['content'][:200]
            click.echo(f"     {ctx}...")
            click.echo()
    except FileNotFoundError as e:
        raise click.ClickException(str(e))


@memory.command("delete")
@click.argument("name")
def delete(name):
    """Delete a memory directory (cannot be undone)."""
    _ensure_config(None)
    if not click.confirm(f'Are you sure you want to delete "{name}"?'):
        return
    try:
        delete_memory(name)
        click.echo(f"Deleted memory: {name}")
    except FileNotFoundError as e:
        raise click.ClickException(str(e))


@memory.command("get")
@click.argument("name")
def get_cmd(name):
    """Show details of a memory directory."""
    _ensure_config(None)
    mem_data = get_memory(name)
    if not mem_data:
        raise click.ClickException(f'Memory "{name}" not found')
    click.echo(f"Memory: {name}")
    click.echo(f"  Description: {mem_data.get('description', '')}")
    click.echo(f"  Created:     {mem_data.get('created_at', '')}")
    click.echo(f"  Updated:     {mem_data.get('updated_at', '')}")
    click.echo(f"  Tags:        {', '.join(mem_data.get('tags', [])) or 'none'}")
    if mem_data.get('files'):
        click.echo(f"  Context files:")
        for f in mem_data['files']:
            click.echo(f"    📄 {f}")


@cli.group()
def llm():
    """Manage LLM providers."""
    pass


@llm.command("list")
def list_llm():
    """List available LLM providers and their models."""
    _ensure_config(None)
    router = LLMRouter()
    providers = router.list_providers()

    click.echo("Available LLM providers:\n")
    for p in providers:
        status = "✅" if p["available"] else "❌"
        click.echo(f"  {status} {p['name']}")
        for m in p.get("models", []):
            click.echo(f"     • {m}")
        click.echo()

    # Show which is active
    active_provider, _ = router.get_active_provider()
    click.echo(f"Active default: {active_provider}")


@llm.command("set")
@click.argument("provider")
@click.argument("model", required=False)
@click.option("--memory", "-m", default=None, help="Set per-memory override")
def set_llm(provider, model, memory):
    """Set the default LLM provider and model."""
    _ensure_config(None)
    config_dir = get_or_create_config_dir()
    config = load_global_config(config_dir)

    if provider == "local":
        model = model or config.get("local_llm", {}).get("model", "llama3.3")
        click.echo(f"Local LLM: {model} (base_url: {config.get('local_llm', {}).get('base_url')})")
    else:
        model = model or "claude-3.5-sonnet-20241022" if provider == "anthropic" else "gpt-4o"
        click.echo(f"{provider}: {model}")

    if memory:
        # Set per-memory LLM config (.llmrc)
        mem_root = get_memory_root(config_dir)
        llmrc_path = mem_root / memory / ".llmrc"
        llmrc = {}
        if llmrc_path.exists():
            with open(llmrc_path, "r") as f:
                llmrc = json.load(f)
        llmrc["provider"] = provider
        llmrc["model"] = model
        with open(llmrc_path, "w") as f:
            json.dump(llmrc, f, indent=2)
        click.echo(f"Set {memory}/.llmrc → {provider}/{model}")
    else:
        # Set global default
        config["default_llm"] = {"provider": provider, "model": model}
        with open(config_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        click.echo(f"Set global default → {provider}/{model}")


@cli.command()
@click.argument("memory_name")
@click.argument("prompt")
@click.option("--provider", "-p", default=None, help="Override LLM provider")
@click.option("--model", "-m", default=None, help="Override model")
@click.option("--system", "-s", default=None, help="System prompt")
@click.option("--stream", "-S", is_flag=True, help="Stream the response")
def query(memory_name, prompt, provider, model, system, stream):
    """Query the LLM with a memory context."""
    _ensure_config(None)
    router = LLMRouter()

    # Build context from memory files
    from memory.manager import get_memory_root, list_files
    config_dir = get_or_create_config_dir()
    memory_root = get_memory_root(config_dir)
    context_files = list_files(memory_name, subfolder="context", config_dir=config_dir)

    context_parts = []
    for f in context_files:
        fp = memory_root / memory_name / f
        if fp.exists():
            try:
                content = fp.read_text(encoding="utf-8")
                context_parts.append(f'--- {f} ---\n{content}')
            except Exception:
                pass

    context_text = "\n\n".join(context_parts) if context_parts else "No context files loaded."

    if context_text:
        system_prompt = system or (
            "You are a helpful assistant. Refer to the following context from the user's memory directory and use it to inform your responses."
            + "\n\n=== CONTEXT ===\n" + context_text
        )
    else:
        system_prompt = system or "You are a helpful assistant."

    from llm.base import ChatMessage
    messages = [ChatMessage("user", prompt)]

    if provider:
        from llm_router import LLMRouter as LR
        r = LR(provider=provider, config_dir=config_dir)
        if model:
            r.default_model = model
            r.default_provider = provider
        click.echo()

    try:
        if stream:
            response = router.chat_stream(messages, memory_name=memory_name, system_prompt=system_prompt)
            for chunk in response:
                click.echo(chunk, nl=False)
            click.echo()
        else:
            response = router.chat(messages, memory_name=memory_name, system_prompt=system_prompt)
            click.echo(response)
    except Exception as e:
        raise click.ClickException(f"LLM call failed: {e}")


@cli.command()
@click.argument("prompt")
@click.option("--memory", "-m", default=None, help="Use specific memory context")
@click.option("--provider", "-p", default=None, help="Override LLM provider")
@click.option("--system", "-s", default=None, help="System prompt")
def chat(prompt, memory, provider, system):
    """Quick chat (alias for query without streaming focus)."""
    _ensure_config(None)
    mem_name = memory or "global"
    router = LLMRouter()

    if provider:
        from llm.base import ChatMessage
        provider_obj = router.create(mem_name)
        messages = [ChatMessage("user", prompt)]
        system_prompt = system or "You are a helpful assistant."
        click.echo(provider_obj.chat(messages, system_prompt=system_prompt))
    else:
        from memory.manager import get_memory_root, list_files
        config_dir = get_or_create_config_dir()
        context_files = list_files(mem_name, subfolder="context", config_dir=config_dir) if mem_name != "global" else []

        context_parts = []
        for f in context_files:
            fp = get_memory_root(config_dir) / mem_name / f
            if fp.exists():
                content = fp.read_text(encoding="utf-8")
                context_parts.append(f'--- {f} ---\n{content}')

        context_text = "\n\n".join(context_parts) if context_parts else ""

        system_prompt = system or "You are a helpful assistant."
        if context_text:
            system_prompt += "\n\n=== CONTEXT ===\n" + context_text

        from llm.base import ChatMessage
        response = router.chat([ChatMessage("user", prompt)], memory_name=mem_name, system_prompt=system_prompt)
        click.echo(response)


if __name__ == "__main__":
    cli()