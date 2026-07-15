"""FastAPI web server for flexible-llm."""

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from memory.manager import (
    create_memory,
    delete_memory,
    list_memories,
    get_memory,
    load_global_config,
    get_or_create_config_dir,
    get_memory_root,
    list_files,
)
from llm_router import LLMRouter
from llm.base import ChatMessage

app = FastAPI(title="flexible-llm")
CONFIG_DIR = get_or_create_config_dir()
templates = Jinja2Templates(directory="templates")


def _read_config():
    p = CONFIG_DIR / "config.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _write_config(cfg):
    with open(CONFIG_DIR / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ─── Memories ────────────────────────────────────────────────

@app.get("/api/memories")
def api_list_memories():
    return list_memories(CONFIG_DIR)


@app.post("/api/memories")
def api_create_memory(
    request: Request,
    name: str = Form(""),
    description: str = Form(""),
    tags: str = Form(""),
):
    try:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        mem_data = create_memory(name, description=description, tags=tag_list, config_dir=CONFIG_DIR)
        return JSONResponse({"ok": True, "name": name, "path": str(mem_data)})
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.delete("/api/memories/{name}")
def api_delete_memory(name: str):
    try:
        delete_memory(name, CONFIG_DIR)
        return {"ok": True, "deleted": name}
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@app.get("/api/memories/{name}")
def api_get_memory(name: str):
    mem = get_memory(name, CONFIG_DIR)
    if not mem:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return mem


@app.get("/api/memories/{name}/files")
def api_memory_files(name: str):
    mem = get_memory(name, CONFIG_DIR)
    if not mem:
        return JSONResponse({"error": "Not found"}, status_code=404)
    files = list_files(name, config_dir=CONFIG_DIR)
    return {"name": name, "files": files}


# ─── LLM ──────────────────────────────────────────────────────

@app.post("/api/query")
def api_query(request: Request):
    body = await request.json()
    memory = body.get("memory", "__global__")
    prompt = body.get("prompt", "")
    provider_override = body.get("provider")

    if not prompt:
        return JSONResponse({"error": "No prompt"}, status_code=400)

    mem_name = None if memory == "__global__" else memory

    router = LLMRouter()
    if provider_override:
        router.default_provider = provider_override

    try:
        response = router.chat(
            [ChatMessage("user", prompt)],
            memory_name=mem_name,
            system_prompt="You are a helpful assistant. Use the memory context to help the user.",
        )
        return {"response": response}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/providers")
def api_providers():
    router = LLMRouter()
    return router.list_providers()


# ─── Config ──────────────────────────────────────────────────

@app.get("/api/config")
def api_config():
    cfg = _read_config()
    cfg["config_dir"] = str(CONFIG_DIR)
    return cfg


@app.post("/api/config/keys")
def api_config_keys(request: Request):
    body = await request.json()
    cfg = _read_config()
    if "api_keys" not in cfg:
        cfg["api_keys"] = {}
    if "ANTHROPIC_API_KEY" in body:
        cfg["api_keys"]["ANTHROPIC_API_KEY"] = body["ANTHROPIC_API_KEY"]
    if "OPENAI_API_KEY" in body:
        cfg["api_keys"]["OPENAI_API_KEY"] = body["OPENAI_API_KEY"]
    _write_config(cfg)
    return {"ok": True}


@app.post("/api/config/llm")
def api_config_llm(request: Request):
    body = await request.json()
    cfg = _read_config()
    cfg["default_llm"] = {
        "provider": body.get("provider", "anthropic"),
        "model": body.get("model", ""),
    }
    _write_config(cfg)
    return {"ok": True}


@app.post("/api/config/local")
def api_config_local(request: Request):
    body = await request.json()
    cfg = _read_config()
    cfg["local_llm"] = {
        "base_url": body.get("base_url", "http://localhost:11434/v1"),
        "model": body.get("model", "llama3.3"),
    }
    _write_config(cfg)
    return {"ok": True}


# ─── Per-memory LLM setting ─────────────────────────────────

@app.post("/api/memories/{name}/llm")
def api_memory_llm(name: str, request: Request):
    body = await request.json()
    provider = body.get("provider")
    model = body.get("model")

    if not name:
        return JSONResponse({"error": "No memory name"}, status_code=400)

    mem_root = get_memory_root(CONFIG_DIR)
    llmrc_path = mem_root / name / ".llmrc"
    llmrc = {}
    if llmrc_path.exists():
        with open(llmrc_path) as f:
            llmrc = json.load(f)
    if provider:
        llmrc["provider"] = provider
    if model:
        llmrc["model"] = model
    with open(llmrc_path, "w") as f:
        json.dump(llmrc, f, indent=2)
    return {"ok": True, "path": str(llmrc_path)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)