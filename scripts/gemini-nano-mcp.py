#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


BASE_URL = os.getenv("NANO_BASE_URL", "http://127.0.0.1:8458").rstrip("/")
MODEL = os.getenv("NANO_MODEL", "gemini-nano")
TIMEOUT = float(os.getenv("NANO_MCP_TIMEOUT_SECONDS", "180"))


def write_message(message: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def read_messages():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
            )


def http_json(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach NanoServer at {BASE_URL}: {exc.reason}") from exc


def text_content(text: str) -> List[Dict[str, str]]:
    return [{"type": "text", "text": text}]


def tool_result(text: str, *, is_error: bool = False) -> Dict[str, Any]:
    return {"content": text_content(text), "isError": is_error}


def nano_health(_: Dict[str, Any]) -> Dict[str, Any]:
    return tool_result(json.dumps(http_json("GET", "/health"), ensure_ascii=False, indent=2))


def nano_state(_: Dict[str, Any]) -> Dict[str, Any]:
    return tool_result(json.dumps(http_json("GET", "/admin/state"), ensure_ascii=False, indent=2))


def ensure_ready() -> None:
    health = http_json("GET", "/health")
    if health.get("ready"):
        return
    raise RuntimeError(
        "NanoServer is not ready. "
        f"worker_connected={health.get('worker_connected')} "
        f"worker_kind={health.get('worker_kind')} "
        f"pending_jobs={health.get('pending_jobs')} "
        f"active_jobs={health.get('active_jobs')}. "
        "Run `./scripts/nano-service.sh restart` and check `./scripts/nano-service.sh status`."
    )


def nano_chat(args: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(args.get("prompt") or "")
    if not prompt.strip():
        raise ValueError("prompt is required")
    ensure_ready()

    messages: List[Dict[str, Any]] = []
    system = args.get("system")
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": prompt})

    response = http_json(
        "POST",
        "/v1/chat/completions",
        {
            "model": str(args.get("model") or MODEL),
            "messages": messages,
            "temperature": args.get("temperature"),
            "max_tokens": args.get("max_tokens"),
        },
    )
    text = (
        response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return tool_result(str(text))


def nano_anthropic_message(args: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(args.get("prompt") or "")
    if not prompt.strip():
        raise ValueError("prompt is required")
    ensure_ready()
    payload: Dict[str, Any] = {
        "model": str(args.get("model") or MODEL),
        "max_tokens": int(args.get("max_tokens") or 1024),
        "messages": [{"role": "user", "content": prompt}],
    }
    if args.get("system"):
        payload["system"] = str(args["system"])
    response = http_json("POST", "/v1/messages", payload)
    text = "\n".join(
        str(block.get("text", ""))
        for block in response.get("content", [])
        if block.get("type") == "text"
    )
    return tool_result(text)


TOOLS = {
    "nano_health": {
        "description": "Check whether the local Chrome Gemini Nano hidden worker is connected and ready.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "handler": nano_health,
    },
    "nano_state": {
        "description": "Return NanoServer diagnostics, including worker kind, extension version, and active jobs.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "handler": nano_state,
    },
    "nano_chat": {
        "description": "Ask local Chrome Gemini Nano a short text question through the OpenAI-compatible endpoint.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "User prompt to send to Gemini Nano."},
                "system": {"type": "string", "description": "Optional system instruction."},
                "model": {"type": "string", "description": "Model name, defaults to gemini-nano."},
                "temperature": {"type": "number", "description": "Optional sampling temperature."},
                "max_tokens": {"type": "integer", "description": "Optional maximum output tokens."},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
        "handler": nano_chat,
    },
    "nano_anthropic_message": {
        "description": "Ask local Chrome Gemini Nano through the Anthropic-compatible Messages endpoint.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "User prompt to send to Gemini Nano."},
                "system": {"type": "string", "description": "Optional system instruction."},
                "model": {"type": "string", "description": "Model name, defaults to gemini-nano."},
                "max_tokens": {"type": "integer", "description": "Maximum output tokens."},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
        "handler": nano_anthropic_message,
    },
}


def list_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "description": spec["description"],
            "inputSchema": spec["inputSchema"],
        }
        for name, spec in TOOLS.items()
    ]


def handle_request(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    if request_id is None:
        return None

    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": params.get("protocolVersion") or "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "chrome-gemini-nano", "version": "1.0.0"},
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": list_tools()}}
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name not in TOOLS:
                raise ValueError(f"Unknown tool: {name}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": TOOLS[name]["handler"](arguments),
            }
        if method in {"resources/list", "prompts/list"}:
            key = "resources" if method == "resources/list" else "prompts"
            return {"jsonrpc": "2.0", "id": request_id, "result": {key: []}}
        raise ValueError(f"Unsupported method: {method}")
    except Exception as exc:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": str(exc)},
        }


def main() -> None:
    for message in read_messages():
        response = handle_request(message)
        if response is not None:
            write_message(response)


if __name__ == "__main__":
    main()
