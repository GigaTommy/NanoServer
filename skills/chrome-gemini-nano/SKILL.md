# Chrome Gemini Nano Local Assistant

Use this skill when the user wants local, private, low-latency help from Chrome's built-in Gemini Nano model, or asks to check whether the local NanoServer bridge is available.

## Service

NanoServer should be running at:

```text
http://127.0.0.1:8458
```

Check readiness before relying on it:

```bash
curl -sS http://127.0.0.1:8458/health
```

The healthy state should include:

- `ready: true`
- `worker_connected: true`
- `worker_kind: extension`
- `capability.sourceVersion`

If it is not ready:

```bash
./scripts/nano-service.sh restart
```

## When To Use

Use Gemini Nano for short local tasks:

- quick summaries
- classification
- small rewrites
- privacy-sensitive snippets
- local fallback checks

Avoid it for large codebase reasoning, long context, tool-heavy workflows, or tasks requiring strong instruction following. In those cases, use the primary agent model and optionally call Nano only as a helper.

## MCP Tool

If the `chrome-gemini-nano` MCP server is configured, prefer the MCP tools:

- `nano_health`
- `nano_state`
- `nano_chat`
- `nano_anthropic_message`

Use `nano_health` first when the service has not been checked recently.

## Direct HTTP Fallback

OpenAI-compatible:

```bash
curl -sS http://127.0.0.1:8458/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-nano","messages":[{"role":"user","content":"Say ok"}]}'
```

Anthropic-compatible:

```bash
curl -sS http://127.0.0.1:8458/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: local' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"gemini-nano","max_tokens":128,"messages":[{"role":"user","content":"Say ok"}]}'
```
