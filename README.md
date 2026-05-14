# Chrome Gemini Nano Service

Local service bridge for Chrome built-in Gemini Nano. It exposes:

- OpenAI Chat Completions: `POST /v1/chat/completions`
- OpenAI legacy Completions: `POST /v1/completions`
- OpenAI Responses: `POST /v1/responses`
- Anthropic Messages: `POST /v1/messages`
- Models: `GET /v1/models`
- Health: `GET /health`
- Warmup: `POST /admin/warmup`
- Diagnostics: `GET /admin/state`

## Start

NanoServer uses worker mode by default. The recommended worker is a Chrome
extension offscreen document, which runs hidden in ordinary Chrome and reuses
the Gemini Nano model already downloaded in your normal Chrome profile.

Install the extension once:

```bash
./scripts/install-extension.sh
```

Then use the service normally:

Foreground:

```bash
source .venv/bin/activate
pip install -r requirements.txt
./scripts/run-foreground.sh
```

macOS background service:

```bash
./scripts/nano-service.sh start
./scripts/nano-service.sh restart
./scripts/nano-service.sh status
./scripts/nano-service.sh logs
./scripts/nano-service.sh stop
```

`status` should show a listener on `127.0.0.1:8458` and
`worker_connected: true`, `worker_kind: extension`, and
`sourceVersion` in the capability block. `restart` waits briefly with the
server stopped so Chrome drops any old offscreen document and reloads the
latest extension worker script.

The service starts ordinary Google Chrome if it is not already running and
tries to wake the extension through `chrome-extension://.../kick.html`. This
does not open the visible `/worker` task page. If your unpacked extension id is
different, set:

```bash
export NANO_EXTENSION_ID=your_extension_id
```

Visible tab fallback is disabled for normal job dispatch when the extension
worker is present, so accidentally opening `/worker` cannot steal Claude Code
or Codex requests:

```bash
./scripts/nano-service.sh worker
```

The default service URL is:

```text
http://127.0.0.1:8458
```

If Gemini Nano is not ready yet, warm it up from the service Chrome profile:

```bash
curl -X POST http://127.0.0.1:8458/admin/warmup
```

## Codex / OpenAI-Compatible Clients

Point the client at:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8458/v1
export OPENAI_API_KEY=local
```

Chat test:

```bash
curl http://127.0.0.1:8458/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local' \
  -d '{
    "model": "gemini-nano",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

Responses test:

```bash
curl http://127.0.0.1:8458/v1/responses \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local' \
  -d '{
    "model": "gemini-nano",
    "input": "Say hello in one sentence."
  }'
```

## Claude Code / Anthropic-Compatible Clients

Point the client at:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8458
export ANTHROPIC_AUTH_TOKEN=local
```

Messages test:

```bash
curl http://127.0.0.1:8458/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: local' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "gemini-nano",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

Note: Gemini Nano is a local text model bridge here. Tool calls, images, and long-context coding-agent behavior are flattened to text rather than executed as native model tool calls.

## MCP Tool Server

For agents that support MCP tools, run:

```bash
./scripts/gemini-nano-mcp.py
```

Claude Code:

```bash
claude mcp add chrome-gemini-nano -- ./scripts/gemini-nano-mcp.py
```

Codex config example:

```toml
[mcp_servers.chrome-gemini-nano]
command = "./scripts/gemini-nano-mcp.py"
```

Available MCP tools:

- `nano_health`: check readiness
- `nano_state`: inspect worker diagnostics
- `nano_chat`: ask Nano through the OpenAI-compatible endpoint
- `nano_anthropic_message`: ask Nano through the Anthropic-compatible endpoint

The MCP server refuses model calls when NanoServer is not ready, so agents do
not silently pile up queued jobs.

## Codex Skill

A repo-local Codex skill is available at:

```text
skills/chrome-gemini-nano/SKILL.md
```

Install or copy it into your Codex skills directory if you want Codex to learn
when to use Nano as a local helper.

## Environment

```bash
NANO_HOST=127.0.0.1
NANO_PORT=8458
NANO_MODEL=gemini-nano
NANO_HEADLESS=0
NANO_CHROME_MODE=worker
NANO_OUTPUT_LANGUAGE=en  # Chrome 148 currently supports en, es, ja
NANO_WORKER_STALE_SECONDS=15
NANO_STRICT_EXTENSION_WORKER=1
NANO_START_CHROME=1
NANO_EXTENSION_ID=bljhlmplinefjciffblfbomfapnollmb
NANO_CDP_URL=http://127.0.0.1:9222  # only used in cdp mode
NANO_CHROME_PROFILE=~/.chrome-nano-server-profile  # only used in launch mode
NANO_WARMUP_TIMEOUT_MS=300000
```
