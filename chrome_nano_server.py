import asyncio
import base64
import contextlib
import json
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from playwright.async_api import async_playwright
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MODEL = os.getenv("NANO_MODEL", "gemini-nano")
SERVER_HOST = os.getenv("NANO_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("NANO_PORT", "8458"))
INTERNAL_BASE_URL = os.getenv("NANO_INTERNAL_BASE_URL", f"http://127.0.0.1:{SERVER_PORT}")
CHROME_MODE = os.getenv("NANO_CHROME_MODE", "worker").strip().lower()
CHROME_CDP_URL = os.getenv("NANO_CDP_URL", "http://127.0.0.1:9222")
CHROME_HEADLESS = os.getenv("NANO_HEADLESS", "0").lower() in {"1", "true", "yes"}
CHROME_PROFILE = os.path.expanduser(
    os.getenv("NANO_CHROME_PROFILE", "~/.chrome-nano-server-profile")
)
WARMUP_TIMEOUT_MS = int(os.getenv("NANO_WARMUP_TIMEOUT_MS", "300000"))
WORKER_STALE_SECONDS = int(os.getenv("NANO_WORKER_STALE_SECONDS", "15"))
NANO_OUTPUT_LANGUAGE = os.getenv("NANO_OUTPUT_LANGUAGE", "en")
NANO_JOB_TIMEOUT_SECONDS = int(os.getenv("NANO_JOB_TIMEOUT_SECONDS", "180"))
NANO_STRICT_EXTENSION_WORKER = os.getenv("NANO_STRICT_EXTENSION_WORKER", "1").lower() in {"1", "true", "yes"}

BASE_CHROME_FEATURES = [
    "AIPromptAPI:langs/*",
    "AIPromptAPIMultimodalInput",
    "AIPromptAPIStructuredOutput",
    "OptimizationGuideModelExecution",
    "OptimizationGuideOnDeviceModel",
    "OnDeviceModelPerformanceParams:compatible_on_device_performance_classes/*/compatible_low_tier_on_device_performance_classes/*",
    "PromptAPIForGeminiNano",
]
PLAYWRIGHT_DISABLE_FEATURES_ARG = (
    "--disable-features="
    "AvoidUnnecessaryBeforeUnloadCheckSync,"
    "BoundaryEventDispatchTracksNodeRemoval,"
    "DestroyProfileOnBrowserClose,"
    "DialMediaRouteProvider,"
    "GlobalMediaControls,"
    "HttpsUpgrades,"
    "LensOverlay,"
    "MediaRouter,"
    "PaintHolding,"
    "ThirdPartyStoragePartitioning,"
    "Translate,"
    "AutoDeElevate,"
    "RenderDocument,"
    "OptimizationHints"
)
PLAYWRIGHT_SERVICE_IGNORE_DEFAULT_ARGS = [
    "--disable-field-trial-config",
    "--disable-background-networking",
    "--disable-client-side-phishing-detection",
    "--disable-component-extensions-with-background-pages",
    "--disable-component-update",
    "--disable-default-apps",
    PLAYWRIGHT_DISABLE_FEATURES_ARG,
    "--no-service-autorun",
]

@contextlib.asynccontextmanager
async def app_lifespan(_: FastAPI) -> AsyncIterator[None]:
    if CHROME_MODE != "worker":
        await startup_chrome()
    try:
        yield
    finally:
        if CHROME_MODE != "worker":
            await shutdown_chrome()


app = FastAPI(
    title="Chrome Gemini Nano Service",
    description=(
        "Local bridge for Chrome built-in Gemini Nano. It exposes OpenAI-compatible, "
        "OpenAI Responses-compatible, and Anthropic Messages-compatible endpoints."
    ),
    version="1.0.0",
    lifespan=app_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


chrome_context: Dict[str, Any] = {
    "playwright": None,
    "browser": None,
    "browser_context": None,
    "page": None,
    "lock": asyncio.Lock(),
    "binding_ready": False,
    "launch_mode": None,
    "owns_browser": False,
    "owns_browser_context": False,
    "last_error": None,
    "last_capability": None,
}
stream_queues: Dict[str, asyncio.Queue] = {}
worker_context: Dict[str, Any] = {
    "last_seen": None,
    "worker_id": None,
    "capability": None,
    "last_error": None,
    "current_job": None,
}
worker_pending_jobs: asyncio.Queue = asyncio.Queue()
worker_jobs: Dict[str, Dict[str, Any]] = {}
worker_events: Dict[str, asyncio.Queue] = {}
worker_cancelled_jobs: set[str] = set()
worker_registry: Dict[str, Dict[str, Any]] = {}


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChatMessage(FlexibleModel):
    role: str
    content: Any = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(FlexibleModel):
    model: Optional[str] = DEFAULT_MODEL
    messages: List[ChatMessage] = Field(default_factory=list)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    response_format: Optional[Dict[str, Any]] = None


class CompletionRequest(FlexibleModel):
    model: Optional[str] = DEFAULT_MODEL
    prompt: Any = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


class AnthropicMessage(FlexibleModel):
    role: str
    content: Any = ""


class AnthropicMessageRequest(FlexibleModel):
    model: Optional[str] = DEFAULT_MODEL
    messages: List[AnthropicMessage] = Field(default_factory=list)
    system: Optional[Any] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None


class ResponsesRequest(FlexibleModel):
    model: Optional[str] = DEFAULT_MODEL
    input: Any = ""
    instructions: Optional[Any] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    text: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None


def now() -> int:
    return int(time.time())


def request_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def model_name(value: Optional[str]) -> str:
    return value or DEFAULT_MODEL


def to_plain_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {"content": value}


def extract_image_payload(block: Dict[str, Any]) -> Optional[Dict[str, str]]:
    image_value: Any = None
    block_type = str(block.get("type", "")).lower()
    if block_type in {"image_url", "input_image"}:
        image_value = block.get("image_url") or block.get("url")
        if isinstance(image_value, dict):
            image_value = image_value.get("url")
    elif block_type == "image":
        image_value = block.get("source") or block.get("data") or block.get("value")

    mime_type = block.get("media_type") or block.get("mime_type") or "image/png"
    data = None
    if isinstance(image_value, dict):
        mime_type = image_value.get("media_type") or image_value.get("mime_type") or mime_type
        data = image_value.get("data") or image_value.get("base64")
    elif isinstance(image_value, str):
        if image_value.startswith("data:"):
            header, _, encoded = image_value.partition(",")
            if ";base64" in header:
                mime_type = header[5:].split(";", 1)[0] or mime_type
                data = encoded
        else:
            data = image_value

    if not data:
        return None
    return {"mimeType": str(mime_type), "data": str(data)}


def collect_images(content: Any) -> List[Dict[str, str]]:
    if content is None:
        return []
    if isinstance(content, list):
        images: List[Dict[str, str]] = []
        for item in content:
            images.extend(collect_images(item))
        return images
    if isinstance(content, dict):
        image = extract_image_payload(content)
        if image:
            return [image]
        return collect_images(content.get("content"))
    return []


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, list):
        parts = [content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        block_type = str(content.get("type", "")).lower()
        if block_type in {"text", "input_text", "output_text"}:
            return str(content.get("text", ""))
        if block_type in {"tool_result", "function_call_output"}:
            tool_id = content.get("tool_use_id") or content.get("call_id") or content.get("id")
            tool_body = content_to_text(content.get("content") or content.get("output"))
            return f"[tool result {tool_id or ''}]\n{tool_body}".strip()
        if block_type in {"tool_use", "function_call"}:
            name = content.get("name") or content.get("function", {}).get("name")
            args = content.get("input") or content.get("arguments") or content.get("function", {}).get("arguments")
            return f"[tool request {name or ''}]\n{json.dumps(args, ensure_ascii=False)}".strip()
        if block_type in {"image_url", "input_image", "image"}:
            return "[image input attached]"
        if "text" in content:
            return str(content.get("text", ""))
        if "content" in content:
            return content_to_text(content.get("content"))
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def map_role(role: str) -> str:
    normalized = (role or "user").lower()
    if normalized in {"system", "developer"}:
        return "system"
    if normalized == "assistant":
        return "assistant"
    return "user"


def normalize_messages(
    messages: Iterable[Dict[str, Any]],
    *,
    system: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    system_parts: List[str] = []
    prompt_messages: List[Dict[str, str]] = []
    transcript_parts: List[str] = []
    images: List[Dict[str, str]] = []

    system_text = content_to_text(system)
    if system_text:
        system_parts.append(system_text)
        transcript_parts.append(f"system: {system_text}")

    if tools:
        tools_note = (
            "The caller provided tool definitions, but this local Gemini Nano bridge "
            "returns text only. If a tool is needed, describe the intended tool call "
            "clearly in text."
        )
        system_parts.append(tools_note)
        transcript_parts.append(f"system: {tools_note}")

    for raw in messages:
        msg = to_plain_dict(raw)
        role = map_role(str(msg.get("role", "user")))
        content = msg.get("content", "")
        images.extend(collect_images(content))
        text = content_to_text(content)
        if msg.get("name"):
            text = f"{msg['name']}: {text}".strip()
        if msg.get("tool_call_id"):
            text = f"tool_call_id={msg['tool_call_id']}\n{text}".strip()
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        else:
            prompt_messages.append({"role": role, "content": text})
        transcript_parts.append(f"{role}: {text}")

    if not prompt_messages:
        prompt_messages.append({"role": "user", "content": "Respond to the instructions above."})
        transcript_parts.append("user: Respond to the instructions above.")

    transcript_parts.append("assistant:")
    initial_prompts = (
        [{"role": "system", "content": "\n\n".join(system_parts)}]
        if system_parts
        else []
    )
    return {
        "initialPrompts": initial_prompts,
        "promptMessages": prompt_messages,
        "fallbackPrompt": "\n".join(transcript_parts),
        "images": images,
    }


def response_constraint_from_openai(format_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not format_config:
        return None
    format_type = format_config.get("type")
    if format_type == "json_schema":
        return (
            format_config.get("json_schema", {}).get("schema")
            or format_config.get("schema")
            or None
        )
    return None


def add_json_instruction(payload: Dict[str, Any], format_config: Optional[Dict[str, Any]]) -> None:
    if not format_config:
        return
    if format_config.get("type") == "json_object":
        instruction = "Return only a valid JSON object."
        if payload["initialPrompts"]:
            payload["initialPrompts"][0]["content"] = (
                f"{payload['initialPrompts'][0]['content']}\n\n{instruction}"
            )
        else:
            payload["initialPrompts"].append({"role": "system", "content": instruction})


def build_payload(
    messages: Iterable[Dict[str, Any]],
    *,
    system: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = normalize_messages(messages, system=system, tools=tools)
    payload["responseConstraint"] = response_constraint_from_openai(response_format)
    payload["outputLanguage"] = NANO_OUTPUT_LANGUAGE
    payload["expectedInputs"] = [{"type": "text", "languages": [NANO_OUTPUT_LANGUAGE]}]
    if payload.get("images"):
        payload["expectedInputs"].append({"type": "image"})
    payload["expectedOutputs"] = [{"type": "text", "languages": [NANO_OUTPUT_LANGUAGE]}]
    add_json_instruction(payload, response_format)
    return payload


def prompt_text_from_payload(payload: Dict[str, Any]) -> str:
    return payload.get("fallbackPrompt", "")


def chrome_args() -> List[str]:
    features = os.getenv("NANO_CHROME_FEATURES")
    feature_list = [item.strip() for item in features.split(",")] if features else BASE_CHROME_FEATURES
    args = [
        f"--enable-features={','.join(item for item in feature_list if item)}",
        "--enable-experimental-web-platform-features",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    extra_args = os.getenv("NANO_CHROME_ARGS")
    if extra_args:
        args.extend(shlex.split(extra_args))
    return args


def cdp_setup_hint() -> str:
    return (
        f"Cannot connect to ordinary Chrome DevTools at {CHROME_CDP_URL}. "
        "Start ordinary Chrome with remote debugging first: "
        "./scripts/chrome-devtools.sh restart"
    )


async def inspect_ai_capability(page: Any) -> Dict[str, Any]:
    return await page.evaluate(
        """async () => {
            function getLanguageModelApi() {
                return globalThis.LanguageModel
                    || globalThis.ai?.languageModel
                    || globalThis.ai?.languageModel?.LanguageModel
                    || null;
            }
            async function availability(api) {
                try {
                    const options = {
                        expectedInputs: [{ type: "text", languages: ["en"] }],
                        expectedOutputs: [{ type: "text", languages: ["en"] }],
                    };
                    if (api?.availability) return await api.availability(options);
                    if (api?.capabilities) {
                        const caps = await api.capabilities(options);
                        return caps.available || caps.availability || "unknown";
                    }
                    return "unknown";
                } catch (error) {
                    return `error: ${error?.message || error}`;
                }
            }
            async function params(api) {
                try {
                    if (api?.params) return await api.params();
                    if (api?.capabilities) {
                        return await api.capabilities({
                            expectedInputs: [{ type: "text", languages: ["en"] }],
                            expectedOutputs: [{ type: "text", languages: ["en"] }],
                        });
                    }
                } catch (error) {
                    return { error: error?.message || String(error) };
                }
                return null;
            }
            const api = getLanguageModelApi();
            return {
                userAgent: navigator.userAgent,
                href: location.href,
                secureContext: globalThis.isSecureContext,
                hasLanguageModel: !!api,
                hasGlobalLanguageModel: typeof globalThis.LanguageModel !== "undefined",
                hasAiLanguageModel: !!globalThis.ai?.languageModel,
                availability: api ? await availability(api) : "not_supported",
                params: api ? await params(api) : null,
            };
        }"""
    )


async def launch_chrome() -> None:
    if chrome_context["page"] and not chrome_context["page"].is_closed():
        return

    p = chrome_context["playwright"] or await async_playwright().start()
    chrome_context["playwright"] = p

    if CHROME_MODE in {"cdp", "devtools", "existing", "ordinary"}:
        try:
            browser = await p.chromium.connect_over_cdp(CHROME_CDP_URL)
            browser_context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = await browser_context.new_page()
            chrome_context.update(
                {
                    "browser": browser,
                    "browser_context": browser_context,
                    "page": page,
                    "binding_ready": False,
                    "launch_mode": f"cdp:{CHROME_CDP_URL}",
                    "owns_browser": False,
                    "owns_browser_context": False,
                    "last_error": None,
                }
            )
            return
        except Exception as exc:
            chrome_context["last_error"] = f"{cdp_setup_hint()} Original error: {exc}"
            raise RuntimeError(chrome_context["last_error"])

    Path(CHROME_PROFILE).mkdir(parents=True, exist_ok=True)
    args = chrome_args()
    candidates = [
        {
            "name": "system-chrome-persistent",
            "kind": "persistent",
            "kwargs": {
                "user_data_dir": CHROME_PROFILE,
                "channel": "chrome",
                "headless": CHROME_HEADLESS,
                "args": args,
                "ignore_default_args": PLAYWRIGHT_SERVICE_IGNORE_DEFAULT_ARGS,
            },
        },
        {
            "name": "system-chrome",
            "kind": "browser",
            "kwargs": {
                "channel": "chrome",
                "headless": CHROME_HEADLESS,
                "args": args,
                "ignore_default_args": PLAYWRIGHT_SERVICE_IGNORE_DEFAULT_ARGS,
            },
        },
        {
            "name": "playwright-chromium",
            "kind": "browser",
            "kwargs": {
                "headless": CHROME_HEADLESS,
                "args": args,
                "ignore_default_args": PLAYWRIGHT_SERVICE_IGNORE_DEFAULT_ARGS,
            },
        },
    ]

    errors: List[str] = []
    for candidate in candidates:
        browser = None
        browser_context = None
        try:
            if candidate["kind"] == "persistent":
                browser_context = await p.chromium.launch_persistent_context(**candidate["kwargs"])
                page = browser_context.pages[0] if browser_context.pages else await browser_context.new_page()
            else:
                browser = await p.chromium.launch(**candidate["kwargs"])
                browser_context = await browser.new_context()
                page = await browser_context.new_page()

            chrome_context.update(
                {
                    "browser": browser,
                    "browser_context": browser_context,
                    "page": page,
                    "binding_ready": False,
                    "launch_mode": candidate["name"],
                    "owns_browser": candidate["kind"] == "browser",
                    "owns_browser_context": True,
                    "last_error": None,
                }
            )
            return
        except Exception as exc:
            errors.append(f"{candidate['name']}: {exc}")
            with contextlib.suppress(Exception):
                if browser_context:
                    await browser_context.close()
            with contextlib.suppress(Exception):
                if browser:
                    await browser.close()

    chrome_context["last_error"] = "; ".join(errors)
    raise RuntimeError(chrome_context["last_error"])


async def stream_push_event(event: Any) -> bool:
    if isinstance(event, str):
        with contextlib.suppress(Exception):
            event = json.loads(event)
    if not isinstance(event, dict):
        event = {"type": "delta", "delta": str(event)}
    queue = stream_queues.get(event.get("requestId"))
    if queue:
        await queue.put(event)
    return True


async def ensure_driver_page() -> Any:
    if not chrome_context["page"] or chrome_context["page"].is_closed():
        await launch_chrome()

    page = chrome_context["page"]
    if not chrome_context["binding_ready"]:
        try:
            await page.expose_function("__nanoPushEvent", stream_push_event)
        except Exception as exc:
            if "has been already registered" not in str(exc):
                raise
        chrome_context["binding_ready"] = True

    target_url = f"{INTERNAL_BASE_URL}/__nano_driver"
    if not page.url.startswith(target_url):
        await page.goto(target_url, wait_until="domcontentloaded")

    capability = await inspect_ai_capability(page)
    chrome_context["last_capability"] = capability
    return page


async def prepare_driver_later() -> None:
    await asyncio.sleep(0.5)
    try:
        async with chrome_context["lock"]:
            await ensure_driver_page()
    except Exception as exc:
        chrome_context["last_error"] = str(exc)


NANO_PROMPT_JS = """async (payload) => {
    function getLanguageModelApi() {
        return globalThis.LanguageModel
            || globalThis.ai?.languageModel
            || globalThis.ai?.languageModel?.LanguageModel
            || null;
    }
    function buildCreateOptions(payload) {
        const options = {};
        const language = payload.outputLanguage || "en";
        options.expectedInputs = payload.expectedInputs || [{ type: "text", languages: [language] }];
        options.expectedOutputs = payload.expectedOutputs || [{ type: "text", languages: [language] }];
        if (payload.initialPrompts && payload.initialPrompts.length) {
            options.initialPrompts = payload.initialPrompts;
        }
        return options;
    }
    async function createSession(api, payload) {
        const options = buildCreateOptions(payload);
        try {
            return await api.create(options);
        } catch (error) {
            if (options.initialPrompts) {
                return await api.create({
                    expectedInputs: options.expectedInputs,
                    expectedOutputs: options.expectedOutputs,
                });
            }
            throw error;
        }
    }
    async function promptWithFallback(session, payload) {
        const options = {};
        if (payload.responseConstraint) {
            options.responseConstraint = payload.responseConstraint;
        }
        try {
            return await session.prompt(payload.promptMessages, options);
        } catch (error) {
            return await session.prompt(payload.fallbackPrompt, options);
        }
    }
    const api = getLanguageModelApi();
    if (!api || !api.create) {
        throw new Error("Chrome built-in LanguageModel / ai.languageModel API is unavailable.");
    }
    const session = await createSession(api, payload);
    try {
        const result = await promptWithFallback(session, payload);
        return String(result ?? "");
    } finally {
        try { session.destroy?.(); } catch (_) {}
    }
}"""


NANO_STREAM_JS = """async (payload) => {
    function getLanguageModelApi() {
        return globalThis.LanguageModel
            || globalThis.ai?.languageModel
            || globalThis.ai?.languageModel?.LanguageModel
            || null;
    }
    function buildCreateOptions(payload) {
        const options = {};
        const language = payload.outputLanguage || "en";
        options.expectedInputs = payload.expectedInputs || [{ type: "text", languages: [language] }];
        options.expectedOutputs = payload.expectedOutputs || [{ type: "text", languages: [language] }];
        if (payload.initialPrompts && payload.initialPrompts.length) {
            options.initialPrompts = payload.initialPrompts;
        }
        return options;
    }
    async function createSession(api, payload) {
        const options = buildCreateOptions(payload);
        try {
            return await api.create(options);
        } catch (error) {
            if (options.initialPrompts) {
                return await api.create({
                    expectedInputs: options.expectedInputs,
                    expectedOutputs: options.expectedOutputs,
                });
            }
            throw error;
        }
    }
    async function sendDelta(payload, delta, fullText) {
        if (!delta) return;
        await globalThis.__nanoPushEvent({
            requestId: payload.requestId,
            type: "delta",
            delta,
            fullText,
        });
    }
    async function consumeStream(stream, payload) {
        let seen = "";
        for await (const chunk of stream) {
            const text = String(chunk ?? "");
            let delta = "";
            if (text.startsWith(seen)) {
                delta = text.slice(seen.length);
                seen = text;
            } else {
                delta = text;
                seen += text;
            }
            await sendDelta(payload, delta, seen);
        }
        return seen;
    }
    async function promptStreamingWithFallback(session, payload) {
        const options = {};
        if (payload.responseConstraint) {
            options.responseConstraint = payload.responseConstraint;
        }
        if (!session.promptStreaming) {
            const text = await session.prompt(payload.promptMessages, options).catch(
                () => session.prompt(payload.fallbackPrompt, options)
            );
            await sendDelta(payload, String(text ?? ""), String(text ?? ""));
            return String(text ?? "");
        }
        try {
            return await consumeStream(session.promptStreaming(payload.promptMessages, options), payload);
        } catch (error) {
            return await consumeStream(session.promptStreaming(payload.fallbackPrompt, options), payload);
        }
    }
    const api = getLanguageModelApi();
    if (!api || !api.create) {
        throw new Error("Chrome built-in LanguageModel / ai.languageModel API is unavailable.");
    }
    const session = await createSession(api, payload);
    try {
        return await promptStreamingWithFallback(session, payload);
    } finally {
        try { session.destroy?.(); } catch (_) {}
    }
}"""


async def nano_complete(payload: Dict[str, Any]) -> str:
    if CHROME_MODE == "worker":
        job_id = request_id("nanojob")
        result_queue: asyncio.Queue = asyncio.Queue()
        worker_events[job_id] = result_queue
        worker_jobs[job_id] = {
            "id": job_id,
            "type": "complete",
            "payload": payload,
            "created": now(),
            "status": "queued",
        }
        await worker_pending_jobs.put(job_id)
        try:
            while True:
                event = await asyncio.wait_for(result_queue.get(), timeout=NANO_JOB_TIMEOUT_SECONDS)
                if event.get("type") == "done":
                    return str(event.get("text", ""))
                if event.get("type") == "error":
                    raise RuntimeError(str(event.get("error", "Unknown worker error")))
        except asyncio.TimeoutError:
            worker_cancelled_jobs.add(job_id)
            raise RuntimeError(f"Chrome Nano worker timed out after {NANO_JOB_TIMEOUT_SECONDS}s")
        finally:
            worker_events.pop(job_id, None)
            worker_jobs.pop(job_id, None)

    async with chrome_context["lock"]:
        page = await ensure_driver_page()
        return await page.evaluate(NANO_PROMPT_JS, payload)


async def nano_stream(payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
    if CHROME_MODE == "worker":
        job_id = request_id("nanojob")
        result_queue: asyncio.Queue = asyncio.Queue()
        worker_events[job_id] = result_queue
        worker_jobs[job_id] = {
            "id": job_id,
            "type": "stream",
            "payload": payload,
            "created": now(),
            "status": "queued",
        }
        await worker_pending_jobs.put(job_id)
        try:
            while True:
                event = await asyncio.wait_for(result_queue.get(), timeout=NANO_JOB_TIMEOUT_SECONDS)
                if event.get("type") == "done":
                    break
                yield event
                if event.get("type") == "error":
                    break
        except asyncio.TimeoutError:
            worker_cancelled_jobs.add(job_id)
            yield {"type": "error", "error": f"Chrome Nano worker timed out after {NANO_JOB_TIMEOUT_SECONDS}s"}
        finally:
            worker_events.pop(job_id, None)
            worker_jobs.pop(job_id, None)
        return

    stream_id = request_id("stream")
    queue: asyncio.Queue = asyncio.Queue()
    stream_queues[stream_id] = queue
    stream_payload = dict(payload, requestId=stream_id)

    async def run_stream() -> None:
        try:
            async with chrome_context["lock"]:
                page = await ensure_driver_page()
                await page.evaluate(NANO_STREAM_JS, stream_payload)
        except Exception as exc:
            await queue.put({"type": "error", "error": str(exc)})
        finally:
            await queue.put({"type": "done"})

    task = asyncio.create_task(run_stream())
    try:
        while True:
            event = await queue.get()
            if event.get("type") == "done":
                break
            yield event
            if event.get("type") == "error":
                break
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(Exception):
                await task
        stream_queues.pop(stream_id, None)


def sse(data: Union[str, Dict[str, Any]], event: Optional[str] = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)
    return f"{prefix}data: {payload}\n\n"


def is_recent_worker(status: Dict[str, Any]) -> bool:
    last_seen = status.get("last_seen")
    return bool(last_seen and time.time() - float(last_seen) < WORKER_STALE_SECONDS)


def worker_kind(status: Dict[str, Any]) -> str:
    worker_id = str(status.get("worker_id") or "")
    href = str((status.get("capability") or {}).get("href") or "")
    if worker_id.startswith("extension-") or href.startswith("chrome-extension://"):
        return "extension"
    if href.startswith(f"{INTERNAL_BASE_URL}/worker") or href.startswith("/worker") or href.startswith("http://127.0.0.1:8458/worker"):
        return "visible-tab"
    return "unknown"


def best_worker_status() -> Optional[Dict[str, Any]]:
    recent = [status for status in worker_registry.values() if is_recent_worker(status)]
    if not recent:
        return None
    extension_workers = [status for status in recent if worker_kind(status) == "extension"]
    if extension_workers:
        return max(extension_workers, key=lambda item: item.get("last_seen") or 0)
    if NANO_STRICT_EXTENSION_WORKER:
        return None
    return max(recent, key=lambda item: item.get("last_seen") or 0)


def sync_worker_context() -> None:
    best = best_worker_status()
    if best:
        worker_context.update(
            {
                "last_seen": best.get("last_seen"),
                "worker_id": best.get("worker_id"),
                "capability": best.get("capability") or {},
                "last_error": best.get("last_error"),
                "kind": worker_kind(best),
            }
        )


def worker_can_take_jobs(status: Dict[str, Any]) -> bool:
    if not is_recent_worker(status):
        return False
    if NANO_STRICT_EXTENSION_WORKER:
        return worker_kind(status) == "extension"
    best = best_worker_status()
    return bool(best and best.get("worker_id") == status.get("worker_id"))


def worker_ready() -> bool:
    sync_worker_context()
    return bool(worker_context.get("last_seen") and worker_context.get("capability", {}).get("hasLanguageModel"))


def worker_available() -> Optional[str]:
    sync_worker_context()
    capability = worker_context.get("capability") or {}
    return capability.get("availability")


def openai_chat_response(
    *,
    completion_id: str,
    created: int,
    model: str,
    prompt: str,
    text: str,
) -> Dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": estimate_tokens(prompt),
            "completion_tokens": estimate_tokens(text),
            "total_tokens": estimate_tokens(prompt) + estimate_tokens(text),
        },
    }


async def openai_chat_stream(
    *,
    completion_id: str,
    created: int,
    model: str,
    payload: Dict[str, Any],
) -> AsyncIterator[str]:
    yield sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )
    async for event in nano_stream(payload):
        if event.get("type") == "error":
            yield sse({"error": event.get("error", "Unknown Chrome Nano stream error")})
            break
        delta = event.get("delta", "")
        if not delta:
            continue
        yield sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            }
        )
    yield sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield "data: [DONE]\n\n"


async def startup_chrome() -> None:
    try:
        await launch_chrome()
        asyncio.create_task(prepare_driver_later())
    except Exception as exc:
        chrome_context["last_error"] = str(exc)


async def shutdown_chrome() -> None:
    with contextlib.suppress(Exception):
        if chrome_context["page"] and not chrome_context["page"].is_closed():
            await chrome_context["page"].close()
    with contextlib.suppress(Exception):
        if chrome_context["browser_context"] and chrome_context.get("owns_browser_context"):
            await chrome_context["browser_context"].close()
    with contextlib.suppress(Exception):
        if chrome_context["browser"] and chrome_context.get("owns_browser"):
            await chrome_context["browser"].close()
    with contextlib.suppress(Exception):
        if chrome_context["playwright"]:
            await chrome_context["playwright"].stop()


@app.get("/", include_in_schema=False)
async def root() -> Dict[str, Any]:
    return {
        "name": "Chrome Gemini Nano Service",
        "model": DEFAULT_MODEL,
        "chrome_mode": CHROME_MODE,
        "cdp_url": CHROME_CDP_URL,
        "endpoints": [
            "/health",
            "/setup",
            "/worker",
            "/admin/warmup",
            "/admin/state",
            "/v1/models",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/messages",
        ],
    }


@app.get("/__nano_driver", include_in_schema=False)
async def nano_driver() -> HTMLResponse:
    return HTMLResponse(
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Chrome Gemini Nano Driver</title>
  <style>
    body { font: 14px/1.5 system-ui, sans-serif; margin: 32px; max-width: 760px; }
    button { font: inherit; padding: 8px 12px; }
    pre { background: #111; color: #d6f5d6; padding: 12px; overflow: auto; }
  </style>
</head>
<body>
  <h1>Chrome Gemini Nano Driver</h1>
  <button id="warmup">Warm up / download model</button>
  <pre id="status">Loading...</pre>
  <script>
    const statusEl = document.getElementById("status");
    function write(value) {
      statusEl.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    }
    function getLanguageModelApi() {
      return globalThis.LanguageModel
        || globalThis.ai?.languageModel
        || globalThis.ai?.languageModel?.LanguageModel
        || null;
    }
    async function availability(api) {
      if (!api) return "not_supported";
      const options = {
        expectedInputs: [{type: "text", languages: ["en"]}],
        expectedOutputs: [{type: "text", languages: ["en"]}],
      };
      if (api.availability) return await api.availability(options);
      if (api.capabilities) {
        const caps = await api.capabilities(options);
        return caps.available || caps.availability || "unknown";
      }
      return "unknown";
    }
    window.__lastWarmup = null;
    window.__nanoWarmup = async function () {
      const api = getLanguageModelApi();
      window.__lastWarmup = { done: false, ok: false, availability: await availability(api) };
      write(window.__lastWarmup);
      if (!api || !api.create) {
        throw new Error("LanguageModel API is unavailable in this Chrome profile.");
      }
      const before = await availability(api);
      const session = await api.create({
        monitor(monitor) {
          monitor.addEventListener("downloadprogress", (event) => {
            window.__lastWarmup = {
              done: false,
              ok: false,
              availability: before,
              progress: event.loaded,
            };
            write(window.__lastWarmup);
          });
        },
      });
      session.destroy?.();
      const after = await availability(api);
      window.__lastWarmup = { done: true, ok: true, availabilityBefore: before, availabilityAfter: after };
      write(window.__lastWarmup);
      return window.__lastWarmup;
    };
    document.getElementById("warmup").addEventListener("click", () => {
      window.__nanoWarmup().catch((error) => {
        window.__lastWarmup = { done: true, ok: false, error: error.message || String(error) };
        write(window.__lastWarmup);
      });
    });
    availability(getLanguageModelApi()).then((value) => write({ availability: value }));
  </script>
</body>
</html>"""
    )


@app.get("/worker", include_in_schema=False)
async def worker_page() -> HTMLResponse:
    return HTMLResponse(
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Chrome Gemini Nano Worker</title>
  <style>
    :root { color-scheme: light dark; }
    body { font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 28px; max-width: 920px; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    pre { background: CanvasText; color: Canvas; padding: 14px; overflow: auto; border-radius: 6px; }
    .ok { color: #15803d; }
    .bad { color: #b91c1c; }
  </style>
</head>
<body>
  <header>
    <h1>Chrome Gemini Nano Worker</h1>
    <strong id="badge">starting</strong>
  </header>
  <p>Keep this tab open. NanoServer sends local API requests here so ordinary Chrome can call <code>LanguageModel</code>.</p>
  <pre id="status">Booting...</pre>
  <script>
    const workerId = crypto.randomUUID();
    const statusEl = document.getElementById("status");
    const badgeEl = document.getElementById("badge");
    let busy = false;
    let handled = 0;

    function write(value) {
      statusEl.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    }
    function setBadge(text, ok = true) {
      badgeEl.textContent = text;
      badgeEl.className = ok ? "ok" : "bad";
      document.title = `Nano Worker: ${text}`;
    }
    function getLanguageModelApi() {
      return globalThis.LanguageModel
        || globalThis.ai?.languageModel
        || globalThis.ai?.languageModel?.LanguageModel
        || null;
    }
    async function availability(api) {
      try {
        if (!api) return "not_supported";
        const options = {
          expectedInputs: [{type: "text", languages: ["en"]}],
          expectedOutputs: [{type: "text", languages: ["en"]}],
        };
        if (api.availability) return await api.availability(options);
        if (api.capabilities) {
          const caps = await api.capabilities(options);
          return caps.available || caps.availability || "unknown";
        }
        return "unknown";
      } catch (error) {
        return `error: ${error?.message || error}`;
      }
    }
    async function capability() {
      const api = getLanguageModelApi();
      return {
        href: location.href,
        userAgent: navigator.userAgent,
        secureContext: globalThis.isSecureContext,
        hasLanguageModel: !!api,
        hasGlobalLanguageModel: typeof globalThis.LanguageModel !== "undefined",
        hasAiLanguageModel: !!globalThis.ai?.languageModel,
        availability: await availability(api),
      };
    }
    function createOptions(payload) {
      const options = {};
      const language = payload.outputLanguage || "en";
      options.expectedInputs = payload.expectedInputs || [{type: "text", languages: [language]}];
      options.expectedOutputs = payload.expectedOutputs || [{type: "text", languages: [language]}];
      if (payload.initialPrompts?.length) options.initialPrompts = payload.initialPrompts;
      return options;
    }
    async function createSession(api, payload) {
      const options = createOptions(payload);
      try {
        return await api.create(options);
      } catch (error) {
        if (options.initialPrompts) {
          return await api.create({
            expectedInputs: options.expectedInputs,
            expectedOutputs: options.expectedOutputs,
          });
        }
        throw error;
      }
    }
    async function promptOnce(session, payload) {
      const options = {};
      if (payload.responseConstraint) options.responseConstraint = payload.responseConstraint;
      try {
        return await session.prompt(payload.promptMessages, options);
      } catch (error) {
        return await session.prompt(payload.fallbackPrompt, options);
      }
    }
    async function postEvent(jobId, event) {
      await fetch(`/worker/jobs/${encodeURIComponent(jobId)}/event`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(event),
      });
    }
    async function complete(job) {
      const api = getLanguageModelApi();
      if (!api?.create) throw new Error("LanguageModel API is unavailable in this Chrome tab.");
      const session = await createSession(api, job.payload);
      try {
        const text = String(await promptOnce(session, job.payload) ?? "");
        await postEvent(job.id, {type: "done", text});
      } finally {
        try { session.destroy?.(); } catch (_) {}
      }
    }
    async function stream(job) {
      const api = getLanguageModelApi();
      if (!api?.create) throw new Error("LanguageModel API is unavailable in this Chrome tab.");
      const session = await createSession(api, job.payload);
      try {
        const options = {};
        if (job.payload.responseConstraint) options.responseConstraint = job.payload.responseConstraint;
        if (!session.promptStreaming) {
          const text = String(await promptOnce(session, job.payload) ?? "");
          if (text) await postEvent(job.id, {type: "delta", delta: text, fullText: text});
          await postEvent(job.id, {type: "done", text});
          return;
        }
        let streamSource;
        try {
          streamSource = session.promptStreaming(job.payload.promptMessages, options);
        } catch (error) {
          streamSource = session.promptStreaming(job.payload.fallbackPrompt, options);
        }
        let seen = "";
        for await (const chunk of streamSource) {
          const text = String(chunk ?? "");
          const delta = text.startsWith(seen) ? text.slice(seen.length) : text;
          seen = text.startsWith(seen) ? text : seen + text;
          if (delta) await postEvent(job.id, {type: "delta", delta, fullText: seen});
        }
        await postEvent(job.id, {type: "done", text: seen});
      } finally {
        try { session.destroy?.(); } catch (_) {}
      }
    }
    async function runJob(job) {
      busy = true;
      setBadge("busy");
      write({workerId, busy, job: job.id, handled, capability: await capability()});
      try {
        if (job.type === "stream") await stream(job);
        else await complete(job);
        handled += 1;
      } catch (error) {
        await postEvent(job.id, {type: "error", error: error?.message || String(error)});
      } finally {
        busy = false;
      }
    }
    async function poll() {
      while (true) {
        try {
          const cap = await capability();
          const response = await fetch("/worker/poll", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({workerId, busy, handled, capability: cap}),
          });
          if (!response.ok) throw new Error(`poll ${response.status}`);
          const data = await response.json();
          setBadge(cap.availability || "connected", cap.hasLanguageModel);
          write({workerId, busy, handled, capability: cap, nextJob: data.job?.id || null});
          if (data.job) await runJob(data.job);
        } catch (error) {
          setBadge("disconnected", false);
          write({workerId, error: error?.message || String(error)});
          await new Promise(resolve => setTimeout(resolve, 1500));
        }
      }
    }
    poll();
  </script>
</body>
</html>"""
    )


@app.get("/setup", include_in_schema=False)
async def setup_page() -> HTMLResponse:
    return await worker_page() if CHROME_MODE == "worker" else await nano_driver()


@app.post("/worker/poll")
async def worker_poll(status: Dict[str, Any]) -> Dict[str, Any]:
    worker_id = status.get("workerId") or "unknown"
    worker_status = {
        "last_seen": time.time(),
        "worker_id": worker_id,
        "capability": status.get("capability") or {},
        "busy": bool(status.get("busy")),
        "handled": status.get("handled"),
        "last_error": None,
    }
    worker_status["kind"] = worker_kind(worker_status)
    worker_registry[str(worker_id)] = worker_status
    sync_worker_context()
    if status.get("busy"):
        return {"job": None}
    if not worker_can_take_jobs(worker_status):
        return {"job": None}
    try:
        job_id = await asyncio.wait_for(worker_pending_jobs.get(), timeout=25)
    except asyncio.TimeoutError:
        return {"job": None}

    job = worker_jobs.get(job_id)
    if not job:
        return {"job": None}
    job["status"] = "running"
    job["worker_id"] = worker_id
    worker_context["current_job"] = job_id
    return {"job": {"id": job_id, "type": job["type"], "payload": job["payload"]}}


@app.post("/worker/jobs/{job_id}/event")
async def worker_job_event(job_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
    if job_id in worker_cancelled_jobs:
        worker_cancelled_jobs.discard(job_id)
        return {"ok": False, "cancelled": True}
    queue = worker_events.get(job_id)
    if not queue:
        return {"ok": False, "ignored": True}
    await queue.put(event)
    if event.get("type") in {"done", "error"}:
        if worker_context.get("current_job") == job_id:
            worker_context["current_job"] = None
        worker_jobs.get(job_id, {})["status"] = event.get("type")
    return {"ok": True}


@app.post("/admin/clear-jobs")
async def clear_jobs() -> Dict[str, Any]:
    cleared = list(worker_jobs.keys())
    for job_id, queue in list(worker_events.items()):
        await queue.put({"type": "error", "error": "Job cleared by admin"})
        worker_cancelled_jobs.add(job_id)
    worker_jobs.clear()
    worker_events.clear()
    while not worker_pending_jobs.empty():
        with contextlib.suppress(Exception):
            worker_pending_jobs.get_nowait()
    worker_context["current_job"] = None
    return {"ok": True, "cleared": cleared}


@app.get("/health")
@app.get("/ready")
@app.get("/v1/health")
async def health() -> Dict[str, Any]:
    if CHROME_MODE == "worker":
        sync_worker_context()
        capability = worker_context.get("capability") or {}
        availability = capability.get("availability")
        last_seen = worker_context.get("last_seen")
        connected = bool(last_seen and time.time() - last_seen < WORKER_STALE_SECONDS)
        return {
            "ok": connected and bool(capability.get("hasLanguageModel")),
            "ready": connected and availability in {"readily", "yes", "available"},
            "model": DEFAULT_MODEL,
            "chrome_mode": CHROME_MODE,
            "worker_url": f"{INTERNAL_BASE_URL}/worker",
            "worker_connected": connected,
            "worker_stale_seconds": WORKER_STALE_SECONDS,
            "worker_id": worker_context.get("worker_id"),
            "worker_kind": worker_context.get("kind"),
            "strict_extension_worker": NANO_STRICT_EXTENSION_WORKER,
            "availability": availability,
            "capability": capability or None,
            "pending_jobs": worker_pending_jobs.qsize(),
            "active_jobs": len(worker_jobs),
            "last_error": worker_context.get("last_error"),
        }

    capability = chrome_context.get("last_capability")
    try:
        async with chrome_context["lock"]:
            page = await ensure_driver_page()
            capability = await inspect_ai_capability(page)
            chrome_context["last_capability"] = capability
    except Exception as exc:
        chrome_context["last_error"] = str(exc)

    availability = (capability or {}).get("availability")
    ready_values = {"readily", "yes", "available"}
    return {
        "ok": bool(capability and capability.get("hasLanguageModel")),
        "ready": availability in ready_values,
        "model": DEFAULT_MODEL,
        "chrome_mode": CHROME_MODE,
        "cdp_url": CHROME_CDP_URL,
        "launch_mode": chrome_context.get("launch_mode"),
        "availability": availability,
        "capability": capability,
        "last_error": chrome_context.get("last_error"),
    }


@app.post("/admin/warmup")
async def warmup() -> Dict[str, Any]:
    if CHROME_MODE == "worker":
        payload = build_payload([{"role": "user", "content": "Reply with OK."}])
        try:
            text = await asyncio.wait_for(nano_complete(payload), timeout=WARMUP_TIMEOUT_MS / 1000)
            return {"ok": True, "text": text, "capability": worker_context.get("capability")}
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "capability": worker_context.get("capability"),
                "worker_url": f"{INTERNAL_BASE_URL}/worker",
            }

    try:
        async with chrome_context["lock"]:
            page = await ensure_driver_page()
            await page.click("#warmup")
            timed_out = False
            try:
                await page.wait_for_function(
                    "window.__lastWarmup && window.__lastWarmup.done === true",
                    timeout=WARMUP_TIMEOUT_MS,
                )
            except Exception as exc:
                if "Timeout" not in str(exc):
                    raise
                timed_out = True
            result = await page.evaluate("window.__lastWarmup")
            capability = await inspect_ai_capability(page)
            chrome_context["last_capability"] = capability
        return {
            "ok": bool(result and result.get("ok")),
            "timeout": timed_out,
            "warmup": result,
            "capability": capability,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chrome Nano warmup failed: {exc}")


@app.get("/admin/state")
async def admin_state() -> Dict[str, Any]:
    if CHROME_MODE == "worker":
        sync_worker_context()
        capability = worker_context.get("capability") or {}
        last_seen = worker_context.get("last_seen")
        return {
            "ok": bool(last_seen and time.time() - last_seen < WORKER_STALE_SECONDS and capability.get("hasLanguageModel")),
            "ready": bool(last_seen and time.time() - last_seen < WORKER_STALE_SECONDS and capability.get("availability") in {"readily", "yes", "available"}),
            "chrome_mode": CHROME_MODE,
            "worker_url": f"{INTERNAL_BASE_URL}/worker",
            "worker_id": worker_context.get("worker_id"),
            "worker_kind": worker_context.get("kind"),
            "last_seen": last_seen,
            "worker_stale_seconds": WORKER_STALE_SECONDS,
            "strict_extension_worker": NANO_STRICT_EXTENSION_WORKER,
            "workers": {
                worker_id: {
                    "kind": status.get("kind"),
                    "last_seen": status.get("last_seen"),
                    "recent": is_recent_worker(status),
                    "busy": status.get("busy"),
                    "handled": status.get("handled"),
                    "href": (status.get("capability") or {}).get("href"),
                    "hasLanguageModel": (status.get("capability") or {}).get("hasLanguageModel"),
                    "availability": (status.get("capability") or {}).get("availability"),
                }
                for worker_id, status in worker_registry.items()
            },
            "warmup": None,
            "capability": capability or None,
            "pending_jobs": worker_pending_jobs.qsize(),
            "active_jobs": len(worker_jobs),
            "current_job": worker_context.get("current_job"),
            "last_error": worker_context.get("last_error"),
        }

    async with chrome_context["lock"]:
        page = await ensure_driver_page()
        warmup_state = await page.evaluate("window.__lastWarmup || null")
        capability = await inspect_ai_capability(page)
        chrome_context["last_capability"] = capability
    return {
        "ok": bool(capability.get("hasLanguageModel")),
        "ready": capability.get("availability") in {"readily", "yes", "available"},
        "chrome_mode": CHROME_MODE,
        "cdp_url": CHROME_CDP_URL,
        "launch_mode": chrome_context.get("launch_mode"),
        "warmup": warmup_state,
        "capability": capability,
        "last_error": chrome_context.get("last_error"),
    }


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    created = now()
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": created,
                "owned_by": "chrome",
                "type": "model",
                "display_name": "Chrome Gemini Nano",
            }
        ],
        "has_more": False,
        "first_id": DEFAULT_MODEL,
        "last_id": DEFAULT_MODEL,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Any:
    model = model_name(request.model)
    completion_id = request_id("chatcmpl")
    created = now()
    payload = build_payload(
        request.messages,
        response_format=request.response_format,
    )

    if request.stream:
        return StreamingResponse(
            openai_chat_stream(
                completion_id=completion_id,
                created=created,
                model=model,
                payload=payload,
            ),
            media_type="text/event-stream",
        )

    try:
        text = await nano_complete(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chrome Nano execution failed: {exc}")
    return openai_chat_response(
        completion_id=completion_id,
        created=created,
        model=model,
        prompt=prompt_text_from_payload(payload),
        text=text,
    )


@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> Any:
    prompt = content_to_text(request.prompt)
    payload = build_payload([{"role": "user", "content": prompt}])
    model = model_name(request.model)
    completion_id = request_id("cmpl")
    created = now()

    if request.stream:
        async def completion_stream() -> AsyncIterator[str]:
            async for chunk in openai_chat_stream(
                completion_id=completion_id,
                created=created,
                model=model,
                payload=payload,
            ):
                yield chunk

        return StreamingResponse(completion_stream(), media_type="text/event-stream")

    try:
        text = await nano_complete(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chrome Nano execution failed: {exc}")
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": estimate_tokens(prompt),
            "completion_tokens": estimate_tokens(text),
            "total_tokens": estimate_tokens(prompt) + estimate_tokens(text),
        },
    }


def responses_input_to_messages(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, dict):
        role = value.get("role", "user")
        return [{"role": role, "content": value.get("content") or value.get("text") or value}]
    if isinstance(value, list):
        messages: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                messages.append({"role": "user", "content": content_to_text(item)})
                continue
            item_type = item.get("type")
            if item_type == "message" or "role" in item:
                messages.append(
                    {
                        "role": item.get("role", "user"),
                        "content": item.get("content") or item.get("text") or "",
                    }
                )
            elif item_type in {"input_text", "output_text"}:
                messages.append({"role": "user", "content": item.get("text", "")})
            elif item_type in {"function_call_output", "tool_result"}:
                messages.append({"role": "tool", "content": item})
            else:
                messages.append({"role": "user", "content": item})
        return messages
    return [{"role": "user", "content": content_to_text(value)}]


def response_object(
    *,
    response_id: str,
    created: int,
    model: str,
    prompt: str,
    text: str,
) -> Dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "model": model,
        "output": [
            {
                "id": request_id("msg"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": text,
        "usage": {
            "input_tokens": estimate_tokens(prompt),
            "output_tokens": estimate_tokens(text),
            "total_tokens": estimate_tokens(prompt) + estimate_tokens(text),
        },
    }


async def responses_stream(
    *,
    response_id: str,
    created: int,
    model: str,
    payload: Dict[str, Any],
) -> AsyncIterator[str]:
    item_id = request_id("msg")
    content_index = 0
    output_index = 0
    full_text = ""
    base_response = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "in_progress",
        "model": model,
        "output": [],
    }
    yield sse({"type": "response.created", "response": base_response}, "response.created")
    yield sse(
        {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []},
        },
        "response.output_item.added",
    )
    yield sse(
        {
            "type": "response.content_part.added",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        "response.content_part.added",
    )
    async for event in nano_stream(payload):
        if event.get("type") == "error":
            yield sse(
                {
                    "type": "response.failed",
                    "response": dict(base_response, status="failed", error={"message": event.get("error")}),
                },
                "response.failed",
            )
            yield "data: [DONE]\n\n"
            return
        delta = event.get("delta", "")
        if not delta:
            continue
        full_text += delta
        yield sse(
            {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "delta": delta,
            },
            "response.output_text.delta",
        )
    yield sse(
        {
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "text": full_text,
        },
        "response.output_text.done",
    )
    yield sse(
        {
            "type": "response.completed",
            "response": response_object(
                response_id=response_id,
                created=created,
                model=model,
                prompt=prompt_text_from_payload(payload),
                text=full_text,
            ),
        },
        "response.completed",
    )
    yield "data: [DONE]\n\n"


@app.post("/v1/responses")
async def responses(request: ResponsesRequest) -> Any:
    model = model_name(request.model)
    response_id = request_id("resp")
    created = now()
    response_format = request.response_format
    if request.text and request.text.get("format"):
        response_format = request.text.get("format")
    payload = build_payload(
        responses_input_to_messages(request.input),
        system=request.instructions,
        tools=request.tools,
        response_format=response_format,
    )

    if request.stream:
        return StreamingResponse(
            responses_stream(
                response_id=response_id,
                created=created,
                model=model,
                payload=payload,
            ),
            media_type="text/event-stream",
        )

    try:
        text = await nano_complete(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chrome Nano execution failed: {exc}")
    return response_object(
        response_id=response_id,
        created=created,
        model=model,
        prompt=prompt_text_from_payload(payload),
        text=text,
    )


def anthropic_prompt_payload(request: AnthropicMessageRequest) -> Dict[str, Any]:
    return build_payload(
        request.messages,
        system=request.system,
        tools=request.tools,
    )


def anthropic_response(
    *,
    message_id: str,
    model: str,
    prompt: str,
    text: str,
) -> Dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": estimate_tokens(prompt),
            "output_tokens": estimate_tokens(text),
        },
    }


async def anthropic_stream(
    *,
    message_id: str,
    model: str,
    payload: Dict[str, Any],
) -> AsyncIterator[str]:
    yield sse(
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": estimate_tokens(prompt_text_from_payload(payload)), "output_tokens": 0},
            },
        },
        "message_start",
    )
    yield sse(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        "content_block_start",
    )
    output_tokens = 0
    async for event in nano_stream(payload):
        if event.get("type") == "error":
            yield sse({"type": "error", "error": {"type": "api_error", "message": event.get("error")}}, "error")
            return
        delta = event.get("delta", "")
        if not delta:
            continue
        output_tokens += estimate_tokens(delta)
        yield sse(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": delta},
            },
            "content_block_delta",
        )
    yield sse({"type": "content_block_stop", "index": 0}, "content_block_stop")
    yield sse(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
        "message_delta",
    )
    yield sse({"type": "message_stop"}, "message_stop")


@app.post("/v1/messages")
async def anthropic_messages(request: AnthropicMessageRequest) -> Any:
    model = model_name(request.model)
    message_id = request_id("msg")
    payload = anthropic_prompt_payload(request)

    if request.stream:
        return StreamingResponse(
            anthropic_stream(message_id=message_id, model=model, payload=payload),
            media_type="text/event-stream",
        )

    try:
        text = await nano_complete(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chrome Nano execution failed: {exc}")
    return anthropic_response(
        message_id=message_id,
        model=model,
        prompt=prompt_text_from_payload(payload),
        text=text,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
