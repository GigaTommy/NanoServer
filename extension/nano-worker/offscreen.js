const BASE_URL = "http://127.0.0.1:8458";
const DEFAULT_LANGUAGE = "en";
const EXTENSION_WORKER_VERSION = "1.0.7";
const workerId = `extension-${chrome.runtime.id}-${crypto.randomUUID()}`;

let busy = false;
let handled = 0;
let consecutiveFailures = 0;
let currentSession = null;
let currentSessionKey = "";
let polling = false;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function resolveLanguageModel() {
  if (globalThis.LanguageModel) return globalThis.LanguageModel;
  if (globalThis.ai?.languageModel) return globalThis.ai.languageModel;
  if (globalThis.window?.ai?.languageModel) return globalThis.window.ai.languageModel;
  return null;
}

function languageOptions(payload = {}) {
  const language = payload.outputLanguage || DEFAULT_LANGUAGE;
  return {
    expectedInputs: payload.expectedInputs || [{ type: "text", languages: [language] }],
    expectedOutputs: payload.expectedOutputs || [{ type: "text", languages: [language] }],
  };
}

function sessionKey(payload = {}) {
  return JSON.stringify({
    language: payload.outputLanguage || DEFAULT_LANGUAGE,
    initialPrompts: payload.initialPrompts || [],
    expectedInputs: payload.expectedInputs || null,
    expectedOutputs: payload.expectedOutputs || null,
  });
}

async function availability(api, payload = {}) {
  try {
    if (!api) return "not_supported";
    const options = languageOptions(payload);
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
  const api = resolveLanguageModel();
  return {
    href: location.href,
    extensionId: chrome.runtime.id,
    sourceVersion: EXTENSION_WORKER_VERSION,
    userAgent: navigator.userAgent,
    secureContext: globalThis.isSecureContext,
    hasLanguageModel: !!api,
    hasGlobalLanguageModel: typeof globalThis.LanguageModel !== "undefined",
    hasAiLanguageModel: !!globalThis.ai?.languageModel,
    availability: await availability(api),
  };
}

function createOptions(payload) {
  const options = languageOptions(payload);
  if (payload.initialPrompts?.length) {
    options.initialPrompts = payload.initialPrompts;
  }
  return options;
}

async function ensureSession(payload) {
  const api = resolveLanguageModel();
  if (!api) throw new Error("LanguageModel is not available in this Chrome extension context.");
  if (typeof api.create !== "function") {
    throw new Error("No compatible LanguageModel.create constructor was found.");
  }

  const key = sessionKey(payload);
  if (currentSession && currentSessionKey === key) {
    return currentSession;
  }

  try { currentSession?.destroy?.(); } catch (_) {}
  currentSession = null;
  currentSessionKey = "";

  const status = await availability(api, payload);
  if (status === "unavailable" || status === "not_supported") {
    throw new Error(`LanguageModel is ${status} in this Chrome extension context.`);
  }

  currentSession = await api.create(createOptions(payload));
  currentSessionKey = key;
  return currentSession;
}

async function promptOnce(session, payload) {
  const options = {};
  if (payload.responseConstraint) {
    options.responseConstraint = payload.responseConstraint;
  }
  const prompt = await promptInput(payload, "promptMessages");
  const fallback = await promptInput(payload, "fallbackPrompt");
  try {
    return await session.prompt(prompt, options);
  } catch (error) {
    return await session.prompt(fallback, options);
  }
}

function base64ToBlob(data, mimeType) {
  const bytes = Uint8Array.from(atob(data), (char) => char.charCodeAt(0));
  return new Blob([bytes], { type: mimeType || "image/png" });
}

async function promptInput(payload, textField) {
  const images = payload.images || [];
  if (!images.length) return payload[textField];
  const text = textField === "promptMessages"
    ? JSON.stringify(payload.promptMessages || [])
    : String(payload.fallbackPrompt || "");
  return [
    {
      role: "user",
      content: [
        { type: "text", value: text },
        ...images.map((image) => ({
          type: "image",
          value: base64ToBlob(image.data, image.mimeType),
        })),
      ],
    },
  ];
}

async function postEvent(jobId, event) {
  const response = await fetch(`${BASE_URL}/worker/jobs/${encodeURIComponent(jobId)}/event`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(event),
  });
  const result = await response.json().catch(() => ({}));
  if (result.cancelled) throw new Error("Job cancelled by NanoServer");
}

async function complete(job) {
  const session = await ensureSession(job.payload);
  const text = String(await promptOnce(session, job.payload) ?? "");
  await postEvent(job.id, { type: "done", text });
}

async function stream(job) {
  const session = await ensureSession(job.payload);
  const options = {};
  if (job.payload.responseConstraint) {
    options.responseConstraint = job.payload.responseConstraint;
  }

  if (!session.promptStreaming) {
    const text = String(await promptOnce(session, job.payload) ?? "");
    if (text) await postEvent(job.id, { type: "delta", delta: text, fullText: text });
    await postEvent(job.id, { type: "done", text });
    return;
  }

  let streamSource;
  const prompt = await promptInput(job.payload, "promptMessages");
  const fallback = await promptInput(job.payload, "fallbackPrompt");
  try {
    streamSource = session.promptStreaming(prompt, options);
  } catch (error) {
    streamSource = session.promptStreaming(fallback, options);
  }

  let seen = "";
  for await (const chunk of streamSource) {
    const text = String(chunk ?? "");
    const delta = text.startsWith(seen) ? text.slice(seen.length) : text;
    seen = text.startsWith(seen) ? text : seen + text;
    if (delta) await postEvent(job.id, { type: "delta", delta, fullText: seen });
  }
  await postEvent(job.id, { type: "done", text: seen });
}

async function runJob(job) {
  busy = true;
  try {
    if (job.type === "stream") await stream(job);
    else await complete(job);
    handled += 1;
  } catch (error) {
    try { currentSession?.destroy?.(); } catch (_) {}
    currentSession = null;
    currentSessionKey = "";
    await postEvent(job.id, { type: "error", error: `${error?.name || "Error"}: ${error?.message || String(error)}` });
  } finally {
    busy = false;
  }
}

async function poll() {
  if (polling) return;
  polling = true;
  while (true) {
    try {
      const response = await fetch(`${BASE_URL}/worker/poll`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workerId,
          busy,
          handled,
          capability: await capability(),
        }),
      });

      if (!response.ok) throw new Error(`poll ${response.status}`);
      const data = await response.json();
      consecutiveFailures = 0;
      if (data.job) await runJob(data.job);
    } catch (error) {
      consecutiveFailures += 1;
      if (consecutiveFailures >= 20) {
        chrome.runtime.sendMessage({ type: "close-offscreen" });
        polling = false;
        return;
      }
      await sleep(1500);
    }
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "nano.ensureLoop") {
    poll()
      .then(() => sendResponse({ status: "started" }))
      .catch((error) => sendResponse({ status: "failed", error: String(error) }));
    return true;
  }
  return false;
});

poll();
