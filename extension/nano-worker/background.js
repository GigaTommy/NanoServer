const OFFSCREEN_DOCUMENT = "offscreen.html";
const REASON = "DOM_PARSER";
const SERVER_HEALTH_URL = "http://127.0.0.1:8458/health";

let creatingOffscreen;

async function hasOffscreenDocument() {
  const offscreenUrl = chrome.runtime.getURL(OFFSCREEN_DOCUMENT);
  if ("getContexts" in chrome.runtime) {
    const contexts = await chrome.runtime.getContexts({
      contextTypes: ["OFFSCREEN_DOCUMENT"],
      documentUrls: [offscreenUrl],
    });
    return contexts.length > 0;
  }

  const clients = await self.clients.matchAll();
  return clients.some((client) => client.url === offscreenUrl);
}

async function ensureOffscreenDocument() {
  if (!(await serverReachable())) {
    await closeOffscreenDocument();
    return;
  }
  if (await hasOffscreenDocument()) return;

  if (!creatingOffscreen) {
    creatingOffscreen = chrome.offscreen.createDocument({
      url: OFFSCREEN_DOCUMENT,
      reasons: [REASON],
      justification: "Keep a hidden ordinary Chrome document available to call the built-in LanguageModel API for local NanoServer requests.",
    });
  }

  try {
    await creatingOffscreen;
  } finally {
    creatingOffscreen = null;
  }
}

async function startWorkerLoop() {
  await ensureOffscreenDocument();
  await withSuppress(() => chrome.runtime.sendMessage({ type: "nano.ensureLoop" }));
}

async function closeOffscreenDocument() {
  if (await hasOffscreenDocument()) {
    await chrome.offscreen.closeDocument();
  }
}

async function withSuppress(action) {
  try {
    await action();
  } catch (_error) {
  }
}

async function serverReachable() {
  try {
    const response = await fetch(SERVER_HEALTH_URL, { cache: "no-store" });
    return response.ok;
  } catch (_error) {
    return false;
  }
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.alarms.create("keep-offscreen", { periodInMinutes: 0.5 });
  startWorkerLoop();
});

chrome.runtime.onStartup.addListener(() => {
  chrome.alarms.create("keep-offscreen", { periodInMinutes: 0.5 });
  startWorkerLoop();
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "keep-offscreen") {
    startWorkerLoop();
  }
});

chrome.action.onClicked.addListener(() => {
  startWorkerLoop();
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "ensure-offscreen") {
    startWorkerLoop().then(
      () => sendResponse({ ok: true }),
      (error) => sendResponse({ ok: false, error: error?.message || String(error) }),
    );
    return true;
  }

  if (message?.type === "close-offscreen") {
    closeOffscreenDocument().then(
      () => sendResponse({ ok: true }),
      (error) => sendResponse({ ok: false, error: error?.message || String(error) }),
    );
    return true;
  }

  return false;
});

startWorkerLoop();
