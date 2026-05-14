async function kick() {
  try {
    await chrome.runtime.sendMessage({ type: "ensure-offscreen" });
  } catch (_error) {
  } finally {
    setTimeout(() => window.close(), 500);
  }
}

kick();
