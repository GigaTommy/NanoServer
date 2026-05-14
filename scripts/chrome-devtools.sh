#!/usr/bin/env bash
set -euo pipefail

PORT="${NANO_CDP_PORT:-9222}"
CDP_URL="http://127.0.0.1:${PORT}"
CHROME_APP="${CHROME_APP:-Google Chrome}"
FEATURES="${NANO_CHROME_FEATURES:-AIPromptAPI:langs/*,AIPromptAPIMultimodalInput,AIPromptAPIStructuredOutput,OptimizationGuideModelExecution,OptimizationGuideOnDeviceModel,OnDeviceModelPerformanceParams:compatible_on_device_performance_classes/*/compatible_low_tier_on_device_performance_classes/*,PromptAPIForGeminiNano}"

cdp_ready() {
  curl -fsS --max-time 2 "${CDP_URL}/json/version" >/dev/null 2>&1
}

chrome_running() {
  pgrep -f "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" >/dev/null 2>&1
}

wait_for_cdp() {
  for _ in {1..30}; do
    if cdp_ready; then
      curl -sS --max-time 2 "${CDP_URL}/json/version"
      echo
      return 0
    fi
    sleep 1
  done
  echo "Chrome DevTools did not appear at ${CDP_URL}" >&2
  echo "Chrome 136+ ignores --remote-debugging-port for the default Chrome data directory." >&2
  echo "Use a non-default --user-data-dir, or run NanoServer in launch/shared-profile mode." >&2
  return 1
}

launch_chrome() {
  open -a "${CHROME_APP}" --args \
    "--remote-debugging-port=${PORT}" \
    "--enable-features=${FEATURES}" \
    "--enable-experimental-web-platform-features"
  wait_for_cdp
}

start_chrome() {
  if cdp_ready; then
    wait_for_cdp
    return 0
  fi

  if chrome_running; then
    echo "Ordinary Chrome is already running, but ${CDP_URL} is not open." >&2
    echo "Run: $0 restart" >&2
    return 1
  fi

  launch_chrome
}

restart_chrome() {
  osascript -e 'tell application "Google Chrome" to quit' >/dev/null 2>&1 || true
  for _ in {1..30}; do
    if ! chrome_running; then
      break
    fi
    sleep 1
  done
  launch_chrome
}

status_chrome() {
  if cdp_ready; then
    echo "Chrome DevTools: ${CDP_URL}"
    curl -sS --max-time 2 "${CDP_URL}/json/version"
    echo
  else
    echo "Chrome DevTools is not listening at ${CDP_URL}"
    chrome_running && echo "Ordinary Chrome is running without remote debugging."
    return 1
  fi
}

case "${1:-status}" in
  start)
    start_chrome
    ;;
  restart)
    restart_chrome
    ;;
  status)
    status_chrome
    ;;
  *)
    echo "Usage: $0 {start|restart|status}" >&2
    exit 2
    ;;
esac
