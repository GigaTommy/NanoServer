#!/usr/bin/env bash
set -euo pipefail

LABEL="com.local.chrome-nano-server"
USER_ID="$(id -u)"
DOMAIN="gui/${USER_ID}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
SERVER_FILE="${ROOT_DIR}/chrome_nano_server.py"
PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_FILE="${PLIST_DIR}/${LABEL}.plist"
LOG_DIR="${HOME}/Library/Logs/chrome-nano-server"
OUT_LOG="${LOG_DIR}/chrome_nano_server.out.log"
ERR_LOG="${LOG_DIR}/chrome_nano_server.err.log"
BASE_URL="${NANO_BASE_URL:-http://127.0.0.1:8458}"
WORKER_URL="${NANO_WORKER_URL:-${BASE_URL}/worker}"
START_CHROME="${NANO_START_CHROME:-1}"
EXTENSION_ID="${NANO_EXTENSION_ID:-bljhlmplinefjciffblfbomfapnollmb}"

ensure_files() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing virtualenv python: ${PYTHON_BIN}" >&2
    echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
  fi
  mkdir -p "${PLIST_DIR}" "${LOG_DIR}"
}

write_plist() {
  ensure_files
  cat > "${PLIST_FILE}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>cd ${ROOT_DIR} &amp;&amp; exec ${PYTHON_BIN} ${SERVER_FILE}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${ROOT_DIR}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>NANO_HOST</key>
    <string>127.0.0.1</string>
    <key>NANO_PORT</key>
    <string>8458</string>
    <key>NANO_HEADLESS</key>
    <string>0</string>
    <key>NANO_CHROME_MODE</key>
    <string>worker</string>
    <key>NANO_OUTPUT_LANGUAGE</key>
    <string>${NANO_OUTPUT_LANGUAGE:-en}</string>
    <key>NANO_JOB_TIMEOUT_SECONDS</key>
    <string>${NANO_JOB_TIMEOUT_SECONDS:-180}</string>
    <key>NANO_STRICT_EXTENSION_WORKER</key>
    <string>${NANO_STRICT_EXTENSION_WORKER:-1}</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${OUT_LOG}</string>
  <key>StandardErrorPath</key>
  <string>${ERR_LOG}</string>
</dict>
</plist>
PLIST
}

start_service() {
  write_plist
  if [[ "${START_CHROME}" == "1" || "${START_CHROME}" == "true" || "${START_CHROME}" == "yes" ]]; then
    open -gj -a "Google Chrome" >/dev/null 2>&1 || true
  fi
  launchctl bootout "${DOMAIN}" "${PLIST_FILE}" >/dev/null 2>&1 || true
  launchctl bootstrap "${DOMAIN}" "${PLIST_FILE}"
  launchctl kickstart -k "${DOMAIN}/${LABEL}"
  for _ in {1..30}; do
    if curl -fsS --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  kick_extension
  wait_for_worker || extension_hint
  status_service
}

stop_service() {
  launchctl bootout "${DOMAIN}" "${PLIST_FILE}" >/dev/null 2>&1 || true
  echo "Stopped ${LABEL}"
}

restart_service() {
  stop_service
  sleep "${NANO_RESTART_OFFSCREEN_WAIT:-40}"
  start_service
}

start_worker() {
  NANO_WORKER_URL="${WORKER_URL}" "${ROOT_DIR}/scripts/open-worker.sh" open >/dev/null
}

stop_worker() {
  NANO_WORKER_URL="${WORKER_URL}" "${ROOT_DIR}/scripts/open-worker.sh" close >/dev/null
}

kick_extension() {
  if [[ -n "${EXTENSION_ID}" ]]; then
    open -gj -a "Google Chrome" "chrome-extension://${EXTENSION_ID}/kick.html" >/dev/null 2>&1 || true
  fi
}

wait_for_worker() {
  for _ in {1..45}; do
    if curl -fsS --max-time 2 "${BASE_URL}/health" \
      | python3 -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("worker_connected") else 1)' >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

extension_hint() {
  cat >&2 <<EOF
Hidden extension worker did not connect within timeout.
Install or reload the unpacked Chrome extension:
  ${ROOT_DIR}/extension/nano-worker

Fallback visible tab command:
  $0 worker
EOF
}

status_service() {
  echo "LaunchAgent: ${PLIST_FILE}"
  launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1 \
    && echo "launchd: loaded" \
    || echo "launchd: not loaded"
  lsof -nP -iTCP:8458 -sTCP:LISTEN || true
  curl -sS --max-time 5 "${BASE_URL}/health" || true
  echo
}

logs_service() {
  echo "== stdout =="
  tail -80 "${OUT_LOG}" 2>/dev/null || true
  echo "== stderr =="
  tail -120 "${ERR_LOG}" 2>/dev/null || true
}

case "${1:-status}" in
  start)
    start_service
    ;;
  restart)
    restart_service
    ;;
  stop)
    stop_service
    ;;
  status)
    status_service
    ;;
  logs)
    logs_service
    ;;
  worker)
    start_worker
    wait_for_worker || true
    status_service
    ;;
  kick)
    kick_extension
    wait_for_worker || true
    status_service
    ;;
  close-worker)
    stop_worker
    status_service
    ;;
  *)
    echo "Usage: $0 {start|restart|stop|status|logs|worker|close-worker|kick}" >&2
    exit 2
    ;;
esac
