#!/usr/bin/env bash
set -euo pipefail

URL="${NANO_WORKER_URL:-http://127.0.0.1:8458/worker}"
MODE="${1:-open}"

open_worker() {
  osascript - "$URL" <<'APPLESCRIPT'
on run argv
  set workerUrl to item 1 of argv
  set previousApp to ""
  try
    tell application "System Events" to set previousApp to name of first application process whose frontmost is true
  end try
  tell application "Google Chrome"
    if it is not running then activate
    if (count of windows) is 0 then make new window

    set foundTab to missing value
    set foundWindow to missing value
    set foundIndex to missing value
    repeat with w in windows
      set tabIndex to 1
      repeat with t in tabs of w
        if (URL of t as text) starts with workerUrl then
          set foundTab to t
          set foundWindow to w
          set foundIndex to tabIndex
          exit repeat
        end if
        set tabIndex to tabIndex + 1
      end repeat
      if foundTab is not missing value then exit repeat
    end repeat

    if foundTab is missing value then
      set foundWindow to window 1
      set foundTab to make new tab at end of tabs of foundWindow with properties {URL:workerUrl}
      set foundIndex to count of tabs of foundWindow
    end if

    set active tab index of foundWindow to foundIndex
    set index of foundWindow to 1
  end tell
  if previousApp is not "" and previousApp is not "Google Chrome" then
    try
      tell application previousApp to activate
    end try
  end if
end run
APPLESCRIPT
  echo "Worker tab is open at ${URL}."
}

close_worker() {
  osascript - "$URL" <<'APPLESCRIPT'
on run argv
  set workerUrl to item 1 of argv
  tell application "Google Chrome"
    if it is not running then return
    repeat with w in windows
      set tabsToClose to {}
      repeat with t in tabs of w
        if (URL of t as text) starts with workerUrl then set end of tabsToClose to t
      end repeat
      repeat with t in tabsToClose
        close t
      end repeat
    end repeat
  end tell
end run
APPLESCRIPT
  echo "Closed worker tab at ${URL}."
}

case "${MODE}" in
  open|start)
    open_worker
    ;;
  close|stop)
    close_worker
    ;;
  restart)
    close_worker
    sleep 1
    open_worker
    ;;
  *)
    echo "Usage: $0 {open|close|restart}" >&2
    exit 2
    ;;
esac
