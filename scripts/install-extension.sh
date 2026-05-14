#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTENSION_DIR="${ROOT_DIR}/extension/nano-worker"

open -a "Google Chrome" "chrome://extensions/"
open -R "${EXTENSION_DIR}/manifest.json"

cat <<EOF
Chrome extensions page opened.

Install once:
1. Enable "Developer mode".
2. Click "Load unpacked".
3. Select:
   ${EXTENSION_DIR}

After installation, NanoServer start/stop runs without a visible worker tab.
EOF
