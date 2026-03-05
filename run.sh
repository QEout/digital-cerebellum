#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
  python3 -m venv "$SCRIPT_DIR/.venv"
  "$SCRIPT_DIR/.venv/bin/pip" install -q -e "$SCRIPT_DIR[mcp]"
fi

exec "$SCRIPT_DIR/.venv/bin/python" -m digital_cerebellum.mcp_server "$@"
