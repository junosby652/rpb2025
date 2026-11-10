#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GO1_VENV="${GO1_VENV:-$PWD/.venv}"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for this tested lockfile"'
if [ ! -x "$GO1_VENV/bin/python" ]; then
    python3 -m venv "$GO1_VENV"
fi
"$GO1_VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
"$GO1_VENV/bin/python" -m pip install -r go1_walk/requirements.lock
"$GO1_VENV/bin/python" -m pip check
"$GO1_VENV/bin/python" -m unittest go1_walk.test_env -v
echo "Setup complete. Use $GO1_VENV/bin/python -m go1_walk.train"
