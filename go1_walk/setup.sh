#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GO1_VENV="${GO1_VENV:-$PWD/.venv}"
GO1_PYTHON="${GO1_PYTHON:-python3}"
"$GO1_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for requirements-local.lock; the original Python 3.12 lock remains in requirements.lock"'
if [ ! -x "$GO1_VENV/bin/python" ]; then
    "$GO1_PYTHON" -m venv "$GO1_VENV" || {
        echo "Install python3.10-venv first: sudo apt install python3.10-venv" >&2
        exit 1
    }
fi
"$GO1_VENV/bin/python" -m pip --version >/dev/null || {
    echo "The venv has no pip. Install python3.10-venv and recreate it, or bootstrap pip inside the venv." >&2
    exit 1
}
"$GO1_VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
"$GO1_VENV/bin/python" -m pip install -r go1_walk/requirements-local.lock
"$GO1_VENV/bin/python" -m pip check
"$GO1_VENV/bin/python" -m unittest go1_walk.test_env -v
echo "Setup complete. Use $GO1_VENV/bin/python -m go1_walk.train"
