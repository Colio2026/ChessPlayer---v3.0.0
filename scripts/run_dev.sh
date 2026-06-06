#!/usr/bin/env bash
# Run ChessPlayer in development mode.
# Must be invoked from the repo root: bash scripts/run_dev.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$REPO_ROOT/src/chessplayer"
VENV="$REPO_ROOT/.venv/bin/activate"

if [ -f "$VENV" ]; then
    # shellcheck disable=SC1090
    source "$VENV"
fi

cd "$APP_DIR"
python main.py "$@"
