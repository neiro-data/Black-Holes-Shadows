#!/usr/bin/env bash
# One process per config: weyl_core.Mat_nu is frozen at first numba compile.
set -euo pipefail
cd "$(dirname "$0")/.."
for cfg in configs/*.toml; do
  echo "== $cfg"
  uv run python run_shadow.py "$cfg" "$@"
done
