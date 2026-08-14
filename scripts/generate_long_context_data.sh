#!/usr/bin/env bash
# Generates the full 4k-2m json_kv sweep and the config yaml that points at
# it, using the lengths/sample count settled on for this extension (4m was
# dropped in favor of a 2m cap; 200 examples/length). This regenerates the
# standard 4k-128k tiers too, rather than relying on the official pre-baked
# data, so the whole range is calibrated against the same (Olmo 3) tokenizer.
#
# Usage: bash generate_long_context_data.sh   (no arguments)
# Requires: uv (https://docs.astral.sh/uv/) -- `uv run` syncs the environment
# from pyproject.toml/uv.lock automatically, no manual venv activation needed.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run python scripts/generate_json_kv_data.py \
  --lengths 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 \
  --num_examples 200 \
  --tokenizer allenai/Olmo-3-1025-7B

uv run python scripts/generate_configs.py

echo "Done. Data in data/json_kv/, config written to configs/recall_long.yaml."
echo "Next: bash scripts/hf_upload.sh to push the data to allenai/helmet-plus."
