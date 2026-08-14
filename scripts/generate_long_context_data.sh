#!/usr/bin/env bash
# Generates the long-context (256k-2m) json_kv data and the config yaml that
# points at it, using the lengths/sample count settled on for this extension
# (4m was dropped in favor of a 2m cap; 200 examples/length).
#
# Usage: bash generate_long_context_data.sh   (no arguments)
# Requires: pip install -r requirements.txt  (transformers is used to
# calibrate num_kvs against a real tokenizer; orjson is picked up
# automatically if installed, but isn't required)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python scripts/generate_json_kv_data.py \
  --lengths 262144 524288 1048576 2097152 \
  --num_examples 200

python scripts/generate_configs.py

echo "Done. Data in data/json_kv/, config written to configs/recall_long.yaml."
echo "Next: bash scripts/hf_upload.sh to push the data to allenai/helmet-plus."
