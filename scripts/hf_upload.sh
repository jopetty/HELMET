#!/usr/bin/env bash
# Uploads the generated long-context json_kv data (see generate_json_kv_data.py)
# to the allenai/helmet-plus dataset repo on the Hub.
#
# Re-running overwrites whatever's already there: num_kvs is deterministic given
# the same tokenizer/lengths/seed, so filenames (and therefore repo paths) stay
# stable across regenerations, and each upload is just a new commit replacing
# the file content at those paths.
#
# Usage: bash hf_upload.sh   (no arguments)
# Requires: pip install -U "huggingface_hub[cli]"  and  hf auth login
set -euo pipefail

REPO_ID="allenai/helmet-plus"
REPO_TYPE="dataset"
DATA_DIR="data"
DATA_INCLUDE="json_kv/**"
MANIFEST="configs/json_kv_long_manifest.json"
MANIFEST_DEST="json_kv/manifest.json"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v hf &> /dev/null; then
  echo "error: the 'hf' CLI is not installed. Install it with: pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

if ! hf auth whoami &> /dev/null; then
  echo "error: not logged in to the Hub. Run 'hf auth login' first." >&2
  exit 1
fi

if [ ! -d "$DATA_DIR/json_kv" ]; then
  echo "error: $DATA_DIR/json_kv not found -- run scripts/generate_json_kv_data.py first" >&2
  exit 1
fi

echo "Uploading $DATA_DIR/json_kv/ -> $REPO_ID ($REPO_TYPE)..."
hf upload-large-folder "$REPO_ID" "$DATA_DIR" \
  --repo-type "$REPO_TYPE" \
  --include "$DATA_INCLUDE" \
  --private

if [ -f "$MANIFEST" ]; then
  echo "Uploading $MANIFEST -> $REPO_ID/$MANIFEST_DEST..."
  hf upload "$REPO_ID" "$MANIFEST" "$MANIFEST_DEST" \
    --repo-type "$REPO_TYPE" \
    --commit-message "Update json_kv manifest"
else
  echo "warning: $MANIFEST not found, skipping manifest upload" >&2
fi

echo "Done: https://huggingface.co/datasets/$REPO_ID"
