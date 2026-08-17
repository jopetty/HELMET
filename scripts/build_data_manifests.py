"""Build manifests for the derived-data directories hosted on allenai/helmet-plus.

The RAG, re-ranking and Cite tasks depend on retrieval that was run once by the
HELMET (and ALCE) authors and shipped inside `data.tar.gz`. Those files are
re-hosted unpacked on the Hub so a consumer can fetch one file per task/length
instead of pulling a 10GB tarball, and these manifests are what let it do that
without hardcoding filenames -- the same convention already used for json_kv
(see generate_json_kv_data.py).

Every value is derived from `master_mapping` in generate_configs.py, so the
manifests cannot drift from the configs.

Usage:
    python scripts/build_data_manifests.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_configs import lengths_mapping, master_mapping  # noqa: E402

# Task -> the hosted directory its files live in.
TASK_DIRS = {
    "kilt_nq": "kilt",
    "kilt_triviaqa": "kilt",
    "kilt_hotpotqa": "kilt",
    "kilt_popqa": "kilt",
    "msmarco_rerank_psg": "msmarco",
    "alce_asqa": "alce",
    "alce_qampari": "alce",
    "alce_asqa_nocite": "alce",
    "alce_qampari_nocite": "alce",
}

# ALCE's few-shot prompts live in the repo rather than the data release, so they
# are uploaded alongside the data to keep the hosted dataset self-contained.
PROMPT_DIR = "alce_prompts"


def _repo_path(local_path: str) -> str:
    """Rewrite a repo-local data path to its path within the hosted dataset."""
    if local_path.startswith("data/"):
        return local_path[len("data/") :]
    if local_path.startswith("prompts/"):
        return f"{PROMPT_DIR}/{Path(local_path).name}"
    return local_path


def _passage_count(task: str, config: dict) -> int | None:
    """Number of retrieved passages for this tier.

    RAG and re-ranking encode it in the filename (`..._k220_...`); ALCE instead
    truncates a fixed 2000-document pool, and records the count in the task
    name suffix (`alce_asqa_75`).
    """
    if task.startswith("alce_"):
        postfix = config.get("name_postfix", "")
        return int(postfix.lstrip("_")) if postfix.lstrip("_").isdigit() else None
    match = re.search(r"_k(\d+)_", config.get("test_files", ""))
    return int(match.group(1)) if match else None


def main() -> None:
    manifests: dict[str, dict] = {}

    for task, directory in TASK_DIRS.items():
        entries = {}
        for length_name in lengths_mapping:
            config = master_mapping[task].get(length_name)
            if config is None:
                continue
            entry = {
                "input_length": config["input_length"],
                "generation_max_length": config["generation_max_length"],
                "test_file": _repo_path(config["test_files"]),
            }
            demo = config.get("demo_files", "")
            if demo:
                entry["demo_file"] = _repo_path(demo)
            count = _passage_count(task, config)
            if count is not None:
                entry["num_passages"] = count
            entries[length_name] = entry
        manifests.setdefault(directory, {})[task] = entries

    for directory, content in manifests.items():
        out = Path("data") / directory / "manifest.json"
        if not out.parent.is_dir():
            print(f"skipping {out}: {out.parent} not present locally")
            continue
        with open(out, "w") as f:
            json.dump(content, f, indent=2, sort_keys=True)
        tiers = sum(len(v) for v in content.values())
        print(f"wrote {out}  ({len(content)} tasks, {tiers} tiers)")


if __name__ == "__main__":
    main()
