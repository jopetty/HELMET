"""Generate synthetic JSON key-value retrieval data for context lengths beyond
what's in the official HELMET data release (which tops out at 128k, see
`data/json_kv/test_k1800_dep6.jsonl` referenced from `configs/recall.yaml`).

This reproduces the same synthetic task (`load_json_kv` in `data.py`, based on
https://github.com/nelson-liu/lost-in-the-middle): a JSON object made of
random UUID keys/values is presented as `context`, and the model must return
the `value` for one `question` key drawn from it. Every row also carries a
pool of unrelated UUID pairs in `demos`, used for few-shot examples.

Because the underlying content is fully synthetic, the task scales to
arbitrary context lengths -- unlike the real-document tasks (LongQA, Summ,
RAG, ...), which are capped by how long the source documents actually are.

For a given target token length, the number of key-value pairs needed is
found by calibrating against a real tokenizer (random UUID text tokenizes
fairly uniformly, so a small calibration sample plus a short binary search
converges quickly, even at multi-million-token lengths).

Usage:
    python scripts/generate_json_kv_data.py \\
        --lengths 262144 524288 1048576 2097152 \\
        --num_examples 200 \\
        --tokenizer NousResearch/Llama-2-7b-hf \\
        --output_dir data/json_kv \\
        --manifest configs/json_kv_long_manifest.json

The manifest written at the end records the resolved num_kvs/file path per
length, and is consumed by `scripts/generate_configs.py` to build the config
yaml (so the two scripts never need their num_kvs guesses to agree by hand).
"""
import argparse
import json
import random
import uuid
from pathlib import Path


DEP_SUFFIX = "dep6"  # kept only to match the naming of the existing 4k-128k files; load_json_kv does not parse it


def random_uuid(rng):
    return str(uuid.UUID(int=rng.getrandbits(128)))


def render_context(kv_pairs):
    # mirrors the JSON-block rendering in nelson-liu/lost-in-the-middle's get_kv_retrieval_prompt
    lines = []
    for i, (k, v) in enumerate(kv_pairs):
        start = "{" if i == 0 else " "
        end = ",\n" if i != len(kv_pairs) - 1 else "}"
        lines.append(f'{start}"{k}": "{v}"{end}')
    return "".join(lines)


def make_example(num_kvs, num_demos, rng):
    keys = set()
    while len(keys) < num_kvs:
        keys.add(random_uuid(rng))
    kv_pairs = [(k, random_uuid(rng)) for k in keys]
    rng.shuffle(kv_pairs)

    gold_key, gold_value = kv_pairs[rng.randrange(num_kvs)]
    demos = [[random_uuid(rng), random_uuid(rng)] for _ in range(num_demos)]

    return {
        "context": render_context(kv_pairs),
        "demos": demos,
        "num_kvs": num_kvs,
        "question": gold_key,
        "answer": [gold_value],
    }


def count_tokens(tokenizer, text):
    return len(tokenizer(text)["input_ids"])


def calibrate_num_kvs(tokenizer, target_length, rng, tolerance=0.02, probe_n=500):
    """Find the num_kvs whose rendered context tokenizes to ~target_length tokens."""
    probe_tokens = count_tokens(tokenizer, make_example(probe_n, 0, rng)["context"])
    per_kv = probe_tokens / probe_n
    guess = max(2, round(target_length / per_kv))

    lo, hi = max(2, int(guess * 0.7)), int(guess * 1.5)
    best, best_diff = guess, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        n_tok = count_tokens(tokenizer, make_example(mid, 0, rng)["context"])
        diff = abs(n_tok - target_length) / target_length
        # prefer meeting-or-exceeding the target slightly, since eval-time truncation
        # (see model_utils.py) trims down to input_max_length anyway, but a file that's
        # too short can't be padded back up
        if diff < best_diff or (diff == best_diff and n_tok >= target_length):
            best, best_diff = mid, diff
        if n_tok < target_length:
            lo = mid + 1
        else:
            hi = mid - 1
        if diff <= tolerance and n_tok >= target_length:
            break
    return best


def length_name(n):
    if n % (1024 * 1024) == 0:
        return f"{n // (1024 * 1024)}m"
    return f"{n // 1024}k"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", type=int, nargs="+", required=True, help="target input lengths in tokens, e.g. 262144 524288 1048576 2097152 4194304")
    parser.add_argument("--num_examples", type=int, default=200, help="number of test examples per length")
    parser.add_argument("--num_demos", type=int, default=10, help="number of few-shot demo kv pairs to attach to each example")
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Llama-2-7b-hf", help="tokenizer used to calibrate num_kvs to the target length; mirrors the meta-llama/Llama-2-7b-hf tokenizer used elsewhere in this repo (see truncate_llama2 in data.py) without requiring gated access")
    parser.add_argument("--output_dir", type=str, default="data/json_kv")
    parser.add_argument("--manifest", type=str, default="configs/json_kv_long_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    if Path(args.manifest).exists():
        with open(args.manifest) as f:
            manifest = json.load(f)

    for target_length in args.lengths:
        rng = random.Random(args.seed)
        name = length_name(target_length)
        print(f"[{name}] calibrating num_kvs for target length {target_length}...")
        num_kvs = calibrate_num_kvs(tokenizer, target_length, rng)
        actual_tokens = count_tokens(tokenizer, make_example(num_kvs, 0, rng)["context"])
        print(f"[{name}] num_kvs={num_kvs} -> ~{actual_tokens} tokens (target {target_length})")

        test_file = output_dir / f"test_k{num_kvs}_{DEP_SUFFIX}.jsonl"
        with open(test_file, "w") as f:
            for _ in range(args.num_examples):
                example = make_example(num_kvs, args.num_demos, rng)
                f.write(json.dumps(example) + "\n")
        print(f"[{name}] wrote {args.num_examples} examples to {test_file}")

        manifest[name] = {
            "input_length": target_length,
            "num_kvs": num_kvs,
            "test_file": str(test_file),
        }

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
