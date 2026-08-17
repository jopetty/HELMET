# Brief: host HELMET's derived data on `allenai/helmet-plus`

> **Status:** steps 1–3 are done. `data.tar.gz` was downloaded and the needed
> subset extracted to `data/` in this repo (21.7 GB across 39 files, all
> present, non-empty, and schema-checked). What remains is the upload (step 4–5)
> and the olmo-eval loaders. Authenticate to the Hub before downloading
> anything from it — unauthenticated transfers are throttled to ~1.7 MB/s
> versus ~85 MB/s authenticated, a ~50x difference.

## Read this first: it is not a retrieval job

The task is framed as "data pre-computation", but almost none of it needs
computing. The retrieval was already run once — by the HELMET authors, and for
ALCE by the same group — and the output ships inside `data.tar.gz` on the
[`princeton-nlp/HELMET`](https://huggingface.co/datasets/princeton-nlp/HELMET)
dataset repo.

It also does not need a cluster. The README's "about 34GB" is the *extracted*
size; the download is **11.3 GB**, so a laptop with ~50GB free handles it.

**So this is an extract-verify-upload job, not an embed-and-index job.** Do not
stand up a retrieval pipeline. If you find yourself downloading a Wikipedia
dump or loading an embedding model, stop — you have misread the task.

The one genuinely open problem is `multi_lexsum` (see the last section).

## Why

`allenai/helmet-plus` currently holds only the regenerated `json_kv` data.
olmo-eval's HELMET integration (allenai/olmo-eval PR #287) covers recall,
LongQA and ICL; the remaining 9 tasks — RAG ×4, rerank ×1, Cite ×2(+2 nocite) —
are blocked purely on these files being reachable from the Hub. Metrics and
task code for them are either done or cheap; the data is the blocker.

## What to upload

39 files. Paths below are as they appear after extracting `data.tar.gz`; the
first 37 are exactly what `scripts/generate_configs.py` references (I generated
this list from `master_mapping`, so it is complete, not a sample).

**RAG — `data/kilt/`** (28 files)
- `nq-dev-multikilt_1000_k{20,50,105,220,440,1000}_dep6.jsonl` + `nq-train-multikilt_1000_k3_dep6.jsonl`
- `triviaqa-dev-multikilt_1000_k{20,50,105,220,440,1000}_dep6.jsonl` + `triviaqa-train-multikilt_1000_k3_dep6.jsonl`
- `hotpotqa-dev-multikilt_1000_k{20,50,105,220,440,1000}_dep3.jsonl` + `hotpotqa-train-multikilt_1000_k3_dep3.jsonl`
- `popqa_test_1000_k{3,20,50,105,220,440,1000}_dep6.jsonl`

The `k` in each filename is the number of retrieved passages per query, chosen
to fill a length tier (k20→4k … k1000→128k). The `k3` files are the few-shot
demo pools. Note `hotpotqa` uses `dep3` where the others use `dep6` — keep the
names byte-exact; the configs match on them literally.

**Rerank — `data/msmarco/`** (7 files)
- `test_reranking_data_k{14,50,130,285,600,1000}_dep3.jsonl` + `test_reranking_data_k10_dep3.jsonl` (demos)

**Cite — `data/alce/`** (2 files)
- `asqa_eval_gtr_top2000.json`, `qampari_eval_gtr_top2000.json`

(The ALCE prompt files live in `prompts/` in this repo already — do not upload.)

**Summ keypoints** (2 files, judge inputs rather than model inputs)
- `data/infbench/longbook_sum_eng_keypoints.jsonl`
- `data/multi_lexsum/multi_lexsum_val.jsonl`

These are consumed by `scripts/eval_gpt4_summ.py`, not by `data.py`, which is
why they are absent from `master_mapping`. Without them the Summarization
judge cannot run at all.

## Use `data.tar.gz`, not `data_v2.tar.gz`

The Hub repo holds two tarballs and only the first is what this repo's code
expects:

| file | size | contents |
|---|---|---|
| `data.tar.gz` (2024-10) | 11.3 GB | what `scripts/download_data.sh` fetches; contains `data/msmarco/`, `data/kilt/`, `data/alce/` under exactly the filenames `generate_configs.py` references |
| `data_v2.tar.gz` (2026-07) | 8.9 GB | "v2 with llama 3 tokenizer" plus added graphwalks/mrcr; leads with `data/ruler_llama3/` |

v2 is a RULER re-tokenization and task expansion. This repo truncates with the
Llama 2 tokenizer (`truncate_llama2` in `data.py`) and its length tiers are
calibrated to match, so v1 is the consistent choice — and RULER is out of scope
here regardless. Verified by streaming the head of both archives.

## How

1. `bash scripts/download_data.sh` (wget + tar), or fetch `data.tar.gz`
   directly and extract only the directories listed above — the archive also
   contains the RULER data, which is the bulk of it and is not needed here.
2. Verify every path in the list above exists and is non-empty. Report anything
   missing rather than substituting — a silently absent file becomes a
   confusing eval-time error later.
3. Schema — **already verified locally**, recorded here so the loaders can rely
   on it:
   - KILT rows: `question`, `answers`, `ctxs`, `positive_ctxs`,
     `hard_negative_ctxs`. Context entries carry `title`/`text` plus either
     `psg_id` (nq/triviaqa/hotpotqa) or `id`/`score`/`has_answer` (popqa) —
     the field sets differ by dataset, so don't assume one shape.
   - MS MARCO rows: `qid`, `query`, `ctxs` with `id`/`text`/`label`.
   - ALCE: a **top-level JSON array** (not an object with a `data` key), 948
     items for asqa, each with `question`, `answer`, `docs` (exactly 2000),
     `qa_pairs`, `annotations`.
   - `k` in each filename equals `len(ctxs)`: confirmed at k=20/50/105/1000.
4. Upload preserving the directory names (`kilt/`, `msmarco/`, `alce/`,
   `infbench/`, `multi_lexsum/`) so they sit alongside the existing `json_kv/`.
   `scripts/hf_upload.sh` already does this for `json_kv` — extend its
   `DATA_INCLUDE` rather than writing a new uploader. It uses
   `hf upload-large-folder`, which is resumable; these files are large.
5. Write a `manifest.json` per directory mirroring `json_kv/manifest.json`
   (length name → file path + parameters), so the olmo-eval loaders resolve
   files from a manifest instead of hardcoding names. For RAG/rerank the useful
   key is the `k` per length tier.

## Verification before you call it done

- Re-download one uploaded file from the Hub and byte-compare against local.
- `python scripts/generate_configs.py` still runs clean.
- Report total uploaded size per directory.

## Out of scope

- **Do not** try to extend RAG/rerank/Cite past 128k. Their length ceiling is
  set by retrieval depth and corpus size, not by a config value; going further
  means genuinely re-running retrieval, which is a separate project.
- **Do not** regenerate `json_kv` — already done and hosted.
- **Do not** touch RULER — covered separately by `allenai/ruler-plus`.

## `multi_lexsum` is NOT blocked after all

Earlier analysis said `multi_lexsum` needed a new data source, because
`load_dataset("allenai/multi_lexsum", name="v20230518")` is broken under
`datasets` ≥4.0 and the parquet fallback fails too. Both of those are still
true — but they no longer matter.

`data/multi_lexsum/multi_lexsum_val.jsonl` in the tarball is not just the
keypoints file: it is the **full validation set**, 312 rows, every one carrying
`sources` (the legal documents), `summary/short` (the gold summary),
`summary/long` (used by the judge's precision call), and both
`summary/short_keypoints` and `summary/long_keypoints`. Verified.

So the olmo-eval loader should read this file directly and never call
`load_dataset` for multi_lexsum. That sidesteps the broken Hub path entirely
and is more reproducible besides.
