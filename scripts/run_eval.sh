for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
    uv run python eval.py --config configs/${task}.yaml
done

# this will run the 8k to 64k versions
for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
    uv run python eval.py --config configs/${task}_short.yaml
done