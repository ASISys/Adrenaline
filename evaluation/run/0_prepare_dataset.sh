#!/bin/bash
set -e

function download_file() {
    local target=$1
    local url=$2
    if [[ -f "$target" ]]; then
        echo "$target already exists, skipping download"
        return
    fi
    mkdir -p "$(dirname "$target")"
    curl -L "$url" -o "$target"
}

function hist_dataset_distribution() {
    local datasets=(
        mooncake
        mooncake-4k
    )
    for dataset in "${datasets[@]}"; do
        python -m evaluation.plots.prepare_datasets.dataset_histogram "${DATASETS_DIR}/${dataset}".ds \
            --output-dir evaluation/plots/prepare_datasets/"${dataset}"
    done
}

DATASETS_DIR=${DATASETS_DIR:=evaluation/datasets}

if [[ ! -f "$DATASETS_DIR/raw/conversation_trace.jsonl" ]]; then
    download_file "$DATASETS_DIR/raw/conversation_trace.jsonl" \
        "https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/conversation_trace.jsonl"
fi

python -m evaluation.run.python.prepare_dataset \
    --dataset mooncake \
    --dataset-path "$DATASETS_DIR/raw/conversation_trace.jsonl" \
    --output-path "$DATASETS_DIR"/mooncake.ds

python -m evaluation.run.python.prepare_dataset \
    --dataset mooncake \
    --dataset-path "$DATASETS_DIR/raw/conversation_trace.jsonl" \
    --max-model-len 4000 \
    --output-path "$DATASETS_DIR"/mooncake-4k.ds

hist_dataset_distribution
