#!/bin/bash

CUR_DIR=$(dirname "$0")
OUTPUT_DIR=${CUR_DIR}/../utils
MODEL=${MODEL:="meta-llama/Llama-2-7b-hf"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
AVG_SEQ_LEN=${AVG_SEQ_LEN:=2048}

function profile_attention_bandwidth() {
    local model_name
    model_name=$(basename "${MODEL}")
    local output_path
    output_path="${OUTPUT_DIR}/attention_bandwidth/${model_name}.csv"
    if [[ -f "${output_path}" ]]; then
        rm "${output_path}"
    fi

    python -m adrenaline.profiler.attention_bandwidth_profiler \
        --model "${MODEL}" \
        --avg-seq-len "${AVG_SEQ_LEN}" \
        --batch-size 32 \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --sm-pcts 10 20 30 40 50 60 70 80 90 100 \
        --output-file "${output_path}"
}

profile_attention_bandwidth
