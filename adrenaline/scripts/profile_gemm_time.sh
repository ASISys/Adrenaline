#!/bin/bash

CUR_DIR=$(dirname "$0")
OUTPUT_DIR=${CUR_DIR}/../utils

function profile_gemm_run_time() {
    local output_path
    output_path="${OUTPUT_DIR}/gemm_run_time.csv"
    if [[ -f "${output_path}" ]]; then
        rm "${output_path}"
    fi

    python -m adrenaline.profiler.decode_gemm_profiler \
        --output-file "${output_path}"
}

profile_gemm_run_time
