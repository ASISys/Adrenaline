#!/bin/bash
set -e

# shellcheck source=../../../evaluation/utils/evaluation_utils.sh
source evaluation/utils/evaluation_utils.sh

function evaluate_with_mooncake_and_llama_3_1_8b() {
    NUM_REQS=60
    DATASET_NAME=mooncake
    MODEL=meta-llama/Llama-3.1-8B
    MAX_MODEL_LEN=32768  # prevent cuda OOM
    local req_rates=(
        1 1.5 2 2.5 3 3.5 4
    )
    for req_rate in "${req_rates[@]}"; do
        PREFILL_INSTANCES=3 REQ_RATE=${req_rate} run_disagg_profile
    done
}

evaluate_with_mooncake_and_llama_3_1_8b
