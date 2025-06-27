#!/bin/bash
set -e

# shellcheck source=../../evaluation/utils/evaluation_utils.sh
source evaluation/utils/evaluation_utils.sh

function evaluate_with_mooncake_and_llama_2_13b() {
    NUM_REQS=50
    DATASET_NAME=mooncake-4k
    MODEL=meta-llama/Llama-2-13b-hf
    local req_rates=(
        1 1.5 2 2.5 3 3.5 4
    )
    for req_rate in "${req_rates[@]}"; do
        PREFILL_INSTANCES=2 REQ_RATE=${req_rate} run_disagg_profile
        PREFILL_INSTANCES=2 NUM_ATTENTION_BLOCKS=1500 REQ_RATE=${req_rate} run_offload_profile
    done
}

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
        PREFILL_INSTANCES=3 NUM_PREFILL_BLOCKS=2048 NUM_ATTENTION_BLOCKS=12000 REQ_RATE=${req_rate} run_offload_profile
    done
}

evaluate_with_mooncake_and_llama_2_13b
evaluate_with_mooncake_and_llama_3_1_8b
