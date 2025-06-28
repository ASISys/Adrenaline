#!/bin/bash
set -e

# shellcheck source=../../../evaluation/utils/evaluation_utils.sh
source evaluation/utils/evaluation_utils.sh

function evaluate_disaggregation_with_mooncake_and_llama_2_13b() {
    NUM_REQS=50
    DATASET_NAME=mooncake-4k
    MODEL=meta-llama/Llama-2-13b-hf
    local req_rates=(
        1 1.5 2 2.5 3 3.5 4
    )
    for req_rate in "${req_rates[@]}"; do
        PREFILL_INSTANCES=2 REQ_RATE=${req_rate} run_disagg_profile
    done
}

evaluate_disaggregation_with_mooncake_and_llama_2_13b
