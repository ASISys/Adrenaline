#!/bin/bash
set -e

# shellcheck source=../../examples/distributed_utils/utils.sh
source examples/distributed_utils/utils.sh

function start_client() {
    NUM_REQS="${NUM_REQS}" \
        REQ_RATE="${REQ_RATE}" \
        DATASET_NAME="${DATASET_NAME}" \
        MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}" \
        DECODE_FILE="${DECODE_FILE}" \
        bash benchmarks/bench_dataset_serving.sh &
}

function wait_for_client_fin() {
    local starttime
    starttime=$(date +%s)
    local timeout=""
    if [ -z "$timeout" ]; then
        while check_process_running bench_dataset_serving; do
            sleep 5
        done
        return 0
    else
        local passtime
        passtime=$(($(date +%s) - starttime))
        while ((passtime < timeout)); do
            sleep 5
            passtime=$(($(date +%s) - starttime))
            if ! check_process_running bench_dataset_serving; then
                return 0
            fi
        done
        return 1
    fi
}

function wait_for_client_running() {
    local starttime
    starttime=$(date +%s)
    local timeout=30
    local passtime
    passtime=$(($(date +%s) - starttime))
    while ((passtime < timeout)); do
        passtime=$(($(date +%s) - starttime))
        if check_process_running bench_dataset_serving; then
            return 0
        fi
    done
    return 1
}
