#!/bin/bash
set -e

# shellcheck source=../../examples/distributed_kv/xp1d/disagg_instances_start.sh
source examples/distributed_kv/xp1d/disagg_instances_start.sh
# shellcheck source=../../examples/adrenaline/xp1d/adrenaline_instances_start.sh
source examples/adrenaline/xp1d/adrenaline_instances_start.sh
# shellcheck source=../../evaluation/utils/client_utils.sh
source evaluation/utils/client_utils.sh

function run_offload_profile() {
    local num_execs=${1:-"1"}
    echo "Profile offload with NUM_REQS=${NUM_REQS}, REQ_RATE=${REQ_RATE}, num_execs=${num_execs}"
    LOAD_FORMAT=dummy
    local cnt=0
    while ((cnt < num_execs)); do
        start_adrenaline_environment
        if [[ ${PREFILL_INSTANCES:-1} -gt 1 ]]; then
            RUN_TYPE=${RUN_TYPE:=offload/${PREFILL_INSTANCES}p1d} start_client
        else
            RUN_TYPE=${RUN_TYPE:=offload} start_client
        fi
        wait_for_client_running
        local pid
        pid=$(pgrep -f bench_dataset_serving)
        if wait_for_client_fin; then
            if wait "$pid"; then
                cnt=$((cnt + 1))
            else
                echo "Client failed."
            fi
        else
            pkill -f benchmark_serving
            wait_for_client_fin
        fi
        stop_adrenaline_environment
    done
    if [[ -n "${RUN_TYPE}" ]]; then
        unset RUN_TYPE
    fi
}

function run_disagg_profile() {
    if [[ -n "${OFFLOAD_RATIO}" ]]; then
        echo "${OFFLOAD_RATIO} should not be set for disagg inference mode"
        return 1
    fi
    LOAD_FORMAT=dummy
    local num_execs=${1:-"1"}
    echo "Profile disagg with NUM_REQS=${NUM_REQS}, REQ_RATE=${REQ_RATE}, num_execs=${num_execs}"
    local cnt=0
    while ((cnt < num_execs)); do
        start_disagg_environemnt
        if [[ ${PREFILL_INSTANCES:-1} -gt 1 ]]; then
            RUN_TYPE=${RUN_TYPE:=disagg/${PREFILL_INSTANCES}p1d} start_client
        else
            RUN_TYPE=${RUN_TYPE:=disagg} start_client
        fi
        wait_for_client_running
        local pid
        pid=$(pgrep -f bench_dataset_serving)
        if wait_for_client_fin; then
            if [[ $(wait "$pid") -eq 0 ]]; then
                cnt=$((cnt + 1))
            else
                echo "Client failed."
            fi
        else
            pkill -f benchmark_serving
            wait_for_client_fin
        fi
        stop_disagg_environment
    done
    if [[ -n "${RUN_TYPE}" ]]; then
        unset RUN_TYPE
    fi
}
