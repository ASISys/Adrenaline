#!/bin/bash
set -e

# shellcheck source=../../examples/distributed_utils/utils.sh
source examples/distributed_utils/utils.sh

function start_disagg_tmux_session() {
    if [ -n "$REMOTE_PROXY" ]; then
        start_remote_session "$REMOTE_PROXY" disagg
    fi
    REMOTE_HOST="$REMOTE_PROXY" remote_or_local_execute \
        tmux new-window -e "HF_HUB_OFFLINE=1" -c ~/adrenaline -n proxy -t disagg: bash

    if [ -n "$REMOTE_PREFILL" ]; then
        start_remote_session "$REMOTE_PREFILL" disagg
    fi
    REMOTE_HOST="$REMOTE_PREFILL" remote_or_local_execute \
        tmux new-window -e "HF_HUB_OFFLINE=1" -c ~/adrenaline -n prefill -t disagg: bash

    if [ -n "$REMOTE_DECODE" ]; then
        start_remote_session "$REMOTE_DECODE" disagg
    fi
    REMOTE_HOST="$REMOTE_DECODE" remote_or_local_execute \
        tmux new-window -e "HF_HUB_OFFLINE=1" -c ~/adrenaline -n decode -t disagg: bash
}

function kill_disagg_tmux_session() {
    if [ -n "$REMOTE_PREFILL" ]; then
        stop_remote_session "$REMOTE_PREFILL" disagg
    fi
    if [ -n "$REMOTE_DECODE" ]; then
        stop_remote_session "$REMOTE_DECODE" disagg
    fi
}

function start_disagg_instances_in_tmux() {
    if [[ -n "${MODEL}" ]]; then
        REMOTE_HOST="$REMOTE_PREFILL" remote_or_local_execute \
            tmux send-key -t disagg:prefill "\"MODEL=${MODEL} \""
        REMOTE_HOST="$REMOTE_DECODE" remote_or_local_execute \
            tmux send-key -t disagg:decode "\"MODEL=${MODEL} \""
    fi

    REMOTE_HOST="$REMOTE_PREFILL" remote_or_local_execute \
        tmux send-key -t disagg:prefill "\"VLLM_HOST_IP=$REMOTE_PREFILL GLOO_SOCKET_IFNAME=ib0 bash examples/distributed_kv/prefill_instance_start.sh | tee prefill.out\"" ENTER

    REMOTE_HOST="$REMOTE_DECODE" remote_or_local_execute \
        tmux send-key -t disagg:decode "\"VLLM_HOST_IP=$REMOTE_PREFILL GLOO_SOCKET_IFNAME=ib0 bash examples/distributed_kv/decode_instance_start.sh | tee decode.out\"" ENTER
}

function wait_disagg_instances_running() {
    local starttime
    starttime=$(date +%s)
    local passtime
    passtime=$(($(date +%s) - "$starttime"))

    local processes=(
        prefill_instance_start
        decode_instance_start
    )
    local remote_hosts=(
        "$REMOTE_PREFILL"
        "$REMOTE_DECODE"
    )
    local flags=(false false)
    while [[ ${flags[*]} =~ false ]]; do
        passtime=$(($(date +%s) - "$starttime"))
        echo "Some processes are not running (${flags[*]}, $passtime s passed). Waiting..."
        local idx
        for idx in "${!processes[@]}"; do
            if REMOTE_HOST="${remote_hosts[$idx]}" check_process_running "${processes[$idx]}"; then
                flags[idx]=true
            fi
        done
        sleep 1
    done
    echo "All disagg processes are running."
}

function wait_disagg_instances_ready() {
    local remote_hosts=(
        "$REMOTE_PREFILL"
        "$REMOTE_DECODE"
    )
    local ports=(
        8100 8200
    )
    local processes=(
        prefill_instance_start
        decode_instance_start
    )
    local starttime
    starttime=$(date +%s)
    local timeout=240
    local passtime
    passtime=$(($(date +%s) - "$starttime"))
    while (("$passtime" < "$timeout")); do
        sleep 10
        passtime=$(($(date +%s) - "$starttime"))
        local flag=true
        local idx
        for idx in "${!processes[@]}"; do
            local endpoint="${remote_hosts[$idx]}:${ports[$idx]}"
            if ! check_endpoint_serving "$endpoint"; then
                echo "Endpoint $endpoint is not serving."
                flag=false
            fi
        done
        if [[ "$flag" = true ]]; then
            echo "All disagg instances are serving."
            return 0
        fi
        for idx in "${!processes[@]}"; do
            if ! REMOTE_HOST=${remote_hosts[$idx]} check_process_running "${processes[$idx]}"; then
                echo "${processes[$idx]} is not running."
                return 1
            fi
        done
        echo "Some instances are not serving (${passtime}s passed with timeout ${timeout}s). Waiting..."
    done
}

function wait_disagg_instances_fin() {
    local starttime
    starttime=$(date +%s)
    local timeout=120
    local passtime
    passtime=$(($(date +%s) - starttime))
    local processes=(
        prefill_instance_start
        decode_instance_start
    )
    local remote_hosts=(
        "$REMOTE_PREFILL"
        "$REMOTE_DECODE"
    )
    while ((passtime < timeout)); do
        sleep 2
        passtime=$(($(date +%s) - starttime))
        local flag=true
        for idx in "${!processes[@]}"; do
            if REMOTE_HOST="${remote_hosts[$idx]}" check_process_running "${processes[$idx]}"; then
                flag=false
            fi
        done
        if [[ "$flag" = true ]]; then
            echo "All disagg instances are finished."
            return 0
        fi
    done
    return 1
}

function start_disagg_proxy_in_tmux() {
    if [[ -n "${MODEL}" ]]; then
        REMOTE_HOST="$REMOTE_PROXY" remote_or_local_execute \
            tmux send-key -t disagg:proxy "\"MODEL=${MODEL} \""
    fi
    REMOTE_HOST="$REMOTE_PROXY" remote_or_local_execute \
        tmux send-key -t disagg:proxy \
        "\"ENDPOINTS=\\\"$ENDPOINTS\\\" bash examples/distributed_kv/proxy_instance_start.sh | tee proxy.out\"" ENTER
}

function kill_disagg_proxy_server() {
    REMOTE_HOST="$REMOTE_PROXY" kill_running_process disagg_prefill_proxy_server
}

function wait_disagg_proxy_ready() {
    local starttime
    starttime=$(date +%s)
    local timeout=30
    local passtime
    passtime=$(($(date +%s) - starttime))
    while (("$passtime" < "$timeout")); do
        local endpoint="${REMOTE_PROXY}:8000"
        if check_endpoint_serving "$endpoint"; then
            return 0
        fi
        passtime=$(($(date +%s) - starttime))
    done
    return 1
}

function start_disagg_environemnt() {
    if [ -z "$HF_HUB_OFFLINE" ]; then
        >&2 echo "HF_HUB_OFFLINE is not set. Please set it globaly to 1."
        exit 1
    fi
    if [ -z "$REMOTE_PREFILL" ]; then
        >&2 echo "Please set REMOTE_PREFILL variable"
        exit 1
    fi
    if [ -z "$REMOTE_DECODE" ]; then
        >&2 echo "Please set REMOTE_DECODE variable"
        exit 1
    fi
    if [ -z "$REMOTE_PROXY" ]; then
        >&2 echo "Please set REMOTE_PROXY variable"
        exit 1
    fi

    start_disagg_tmux_session
    start_disagg_instances_in_tmux
    wait_disagg_instances_running
    while ! wait_disagg_instances_ready; do
        echo "Some instances have failed to start. Restarting..."
        REMOTE_HOST=$REMOTE_PREFILL kill_openai_servers
        REMOTE_HOST=$REMOTE_DECODE kill_openai_servers
        wait_disagg_instances_fin
        sleep 5
        start_disagg_instances_in_tmux
        wait_disagg_instances_running
    done
    ENDPOINTS="$REMOTE_PREFILL:8100 $REMOTE_DECODE:8200" start_disagg_proxy_in_tmux
    wait_disagg_proxy_ready
}

function stop_disagg_environment() {
    REMOTE_HOST=$REMOTE_PREFILL kill_openai_servers
    REMOTE_HOST=$REMOTE_DECODE kill_openai_servers
    kill_disagg_proxy_server
    wait_disagg_instances_fin
    kill_disagg_tmux_session
}
