#!/bin/bash
set -e

# shellcheck source=../../examples/distributed_utils/utils.sh
source examples/distributed_utils/utils.sh

function start_adrenaline_tmux_session() {
    if [ -n "$REMOTE_PROXY" ]; then
        start_remote_session "$REMOTE_PROXY" adrenaline
    fi
    REMOTE_HOST="$REMOTE_PROXY" remote_or_local_execute \
        tmux new-window -e "HF_HUB_OFFLINE=1" -c ~/adrenaline -n proxy -t adrenaline: bash

    if [ -n "$REMOTE_PREFILL" ]; then
        start_remote_session "$REMOTE_PREFILL" adrenaline
    fi
    REMOTE_HOST="$REMOTE_PREFILL" remote_or_local_execute \
        tmux new-window -e "HF_HUB_OFFLINE=1" -c ~/adrenaline -n prefill -t adrenaline: bash
    REMOTE_HOST="$REMOTE_PREFILL" remote_or_local_execute \
        tmux new-window -e "HF_HUB_OFFLINE=1" -c ~/adrenaline -n attention -t adrenaline: bash

    if [ -n "$REMOTE_DECODE" ]; then
        start_remote_session "$REMOTE_DECODE" adrenaline
    fi
    REMOTE_HOST="$REMOTE_DECODE" remote_or_local_execute \
        tmux new-window -e "HF_HUB_OFFLINE=1" -c ~/adrenaline -n decode -t adrenaline: bash
}

function kill_adrenaline_tmux_session() {
    if [ -n "$REMOTE_PREFILL" ]; then
        stop_remote_session "$REMOTE_PREFILL" adrenaline
    fi
    if [ -n "$REMOTE_DECODE" ]; then
        stop_remote_session "$REMOTE_DECODE" adrenaline
    fi
}

function start_adrenaline_instances_in_tmux() {
    if [[ -n "${MODEL}" ]]; then
        REMOTE_HOST="$REMOTE_PREFILL" remote_or_local_execute \
            tmux send-key -t adrenaline:prefill "\"MODEL=${MODEL} \""
        REMOTE_HOST="$REMOTE_PREFILL" remote_or_local_execute \
            tmux send-key -t adrenaline:attention "\"MODEL=${MODEL} \""
        REMOTE_HOST="$REMOTE_DECODE" remote_or_local_execute \
            tmux send-key -t adrenaline:decode "\"MODEL=${MODEL} \""
    fi

    REMOTE_HOST="$REMOTE_PREFILL" remote_or_local_execute \
        tmux send-key -t adrenaline:prefill "\"OFFLOAD_RATIO=${OFFLOAD_RATIO} VLLM_HOST_IP=$REMOTE_PREFILL GLOO_SOCKET_IFNAME=ib0 \
        bash examples/adrenaline/prefill_instance_start.sh | tee prefill.out\"" ENTER

    REMOTE_HOST="$REMOTE_DECODE" remote_or_local_execute \
        tmux send-key -t adrenaline:decode "\"OFFLOAD_RATIO=${OFFLOAD_RATIO} VLLM_HOST_IP=$REMOTE_PREFILL GLOO_SOCKET_IFNAME=ib0 \
        bash examples/adrenaline/decode_instance_start.sh | tee decode.out\"" ENTER

    if [[ -n "${NUM_ATTENTION_BLOCKS}" ]]; then
        REMOTE_HOST="$REMOTE_PREFILL" remote_or_local_execute \
            tmux send-key -t adrenaline:attention \
            "\"NUM_BLOCKS=${NUM_ATTENTION_BLOCKS} \""
    fi
    REMOTE_HOST="$REMOTE_PREFILL" remote_or_local_execute \
        tmux send-key -t adrenaline:attention \
        "\"OFFLOAD_RATIO=${OFFLOAD_RATIO} VLLM_HOST_IP=$REMOTE_PREFILL GLOO_SOCKET_IFNAME=ib0 \
        bash examples/adrenaline/attention_instance_start.sh | tee attention.out\"" ENTER
}

function wait_adrenaline_instances_running() {
    local starttime
    starttime=$(date +%s)
    local passtime

    local processes=(
        prefill_instance_start
        decode_instance_start
        attention_instance_start
    )
    local remote_hosts=(
        "$REMOTE_PREFILL"
        "$REMOTE_DECODE"
        "$REMOTE_PREFILL"
    )
    local flags=(false false false)
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
    echo "All adrenaline processes are running."
}

function wait_adrenaline_instances_ready() {
    local starttime
    starttime=$(date +%s)
    local timeout=300
    local remote_hosts=(
        "$REMOTE_PREFILL"
        "$REMOTE_DECODE"
        "$REMOTE_PREFILL"
    )
    local ports=(
        8100
        8200
        8300
    )
    local processes=(
        prefill_instance_start
        decode_instance_start
        attention_instance_start
    )
    local passtime
    passtime=$(($(date +%s) - "$starttime"))
    while (("$passtime" < "$timeout")); do
        sleep 10
        passtime=$(($(date +%s) - "$starttime"))
        local flag=true
        local idx
        for idx in "${!processes[@]}"; do
        local endpoint="${remote_hosts[$idx]}:${ports[$idx]}"
            if ! check_endpoint_serving "${endpoint}"; then
                echo "Endpoint $endpoint is not serving."
                flag=false
            fi
        done
        if [[ "$flag" = true ]]; then
            echo "All adrenaline instances are serving."
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

function wait_adrenaline_instances_fin() {
    local starttime
    starttime=$(date +%s)
    local timeout=120
    local passtime
    passtime=$(($(date +%s) - starttime))
    local processes=(
        prefill_instance_start
        decode_instance_start
        attention_instance_start
    )
    local remote_hosts=(
        "$REMOTE_PREFILL"
        "$REMOTE_DECODE"
        "$REMOTE_PREFILL"
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
            echo "All adrenaline instances are finished."
            return 0
        fi
    done
    return 1
}

function start_adrenaline_proxy_in_tmux() {
    # shellcheck disable=SC2088
    local working_directory="~/adrenaline"
    # shellcheck disable=SC2029
    if ssh "$REMOTE_DECODE" "cd $working_directory; exec [ ! -f decode.out ]"; then
        echo "decode.out is not found. Please run the decode instance first."
        exit 1
    fi
    local num_decode_blocks
    # shellcheck disable=SC2029
    num_decode_blocks=$(ssh "$REMOTE_DECODE" "cd $working_directory; exec grep \"GPU blocks\" decode.out | sed \"s/.*GPU blocks: //g\" | sed \"s/,.*//g\"")
    local num_attention_blocks
    num_attention_blocks=$(echo "(${num_decode_blocks} * ${OFFLOAD_RATIO}) / 1" | bc)
    if [[ -n "${MODEL}" ]]; then
        REMOTE_HOST="$REMOTE_PROXY" remote_or_local_execute \
            tmux send-key -t adrenaline:proxy "\"MODEL=${MODEL} \""
    fi
    if [[ -n "${OFFLOAD_BACKEND}" ]]; then
        REMOTE_HOST="$REMOTE_PROXY" remote_or_local_execute \
            tmux send-key -t adrenaline:proxy "\"OFFLOAD_BACKEND=${OFFLOAD_BACKEND} \""
    fi
    REMOTE_HOST="$REMOTE_PROXY" remote_or_local_execute \
        tmux send-key -t adrenaline:proxy \
        "\"OFFLOAD_RATIO=${OFFLOAD_RATIO} \
        NUM_DECODE_BLOCKS=${num_decode_blocks} \
        NUM_ATTENTION_BLOCKS=${num_attention_blocks} \
        ENDPOINTS=\\\"$ENDPOINTS\\\" \
        bash examples/adrenaline/proxy_instance_start.sh \
        | tee proxy.out\"" ENTER
}

function kill_adrenaline_proxy_server() {
    REMOTE_HOST="$REMOTE_PROXY" kill_running_process adrenaline_proxy_server
}

function wait_adrenaline_proxy_ready() {
    local starttime
    starttime=$(date +%s)
    local timeout=30
    local passtime
    passtime=$(($(date +%s) - "$starttime"))
    while (("$passtime" < "$timeout")); do
        local endpoint="${REMOTE_PROXY}:8000"
        if check_endpoint_serving "$endpoint"; then
            return 0
        fi
        passtime=$(($(date +%s) - "$starttime"))
    done
    return 1
}

function start_adrenaline_environemnt() {
    if [ -z "$HF_HUB_OFFLINE" ]; then
        echo "HF_HUB_OFFLINE is not set. Please set it globaly to 1."
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

    start_adrenaline_tmux_session
    start_adrenaline_instances_in_tmux
    wait_adrenaline_instances_running
    while ! wait_adrenaline_instances_ready; do
        echo "Some instances have failed to start. Restarting..."
        REMOTE_HOST=$REMOTE_PREFILL kill_openai_servers
        REMOTE_HOST=$REMOTE_DECODE kill_openai_servers
        wait_adrenaline_instances_fin
        sleep 5
        start_adrenaline_instances_in_tmux
        wait_adrenaline_instances_running
    done
    ENDPOINTS="$REMOTE_PREFILL:8100 $REMOTE_DECODE:8200 $REMOTE_PREFILL:8300" start_adrenaline_proxy_in_tmux
    wait_adrenaline_proxy_ready
}

function stop_adrenaline_environment() {
    REMOTE_HOST=$REMOTE_PREFILL kill_openai_servers
    REMOTE_HOST=$REMOTE_DECODE kill_openai_servers
    kill_adrenaline_proxy_server
    wait_adrenaline_instances_fin
    kill_adrenaline_tmux_session
}
