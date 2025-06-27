#!/bin/bash
set -e

# shellcheck source=../../examples/distributed_utils/utils.sh
source examples/distributed_utils/utils.sh

function start_adrenaline_tmux_session() {
    # tmux new-session -d -e "HF_HUB_OFFLINE=1" -s adrenaline bash
    tmux new-session -d -n proxy -s adrenaline bash
    tmux new-window -n prefill -t adrenaline: bash
    tmux new-window -n decode -t adrenaline: bash
    tmux new-window -n attention -t adrenaline: bash
}

function kill_adrenaline_tmux_session() {
    tmux kill-session -t adrenaline
}

function start_adrenaline_instances() {
    OFFLOAD_RATIO=${OFFLOAD_RATIO} \
        bash examples/adrenaline/prefill_instance_start.sh |
        tee prefill.out &

    OFFLOAD_RATIO=${OFFLOAD_RATIO} \
        bash examples/adrenaline/decode_instance_start.sh |
        tee decode.out &

    OFFLOAD_RATIO=${OFFLOAD_RATIO} \
        bash examples/adrenaline/attention_instance_start.sh |
        tee attention.out &
}

function start_adrenaline_instances_in_tmux() {
    if [[ -n "${MODEL}" ]]; then
        tmux send-key -t adrenaline:prefill "MODEL=${MODEL} "
        tmux send-key -t adrenaline:decode "MODEL=${MODEL} "
        tmux send-key -t adrenaline:attention "MODEL=${MODEL} "
    fi

    tmux send-key -t adrenaline:prefill \
        "OFFLOAD_RATIO=${OFFLOAD_RATIO} \
        bash examples/adrenaline/prefill_instance_start.sh \
        | tee prefill.out" ENTER

    tmux send-key -t adrenaline:decode \
        "OFFLOAD_RATIO=${OFFLOAD_RATIO} \
        bash examples/adrenaline/decode_instance_start.sh \
        | tee decode.out" ENTER

    if [[ -n "${NUM_ATTENTION_BLOCKS}" ]]; then
        tmux send-key -t adrenaline:attention \
            "NUM_BLOCKS=${NUM_ATTENTION_BLOCKS} "
    fi
    tmux send-key -t adrenaline:attention \
        "OFFLOAD_RATIO=${OFFLOAD_RATIO} \
        bash examples/adrenaline/attention_instance_start.sh \
        | tee attention.out" ENTER
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
    local flags=(false false false)
    while [[ ${flags[*]} =~ false ]]; do
        passtime=$(($(date +%s) - "$starttime"))
        for ident in "${!processes[@]}"; do
            if check_process_running "${processes[$ident]}"; then
                flags[ident]=true
            fi
        done
        echo "Some processes are not running (${flags[*]}, $passtime s passed). Waiting..."
        sleep 1
    done
    echo "All adrenaline processes are running."
}

function wait_adrenaline_instances_ready() {
    local starttime
    starttime=$(date +%s)
    local timeout=300
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
        for port in "${ports[@]}"; do
            if ! check_port_serving "${port}"; then
                echo "Port ${port} is not serving."
                flag=false
            fi
        done
        if [[ "$flag" = true ]]; then
            echo "All adrenaline instances are serving."
            return 0
        fi
        for ident in "${processes[@]}"; do
            if ! check_process_running "$ident"; then
                echo "$ident is not running."
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
    while ((passtime < timeout)); do
        sleep 2
        passtime=$(($(date +%s) - starttime))
        local flag=true
        for ident in "${!processes[@]}"; do
            if check_process_running "${processes[$ident]}"; then
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

function start_adrenaline_proxy() {
    if [ ! -f decode.out ]; then
        echo "decode.out is not found. Please run the decode instance first."
        exit 1
    fi
    local num_decode_blocks
    num_decode_blocks=$(grep "GPU blocks" decode.out | sed "s/.*GPU blocks: //g" | sed "s/,.*//g")
    local num_attention_blocks
    num_attention_blocks=$(echo "(${num_decode_blocks} * ${OFFLOAD_RATIO}) / 1" | bc)
    OFFLOAD_RATIO=${OFFLOAD_RATIO} \
        NUM_DECODE_BLOCKS=${num_decode_blocks} \
        NUM_ATTENTION_BLOCKS=${num_attention_blocks} \
        bash examples/adrenaline/proxy_instance_start.sh |
        tee proxy.out &
}

function start_adrenaline_proxy_in_tmux() {
    if [ ! -f decode.out ]; then
        echo "decode.out is not found. Please run the decode instance first."
        exit 1
    fi
    local num_decode_blocks
    num_decode_blocks=$(grep "GPU blocks" decode.out | sed "s/.*GPU blocks: //g" | sed "s/,.*//g")
    local num_attention_blocks=${NUM_ATTENTION_BLOCKS}
    if [[ -n "${MODEL}" ]]; then
        tmux send-key -t adrenaline:proxy "MODEL=${MODEL} "
    fi
    if [[ -n "${OFFLOAD_BACKEND}" ]]; then
        tmux send-key -t adrenaline:proxy \
            "OFFLOAD_BACKEND=${OFFLOAD_BACKEND} "
    fi
    tmux send-key -t adrenaline:proxy \
        "OFFLOAD_RATIO=${OFFLOAD_RATIO} \
        NUM_DECODE_BLOCKS=${num_decode_blocks} \
        NUM_ATTENTION_BLOCKS=${num_attention_blocks} \
        bash examples/adrenaline/proxy_instance_start.sh \
        | tee proxy.out" ENTER
}

function kill_adrenaline_proxy_server() {
    pgrep -f "adrenaline_proxy_server" | xargs kill -s TERM
}

function wait_adrenaline_proxy_ready() {
    local starttime
    starttime=$(date +%s)
    local timeout=30
    local passtime
    passtime=$(($(date +%s) - "$starttime"))
    while (("$passtime" < "$timeout")); do
        if check_port_serving 8000; then
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
    start_adrenaline_tmux_session

    start_adrenaline_instances_in_tmux
    wait_adrenaline_instances_running
    while ! wait_adrenaline_instances_ready; do
        echo "Some instances have failed to start. Restarting..."
        kill_openai_servers
        wait_adrenaline_instances_fin
        sleep 5
        start_adrenaline_instances_in_tmux
        wait_adrenaline_instances_running
    done

    start_adrenaline_proxy_in_tmux
    wait_adrenaline_proxy_ready
}

function stop_adrenaline_environment() {
    kill_openai_servers
    kill_adrenaline_proxy_server
    wait_adrenaline_instances_fin
    kill_adrenaline_tmux_session
}
