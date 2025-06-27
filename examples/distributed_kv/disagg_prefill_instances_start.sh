#!/bin/bash
set -e

# shellcheck source=../../examples/distributed_utils/utils.sh
source examples/distributed_utils/utils.sh

function start_disagg_tmux_session() {
    # tmux new-session -d -e "HF_HUB_OFFLINE=1" -s disagg fish
    tmux new-session -d -n proxy -s disagg fish
    tmux new-window -n prefill -t disagg: fish
    tmux new-window -n decode -t disagg: fish
}

function kill_disagg_tmux_session() {
    tmux kill-session -t disagg
}

function start_disagg_instances() {
    bash examples/distributed_kv/prefill_instance_start.sh |
        tee prefill.out &

    bash examples/distributed_kv/decode_instance_start.sh |
        tee decode.out &
}

function start_disagg_instances_in_tmux() {
    if [[ -n "${MODEL}" ]]; then
        tmux send-key -t disagg:prefill "MODEL=${MODEL} "
        tmux send-key -t disagg:decode "MODEL=${MODEL} "
    fi

    tmux send-key -t disagg:prefill \
        "bash examples/distributed_kv/prefill_instance_start.sh \
        | tee prefill.out" ENTER

    tmux send-key -t disagg:decode \
        "bash examples/distributed_kv/decode_instance_start.sh \
        | tee decode.out" ENTER
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
    local flags=(false false)
    while [[ ${flags[*]} =~ false ]]; do
        for ident in "${!processes[@]}"; do
            if check_process_running "${processes[$ident]}"; then
                flags[ident]=true
            fi
        done
        echo "Some processes are not running (${flags[*]}, $passtime s passed). Waiting..."
        sleep 1
    done
    echo "All disagg processes are running."
}

function wait_disagg_instances_ready() {
    local ports=(
        8100 8200
    )
    local processes=(
        prefill_instance_start decode_instance_start
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
        for port in "${ports[@]}"; do
            if ! check_port_serving "${port}"; then
                echo "Port ${port} is not serving."
                flag=false
            fi
        done
        if [[ "$flag" = true ]]; then
            echo "All disagg instances are serving."
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

function wait_disagg_instances_fin() {
    local starttime
    starttime=$(date +%s)
    local timeout=120
    local passtime
    passtime=$(($(date +%s) - starttime))
    processes=(
        prefill_instance_start
        decode_instance_start
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
            echo "All disagg instances are finished."
            return 0
        fi
    done
    return 1
}

function start_disagg_proxy() {
    bash examples/distributed_kv/proxy_instance_start.sh |
        tee proxy.out &
}

function start_disagg_proxy_in_tmux() {
    if [[ -n "${MODEL}" ]]; then
        tmux send-key -t disagg:proxy "MODEL=${MODEL} "
    fi
    tmux send-key -t disagg:proxy \
        "bash examples/distributed_kv/proxy_instance_start.sh \
        | tee proxy.out" ENTER
}

function kill_disagg_proxy_server() {
    pgrep -f "disagg_prefill_proxy_server" | xargs kill -s TERM
}

function wait_disagg_proxy_ready() {
    local starttime
    starttime=$(date +%s)
    local timeout=30
    local passtime
    passtime=$(($(date +%s) - starttime))
    while (("$passtime" < "$timeout")); do
        if check_port_serving 8000; then
            return 0
        fi
        passtime=$(($(date +%s) - starttime))
    done
    return 1
}

function start_disagg_environemnt() {
    if [ -z "$HF_HUB_OFFLINE" ]; then
        echo "HF_HUB_OFFLINE is not set. Please set it globaly to 1."
        exit 1
    fi
    start_disagg_tmux_session

    start_disagg_instances_in_tmux
    wait_disagg_instances_running
    while ! wait_disagg_instances_ready; do
        echo "Some instances have failed to start. Restarting..."
        kill_openai_servers
        wait_disagg_instances_fin
        sleep 5
        start_disagg_instances_in_tmux
        wait_disagg_instances_running
    done

    start_disagg_proxy_in_tmux
    wait_disagg_proxy_ready
}

function stop_disagg_environment() {
    kill_openai_servers
    kill_disagg_proxy_server
    wait_disagg_instances_fin
    kill_disagg_tmux_session
}
