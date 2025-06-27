#!/bin/bash
set -e

function check_process_running() {
    local ident=$1
    if [ -n "$REMOTE_HOST" ]; then
        # shellcheck disable=SC2029
        ssh "$REMOTE_HOST" "pgrep -a -f $ident | grep -v ssh" > /dev/null || return 1
    else
        pgrep -f "${ident}" > /dev/null || return 1
    fi
}

function kill_running_process() {
    local ident=$1
    if [ -n "$REMOTE_HOST" ]; then
        # ref: https://stackoverflow.com/questions/40473088/pkill-returns-255-in-combination-with-another-command-via-remote-ssh
        # NOTE: prevent kill ssh session
        # shellcheck disable=SC2029
        ssh "$REMOTE_HOST" "pgrep -a -f $ident | grep -v ssh | awk '{print \$1}' | xargs kill"
    else
        pkill -f "$ident"
    fi
}

function check_port_serving() {
    local port=$1
    curl -s "localhost:${port}/v1/completions" > /dev/null || return 1
}

function check_endpoint_serving() {
    local endpoint=$1
    curl -s "${endpoint}/v1/completions" > /dev/null || return 1
}

function kill_openai_servers() {
    if [ -n "$REMOTE_HOST" ]; then
        if check_process_running spawn; then
            kill_running_process spawn
        fi
        if check_process_running openai.api_server; then
            kill_running_process openai.api_server
        fi
    else
        if pgrep -f "spawn" > /dev/null ; then
            pkill -f "spawn"
        fi
        if pgrep -f "openai.api_server" > /dev/null ; then
            pkill -f "openai.api_server"
        fi
    fi
}

function remote_or_local_execute() {
    echo "$@"
    if [ -n "$REMOTE_HOST" ]; then
        # shellcheck disable=SC2029
        ssh "$REMOTE_HOST" "$@"
    else
        "$@"
    fi
}

function start_remote_session() {
    local remote_host=$1
    local session_name=$2
    # shellcheck disable=SC2029
    if ! ssh "$remote_host" "tmux has-session -t $session_name" > /dev/null 2>&1; then
        ssh "$remote_host" "tmux new-session -e HF_HUB_OFFLINE=1 -s $session_name -d bash"
    fi
}

function stop_remote_session() {
    local remote_host=$1
    local session_name=$2
    # shellcheck disable=SC2029
    if ssh "$remote_host" "tmux has-session -t $session_name"; then
        ssh "$remote_host" "tmux kill-session -t $session_name"
    fi
}
