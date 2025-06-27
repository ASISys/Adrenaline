#!/bin/bash
# This file demonstrates the example usage of configurable xP1D adrenaline setup
# We will launch (x+1) vllm instances (x for prefill and 1 for decode),
# along with x attention instances for KV offloading.
#
# CUDA Device Allocation with Tensor Parallelism Support:
# Set TENSOR_PARALLEL_SIZE to specify how many GPUs each process should use (default: 1)
#
# Option 1 - Automatic distribution:
#   Set AVAILABLE_CUDA_DEVICES environment variable to specify which GPUs to use.
#   Example: TENSOR_PARALLEL_SIZE=2 AVAILABLE_CUDA_DEVICES="0,1,2,3,4,5" ./adrenaline_instances_start.sh
#   If not set, defaults to "0,1,2,3,4,5,6,7"
#   The script will automatically distribute device ranges between instances.
#
# Option 2 - Explicit assignment:
#   Set PREFILL_CUDA_DEVICES_EXPLICIT for prefill instances (semicolon-separated device ranges)
#   Set ATTENTION_CUDA_DEVICES_EXPLICIT for attention instances (semicolon-separated device ranges)
#   Set DECODE_CUDA_DEVICE_EXPLICIT for decode instance (comma-separated device range)
#   Example: PREFILL_CUDA_DEVICES_EXPLICIT="0,1;2,3" ATTENTION_CUDA_DEVICES_EXPLICIT="0,1;2,3" DECODE_CUDA_DEVICE_EXPLICIT="4,5" ./adrenaline_instances_start.sh
#
# Tensor Parallelism Examples:
#   TENSOR_PARALLEL_SIZE=2 PREFILL_INSTANCES=2 -> Prefill 0: GPUs 0,1; Prefill 1: GPUs 2,3; Decode: GPUs 4,5
#   Attention instances co-locate with prefill instances (share same GPUs)

set -e

# shellcheck source=../../../examples/distributed_utils/utils.sh
source examples/distributed_utils/utils.sh

# Function to parse and distribute CUDA devices
function distribute_adrenaline_cuda_devices() {
    local available_devices="${AVAILABLE_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
    local prefill_instances="${PREFILL_INSTANCES:-1}"
    local tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}"

    # Convert comma-separated string to array
    IFS=',' read -ra DEVICE_ARRAY <<< "$available_devices"
    local total_devices=${#DEVICE_ARRAY[@]}

    # Calculate total devices needed
    local devices_per_prefill=$tensor_parallel_size
    local devices_per_decode=$tensor_parallel_size
    # Prefill and attention instances share devices, decode uses separate devices
    local total_devices_needed=$((prefill_instances * devices_per_prefill + devices_per_decode))

    if [ "$total_devices" -lt "$total_devices_needed" ]; then
        echo "Warning: Not enough CUDA devices ($total_devices) for tensor parallelism"
        echo "Need $total_devices_needed devices ($prefill_instances prefill × $devices_per_prefill + 1 decode × $devices_per_decode)"
        echo "Consider reducing TENSOR_PARALLEL_SIZE or PREFILL_INSTANCES"
    fi

    # Check if explicit device assignments are provided
    if [[ -n "${PREFILL_CUDA_DEVICES_EXPLICIT}" ]]; then
        # Use explicit assignments for prefill instances
        IFS=';' read -ra PREFILL_CUDA_DEVICES <<< "$PREFILL_CUDA_DEVICES_EXPLICIT"
        echo "Using explicit prefill device assignments: $PREFILL_CUDA_DEVICES_EXPLICIT"
    else
        # Distribute device ranges to prefill instances
        PREFILL_CUDA_DEVICES=()
        local device_offset=0
        for ((i=0; i<prefill_instances; i++)); do
            local device_range=""
            for ((j=0; j<tensor_parallel_size; j++)); do
                local device_idx=$(((device_offset + j) % total_devices))
                if [ $j -eq 0 ]; then
                    device_range="${DEVICE_ARRAY[device_idx]}"
                else
                    device_range="${device_range},${DEVICE_ARRAY[device_idx]}"
                fi
            done
            PREFILL_CUDA_DEVICES[i]="$device_range"
            device_offset=$((device_offset + tensor_parallel_size))
        done
    fi

    # Handle attention device assignment (co-located with prefill instances)
    if [[ -n "${ATTENTION_CUDA_DEVICES_EXPLICIT}" ]]; then
        IFS=';' read -ra ATTENTION_CUDA_DEVICES <<< "$ATTENTION_CUDA_DEVICES_EXPLICIT"
        echo "Using explicit attention device assignments: $ATTENTION_CUDA_DEVICES_EXPLICIT"
    else
        # Use same device ranges as prefill instances for attention instances
        ATTENTION_CUDA_DEVICES=("${PREFILL_CUDA_DEVICES[@]}")
    fi

    # Handle decode device assignment (separate from prefill/attention)
    if [[ -n "${DECODE_CUDA_DEVICE_EXPLICIT}" ]]; then
        DECODE_CUDA_DEVICE="$DECODE_CUDA_DEVICE_EXPLICIT"
        echo "Using explicit decode device assignment: $DECODE_CUDA_DEVICE_EXPLICIT"
    else
        # Assign device range to decode instance (non-overlapping with prefill)
        local decode_device_range=""
        local decode_offset=$((prefill_instances * tensor_parallel_size))
        for ((j=0; j<tensor_parallel_size; j++)); do
            local device_idx=$(((decode_offset + j) % total_devices))
            if [ $j -eq 0 ]; then
                decode_device_range="${DEVICE_ARRAY[device_idx]}"
            else
                decode_device_range="${decode_device_range},${DEVICE_ARRAY[device_idx]}"
            fi
        done
        DECODE_CUDA_DEVICE="$decode_device_range"
    fi

    # Export for use in sub-scripts
    export PREFILL_CUDA_DEVICES
    export ATTENTION_CUDA_DEVICES
    export DECODE_CUDA_DEVICE
    export TENSOR_PARALLEL_SIZE

    echo "CUDA Device Distribution (Tensor Parallel Size: $tensor_parallel_size):"
    echo "Available devices: $available_devices"
    echo "Prefill instances: $prefill_instances"
    for ((i=0; i<prefill_instances; i++)); do
        echo "  Prefill instance $i: GPUs ${PREFILL_CUDA_DEVICES[i]}"
    done
    for ((i=0; i<prefill_instances; i++)); do
        echo "  Attention instance $i: GPUs ${ATTENTION_CUDA_DEVICES[i]} (co-located with prefill $i)"
    done
    echo "  Decode instance: GPUs $DECODE_CUDA_DEVICE"
}

function start_adrenaline_instances() {
    local prefill_instances="${PREFILL_INSTANCES:-1}"

    # Distribute CUDA devices before starting instances
    distribute_adrenaline_cuda_devices

    # Start prefill instances
    for ((i=0; i<prefill_instances; i++)); do
        ASSIGNED_CUDA_DEVICE=${PREFILL_CUDA_DEVICES[i]} \
        INSTANCE_RANK=$i \
        TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} \
            bash examples/adrenaline/xp1d/prefill_instance_start.sh $i |
            tee "prefill_${i}.out" &
    done

    # Start attention instances
    for ((i=0; i<prefill_instances; i++)); do
        ASSIGNED_CUDA_DEVICE=${ATTENTION_CUDA_DEVICES[i]} \
        INSTANCE_RANK=$((prefill_instances + 1 + i)) \
        TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} \
            bash examples/adrenaline/xp1d/attention_instance_start.sh $i |
            tee "attention_${i}.out" &
    done

    # Start decode instance
    ASSIGNED_CUDA_DEVICE=$DECODE_CUDA_DEVICE \
    INSTANCE_RANK=$prefill_instances \
    TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} \
        bash examples/adrenaline/xp1d/decode_instance_start.sh |
        tee decode.out &
}

function start_adrenaline_tmux_session() {
    local prefill_instances="${PREFILL_INSTANCES:=1}"
    
    tmux new-session -d -n proxy -s adrenaline bash
    
    # Create windows for prefill instances
    for ((i=0; i<prefill_instances; i++)); do
        tmux new-window -n "p${i}" -t adrenaline: bash
    done
    
    # Create windows for attention instances
    for ((i=0; i<prefill_instances; i++)); do
        tmux new-window -n "a${i}" -t adrenaline: bash
    done
    
    # Create window for decode instance
    tmux new-window -n d0 -t adrenaline: bash
}

function kill_adrenaline_tmux_session() {
    tmux kill-session -t adrenaline
}

function start_adrenaline_instances_in_tmux() {
    local prefill_instances="${PREFILL_INSTANCES:-1}"

    # Distribute CUDA devices before starting instances
    distribute_adrenaline_cuda_devices

    # Set environment variables for all instances
    local all_windows=()
    for ((i=0; i<prefill_instances; i++)); do
        all_windows+=("p${i}")
    done
    for ((i=0; i<prefill_instances; i++)); do
        all_windows+=("a${i}")
    done
    all_windows+=("d0")

    # Set MODEL environment variable
    if [[ -n "${MODEL}" ]]; then
        for window in "${all_windows[@]}"; do
            tmux send-key -t "adrenaline:${window}" "MODEL=${MODEL} "
        done
    fi

    # Set MAX_MODEL_LEN environment variable
    if [[ -n "${MAX_MODEL_LEN}" ]]; then
        for window in "${all_windows[@]}"; do
            tmux send-key -t "adrenaline:${window}" "MAX_MODEL_LEN=${MAX_MODEL_LEN} "
        done
    fi

    # Set LOAD_FORMAT environment variable
    if [[ -n "${LOAD_FORMAT}" ]]; then
        for window in "${all_windows[@]}"; do
            tmux send-key -t "adrenaline:${window}" "LOAD_FORMAT=${LOAD_FORMAT} "
        done
    fi

    if [[ -n "$TENSOR_PARALLEL_SIZE" ]]; then
        for window in "${all_windows[@]}"; do
            tmux send-key -t "adrenaline:${window}" "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} "
        done
    fi

    for window in "${all_windows[@]}"; do
        tmux send-key -t "adrenaline:${window}" "PREFILL_INSTANCES=${PREFILL_INSTANCES} "
    done

    if [[ -n "${ATTENTION_SM_PCT}" ]]; then
        local prefill_sm_pct=$((100 - ATTENTION_SM_PCT))
        for ((i=0; i<prefill_instances; i++)); do
            tmux send-key -t "adrenaline:p${i}" "SM_PCT=$prefill_sm_pct "
            tmux send-key -t "adrenaline:a${i}" "SM_PCT=$ATTENTION_SM_PCT "
        done
    fi

    # Set NUM_PREFILL_BLOCKS for prefill instances
    if [[ -n "${NUM_PREFILL_BLOCKS}" ]]; then
        for ((i=0; i<prefill_instances; i++)); do
            tmux send-key -t "adrenaline:p${i}" "NUM_BLOCKS=${NUM_PREFILL_BLOCKS} "
        done
    fi

    # Set NUM_ATTENTION_BLOCKS for attention instances
    if [[ -n "${NUM_ATTENTION_BLOCKS}" ]]; then
        for ((i=0; i<prefill_instances; i++)); do
            tmux send-key -t "adrenaline:a${i}" "NUM_BLOCKS=${NUM_ATTENTION_BLOCKS} "
        done
    fi

    # Start prefill instances
    for ((i=0; i<prefill_instances; i++)); do
        tmux send-key -t "adrenaline:p${i}" \
            "ASSIGNED_CUDA_DEVICE=${PREFILL_CUDA_DEVICES[i]} INSTANCE_RANK=$i bash examples/adrenaline/xp1d/prefill_instance_start.sh $i 2>&1 | tee prefill_${i}.out" ENTER
    done

    # Start decode instance
    tmux send-key -t adrenaline:d0 \
        "ASSIGNED_CUDA_DEVICE=$DECODE_CUDA_DEVICE INSTANCE_RANK=$prefill_instances bash examples/adrenaline/xp1d/decode_instance_start.sh 2>&1 | tee decode.out" ENTER

    # Start attention instances
    for ((i=0; i<prefill_instances; i++)); do
        local attention_rank=$((prefill_instances + 1 + i))
        tmux send-key -t "adrenaline:a${i}" \
            "ASSIGNED_CUDA_DEVICE=${ATTENTION_CUDA_DEVICES[i]} INSTANCE_RANK=$attention_rank bash examples/adrenaline/xp1d/attention_instance_start.sh $i 2>&1 | tee attention_${i}.out" ENTER
    done
}

function wait_adrenaline_instances_running() {
    local prefill_instances="${PREFILL_INSTANCES:-1}"
    local starttime
    starttime=$(date +%s)
    local passtime

    # Build process list dynamically
    local processes=()
    for ((i=0; i<prefill_instances; i++)); do
        processes+=("prefill_instance_start.sh $i")
    done
    for ((i=0; i<prefill_instances; i++)); do
        processes+=("attention_instance_start.sh $i")
    done
    processes+=("decode_instance_start")

    # Build flags array
    local flags=()
    for ((i=0; i<${#processes[@]}; i++)); do
        flags+=(false)
    done

    while [[ ${flags[*]} =~ false ]]; do
        passtime=$(($(date +%s) - "$starttime"))
        for ident in "${!processes[@]}"; do
            if check_process_running "${processes[$ident]}"; then
                flags[ident]=true
            fi
        done
        echo "Some processes are not running (${flags[*]}, ${passtime}s passed). Waiting..."
        sleep 1
    done
    echo "All adrenaline processes are running."
}

function wait_adrenaline_instances_ready() {
    local prefill_instances="${PREFILL_INSTANCES:-1}"
    local starttime
    starttime=$(date +%s)
    local timeout=300

    # Build port list dynamically
    local ports=()
    for ((i=0; i<prefill_instances; i++)); do
        ports+=($((8100 + i * 100)))
    done
    ports+=($((8100 + prefill_instances * 100)))  # decode port
    for ((i=0; i<prefill_instances; i++)); do
        ports+=($((8100 + (prefill_instances + 1 + i) * 100)))  # attention ports
    done

    # Build process list dynamically
    local processes=()
    for ((i=0; i<prefill_instances; i++)); do
        processes+=("prefill_instance_start.sh $i")
    done
    processes+=("decode_instance_start")
    for ((i=0; i<prefill_instances; i++)); do
        processes+=("attention_instance_start.sh $i")
    done

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
        for process in "${processes[@]}"; do
            if ! check_process_running "$process"; then
                echo "$process is not running."
                return 1
            fi
        done
        echo "Some instances are not serving (${passtime}s passed with timeout ${timeout}s). Waiting..."
    done
}

function wait_adrenaline_instances_fin() {
    local prefill_instances="${PREFILL_INSTANCES:-1}"
    local starttime
    starttime=$(date +%s)
    local timeout=120
    local passtime
    passtime=$(($(date +%s) - starttime))

    # Build process list dynamically
    local processes=()
    for ((i=0; i<prefill_instances; i++)); do
        processes+=("prefill_instance_start.sh $i")
    done
    for ((i=0; i<prefill_instances; i++)); do
        processes+=("attention_instance_start.sh $i")
    done
    processes+=("decode_instance_start")

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
    local prefill_instances="${PREFILL_INSTANCES:-1}"

    if [ ! -f decode.out ]; then
        echo "decode.out is not found. Please run the decode instance first."
        exit 1
    fi
    local num_decode_blocks
    num_decode_blocks=$(grep "GPU blocks" decode.out | sed "s/.*GPU blocks: //g" | sed "s/,.*//g")
    local num_attention_blocks=${NUM_ATTENTION_BLOCKS}

    # Build endpoint list dynamically
    local endpoints=""
    for ((i=0; i<prefill_instances; i++)); do
        endpoints+="localhost:$((8100 + i * 100)) "
    done
    endpoints+="localhost:$((8100 + prefill_instances * 100)) "  # decode endpoint
    for ((i=0; i<prefill_instances; i++)); do
        endpoints+="localhost:$((8100 + (prefill_instances + 1 + i) * 100)) "  # attention endpoints
    done

    NUM_DECODE_BLOCKS=${num_decode_blocks} \
        NUM_ATTENTION_BLOCKS=${num_attention_blocks} \
        ENDPOINTS="$endpoints" \
        bash examples/adrenaline/xp1d/proxy_instance_start.sh |
        tee proxy.out &
}

function start_adrenaline_proxy_in_tmux() {
    local prefill_instances="${PREFILL_INSTANCES:-1}"

    if [ ! -f decode.out ]; then
        echo "decode.out is not found. Please run the decode instance first."
        exit 1
    fi
    local num_decode_blocks
    num_decode_blocks=$(grep "GPU blocks" decode.out | sed "s/.*GPU blocks: //g" | sed "s/,.*//g")
    local num_attention_blocks=${NUM_ATTENTION_BLOCKS}

    # Build endpoint list dynamically
    local endpoints=""
    for ((i=0; i<prefill_instances; i++)); do
        endpoints+="localhost:$((8100 + i * 100)) "
    done
    endpoints+="localhost:$((8100 + prefill_instances * 100)) "  # decode endpoint
    for ((i=0; i<prefill_instances; i++)); do
        endpoints+="localhost:$((8100 + (prefill_instances + 1 + i) * 100)) "  # attention endpoints
    done

    if [[ -n "${MODEL}" ]]; then
        tmux send-key -t adrenaline:proxy "MODEL=${MODEL} "
    fi

    if [[ -n "${OFFLOAD_BACKEND}" ]]; then
        tmux send-key -t adrenaline:proxy "OFFLOAD_BACKEND=${OFFLOAD_BACKEND} "
    fi

    if [[ -n "${OFFLOAD_RATIO}" ]]; then
        tmux send-key -t adrenaline:proxy "OFFLOAD_RATIO=${OFFLOAD_RATIO} "
    fi

    if [[ -n "${ATTENTION_SM_PCT}" ]]; then
        tmux send-key -t "adrenaline:proxy" "ATTENTION_SM_PCT=$ATTENTION_SM_PCT "
    fi

    tmux send-key -t adrenaline:proxy "PREFILL_INSTANCES=${PREFILL_INSTANCES} "
    tmux send-key -t adrenaline:proxy \
        "NUM_DECODE_BLOCKS=${num_decode_blocks} \
        NUM_ATTENTION_BLOCKS=${num_attention_blocks} \
        ENDPOINTS=\"$endpoints\" \
        bash examples/adrenaline/xp1d/proxy_instance_start.sh \
        | tee proxy.out" ENTER
}

function kill_adrenaline_proxy_server() {
    pkill -f "adrenaline_proxy_server"
}

function wait_adrenaline_proxy_ready() {
    local starttime
    starttime=$(date +%s)
    local timeout=60
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

function start_adrenaline_environment() {
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
