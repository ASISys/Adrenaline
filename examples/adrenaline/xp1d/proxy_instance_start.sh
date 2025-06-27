#!/bin/bash
# This file demonstrates the example usage of configurable xP1D adrenaline setup
# This is the proxy server script that dynamically handles endpoints

VLLM_HOST_IP=${VLLM_HOST_IP:=$(hostname -I | awk '{print $1}')}
export VLLM_HOST_IP
VLLM_PORT=${VLLM_PORT:=12345}
export VLLM_PORT

MODEL=${MODEL:=meta-llama/Llama-2-7b-hf}
OFFLOAD_BACKEND=${OFFLOAD_BACKEND:=loadaware}
PREFILL_INSTANCES=${PREFILL_INSTANCES:-1}

# Build endpoints dynamically if not provided
if [[ -z "${ENDPOINTS}" ]]; then
    ENDPOINTS=""
    # Add prefill endpoints
    for ((i=0; i<PREFILL_INSTANCES; i++)); do
        ENDPOINTS+="localhost:$((8100 + i * 100)) "
    done
    # Add decode endpoint
    ENDPOINTS+="localhost:$((8100 + PREFILL_INSTANCES * 100)) "
    # Add attention endpoints
    for ((i=0; i<PREFILL_INSTANCES; i++)); do
        ENDPOINTS+="localhost:$((8100 + (PREFILL_INSTANCES + 1 + i) * 100)) "
    done
fi

# Build attention blocks array dynamically
ATTENTION_BLOCKS_ARRAY=""
for ((i=0; i<PREFILL_INSTANCES; i++)); do
    ATTENTION_BLOCKS_ARRAY+="${NUM_ATTENTION_BLOCKS} "
done

ATTENTION_SM_PCT=${ATTENTION_SM_PCT:-20}

flags=()
if [ -n "$OFFLOAD_RATIO" ]; then
    flags+=("--offload-ratio" "$OFFLOAD_RATIO")
fi

echo "Starting proxy server with endpoints: $ENDPOINTS"
echo "Attention blocks: $ATTENTION_BLOCKS_ARRAY"

# launch a proxy server that opens the service at port 8000
python -m adrenaline.entrypoints.adrenaline_proxy_server \
    --tokenizer "${MODEL}" \
    --end-points $ENDPOINTS \
    --num-decode-blocks "${NUM_DECODE_BLOCKS}" \
    --num-attention-blocks $ATTENTION_BLOCKS_ARRAY \
    --offload-backend "${OFFLOAD_BACKEND}" \
    --offload-sm-pct ${ATTENTION_SM_PCT} \
    "${flags[@]}"
