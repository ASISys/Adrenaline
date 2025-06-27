#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi 

OFFLOAD_RATIO=${OFFLOAD_RATIO:=0.6}
MODEL=${MODEL:=meta-llama/Llama-2-7b-hf}
OFFLOAD_BACKEND=${OFFLOAD_BACKEND:=loadaware}

flags=()
if [ -n "$HOST" ]; then
    flags+=("--host" "$HOST")
fi
if [ -n "$ENDPOINTS" ]; then
    # shellcheck disable=SC2206
    flags+=("--end-points" $ENDPOINTS)
fi

# launch a proxy server that opens the service at port 8000
python -m adrenaline.entrypoints.adrenaline_proxy_server \
    --tokenizer "${MODEL}" \
    --offload-backend "${OFFLOAD_BACKEND}" \
    --offload-ratio "${OFFLOAD_RATIO}" \
    --num-decode-blocks "${NUM_DECODE_BLOCKS}" \
    --num-attention-blocks "${NUM_ATTENTION_BLOCKS}" "${flags[@]}"
