#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch (x+1) vllm instances (x for prefill and 1 for decode),
# and then transfer the KV cache between them.

VLLM_HOST_IP=${VLLM_HOST_IP:=$(hostname -I | awk '{print $1}')}
export VLLM_HOST_IP
VLLM_PORT=${VLLM_PORT:=12345}
export VLLM_PORT

# install quart first -- required for disagg prefill proxy serve
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi

# Generate endpoints list
endpoints=()
for ((i=0; i<PREFILL_INSTANCES; i++)); do
    endpoints+=("localhost:$((8100 + i * 100))")
done
endpoints+=("localhost:$((8100 + PREFILL_INSTANCES * 100))")

# launch a proxy server that opens the service at port 8000
# the workflow of this proxy:
# - send the request to prefill vLLM instances, change max_tokens to 1
# - after the prefill vLLM finishes prefill, send the request to decode vLLM instance
python -m benchmarks.disagg_benchmarks.disagg_prefill_proxy_server \
    --end-points "${endpoints[@]}" \
    --num-prefills ${PREFILL_INSTANCES} \
    --num-decodes 1
