#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

# install quart first -- required for disagg prefill proxy serve
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi 

flags=()
if [ -n "$HOST" ]; then
    flags+=("--host" "$HOST")
fi
if [ -n "$ENDPOINTS" ]; then
    # shellcheck disable=SC2206
    flags+=("--end-points" $ENDPOINTS)
fi
# launch a proxy server that opens the service at port 8000
# the workflow of this proxy:
# - send the request to prefill vLLM instance (port 8100), change max_tokens to 1
# - after the prefill vLLM finishes prefill, send the request to decode vLLM instance
python -m benchmarks.disagg_benchmarks.disagg_prefill_proxy_server "${flags[@]}"
