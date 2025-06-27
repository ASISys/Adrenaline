#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

VLLM_HOST_IP=${VLLM_HOST_IP:=$(hostname -I | awk '{print $1}')}
export VLLM_HOST_IP
VLLM_PORT=${VLLM_PORT:=12345}
export VLLM_PORT

OFFLOAD_RATIO=${OFFLOAD_RATIO:=0.6}
MODEL=${MODEL:=meta-llama/Llama-2-7b-hf}
LOAD_FORMAT=${LOAD_FORMAT:=auto}

# decoding instance, which is the KV consumer
GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_ATTENTION_BACKEND=ADRENALINE_FLASHINFER \
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_ENABLE_KV_OFFLOAD=1 \
VLLM_OFFLOAD_RATIO="${OFFLOAD_RATIO}" \
VLLM_DISTRIBUTED_KV_ROLE=consumer \
CUDA_VISIBLE_DEVICES=1 \
python3 \
    -m vllm.entrypoints.openai.api_server \
    --disable-log-stats \
    --model "${MODEL}" \
    --load-format "${LOAD_FORMAT}" \
    --port 8200 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --preemption-mode swap \
    --swap-space 40
