#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

VLLM_HOST_IP=${VLLM_HOST_IP:=$(hostname -I | awk '{print $1}')}
export VLLM_HOST_IP
VLLM_PORT=${VLLM_PORT:=12345}
export VLLM_PORT

MODEL=${MODEL:=meta-llama/Llama-2-7b-hf}

# decoding instance, which is the KV consumer
GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_ATTENTION_BACKEND=ADRENALINE_FLASHINFER \
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_DISTRIBUTED_KV_ROLE=consumer \
CUDA_VISIBLE_DEVICES=1 \
python3 \
    -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --load-format dummy \
    --port 8200 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --preemption-mode swap \
    --swap-space 40
