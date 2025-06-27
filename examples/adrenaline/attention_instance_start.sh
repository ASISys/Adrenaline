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
NUM_BLOCKS=${NUM_BLOCKS:=5632}
LOAD_FORMAT=${LOAD_FORMAT:=auto}

# attention instance, which is the KV producer
GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_ATTENTION_BACKEND=ADRENALINE_FLASHINFER \
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_ENABLE_KV_OFFLOAD=1 \
VLLM_OFFLOAD_RATIO="${OFFLOAD_RATIO}" \
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=30 \
VLLM_DISTRIBUTED_KV_ROLE=offload \
CUDA_VISIBLE_DEVICES=0 \
python3 \
    -m vllm.entrypoints.openai.api_server \
    --disable-log-stats \
    --model "${MODEL}" \
    --load-format "${LOAD_FORMAT}" \
    --port 8300 \
    --max-model-len 4096 \
    --num-gpu-blocks-override "${NUM_BLOCKS}"
