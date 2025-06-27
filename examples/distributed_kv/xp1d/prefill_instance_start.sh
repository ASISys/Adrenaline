#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch x vllm instances (x-1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

VLLM_HOST_IP=${VLLM_HOST_IP:=$(hostname -I | awk '{print $1}')}
export VLLM_HOST_IP
VLLM_PORT=${VLLM_PORT:=12345}
export VLLM_PORT

MODEL=${MODEL:=meta-llama/Llama-2-7b-hf}
LOAD_FORMAT=${LOAD_FORMAT:=auto}
MAX_MODEL_LEN=${MAX_MODEL_LEN:=4096}

if [ "$MAX_MODEL_LEN" -gt 8192 ]; then
    MAX_SEQ_LEN_TO_CAPTURE=$MAX_MODEL_LEN
else
    MAX_SEQ_LEN_TO_CAPTURE=8192
fi

PREFILL_INSTANCES=${PREFILL_INSTANCES:-1}
RANK=${1:-0}
PORT=${PORT:-8100}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

# Use assigned CUDA device if provided, otherwise fall back to RANK-based allocation
CUDA_DEVICE=${ASSIGNED_CUDA_DEVICE:-${RANK}}

echo "Starting prefill instance $RANK on CUDA devices $CUDA_DEVICE (tensor parallel size: $TENSOR_PARALLEL_SIZE)"

# prefilling instance, which is the KV producer
GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_ATTENTION_BACKEND=ADRENALINE_FLASHINFER \
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_DISTRIBUTED_KV_ROLE=producer \
VLLM_PREFILL_INSTANCES=${PREFILL_INSTANCES} \
VLLM_RANK_IN_INSTANCES=${RANK} \
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
python3 \
    -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --load-format "${LOAD_FORMAT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-seq-len-to-capture ${MAX_SEQ_LEN_TO_CAPTURE} \
    --enforce-eager \
    --port "$PORT" \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization 0.8
