#!/bin/bash
# This file demonstrates the example usage of configurable xP1D adrenaline setup
# This is the decode instance script

VLLM_HOST_IP=${VLLM_HOST_IP:=$(hostname -I | awk '{print $1}')}
export VLLM_HOST_IP
VLLM_PORT=${VLLM_PORT:=12345}
export VLLM_PORT

OFFLOAD_RATIO=${OFFLOAD_RATIO:=0.6}
OFFLOAD_MAX_CAPTURE_BATCH=${OFFLOAD_MAX_CAPTURE_BATCH:=56}
MODEL=${MODEL:=meta-llama/Llama-2-7b-hf}
LOAD_FORMAT=${LOAD_FORMAT:=auto}
MAX_MODEL_LEN=${MAX_MODEL_LEN:=4096}

# Get instance parameters
PREFILL_INSTANCES=${PREFILL_INSTANCES:-1}
INSTANCE_RANK=${INSTANCE_RANK:-$PREFILL_INSTANCES}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

# Use assigned CUDA device if provided, otherwise fall back to prefill instances count
CUDA_DEVICE=${ASSIGNED_CUDA_DEVICE:-${PREFILL_INSTANCES}}

# Calculate port dynamically based on prefill instances
PORT=$((8100 + PREFILL_INSTANCES * 100))

if [ "$MAX_MODEL_LEN" -gt 8192 ]; then
    MAX_SEQ_LEN_TO_CAPTURE=$MAX_MODEL_LEN
else
    MAX_SEQ_LEN_TO_CAPTURE=8192
fi

echo "Starting decode instance (rank $INSTANCE_RANK) on CUDA devices $CUDA_DEVICE, port $PORT (tensor parallel size: $TENSOR_PARALLEL_SIZE)"

# decoding instance, which is the KV consumer
GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_ATTENTION_BACKEND=ADRENALINE_FLASHINFER \
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_ENABLE_KV_OFFLOAD=1 \
VLLM_OFFLOAD_RATIO="${OFFLOAD_RATIO}" \
VLLM_OFFLOAD_MAX_CAPTURE_BATCH="${OFFLOAD_MAX_CAPTURE_BATCH}" \
VLLM_DISTRIBUTED_KV_ROLE=consumer \
VLLM_PREFILL_INSTANCES=${PREFILL_INSTANCES} \
VLLM_RANK_IN_INSTANCES=${INSTANCE_RANK} \
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
python3 \
    -m vllm.entrypoints.openai.api_server \
    --disable-log-stats \
    --model "${MODEL}" \
    --load-format "${LOAD_FORMAT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-seq-len-to-capture ${MAX_SEQ_LEN_TO_CAPTURE} \
    --port ${PORT} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --preemption-mode swap \
    --swap-space 10 \
    --gpu-memory-utilization 0.8
