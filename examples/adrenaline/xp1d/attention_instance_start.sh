#!/bin/bash
# This file demonstrates the example usage of configurable xP1D adrenaline setup
# This is a template script for attention instances that can be used for any number of attention instances

VLLM_HOST_IP=${VLLM_HOST_IP:=$(hostname -I | awk '{print $1}')}
export VLLM_HOST_IP
VLLM_PORT=${VLLM_PORT:=12345}
export VLLM_PORT

OFFLOAD_RATIO=${OFFLOAD_RATIO:=0.6}
OFFLOAD_MAX_CAPTURE_BATCH=${OFFLOAD_MAX_CAPTURE_BATCH:=56}
MODEL=${MODEL:=meta-llama/Llama-2-7b-hf}
NUM_BLOCKS=${NUM_BLOCKS:=5000}
LOAD_FORMAT=${LOAD_FORMAT:=auto}
MAX_MODEL_LEN=${MAX_MODEL_LEN:=4096}

# Get instance parameters
PREFILL_INSTANCES=${PREFILL_INSTANCES:-1}
INSTANCE_INDEX=${1:-0}
INSTANCE_RANK=${INSTANCE_RANK:-$((PREFILL_INSTANCES + 1 + INSTANCE_INDEX))}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
SM_PCT=${SM_PCT:-20}

# Use assigned CUDA device if provided, otherwise fall back to instance index
CUDA_DEVICE=${ASSIGNED_CUDA_DEVICE:-${INSTANCE_INDEX}}

# Calculate port dynamically based on instance rank
PORT=$((8100 + INSTANCE_RANK * 100))

if [ "$MAX_MODEL_LEN" -gt 8192 ]; then
    MAX_SEQ_LEN_TO_CAPTURE=$MAX_MODEL_LEN
else
    MAX_SEQ_LEN_TO_CAPTURE=8192
fi

echo "Starting attention instance $INSTANCE_INDEX (rank $INSTANCE_RANK) on CUDA devices $CUDA_DEVICE, port $PORT (tensor parallel size: $TENSOR_PARALLEL_SIZE)"

# attention instance, which is the KV offload handler
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_ATTENTION_BACKEND=ADRENALINE_FLASHINFER \
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_ENABLE_KV_OFFLOAD=1 \
VLLM_OFFLOAD_RATIO="${OFFLOAD_RATIO}" \
VLLM_OFFLOAD_MAX_CAPTURE_BATCH="${OFFLOAD_MAX_CAPTURE_BATCH}" \
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
VLLM_DISTRIBUTED_KV_ROLE=offload \
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
    --num-gpu-blocks-override "${NUM_BLOCKS}"
