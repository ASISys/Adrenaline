#!/bin/bash

INPUT_LEN=${INPUT_LEN:=100}
OUTPUT_LEN=${OUTPUT_LEN:=200}
RNG_RATIO=${RNG_RATIO:=0.5}
REQ_RATE=${REQ_RATE:=1}
NUM_REQS=${NUM_REQS:=100}
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_TYPE=${RUN_TYPE:="."}

MODEL=${MODEL:=meta-llama/Llama-2-7b-hf}
CONFIGS="in_${INPUT_LEN}-out_${OUTPUT_LEN}-qps_${REQ_RATE}-numreqs_${NUM_REQS}-rngratio_${RNG_RATIO}"
if [[ -n "${OFFLOAD_RATIO}" ]]; then
    RUN_TYPE="${RUN_TYPE}/ratio_${OFFLOAD_RATIO}"
fi
DEFAULT_DIR="outputs/e2e_perf/${RUN_TYPE}/${CONFIGS}-${TIME}"
RESULT_DIR=${RESULT_DIR:=${DEFAULT_DIR}}
mkdir -p "${RESULT_DIR}"

python -m benchmarks.benchmark_serving \
    --backend vllm \
    --model "${MODEL}" \
    --dataset-name random \
    --request-rate "${REQ_RATE}" \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --random-range-ratio "${RNG_RATIO}" \
    --save-result \
    --result-dir "${RESULT_DIR}" \
    --ignore-eos \
    --num-prompts "${NUM_REQS}" | tee "${RESULT_DIR}/output.txt"

if [[ $? -eq 0 ]]; then
    echo "Benchmarking finished successfully (store results in ${RESULT_DIR})."
    cp ./*.out "${RESULT_DIR}"/
else
    echo "Benchmarking failed."
    exit 1
fi
