#!/bin/bash

DATASET_DIR=${DATASET_DIR:=evaluation/datasets}
DATASET_NAME=${DATASET_NAME:=sharegpt}
REQ_RATE=${REQ_RATE:=4}
NUM_REQS=${NUM_REQS:=200}
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_TYPE=${RUN_TYPE:="."}

MODEL=${MODEL:=meta-llama/Llama-2-7b-hf}
CONFIGS="${DATASET_NAME}-qps_${REQ_RATE}-numreqs_${NUM_REQS}"
DEFAULT_DIR="outputs/e2e_perf/${RUN_TYPE}/${MODEL}/${CONFIGS}-${TIME}"
RESULT_DIR=${RESULT_DIR:=${DEFAULT_DIR}}
mkdir -p "${RESULT_DIR}"
DECODE_FILE=${DECODE_FILE:="decode.out"}

python -m benchmarks.benchmark_serving \
    --backend vllm \
    --model "${MODEL}" \
    --dataset-name custom-dataset \
    --dataset-path "${DATASET_DIR}/${DATASET_NAME}.ds" \
    --request-rate "${REQ_RATE}" \
    --save-result \
    --result-dir "${RESULT_DIR}" \
    --ignore-eos \
    --stable_throughput \
    --num-prompts "${NUM_REQS}" | tee "${RESULT_DIR}/output.txt"

if [[ $? -eq 0 ]]; then
    num_fin=$(grep "Successful requests" "${RESULT_DIR}/output.txt" | awk '{print $3}')
    if [[ "${num_fin}" -ne "${NUM_REQS}" ]]; then
        echo "Benchmarking failed: ${num_fin} out of ${NUM_REQS} requests finished."
        exit 1
    fi
    echo "Benchmarking finished successfully (store results in ${RESULT_DIR})."
    cp ./*.out "${RESULT_DIR}"/
else
    echo "Benchmarking failed."
    exit 1
fi
