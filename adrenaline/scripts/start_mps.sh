#!/bin/bash
set -e

# Default CUDA_MPS_LOG_DIRECTORY is set to /var/log/nvidia-mps.
# User can override the default value if not have sufficient permission.
if [ -n "$CUDA_MPS_LOG_DIRECTORY" ]; then
    mkdir -p "$CUDA_MPS_LOG_DIRECTORY"
fi
if [[ $EUID -eq 0 ]]; then
    nvidia-smi -c EXCLUSIVE_PROCESS
fi
nvidia-cuda-mps-control -d
pgrep -a -f "mps"
