#!/bin/bash
set -e

echo quit | nvidia-cuda-mps-control
nvidia-smi -c DEFAULT
pgrep -a -f "mps"
