#!/bin/bash
set -e

function serve_requests() {
    for i in {1..10}; do
        curl -s http://localhost:8000/v1/completions \
            -H "Content-Type: application/json" \
            -d '{
            "model": "meta-llama/Llama-2-7b-hf",
            "prompt": "San Francisco is a",
            "max_tokens": 10,
            "stream": true,
            "temperature": 0
            }' > /tmp/adrenaline_req_"$i".out &
    done
    wait
    for i in {1..10}; do
        cat "/tmp/adrenaline_req_$i.out"
    done
}

serve_requests
