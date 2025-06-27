#!/bin/bash
# serve two example requests
output1=$(curl -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Llama-2-7b-hf",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}')

output2=$(curl -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Llama-2-7b-hf",
"prompt": "Santa Clara is a",
"max_tokens": 10,
"temperature": 0
}')

# Print the outputs of the curl requests
echo ""
echo "Output of first request: $output1"
echo "Output of second request: $output2"

echo "Successfully finished 2 test requests!"
echo ""

# Cleanup commands, suppressing their output
# ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9 > /dev/null 2>&1
# pkill -f python3 > /dev/null 2>&1
