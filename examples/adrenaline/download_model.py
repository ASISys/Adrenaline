import argparse

from vllm import LLM
from vllm.config import LoadFormat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-weight", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    download_weight: bool = args.download_weight
    if download_weight:
        llm = LLM(model="meta-llama/Llama-2-7b-hf")
    else:
        llm = LLM(model="meta-llama/Llama-2-7b-hf", load_format=LoadFormat.DUMMY)
