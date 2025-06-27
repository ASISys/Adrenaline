import argparse
import os
import statistics

from pathlib import Path
from matplotlib import pyplot
from benchmarks.dataset_utils import Dataset


def hist_data(data: list, bin_size: int = 100, output_path: str = "untitled.pdf"):
    min_data = min(data)
    max_data = max(data)
    min_bound = min_data - min_data % bin_size
    max_bound = max_data + bin_size - max_data % bin_size
    num_bins = (max_bound - min_bound) // bin_size
    pyplot.hist(data, bins=num_bins, range=(min_bound, max_bound))
    pyplot.savefig(output_path)
    pyplot.clf()


def hist_prompt_len_and_output_len(
    prompt_lens: list[int],
    output_lens: list[int],
    bin_size: int = 100,
    output_path: str = "prompt_and_output_len.pdf",
):
    min_data = min(min(prompt_lens), min(output_lens))
    max_data = max(max(prompt_lens), max(output_lens))
    min_bound = min_data - min_data % bin_size
    max_bound = max_data + bin_size - max_data % bin_size
    num_bins = (max_bound - min_bound) // bin_size
    pyplot.hist(prompt_lens, bins=num_bins, range=(min_bound, max_bound), label="prompt_len", alpha=0.5)
    pyplot.hist(output_lens, bins=num_bins, range=(min_bound, max_bound), label="output_len", alpha=0.5)
    pyplot.xlabel("# Tokens")
    pyplot.ylabel("Density")
    pyplot.legend()
    pyplot.savefig(output_path)
    pyplot.clf()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, default="evaluation/datasets/sharegpt.ds")
    parser.add_argument("--output-dir", type=str, default="evaluations/plots/prepare_datasets")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    dataset_path = args.dataset
    dataset = Dataset.load(dataset_path)

    prompt_lens = [req.prompt_len for req in dataset.reqs]
    output_lens = [req.output_len for req in dataset.reqs]
    total_lens = [req.prompt_len + req.output_len for req in dataset.reqs]

    os.makedirs(args.output_dir, exist_ok=True)
    hist_prompt_len_and_output_len(
        prompt_lens, output_lens, output_path=f"{args.output_dir}/prompt_and_output_len.pdf"
    )
    hist_data(total_lens, output_path=f"{args.output_dir}/total_len_hist.pdf")

    dataset_name = Path(dataset_path).stem
    print(f"{dataset_name=}")
    avg_prompt_len = statistics.mean(prompt_lens)
    avg_output_len = statistics.mean(output_lens)
    num_reqs = len(dataset.reqs)
    print(f"{avg_prompt_len=:.2f}, {avg_output_len=:.2f}, {num_reqs=}")
