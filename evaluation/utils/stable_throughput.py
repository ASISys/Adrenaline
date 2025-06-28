import argparse
import os
import statistics

from matplotlib import pyplot
from pathlib import Path
from typing import Optional

from evaluation.plots.dataset_performance.parse_performance_metrics import (
    get_avg_throughput,
    get_stable_range,
    get_first_index_by_cond,
)
from evaluation.utils.output_parser import (
    parse_output_tokens,
    parse_num_swaps,
    parse_step_time,
    parse_local_tokens,
    parse_attn_tokens,
    parse_avg_seq_len,
)


def get_avg_seq_lens(step_lines: list[str]):
    stable_avg_seq_lens = [parse_avg_seq_len(line) for line in step_lines]
    mean_avg_seq_lens = statistics.fmean(stable_avg_seq_lens)
    return mean_avg_seq_lens


def plot_output_tokens(
    step_lines: list[str],
    stable_range: Optional[tuple[int, int]] = None,
    last_req_idx: Optional[int] = None,
    output_path: str = "output_tokens.pdf",
):
    output_tokens = [parse_output_tokens(line) for line in step_lines]
    pyplot.plot(output_tokens)
    if stable_range is not None:
        start_idx, end_idx = stable_range
        pyplot.vlines(start_idx, 0, max(output_tokens), colors="r", linestyles="dashed", label="stable range")
        pyplot.vlines(end_idx, 0, max(output_tokens), colors="r", linestyles="dashed")
    if last_req_idx is not None:
        pyplot.vlines(last_req_idx, 0, max(output_tokens), colors="g", linestyles="dashed", label="last req added")
    pyplot.legend()
    path = Path(output_path)
    os.makedirs(path.parent, exist_ok=True)
    pyplot.savefig(output_path)


def plot_step_times(
    step_lines: list[str],
    stable_range: tuple[int, int],
    output_path: str = "step_times.pdf",
):
    step_times = [parse_step_time(line) for line in step_lines]
    pyplot.plot(step_times)
    start_idx, end_idx = stable_range
    pyplot.vlines(start_idx, 0, max(step_times), colors="r", linestyles="dashed")
    pyplot.vlines(end_idx, 0, max(step_times), colors="r", linestyles="dashed")
    if max(step_times) > 100:
        pyplot.ylim([0, 100])
    path = Path(output_path)
    os.makedirs(path.parent, exist_ok=True)
    pyplot.savefig(output_path)


def get_last_req_step_idx(lines: list[str]):
    last_req_line_idx = get_first_index_by_cond(
        lines, lambda line: "Added request cmpl" in line, rev=True
    )
    remain_lines = lines[:last_req_line_idx]
    step_lines = [line for line in remain_lines if "step_time" in line]
    return len(step_lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str)
    parser.add_argument("--decode-file", type=str, default="decode.out")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.result_dir
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    with open(f"{args.result_dir}/{args.decode_file}") as f:
        lines = f.readlines()
    step_lines = [line for line in lines if "step_time" in line]
    last_req_step_idx = get_last_req_step_idx(lines)

    start_idx, end_idx = get_stable_range(step_lines)
    print(f"{start_idx=}, {end_idx=}")
    assert start_idx < end_idx
    stable_step_lines = step_lines[start_idx:end_idx]

    plot_output_tokens(
        step_lines,
        stable_range=(start_idx, end_idx),
        last_req_idx=last_req_step_idx,
        output_path=f"{args.output_dir}/output_tokens_distribution.pdf",
    )
    pyplot.clf()
    plot_step_times(step_lines, (start_idx, end_idx), f"{args.output_dir}/step_times_distribution.pdf")
    pyplot.clf()

    avg_throughput = get_avg_throughput(stable_step_lines)
    output_tokens = [parse_output_tokens(line) for line in stable_step_lines]
    step_times = [parse_step_time(line) for line in stable_step_lines]
    avg_output_tokens = statistics.mean(output_tokens)
    avg_step_times = statistics.mean(step_times)
    print(f"{avg_throughput=:.2f}, {avg_output_tokens=:.2f}, {avg_step_times=:.2f}")

    num_swaps = [parse_num_swaps(line) for line in stable_step_lines]
    prev_num_swaps = [0] + num_swaps[: len(num_swaps) - 1]
    num_swap_steps = sum([1 if num_swap != prev_num_swap else 0 for num_swap, prev_num_swap in zip(num_swaps, prev_num_swaps)])
    num_steps = len(stable_step_lines)
    swap_pct = num_swap_steps / num_steps
    print(f"{num_swap_steps=}, {num_steps=}, {swap_pct=:.2f}")

    outlier_step_times = [step_time for step_time in step_times if step_time > 60]
    outlier_pct = len(outlier_step_times) / num_steps
    avg_outlier_time = statistics.fmean(outlier_step_times) if len(outlier_step_times) > 0 else 0
    print(f"{len(outlier_step_times)=}, {num_steps=}, {outlier_pct=}, {avg_outlier_time=}")

    local_tokens = [parse_local_tokens(line) for line in stable_step_lines]
    avg_local_tokens = statistics.mean(local_tokens)
    attn_tokens = [parse_attn_tokens(line) for line in stable_step_lines]
    avg_attn_tokens = statistics.mean(attn_tokens)
    print(f"{avg_local_tokens=:.2f}, {avg_attn_tokens=:.2f}")

    avg_seq_lens = get_avg_seq_lens(stable_step_lines)
    print(f"{avg_seq_lens=:.2f}")
