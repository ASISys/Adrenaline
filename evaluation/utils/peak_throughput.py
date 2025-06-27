import argparse
import statistics

from collections import Counter
from evaluation.utils.output_parser import (
    parse_output_tokens,
    parse_step_time,
)


def sample_middle_range(lst, percent=0.1):
    sorted_lst = sorted(lst)
    return sorted_lst[
        int(len(sorted_lst) * percent) : int(len(sorted_lst) * (1 - percent))
    ]


def get_peak_batch_and_time_from_log(path: str):
    with open(path) as f:
        lines = f.readlines()
    step_lines = [line for line in lines if "step_time" in line]
    output_tokens = [parse_output_tokens(line) for line in step_lines]
    output_token_counter = Counter(output_tokens)
    sorted_output_tokens = sorted(output_token_counter.keys(), reverse=True)
    output_token, mean_step_time = None, None
    for output_token in sorted_output_tokens:
        if output_token_counter[output_token] < 10:
            continue
        lines_with_output_token = [
            line for line in step_lines if f"output_tokens={output_token}" in line
        ]
        step_times = [parse_step_time(line) for line in lines_with_output_token]
        mean_step_time = statistics.fmean(sample_middle_range(step_times))
        break

    assert output_token is not None
    assert mean_step_time is not None
    return output_token, mean_step_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str)
    parser.add_argument("--decode-file", type=str, default="decode.out")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    output_token, step_time = get_peak_batch_and_time_from_log(
        f"{args.result_dir}/{args.decode_file}"
    )
    throughput = output_token / step_time
    print(f"{throughput=}, {output_token=}, {step_time=}")
