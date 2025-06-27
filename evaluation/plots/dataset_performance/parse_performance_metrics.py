from evaluation.utils.output_parser import (
    get_first_index_by_cond,
    parse_output_tokens,
    parse_num_swaps,
    parse_step_time,
    parse_mean_ttft,
    parse_mean_tpot,
    parse_p99_tpot,
)

def get_stable_range_by_batch_pct(step_lines: list[str], pct: float = 0.8):
    output_tokens = [parse_output_tokens(line) for line in step_lines]
    max_output_token = max(output_tokens)
    start_idx = get_first_index_by_cond(
        output_tokens, lambda output_token: output_token >= int(max_output_token * pct)
    )
    end_idx = get_first_index_by_cond(
        output_tokens,
        lambda output_token: output_token >= int(max_output_token * pct),
        rev=True,
    )
    return start_idx, end_idx


def get_stable_range_by_swap(step_lines: list[str]):
    start_idx = get_first_index_by_cond(
        step_lines, lambda line: parse_num_swaps(line) > 0
    )
    end_idx = get_first_index_by_cond(
        step_lines, lambda line: parse_num_swaps(line) > 0, rev=True
    )
    return start_idx, end_idx


def get_stable_range(step_lines: list[str]):
    swap_start_idx, swap_end_idx = get_stable_range_by_swap(step_lines)
    batch_pct_start_idx, batch_pct_end_idx = get_stable_range_by_batch_pct(step_lines)
    if swap_start_idx is None or swap_end_idx is None:
        start_idx, end_idx = batch_pct_start_idx, batch_pct_end_idx
    else:
        start_idx = min(swap_start_idx, batch_pct_start_idx)
        end_idx = max(swap_end_idx, batch_pct_end_idx)
    assert start_idx is not None and end_idx is not None
    assert start_idx < end_idx
    return start_idx, end_idx


def get_avg_throughput(step_lines: list[str]):
    output_tokens = [parse_output_tokens(line) for line in step_lines]
    step_times = [parse_step_time(line) for line in step_lines]
    avg_throughput = sum(output_tokens) / sum(step_times)
    return avg_throughput


def get_stable_throughput(filepath: str):
    with open(filepath) as f:
        lines = f.readlines()
    step_lines = [line for line in lines if "step_time" in line]
    assert len(step_lines) > 0, f"{filepath}"
    start_idx, end_idx = get_stable_range(step_lines)
    stable_step_lines = step_lines[start_idx:end_idx]
    stable_throughput = get_avg_throughput(stable_step_lines)
    return stable_throughput


def get_ttft_for_path(filepath: str):
    with open(filepath) as f:
        ttft_lines = [line for line in f.readlines() if "TTFT" in line]
    assert len(ttft_lines) > 0, f"{filepath}"
    mean_ttft = parse_mean_ttft(ttft_lines[0])
    return mean_ttft


def get_tpot_for_path(filepath: str):
    with open(filepath) as f:
        tpot_lines = [line for line in f.readlines() if "TPOT" in line]
    assert len(tpot_lines) > 0, f"{filepath}"
    mean_tpot = parse_mean_tpot(tpot_lines[0])
    return mean_tpot

def get_p99_tpot_for_path(filepath: str):
    with open(filepath) as f:
        tpot_lines = [line for line in f.readlines() if "TPOT" in line]
    assert len(tpot_lines) > 0, f"{filepath}"
    p99_tpot = parse_p99_tpot(tpot_lines[2])
    return p99_tpot

def get_thpt_for_path(filepath: str):
    return get_stable_throughput(filepath)
