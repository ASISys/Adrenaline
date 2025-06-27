from typing import Callable

def get_first_index_by_cond(lst: list, cond: Callable[..., bool], rev: bool = False):
    iter = lst if not rev else reversed(lst)
    index = next((idx for idx, item in enumerate(iter) if cond(item)), None)
    if rev and index is not None:
        index = len(lst) - index - 1
    return index


def parse_list(list_str: str):
    return list_str.strip("[]").split(", ")


def parse_field(line: str, field: str):
    assert field in line, f"{field} not in {line}"
    start_idx = line.find(field)
    sub_line = line[start_idx:]
    end_idx = sub_line.find(",")
    if end_idx != -1:
        sub_line = sub_line[:end_idx]
    else:
        sub_line = sub_line.strip()
    return sub_line.split("=")[1]


def parse_output_tokens(line: str):
    return int(parse_field(line, "output_tokens").split("(")[0])


def parse_num_swaps(line: str):
    return int(parse_field(line, "len(scheduler.swapped)"))


def parse_step_time(line: str):
    return float(parse_field(line, "step_time"))


def parse_gpu_time(line: str):
    return float(parse_field(line, "gpu_time"))


def parse_local_tokens(line: str):
    return int(parse_field(line, "output_tokens").split("(")[1].split("+")[0])


def parse_attn_tokens(line: str):
    return int(parse_field(line, "output_tokens").split("(")[1].split("+")[1].split(")")[0])


def parse_avg_seq_len(line: str):
    return float(parse_field(line, "avg_seq_len"))


def parse_num_blocks(line: str):
    return int(parse_field(line, "num_blocks"))


def parse_timestamp(line: str):
    return float(parse_field(line, "timestamp"))


def parse_scheduled_request_ids(line: str):
    field = "scheduled_request_ids" 
    assert field in line
    start_idx = line.find(field) + len(field) + 1
    assert line[start_idx] == "["
    request_ids_list_str = line[start_idx:]
    end_idx = request_ids_list_str.find("]") + 1
    request_ids_list_str = request_ids_list_str[:end_idx]
    request_ids_str = parse_list(request_ids_list_str)
    request_ids = [request_id_str.strip("'") for request_id_str in request_ids_str]
    return request_ids


def parse_is_recv_fin(line: str):
    return bool(parse_field(line, "is_recv_fin") == "True")


def parse_step_start(line: str):
    return float(parse_field(line, "step_start_time"))


def parse_step_stop(line: str):
    return float(parse_field(line, "step_stop_time"))


def parse_field_by_whitespace(line: str, field: str):
    assert field in line, f"{field} not in {line}"
    value_str = line.split(field)[1].strip()
    return value_str


def parse_mean_ttft(line: str):
    return float(parse_field_by_whitespace(line, "Mean TTFT (ms):"))


def parse_mean_tpot(line: str):
    return float(parse_field_by_whitespace(line, "Mean TPOT (ms):"))


def parse_p99_tpot(line: str):
    return float(parse_field_by_whitespace(line, "P99 TPOT (ms):"))


def parse_request_id(line: str):
    return parse_field_by_whitespace(line, "Added request").strip(".")
