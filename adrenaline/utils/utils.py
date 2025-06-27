import csv
import torch
import os

from pathlib import Path
from vllm.logger import init_logger
from vllm.config import ModelConfig, ParallelConfig

logger = init_logger("vllm.utils.utils")


def get_token_size(model_config: ModelConfig, parallel_config: ParallelConfig, kv_dtype: torch.dtype):
    num_layers = model_config.get_num_layers(parallel_config)
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    head_dim = model_config.get_head_size()
    token_size = num_layers * 2 * num_kv_heads * head_dim * kv_dtype.itemsize  # bytes
    token_size /= 1024 * 1024  # convert to MB
    return token_size


def get_offload_ratio_by_profile(profile_path: Path, attn_sm_pct: int = 30):
    with open(profile_path) as f:
        reader = csv.DictReader(f)
        bandwidths = {int(row["sm_pct"]): float(row["bandwidth"]) for row in reader}
    offload_ratio = bandwidths[attn_sm_pct] / bandwidths[100]
    return offload_ratio


def get_offload_ratio(model_name: str):
    model_name = Path(model_name).stem
    bandwidth_profile_csv = Path(f"adrenaline/utils/attention_bandwidth/{model_name}.csv")
    if bandwidth_profile_csv.exists():
        return get_offload_ratio_by_profile(bandwidth_profile_csv)
    logger.warning(f"Can't find bandwidth profile result in {bandwidth_profile_csv}")
    assert "VLLM_OFFLOAD_RATIO" in os.environ
    return float(os.environ["VLLM_OFFLOAD_RATIO"])
