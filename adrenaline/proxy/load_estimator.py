import csv

from pathlib import Path
from vllm.engine.arg_utils import EngineArgs

from adrenaline.proxy.storage_manager import StorageManager
from adrenaline.proxy.utils import RequestRuntimeData
from adrenaline.utils.utils import get_token_size


def get_bandwidth_by_sm(offload_sm_pct: int, model_name: str):
    model_name = Path(model_name).stem
    bandwidth_profiler_path = Path(f"adrenaline/utils/attention_bandwidth/{model_name}.csv")
    assert bandwidth_profiler_path.exists()
    with open(bandwidth_profiler_path) as f:
        reader = csv.DictReader(f)
        bandwidths = {int(row["sm_pct"]): float(row["bandwidth"]) for row in reader}
    offload_bandwidth = bandwidths[offload_sm_pct]
    decode_bandwidth = bandwidths[100]
    return offload_bandwidth, decode_bandwidth


def get_profile_gemm_time():
    gemm_profiler_path = Path("adrenaline/utils/gemm_run_time.csv")
    if not gemm_profiler_path.exists():
        return None
    with open(gemm_profiler_path) as f:
        reader = csv.DictReader(f)
        gemm_time = {int(row["batch_size"]): float(row["gemm_time"]) for row in reader}
    return gemm_time


def get_delta_estimator():
    misc_profiler_path = Path("adrenaline/utils/misc_time.csv")
    if not misc_profiler_path.exists():
        return None
    with open(misc_profiler_path) as f:
        reader = csv.DictReader(f)
        row = reader.__next__()
    delta_coef = float(row["delta_coef"])
    delta_intercept = float(row["delta_intercept"])
    def estimator(batch_size: int):
        return delta_coef * batch_size + delta_intercept
    return estimator


def get_transfer_bandwidth():
    misc_profiler_path = Path("adrenaline/utils/misc_time.csv")
    if not misc_profiler_path.exists():
        return None
    with open(misc_profiler_path) as f:
        reader = csv.DictReader(f)
        row = reader.__next__()
    if "transfer_bandiwdth" not in row:
        return None
    transfer_bandwidth = float(row["transfer_bandwidth"])
    return transfer_bandwidth


class LoadEstimator:

    def __init__(
        self,
        storage_manager: StorageManager,
        offload_sm: int,
        model_name: str,
        tensor_parallel_size: int = 1,
    ):
        engine_args = EngineArgs(
            model=model_name, tensor_parallel_size=tensor_parallel_size
        )
        config = engine_args.create_engine_config()
        self.storage_manager = storage_manager
        self.token_size = get_token_size(
            config.model_config, config.parallel_config, config.model_config.dtype
        )

        self.offload_bandwidth, self.decode_bandwidth = get_bandwidth_by_sm(offload_sm, model_name)  # GB/s
        self.gemm_time = get_profile_gemm_time()
        self.delta_estimator = get_delta_estimator()
        self.transfer_bandwidth = get_transfer_bandwidth()

    def estimate_offload_attn_time(self, req_data: RequestRuntimeData, attn_rank: int):
        attn_loc = self.storage_manager.get_attn_location(attn_rank)
        assert attn_loc >= self.storage_manager.num_decode_ranks
        req_need_max_blocks = self.storage_manager._calc_num_blocks(
            req_data.max_seq_len
        )
        if req_need_max_blocks > self.storage_manager.num_total_blocks[attn_loc]:
            return float("inf")
        req_need_blocks = self.storage_manager._calc_num_blocks(req_data.seq_len)
        num_blocks = (
            self.storage_manager.num_used_blocks[attn_loc] + req_need_blocks
        )
        attn_size = num_blocks * self.storage_manager.block_size * self.token_size  # MB
        offload_attn_time = attn_size / self.offload_bandwidth  # ms
        return offload_attn_time

    def estimate_decode_attn_time(self, req_data: RequestRuntimeData, decode_rank: int):
        decode_loc = self.storage_manager.get_decode_location(decode_rank)
        req_need_max_blocks = self.storage_manager._calc_num_blocks(
            req_data.max_seq_len
        )
        if req_need_max_blocks > self.storage_manager.num_total_blocks[decode_loc]:
            return float("inf")
        req_need_blocks = self.storage_manager._calc_num_blocks(req_data.seq_len)
        num_blocks = (
            self.storage_manager.num_used_blocks[decode_loc] + req_need_blocks
        )
        attn_size = num_blocks * self.storage_manager.block_size * self.token_size  # MB
        decode_attn_time = attn_size / self.decode_bandwidth  # ms
        return decode_attn_time

    def estimate_decode_time(
        self, req_data: str, is_offload: bool, attn_rank: int, decode_rank: int
    ):
        if is_offload:
            attn_time = self.estimate_offload_attn_time(req_data, attn_rank)
        else:
            attn_time = self.estimate_decode_attn_time(req_data, decode_rank)
        attn_loc = self.storage_manager.get_attn_location(attn_rank)
        decode_loc = self.storage_manager.get_decode_location(decode_rank)
        decode_batch = self.storage_manager.num_requests[decode_loc]
        attn_batch = self.storage_manager.num_requests[attn_loc]
        batch_size = decode_batch + attn_batch + 1
        assert batch_size > 0
        gemm_time = self.gemm_time[batch_size]
        delta_time = self.delta_estimator(batch_size)
        decode_time = gemm_time + attn_time + delta_time
        return decode_time
