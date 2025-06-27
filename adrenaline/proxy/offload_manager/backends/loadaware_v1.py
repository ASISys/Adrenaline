from typing import Optional
from vllm.logger import init_logger

from adrenaline.proxy.offload_manager.backends.backend import OffloadManager
from adrenaline.proxy.request_dispatcher import RequestDispatcher
from adrenaline.proxy.storage_manager import StorageManager
from adrenaline.proxy.utils import RequestRuntimeData

logger = init_logger("vllm.proxy.offload.backends.loadaware")


def is_decode_instance_available(
    m: StorageManager, req_id: str, req_data: RequestRuntimeData, decode_rank: int
):
    location = m.get_decode_location(decode_rank)
    req_num_max_used_blocks = m._calc_num_blocks(req_data.max_seq_len)
    max_used_block = req_num_max_used_blocks + m.num_used_blocks[location]
    decode_available = max_used_block < m.num_total_blocks[location]
    return decode_available


def is_attn_instance_available(
    m: StorageManager,
    req_id: str,
    req_data: RequestRuntimeData,
    offload_ratio: float,
    attn_rank: int,
    decode_rank: int,
):
    decode_loc = m.get_decode_location(decode_rank)
    attn_loc = m.get_attn_location(attn_rank)
    assert m.num_decode_ranks <= attn_loc
    req_num_max_used_blocks = m._calc_num_blocks(req_data.max_seq_len)
    if req_num_max_used_blocks > m.num_total_blocks[attn_loc]:
        return False
    attn_max_used_block = req_num_max_used_blocks + m.num_max_used_blocks[attn_loc]
    if attn_max_used_block < m.num_used_blocks[decode_loc] * offload_ratio:
        return True
    req_num_used_block = m._calc_num_blocks(req_data.seq_len)
    attn_num_used_block = req_num_used_block + m.num_used_blocks[attn_loc]
    attn_num_used_block_threshold = m.num_used_blocks[decode_loc] * offload_ratio
    attn_num_requests_threshold = m.num_requests[decode_loc] * offload_ratio
    if (
        attn_num_used_block < attn_num_used_block_threshold
        and m.num_requests[attn_loc] + 1 < attn_num_requests_threshold
    ):
        return True
    else:
        return False


class LoadawareOffloadManager(OffloadManager):
    def __init__(
        self,
        storage_manager: StorageManager,
        request_dispatcher: RequestDispatcher,
        offload_ratio: float,
    ):
        self.storage_manager = storage_manager
        self.request_dispatcher = request_dispatcher
        self.offload_ratio = offload_ratio

    def make_offload_decision(
        self, req_id: Optional[str], req_data: Optional[RequestRuntimeData]
    ):
        prefill_rank = self.request_dispatcher.target_prefill_instance_rank
        decode_rank = self.request_dispatcher.target_decode_instance_rank
        if is_attn_instance_available(
            self.storage_manager,
            req_id,
            req_data,
            self.offload_ratio,
            attn_rank=prefill_rank,
            decode_rank=decode_rank,
        ):
            return True
        else:
            if not is_decode_instance_available(
                self.storage_manager, req_id, req_data, decode_rank
            ):
                logger.warning("No available decode instance")
            return False
