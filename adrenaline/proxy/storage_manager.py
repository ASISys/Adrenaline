from vllm.utils import cdiv

from adrenaline.proxy.utils import RequestRuntimeData


class StorageManager:

    def __init__(
        self,
        num_decode_blocks: list[int],
        num_attn_blocks: list[int],
        block_size: int,
    ):
        self.block_size = block_size

        # storage static infomation
        self.num_decode_ranks = len(num_decode_blocks)
        self.num_attn_ranks = len(num_attn_blocks)
        self.num_total_blocks = num_decode_blocks + num_attn_blocks

        # storage runtime infomation
        self.num_used_blocks = [
            0 for _ in range(self.num_decode_ranks + self.num_attn_ranks)
        ]
        self.num_max_used_blocks = [
            0 for _ in range(self.num_decode_ranks + self.num_attn_ranks)
        ]
        self.num_requests = [
            0 for _ in range(self.num_decode_ranks + self.num_attn_ranks)
        ]

        # request runtime information related to storage
        # map request_id to execution location [0, #decodes + #attentions - 1]
        self.reqs_location: dict[str, int] = {}
        # map request_id to current numbers of blocks
        self.req_num_blocks: dict[str, int] = {}
        # map request_id to max numbers of blocks
        self.req_max_num_blocks: dict[str, int] = {}

    def setup_location_for_new_request(
        self, req_id: str, prefill_rank: int, decode_rank: int, is_offload: bool
    ):
        if is_offload:
            location = self.get_attn_location(prefill_rank)
        else:
            location = self.get_decode_location(decode_rank)
        self.reqs_location[req_id] = location

    def update_blocks_for_new_request(self, req_id: str, req_data: RequestRuntimeData):
        assert (
            req_id in self.reqs_location
        ), "please call setup_location_for_new_requests before update blocks for new request"
        location = self.reqs_location[req_id]
        self.req_num_blocks[req_id] = self._calc_num_blocks(req_data.seq_len)
        self.req_max_num_blocks[req_id] = self._calc_num_blocks(req_data.max_seq_len)
        # update storage runtime information
        self.num_used_blocks[location] += self.req_num_blocks[req_id]
        self.num_max_used_blocks[location] += self.req_max_num_blocks[req_id]
        self.num_requests[location] += 1

    def update_blocks_for_decode_request(self, req_id: str, seq_len: int):
        num_blocks_needed = self._calc_num_blocks(seq_len)
        num_current_blocks = self.req_num_blocks[req_id]
        if num_current_blocks < num_blocks_needed:
            location = self.reqs_location[req_id]
            self.num_used_blocks[location] += num_blocks_needed - num_current_blocks
            self.req_num_blocks[req_id] = num_blocks_needed

    def release_request(self, req_id: str):
        location = self.reqs_location[req_id]
        self.num_used_blocks[location] -= self.req_num_blocks[req_id]
        self.num_max_used_blocks[location] -= self.req_max_num_blocks[req_id]
        self.num_requests[location] -= 1

        self.reqs_location.pop(req_id)
        self.req_num_blocks.pop(req_id)

    def _calc_num_blocks(self, seq_len: int):
        return cdiv(seq_len, self.block_size)

    def __repr__(self):
        return f"{self.num_used_blocks=}, {self.num_max_used_blocks=}, {self.num_requests=}"

    def get_decode_location(self, decode_instance_rank: int):
        return decode_instance_rank

    def get_attn_location(self, attn_instance_rank: int):
        return self.num_decode_ranks + attn_instance_rank
