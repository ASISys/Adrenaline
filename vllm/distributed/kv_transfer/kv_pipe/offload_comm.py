import torch

from torch.distributed import Backend
from typing import Union

from vllm.logger import init_logger

logger = init_logger(__name__)


class OffloadComm:
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
    ):
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend
            )
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.src_rank = ranks[0]

        if self.device_group is None:
            logger.debug(f"rank {self.rank} not in any group")

    def close(self):
        if self.device_group is not None:
            torch.distributed.destroy_process_group(self.device_group)
