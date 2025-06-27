import torch


from torch.distributed import Backend

from vllm.distributed.kv_transfer.kv_pipe.offload_comm import OffloadComm
from vllm.distributed.kv_transfer.vllm_adapter import *
from typing import List, Optional, Union


class Offload_attn_exec_agent:
    """
    A class designated for KV cache offloading
    """

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        dist_backend: Union[str, Backend] = Backend.NCCL,
        decode_master_rank: int = -1,
    ):
        data_comm = OffloadComm(group_ranks, local_rank, dist_backend)
        fin_comm = OffloadComm(group_ranks, local_rank, dist_backend)
        sche_comm = OffloadComm(group_ranks, local_rank, Backend.GLOO)
        sync_comm = OffloadComm(group_ranks, local_rank, Backend.GLOO)
        assert decode_master_rank != -1
        self.decode_master_rank = decode_master_rank
        if IS_KV_CONSUMER:
            self.data_comm = data_comm
            self.sche_comm = sche_comm
            self.sync_comm = sync_comm
            self.fin_comm = fin_comm
        elif IS_ATTN_EXECUTOR:
            self.data_comm = data_comm
            self.sche_comm = sche_comm
            self.sync_comm = sync_comm
            self.fin_comm = fin_comm

    def destroy(self):
        if self.data_comm is not None:
            self.data_comm.close()
        if self.sche_comm is not None:
            self.sche_comm.close()
        if self.sync_comm is not None:
            self.sync_comm.close()

    def boardcast_sche_out(self, objects: list):
        torch.distributed.broadcast_object_list(
            objects,
            src=self.decode_master_rank,
            group=self.sche_comm.device_group,
        )
        return objects

    def broadcast_tensor_to_offload(self, tensor: torch.Tensor, async_op: bool = False):
        return torch.distributed.broadcast(
            tensor, src=self.decode_master_rank, group=self.fin_comm.device_group, async_op=async_op
        )

    def scatter_tensor_to_offload(
        self,
        output_tensor: Optional[torch.Tensor] = None,
        input_tensors: Optional[list[torch.Tensor]] = None,
        async_op: bool = False,
    ):
        if IS_KV_CONSUMER:
            assert len(input_tensors) == NUM_DECODE_INSTANCES + NUM_ATTN_INSTANCES
            output_tensor = torch.empty_like(input_tensors[1])
            input_tensors = [output_tensor] + input_tensors[1:]

        work = torch.distributed.scatter(
            output_tensor,
            input_tensors,
            src=self.data_comm.src_rank,
            group=self.data_comm.device_group,
            async_op=async_op,
        )
        return work

    def gather_tensor_to_decode(
        self,
        input_tensor: Optional[torch.Tensor] = None,
        output_tensors: Optional[list[torch.Tensor]] = None,
        async_op: bool = False,
    ):
        if IS_KV_CONSUMER:
            assert len(output_tensors) == NUM_DECODE_INSTANCES + NUM_ATTN_INSTANCES
            input_tensor = torch.empty_like(output_tensors[1])
            output_tensors = [input_tensor] + output_tensors[1:]

        work = torch.distributed.gather(
            input_tensor,
            output_tensors,
            dst=self.data_comm.src_rank,
            group=self.data_comm.device_group,
            async_op=async_op,
        )
        return work
