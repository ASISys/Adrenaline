import threading
import time
import torch

from concurrent.futures import ThreadPoolExecutor
from torch.distributed import Backend
from typing import List, Optional, Union
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger


logger = init_logger(__name__)

# if the tensor is only one-element and only contains NONE_INT
# this means that the sended object is None.
NONE_INT = -150886311

# Mapping tensor dtype to INT64, used for tensor metadata transmission
FLOAT16_INT = -543205003776624
INT64_INT = -375623078607432
BOOL_INT = -28035262008646
BFLOAT16_INT = -452084912267662
FLOAT32_INT = -1049557997456592
FLOAT64_INT = -452201007054137
FLOAT8_E4M3FN_INT = -1066697177659525
FLOAT8_E5M2_INT = -618182574682355

DTYPE2INT = {
    torch.float16: FLOAT16_INT,
    torch.int64: INT64_INT,
    torch.bool: BOOL_INT,
    torch.bfloat16: BFLOAT16_INT,
    torch.float32: FLOAT32_INT,
    torch.float64: FLOAT64_INT,
    torch.float8_e4m3fn: FLOAT8_E4M3FN_INT,
    torch.float8_e5m2: FLOAT8_E5M2_INT,
}

INT2DTYPE = {
    FLOAT16_INT: torch.float16,
    INT64_INT: torch.int64,
    BOOL_INT: torch.bool,
    BFLOAT16_INT: torch.bfloat16,
    FLOAT32_INT: torch.float32,
    FLOAT64_INT: torch.float64,
    FLOAT8_E4M3FN_INT: torch.float8_e4m3fn,
    FLOAT8_E5M2_INT: torch.float8_e5m2,
}


class TorchWithinDevicePipe(KVPipeBase):
    METADATA_LENGTH = 16
    MAX_TENSOR_DIMENSIONS = 14
    METADATA_DTYPE = torch.int64

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend = Backend.NCCL
    ):
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.comm_group = None

        logger.debug(f"init pipe {group_ranks=}, {self.rank=}")
        for ranks in group_ranks:
            # transfer tensors within the same device can't use the nccl backend
            comm_group = torch.distributed.new_group(
                ranks, backend=Backend.GLOO
            )
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.comm_group = comm_group

        if self.comm_group is None:
            logger.debug(f"rank {self.rank} not in any group")
            return
        assert self.comm_group is not None
        assert self.rank_in_group <= 1

        self.device = self._select_device(torch_distributed_backend)
        self.device_type = "cpu" if self.device == "cpu" else "cuda"
        logger.debug(f"select device {self.device}")

        self.target_rank_for_send = self.ranks[
            (self.rank_in_group + 1) % self.world_size
        ]
        self.target_rank_for_recv = self.ranks[
            (self.rank_in_group - 1) % self.world_size
        ]
        logger.debug(f"init device pipe with {self.rank=}, {self.target_rank_for_send=}, {self.target_rank_for_recv=}")

        self.transport_thread: Optional[ThreadPoolExecutor] = None
        self.buffer_size = 0
        self.buffer_size_lock = threading.Lock()

        self.none_tensor = torch.tensor([NONE_INT], device=self.device)
        self.ack_tensor = torch.tensor([1], device="cpu")
        self.recv_metadata_buffer = torch.zeros(
            self.METADATA_LENGTH, dtype=self.METADATA_DTYPE, device="cpu"
        )

    def _select_device(self, torch_distributed_backend: Backend):
        if torch_distributed_backend == Backend.NCCL:
            return f"cuda:{self.local_rank}" 
        else:
            return "cpu"

    def _make_metadata(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create the metadata on based on the input tensor
        Parameters:
            - tensor: the input tensor

        Returns:
            - metadata: the cuda rebuild metadata of the tensor
        """
        raise NotImplementedError
        if tensor.device.type == "cuda":
            metadata = share_cuda_tensor(tensor)
        else:
            metadata = torch.empty(
                self.METADATA_LENGTH, dtype=self.METADATA_DTYPE, device="cpu"
            )
            metadata[0] = DTYPE2INT[tensor.dtype]
            ndims = len(tensor.shape)
            metadata[1] = len(tensor.shape)
            metadata[2 : 2 + ndims] = torch.tensor(
                tensor.shape, dtype=self.METADATA_DTYPE
            )
        return metadata

    def _send_metadata(self, metadata: torch.Tensor):
        """Send the metadata to the target rank.
        """
        raise NotImplementedError
        if isinstance(metadata, CudaRebuildMetadata):
            torch.distributed.send_object_list(
                [metadata], dst=self.target_rank_for_send, group=self.comm_group
            )
        else:
            assert isinstance(metadata, torch.Tensor)
            assert metadata.device.type == "cpu", f"{metadata.device}"
            torch.distributed.send(
                metadata, dst=self.target_rank_for_send, group=self.comm_group
            )

    def _recv_metadata(self) -> torch.Tensor:
        """Receive the metadata from the target rank.

        Returns:
            - metadata: the cuda rebuild metadata of the tensor

        Note:
            The current implementation uses the assumption that there is no
            race conditions during sending/receiving. Therefore, the metadata
            buffer can be reused
        """
        raise NotImplementedError
        if self.device_type == "cuda":
            recv_objects = [None]
            torch.distributed.recv_object_list(
                recv_objects, src=self.target_rank_for_recv, group=self.comm_group
            )
            metadata = recv_objects[0]
            return metadata
        else:
            torch.distributed.recv(
                self.recv_metadata_buffer,
                src=self.target_rank_for_recv,
                group=self.comm_group,
            )
            return self.recv_metadata_buffer

    def _send_impl(self, tensor: torch.Tensor):
        """
        The actual implementation of sending the tensor to the target rank.
        This function will first send the metadata, and then send the tensor.

        Parameters:
            - tensor: the input tensor to be sent
        """
        assert tensor is not None, "Sending None tensor is not supported"
        if self.device_type == "cuda":
            assert tensor.device.type == "cuda", f"{tensor.device=}, {self.device=}"
        metadata = self._make_metadata(tensor)
        self._send_metadata(metadata)
        if tensor.device.type == "cuda":
            # synchronize to preserve the tensor
            # ack_tensor = torch.empty(1)
            # torch.distributed.recv(ack_tensor, src=self.target_rank_for_send, group=self.comm_group)
            pass
        else:
            assert tensor.device.type == "cpu"
            torch.distributed.send(
                tensor, dst=self.target_rank_for_send, group=self.comm_group
            )

    def _prepare_recv_buffer(self, metadata: torch.Tensor) -> torch.Tensor:
        """Create a buffer to receive the tensor based on the metadata.

        Parameters:
            - d_metadata_buffer: the metadata tensor on self.device

        Returns:
            - buffer: the buffer tensor to receive the tensor, on self.device
        """
        h_buffer = metadata.numpy()
        dtype = INT2DTYPE[h_buffer[0]]
        ndims = h_buffer[1]
        shape = tuple(h_buffer[2:2 + ndims])
        return torch.empty(shape, dtype=dtype, device="cpu")

    def _recv_impl(self) -> torch.Tensor:
        """
        The actual implementation of receiving the tensor from the target rank.
        This function will first receive the metadata, then receive the tensor.

        This function will block if there is no tensor to receive.

        Returns:
            - buffer: the received tensor, on self.device
        """
        raise NotImplementedError
        metadata = self._recv_metadata()
        if self.device_type == "cuda":
            assert isinstance(metadata, CudaRebuildMetadata)
            # reuse the storage of the tensor from sender
            recv_tensor = rebuild_cuda_tensor(metadata)
            tensor = recv_tensor
            # tensor = recv_tensor.clone()
            # del recv_tensor
            # ack_tensor = torch.zeros(1)
            # torch.distributed.send(ack_tensor, dst=self.target_rank_for_recv, group=self.comm_group)
        else:
            assert isinstance(metadata, torch.Tensor)
            tensor = self._prepare_recv_buffer(metadata)
            torch.distributed.recv(
                tensor, src=self.target_rank_for_recv, group=self.comm_group
            )
        return tensor

    def send_tensor_wrapper(self, tensor: torch.Tensor, tensor_size: int):
        try:
            """Wrapper for send_tensor_dict"""
            self._send_impl(tensor)

            with self.buffer_size_lock:
                self.buffer_size = self.buffer_size - tensor_size
        except Exception as e:
            logger.error("[rank%d]: Exception when trying to send %s, msg: %s",
                         torch.distributed.get_rank(), str(tensor), str(e))
            import traceback
            traceback.print_exc()

    def block_if_full(self):
        """Block the current thread if the buffer size is larger than 1e9."""
        # TODO: replace this 1e9 with a configurable parameter or a constant
        while self.buffer_size > 4.5e9:
            logger.debug("KV cache transfer pipe is full. Waiting...")
            time.sleep(0.05)

    def send_tensor(self, tensor: Optional[torch.Tensor]):
        """Sends a tensor to the destination rank in a non-blocking way.
        Flow: send tensor dim -- send tensor shape -- send tensor data
        """
        if tensor is not None:
            tensor_size = tensor.element_size() * tensor.numel()
        else:
            tensor_size = 0

        self.block_if_full()

        with self.buffer_size_lock:
            self.buffer_size = self.buffer_size + tensor_size

        try:
            self.send_tensor_wrapper(tensor, tensor_size)
        except Exception as e:
            logger.error(f"Encountering exception in sending thread, {e}")
            import traceback

            traceback.print_exc()
            raise e

    def recv_tensor(self) -> Optional[torch.Tensor]:
        """Receives a tensor from the src rank. Blocking."""
        try:
            tensor = self._recv_impl()
        except Exception as e:
            # the underlying pipe is likely broken
            logger.error("Encountering exception in KV receiving thread")
            logger.error("%s", e)
            # fault tolerance: if the pipe is broken, return None
            return None

        if tensor.numel() == 1 and tensor.item() == NONE_INT:
            return None
        else:
            return tensor

    def close(self):
        """Close the pipe and release the resources."""
        if (hasattr(self, "transport_thread")
                and self.transport_thread is not None):
            self.transport_thread.shutdown()
