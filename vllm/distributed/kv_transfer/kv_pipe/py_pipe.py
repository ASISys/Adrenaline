"""
    This module implements a PyNccl pipe for sending and receiving 
    Optional[torch.Tensor] between distributed ranks with advanced 
    communication features.

    Key Features:
    - Supports sending and receiving tensors with metadata
    - Handles both CUDA and CPU device communications
    - Implements a non-blocking tensor transfer mechanism
    - Manages buffer size and provides backpressure control
    - Supports distributed process groups with configurable parameters
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional, Tuple, List, Union

import torch
from torch.distributed import ProcessGroup, Backend

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
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


class BrokenPipeException(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


Metadata = Dict[str, Optional[torch.Tensor]]


class PyPipe(KVPipeBase):

    METADATA_LENGTH = 16
    MAX_TENSOR_DIMENSIONS = 14
    METADATA_DTYPE = torch.int64

    def __init__(self,
                 group_ranks: List[List[int]],
                 local_rank: int,
                 torch_distributed_backend: Union[str, Backend]):
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None

        for ranks in group_ranks:
            logger.debug(f"torch.distributed.new_group: ranks={ranks}, "
                         f"{torch_distributed_backend}")
            group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend)
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = group

        if self.device_group is None:
            logger.debug(f"rank {self.rank} not in {group_ranks=} group")
            return
        assert self.device_group is not None
        assert self.rank_in_group <= 1

        self.device = self._select_device(torch_distributed_backend)

        self.target_rank_for_send = (self.rank_in_group + 1) % self.world_size
        self.target_rank_for_recv = (self.rank_in_group + 1) % self.world_size

        # 设定 send / recv 函数 (cuda: PyNcclCommunicator, cpu: group.send_obj / group.recv_obj)
        impl = self._get_device_send_recv_impl(self.device_group)
        self.device_send_func, self.device_recv_func = impl

        # transportation-related variables
        self.buffer_size = 0
        self.buffer_size_lock = threading.Lock()
        # TODO: replace this 1e9 with a configurable parameter or a constant
        self.buffer_size_thresh = 1e9

        self.none_tensor = torch.tensor([NONE_INT], device=self.device)

        # On-device tensors to be reused for recv
        self.rcv_metadata_buffer = torch.zeros(self.METADATA_LENGTH,
                                               dtype=self.METADATA_DTYPE,
                                               device=self.device)

    def _get_device_send_recv_impl(
        self, group: ProcessGroup
    ) -> Tuple[Callable[[torch.Tensor, int], None], Callable[
        [torch.Tensor, int], None]]:

        send: Callable[[torch.Tensor, int], None]
        recv: Callable[[torch.Tensor, int], None]

        # Currently, PyPipe is designed to use pynccl
        assert self.device.type == "cuda"

        if self.device.type == "cuda":
            # use PyNCCL for send / recv
            comm = PyNcclCommunicator(group, device=self.local_rank)
            comm.enable()
            send, recv = comm.send, comm.recv  # type: ignore
        else:
            # This send / recv implementation here is NOT intended to transfer
            # KV caches (and should NOT be repurposed to transfer KV caches).
            # Currently it is only used to transmit control-plane messages
            # for PyNcclBuffer.
            send = group.send_obj

            def my_recv(x, src):
                x[...] = group.recv_obj(src)

            recv = my_recv

        return send, recv

    def _select_device(self, device: str):
        logger.info("Selecting device: %s", device)
        if device == "nccl":
            return torch.device(f"cuda:{self.local_rank}")
        else:
            return torch.device("cpu")

    def _make_metadata(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create the metadata on based on the input tensor, and move it to GPU.
        The metadata's length is `TorchDistributedPipe.METADATA_LENGTH`.

        Currently, the metadata is a int64 tensor and it includes dtype, number
        of dimensions, and the shape information of the input tensor.


        The information follows the layout below:
        - metadata[0] -- dtype
        - metadata[1] -- number of dimensions
        - metadata[2 : 2+ndims] -- the shape of the input tensor

        Parameters:
            - tensor: the input tensor

        Returns:
            - metadata: the metadata tensor, on self.device
        """
        buffer = torch.empty(
            self.METADATA_LENGTH, dtype=self.METADATA_DTYPE, device="cpu"
        )
        buffer[0] = DTYPE2INT[tensor.dtype]
        ndims = len(tensor.shape)
        buffer[1] = len(tensor.shape)
        buffer[2 : 2 + ndims] = torch.tensor(tensor.shape, dtype=self.METADATA_DTYPE)
        return buffer.to(self.device)

    def _prepare_recv_buffer(self,
                             d_metadata_buffer: torch.Tensor) -> torch.Tensor:
        """Create a buffer to receive the tensor based on the metadata.

        Parameters:
            - d_metadata_buffer: the metadata tensor on self.device

        Returns:
            - buffer: the buffer tensor to receive the tensor, on self.device
        """
        h_buffer = d_metadata_buffer.cpu().numpy()
        dtype = INT2DTYPE[h_buffer[0]]
        ndims = h_buffer[1]
        shape = tuple(h_buffer[2:2 + ndims])
        return torch.empty(shape, dtype=dtype, device=self.device)

    def _send_metadata(self, d_metadata_buffer: torch.Tensor):
        """Send the metadata buffer to the target rank.
        """
        self.device_send_func(d_metadata_buffer,
                              self.target_rank_for_send)

    def _recv_metadata(self) -> Metadata:
        """Receive the metadata buffer from the target rank.

        Returns:
            - metadata_buffer: the metadata buffer tensor, on self.device

        Note:
            The current implementation uses the assumption that there is no
            race conditions during sending/receiving. Therefore, the metadata
            buffer can be reused
        """
        self.device_recv_func(self.rcv_metadata_buffer,
                              self.target_rank_for_recv)

        return self.rcv_metadata_buffer

    def _send_impl(self, tensor: Optional[torch.Tensor]) -> None:
        """
        The actual implementation of sending the tensor and its metadata to the 
        target rank.

        Parameters:
            - tensor: The input tensor to be sent, or None if no tensor is 
              being sent.
        """
        metadata = self._make_metadata(tensor)
        self._send_metadata(metadata)
        if tensor is not None:
            self.device_send_func(tensor.to(self.device),
                                  self.target_rank_for_send)
        else:
            raise ValueError("Sending None tensor is not supported")

    def _recv_impl(self) -> Optional[torch.Tensor]:
        """
        The actual implementation of receiving a tensor and its metadata from 
        the target rank.

        Returns:
            - buffer: The received tensor, or None if no tensor is received.
        """
        metadata = self._recv_metadata()
        assert metadata is not None

        assert torch.cuda.current_stream() != torch.cuda.default_stream()
        buffer = self._prepare_recv_buffer(metadata)
        self.device_recv_func(buffer, self.target_rank_for_recv)

        return buffer

    def send_tensor_wrapper(self, tensor: Optional[torch.Tensor],
                            tensor_size: int) -> None:
        """
        Wrapper for _send_impl to handle exceptions and update buffer size.
        """
        try:
            self._send_impl(tensor)

            with self.buffer_size_lock:
                self.buffer_size -= tensor_size
        except Exception as e:
            logger.error("[rank%d]: Exception when trying to send %s, msg: %s",
                         torch.distributed.get_rank(), str(tensor), str(e))
            import traceback
            traceback.print_exc()

    def block_if_full(self):
        """
        Block the current thread if the buffer size is larger than the 
        threshold.
        """
        while self.buffer_size > self.buffer_size_thresh:
            logger.debug("KV cache transfer pipe is full. Waiting...")
            time.sleep(0.05)

    # # No need to update target rank ID for attn offload
    # def update_send_target(self, target_rank: int) -> None:
    #     self.target_rank_for_send = target_rank

    # # No need to update target rank ID for attn offload
    # def update_recv_target(self, target_rank: int) -> None:
    #     self.target_rank_for_recv = target_rank

    def send_tensor(self, tensor: Optional[torch.Tensor]) -> None:
        """
        Sends a tensor and its metadata to the destination rank in a 
        non-blocking way.

        Parameters:
            - tensor: The tensor to send, or None if no tensor is being sent.
        """
        if tensor is not None:
            tensor_size = tensor.element_size() * tensor.numel()
        else:
            tensor_size = 0

        self.block_if_full()
        with self.buffer_size_lock:
            self.buffer_size += tensor_size

        try:
            self.send_tensor_wrapper(tensor, tensor_size)
        except Exception as e:
            logger.error(f"Encountering exception in sending thread")
            import traceback

            traceback.print_exc()
            raise e

    def recv_tensor(self, num_tokens: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Receives a tensor and its metadata from the source rank. Blocking call.

        Returns:
            - tensor: The received tensor, or None if no tensor is received.
        """
        try:
            tensor = self._recv_impl()
            if num_tokens is not None:
                assert (
                    tensor.dim() == 4
                ), f"Tensor dimensions are {tensor.dim()}, expected 4"
                assert (
                    tensor.size(2) == num_tokens
                ), f"Dimension at index 2 is {tensor.size(2)}, expected {num_tokens}"
                logger.debug(
                    f"[Pipe RECV] require {num_tokens} tokens, does get shape {tensor.shape}"
                )
        except Exception as e:
            logger.error("Encountering exception in KV receiving thread")
            logger.error("%s", e)
            logger.error("My device: %s", self.device)
            import traceback
            traceback.print_exc()
            raise e

        return tensor

    def close(self):
        """
        Close the pipe and release associated resources.
        """
        pass
