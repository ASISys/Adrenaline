import enum
import time
import threading
import torch

from torch import distributed as dist
from typing import List, Optional
from vllm.distributed.utils import TransferUtils, get_element_size
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.distributed.kv_transfer import vllm_adapter as dist_kv
from vllm.logger import init_logger

from adrenaline import custom_ops as adrenaline_ops
from adrenaline.custom_ops import IpcHandle, DevPtr

logger = init_logger(__name__)


class TransferSignal(enum.IntEnum):
    NORMAL = enum.auto()
    REQ = enum.auto()
    ACK = enum.auto()


class IpcTransferBuffer(KVLookupBufferBase):

    def __init__(
        self,
        signal_pipe: KVPipeBase,
        data_pipe: KVPipeBase,
        buffer_size_thresh: int,
        ipc_buffer_size: int,
        send_stream: torch.cuda.Stream = None,
        transfer_utils: Optional[TransferUtils] = None,
    ):
        """
        signal_pipe: on CPU 
        
        NOTE: on-device recv will block all threads in the process, making the 
        KV cache producer unable to listen to new request while transmitting 
        KV cache. Luckily CPU recv only blocks the current thread so we use 
        CPU recv to listen to new request.
        
        data_pipe: on device (e.g. GPU)
        """

        self.buffer: dict[str, list[torch.Tensor]] = {}
        self.buffer_size = 0
        self.buffer_size_threshold = buffer_size_thresh
        self.buffer_lock = threading.Lock()
        self.signal_pipe = signal_pipe
        self.data_pipe = data_pipe
        self.request_handling_thread: Optional[threading.Thread] = None

        self.normal_signal = torch.tensor([TransferSignal.NORMAL], dtype=torch.uint8)
        self.recv_signal_tensor = torch.empty_like(self.normal_signal)

        self.req_tensor = torch.tensor([TransferSignal.REQ], dtype=torch.uint8)
        self.ack_tensor = torch.tensor([TransferSignal.ACK], dtype=torch.uint8)
        self.recv_ipc_tensor = torch.empty_like(self.req_tensor)
        self.end_signal = None

        self.send_stream = send_stream
        self.ipc_buffer = torch.empty(
            ipc_buffer_size, device="cuda", dtype=torch.uint8
        )
        self.remote_buffer_addr: Optional[DevPtr] = None

        self.hidden_size = transfer_utils.hidden_size
        self.num_heads = transfer_utils.num_kv_heads
        self.num_layers = transfer_utils.num_layers
        self.head_size = transfer_utils.head_size

    def export_ipc_buffer_handle(self):
        ipc_handle = adrenaline_ops.get_ipc_mem_handle(self.ipc_buffer)
        dist.send_object_list(
            [ipc_handle],
            dst=self.data_pipe.target_rank_for_send,
            group=self.data_pipe.comm_group,
        )

    def register_remote_ipc_buffer(self):
        recv_objects = [None]
        dist.recv_object_list(
            recv_objects,
            src=self.data_pipe.target_rank_for_recv,
            group=self.data_pipe.comm_group,
        )
        ipc_handle: IpcHandle = recv_objects[0]
        self.remote_buffer_addr = adrenaline_ops.open_ipc_mem_handle(ipc_handle)

    def unregister_remote_ipc_buffer(self):
        adrenaline_ops.close_ipc_mem_handle(self.remote_buffer_addr)
        self.remote_buffer_addr = None

    def _add_to_buffer(self, input_tokens: torch.Tensor,
                       key: torch.Tensor, value: torch.Tensor,
                       hidden: torch.Tensor, request_id: str):

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(key, torch.Tensor):
            key = key.clone()
        if isinstance(value, torch.Tensor):
            value = value.clone()
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.clone()

        buffer_item = [input_tokens, key, value, hidden]

        with self.buffer_lock:
            for data in buffer_item:
                self.buffer_size += get_element_size(data)
            self.buffer[request_id] = buffer_item

    def _is_end_signal(self, signal):
        return signal is None

    def send_tensor(self, tensor: torch.Tensor):
        adrenaline_ops.copy_tensor_to_buffer(self.ipc_buffer, tensor, async_op=True)
        torch.cuda.current_stream().synchronize()
        dist.send(self.req_tensor, dst=self.data_pipe.target_rank_for_send, group=self.data_pipe.comm_group)
        dist.recv(self.recv_ipc_tensor, src=self.data_pipe.target_rank_for_send, group=self.data_pipe.comm_group)
        assert self.recv_ipc_tensor.item() == TransferSignal.ACK

    def recv_tensor(self, output_buffer: torch.Tensor):
        dist.recv(self.recv_ipc_tensor, src=self.data_pipe.target_rank_for_recv, group=self.data_pipe.comm_group)
        assert self.recv_ipc_tensor.item() == TransferSignal.REQ
        adrenaline_ops.copy_tensor_from_buffer_addr(output_buffer, self.remote_buffer_addr, async_op=True)
        torch.cuda.current_stream().synchronize()
        dist.send(self.ack_tensor, dst=self.data_pipe.target_rank_for_recv, group=self.data_pipe.comm_group)

    def drop_select_handler(self):
        with torch.cuda.stream(self.send_stream):
            self.drop_select_handler_impl()

    def drop_select_handler_impl(self):
        try:
            while True:
                torch.distributed.recv(
                    self.recv_signal_tensor,
                    self.signal_pipe.target_rank_for_recv,
                    self.signal_pipe.comm_group,
                )
                if self._is_end_signal(self.recv_signal_tensor):
                    logger.info("Received end signal!")
                    break

                recv_objects = [None]
                torch.distributed.recv_object_list(
                    recv_objects,
                    src=self.data_pipe.target_rank_for_send,
                    group=self.data_pipe.comm_group,
                )
                request_id: str = recv_objects[0]
                assert request_id in self.buffer, f"{self.buffer}"
                with self.buffer_lock:
                    tensors = self.buffer.pop(request_id)
                    tensors_size = sum(get_element_size(t) for t in tensors)
                    self.buffer_size -= tensors_size
                for tensor in tensors:
                    self.send_tensor(tensor)
                tensors = None

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_handler")

    def drop_select(
        self,
        input_tokens: torch.Tensor,
        request_id: str,
        recv_rank: int = -1,
    ) -> List[Optional[torch.Tensor]]:

        assert self.request_handling_thread is None, \
            "drop_select should be called by the KV cache consumer "\
            "(e.g. the decode vLLM instance)"
        assert dist_kv.IS_ATTN_EXECUTOR

        num_tokens = input_tokens.size(0)

        torch.distributed.send(
            self.normal_signal,
            self.signal_pipe.target_rank_for_send,
            self.signal_pipe.comm_group,
        )
        torch.distributed.send_object_list(
            [request_id],
            dst=self.data_pipe.target_rank_for_recv,
            group=self.data_pipe.comm_group,
        )
        key = torch.empty(self.num_layers, num_tokens, self.num_heads, self.head_size, dtype=torch.half, device="cuda")
        value = torch.empty(self.num_layers, num_tokens, self.num_heads, self.head_size, dtype=torch.half, device="cuda")
        hidden = torch.empty(num_tokens, self.hidden_size, dtype=torch.half, device="cuda")
        tensors = [input_tokens, key, value, hidden]
        for t in tensors:
            self.recv_tensor(t)

        return tensors

    def full_handler(self):
        time.sleep(0.001)

    def insert(
        self,
        input_tokens: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        hidden: torch.Tensor,
        request_id: Optional[str] = None,
    ) -> None:
        logger.debug(f"Inserting tensor to buffer {self.buffer_size=}, {request_id=}")
        if self.buffer_size > self.buffer_size_threshold:
            # log outside the while loop to avoid this message being logged
            # repeatedly.
            logger.debug("KV transfer buffer is full. Handling...")
        while self.buffer_size > self.buffer_size_threshold:
            self.full_handler()

        self._add_to_buffer(input_tokens, key, value, hidden, request_id)

        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        if self.request_handling_thread is None:
            self.request_handling_thread = threading.Thread(
                target=self.drop_select_handler)
            self.request_handling_thread.start()

    def clear(self):
        while len(self.buffer) > 0:
            tensors: list[torch.Tensor] = self.buffer.pop()
            self.buffer_size -= sum(get_element_size(t) for t in tensors)

    def close(self):
        if self.ipc_buffer is not None:
            self.unregister_remote_ipc_buffer()

        if hasattr(self, "request_handling_thread"
                   ) and self.request_handling_thread is not None:
            self.request_handling_thread.join()

        else:
            # TODO: have a explicit close signal and have a explicit way to
            # check if it's requester
            self.signal_pipe.send_tensor(self.end_signal)
