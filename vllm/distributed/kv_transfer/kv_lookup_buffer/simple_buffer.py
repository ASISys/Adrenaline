import threading
import time
from collections import deque
from typing import Deque, List, Optional, Union

import torch

from vllm.distributed.utils import TransferUtils, get_element_size
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.distributed.kv_transfer import vllm_adapter as dist_kv
from vllm.logger import init_logger

logger = init_logger(__name__)


class SimpleKVLookupBuffer(KVLookupBufferBase):

    def __init__(
        self,
        signal_pipe: KVPipeBase,
        data_pipe: KVPipeBase,
        buffer_size_thresh: int,
        signal_pipes: Optional[list[KVPipeBase]] = None,
        data_pipes: Optional[list[KVPipeBase]] = None,
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

        self.buffer: Deque[List[torch.Tensor]] = deque()

        self.buffer_size = 0
        self.buffer_size_threshold = buffer_size_thresh
        self.buffer_lock = threading.Lock()
        self.signal_pipe = signal_pipe
        self.data_pipe = data_pipe
        self.data_pipes = data_pipes
        self.signal_pipes = signal_pipes
        self.request_handling_thread: Optional[threading.Thread] = None

        self.normal_signal = torch.tensor([0])
        self.end_signal = None
        self.send_stream = send_stream

        self.hidden_size = transfer_utils.hidden_size
        self.num_heads = transfer_utils.num_kv_heads
        self.num_layers = transfer_utils.num_layers
        self.head_size = transfer_utils.head_size

    def _matches(self, tokens_roi_sender: List[torch.Tensor],
                 tokens_roi_recver: List[torch.Tensor]):

        tokens_sender = tokens_roi_sender[0]
        tokens_recver = tokens_roi_recver[0]

        if tokens_recver is None:
            # consumer sends an empty request
            # semantics: DROP SELECT * LIMIT 1
            # so any of the data in the buffer can be drop-selected
            return True

        # simple common prefix matching
        min_length = min(len(tokens_sender), len(tokens_recver))
        if torch.allclose(tokens_sender[:min_length],
                          tokens_recver[:min_length]):
            return min_length

        return 0


    def _get_element_size(self, data: Optional[Union[List, torch.Tensor]]):

        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()
        if not data:
            # cannot perform `not data` on a tensor
            # so this check needs to go after the check above
            return 0

        raise AssertionError("Unknown data type %s" % type(data))

    def _add_to_buffer(
        self,
        input_tokens: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        hidden: torch.Tensor
    ):
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
                self.buffer_size += self._get_element_size(data)
            self.buffer.append(buffer_item)

    def _is_end_signal(self, signal):
        return signal is None

    def drop_select_handler(self):
        with torch.cuda.stream(self.send_stream):
            self.drop_select_handler_impl()

    def drop_select_handler_impl(self):
        try:
            while True:
                signal = self.signal_pipe.recv_tensor()
                if self._is_end_signal(signal):
                    logger.info("Received end signal!")
                    break

                input_tokens = self.data_pipe.recv_tensor()
                tokens_roi_recver = [input_tokens]
                matched_length = 0

                # perform input tokens and roi matching
                # FIXME: this matching is O(n), ideally it should be O(1)
                # but this buffer size won't (and shouldn't) be too large so
                # the fix is not urgent.
                with self.buffer_lock:
                    for _ in range(len(self.buffer)):
                        temp_length = self._matches(self.buffer[0], tokens_roi_recver)
                        if temp_length > 0:
                            matched_length = temp_length
                            break
                        # rotate the element we just accessed to the end
                        self.buffer.rotate(-1)

                    assert matched_length > 0
                    matched_item = self.buffer.popleft()
                    for item in matched_item:
                        if item is not None:
                            self.buffer_size -= item.element_size() * item.numel()
                            self.data_pipe.send_tensor(item)

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_handler")

    def drop_select(
        self,
        input_tokens: torch.Tensor,
        recv_rank: int = -1,
        is_offload: bool = False,
    ) -> List[Optional[torch.Tensor]]:

        assert self.request_handling_thread is None, \
            "drop_select should be called by the KV cache consumer "\
            "(e.g. the decode vLLM instance)"

        assert recv_rank != -1, "recv_rank should be specified"
        self.signal_pipes[recv_rank].send_tensor(self.normal_signal)
        self.data_pipes[recv_rank].send_tensor(input_tokens)

        if is_offload:
            input_tokens = self.data_pipes[recv_rank].recv_tensor()
            hidden = self.data_pipes[recv_rank].recv_tensor()
            return [input_tokens, None, None, hidden]

        input_tokens = self.data_pipes[recv_rank].recv_tensor()
        key = self.data_pipes[recv_rank].recv_tensor()
        value = self.data_pipes[recv_rank].recv_tensor()
        hidden = self.data_pipes[recv_rank].recv_tensor()
        return [input_tokens, key, value, hidden]


    def full_handler(self):
        time.sleep(0.001)

    def insert(
        self,
        input_tokens: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        hidden: torch.Tensor,
        request_id: Optional[str] = None,
    ) -> None:

        if request_id is not None:
            logger.debug(f"Inserting tensor to buffer {self.buffer_size=}, {request_id=}")
        if self.buffer_size > self.buffer_size_threshold:
            # log outside the while loop to avoid this message being logged
            # repeatedly.
            logger.debug("KV transfer buffer is full. Handling...")
        while self.buffer_size > self.buffer_size_threshold:
            self.full_handler()

        self._add_to_buffer(input_tokens, key, value, hidden)

        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        if self.request_handling_thread is None:
            self.request_handling_thread = threading.Thread(
                target=self.drop_select_handler)
            self.request_handling_thread.start()

    def clear(self):
        while len(self.buffer) > 0:
            tensors = self.buffer.pop()
            self.buffer_size -= sum((t.element_size() * t.numel() for t in tensors))


    def close(self):

        if hasattr(self, "request_handling_thread"
                   ) and self.request_handling_thread is not None:
            self.request_handling_thread.join()

        else:
            # TODO: have a explicit close signal and have a explicit way to
            # check if it's requester
            self.signal_pipe.send_tensor(self.end_signal)
