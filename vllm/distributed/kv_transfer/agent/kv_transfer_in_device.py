import threading
import time
import torch

from concurrent.futures import ThreadPoolExecutor
from torch.distributed import Backend

from vllm import _custom_ops as ops
from vllm.distributed.utils import TransferUtils
from vllm.distributed.kv_transfer.kv_lookup_buffer.ipc_buffer import IpcTransferBuffer
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import KVLookupBufferBase
from vllm.distributed.kv_transfer.kv_pipe.torch_within_device_pipe import (
    TorchWithinDevicePipe,
)
from vllm.distributed.kv_transfer.agent.utils import handle_error
from vllm.distributed.kv_transfer.vllm_adapter import *
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata


logger = init_logger(__name__)

def in_comm_group(pipe: TorchWithinDevicePipe):
    return pipe.comm_group is not None

class KV_transfer_agent_in_same_device:
    """
    A class designated for transfering metadata within the same CUDA device
    """

    def __init__(
        self,
        group_ranks: list[list[list[int]]],
        local_rank: int,
        # FIXME(Kuntai): remove this hardcoding
        lookup_buffer_size: int = int(4.5e9),
        transfer_utils: Optional[TransferUtils] = None,
    ):

        self.lookup_buffer_size = lookup_buffer_size

        self.send_buffer: Optional[KVLookupBufferBase] = None
        self.recv_buffer: Optional[KVLookupBufferBase] = None
        self.send_pipe: Optional[TorchWithinDevicePipe] = None
        self.recv_pipe: Optional[TorchWithinDevicePipe] = None
        self.send_signal_pipe: Optional[TorchWithinDevicePipe] = None
        self.recv_signal_pipe: Optional[TorchWithinDevicePipe] = None

        producer_send_pipes = [
            TorchWithinDevicePipe(group_rank_per_instance, local_rank, Backend.NCCL)
            for group_rank_per_instance in group_ranks
        ]
        consumer_send_pipes = [
            TorchWithinDevicePipe(group_rank_per_instance, local_rank, Backend.NCCL)
            for group_rank_per_instance in group_ranks
        ]
        producer_send_signal_pipes = [
            TorchWithinDevicePipe(group_rank_per_instance, local_rank, Backend.GLOO)
            for group_rank_per_instance in group_ranks
        ]
        consumer_send_signal_pipes = [
            TorchWithinDevicePipe(group_rank_per_instance, local_rank, Backend.GLOO)
            for group_rank_per_instance in group_ranks
        ]

        self.recv_condition = threading.Condition()
        self.recv_batch_queue: list[tuple[
            torch.nn.Module,
            int,
            int,
            ModelInputForGPUWithSamplingMetadata,
            list[torch.Tensor],
        ]] = []
        self.recv_handler_thread_executor = ThreadPoolExecutor(max_workers=1)

        self.received_requests_lock = threading.Lock()
        self.received_requests: dict[str, torch.Tensor] = {}

        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        if IS_KV_PRODUCER:
            send_pipes = list(filter(in_comm_group, producer_send_pipes))
            assert len(send_pipes) == 1
            self.send_pipe = send_pipes[0]

            send_signal_pipes = list(filter(in_comm_group, producer_send_signal_pipes))
            assert len(send_signal_pipes) == 1
            self.send_signal_pipe = send_signal_pipes[0]

            recv_pipes = list(filter(in_comm_group, consumer_send_pipes))
            assert len(recv_pipes) == 1
            self.recv_pipe = recv_pipes[0]

            recv_signal_pipes = list(filter(in_comm_group, consumer_send_signal_pipes))
            assert len(recv_signal_pipes) == 1
            self.recv_signal_pipe = recv_signal_pipes[0]

            # NOTE: ipc_buffer of prefill instance is used to send key / value cache,
            # the maximal size of tensor is max_model_len * token_kv_size / 2
            ipc_buffer_size = transfer_utils.max_model_len * transfer_utils.token_kv_size // 2
            self.send_buffer = IpcTransferBuffer(
                self.send_signal_pipe,
                self.send_pipe,
                self.lookup_buffer_size,
                ipc_buffer_size,
                self.send_stream,
                transfer_utils=transfer_utils,
            )
        elif IS_ATTN_EXECUTOR:
            recv_pipes = list(filter(in_comm_group, producer_send_pipes))
            assert len(recv_pipes) == 1
            self.recv_pipe = recv_pipes[0]

            recv_signal_pipes = list(filter(in_comm_group, producer_send_signal_pipes))
            assert len(recv_signal_pipes) == 1
            self.recv_signal_pipe = recv_signal_pipes[0]

            send_pipes = list(filter(in_comm_group, consumer_send_pipes))
            assert len(send_pipes) == 1
            self.send_pipe = send_pipes[0]

            send_signal_pipes = list(filter(in_comm_group, consumer_send_signal_pipes))
            assert len(send_signal_pipes) == 1
            self.send_signal_pipe = send_signal_pipes[0]

            # NOTE: ipc_buffer of attn instance is used to send input_tokens to prefill instance
            ipc_buffer_size = transfer_utils.max_model_len * torch.empty(1, dtype=torch.long).element_size()
            self.recv_buffer = IpcTransferBuffer(
                self.recv_signal_pipe,
                self.recv_pipe,
                self.lookup_buffer_size,
                ipc_buffer_size,
                self.recv_stream,
                transfer_utils=transfer_utils,
            )

        if IS_KV_PRODUCER:
            self.send_buffer.export_ipc_buffer_handle()
            self.send_buffer.register_remote_ipc_buffer()
        elif IS_ATTN_EXECUTOR:
            self.recv_buffer.register_remote_ipc_buffer()
            self.recv_buffer.export_ipc_buffer_handle()

    def destroy(self) -> None:
        if self.send_buffer is not None:
            self.send_buffer.close()
        if self.recv_buffer is not None:
            self.recv_buffer.close()
        if self.send_pipe is not None:
            self.send_pipe.close()
        if self.recv_pipe is not None:
            self.recv_pipe.close()
        if self.send_signal_pipe is not None:
            self.send_signal_pipe.close()
        if self.recv_signal_pipe is not None:
            self.recv_signal_pipe.close()
        self.recv_handler_thread_executor.shutdown()

    def send_kv_caches_and_hidden_states(
        self,
        start_layer: int,
        end_layer: int,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor, IntermediateTensors],
    ) -> None:

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()

        # query_lens contains new KV caches that are added to vLLM.
        # so we will send them to decode instance
        # FIXME(Kuntai): This assume that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            is_offload = idx >= model_input.attn_metadata.num_locals
            if not is_offload:
                continue
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]

            keys, values = [], []

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]

                _, _, num_heads, head_size = kv_cache[:, 0].shape
                key_cache = kv_cache[:, 0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[:, 1].reshape(-1, num_heads, head_size)

                # FIXME: assert using flashinfer here
                # _, _, num_heads, head_size = kv_cache[0].shape
                # key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                # value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))

            request_id = model_input.request_ids[idx]
            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)
            if self.send_buffer is not None:
                self.send_buffer.insert(
                    current_tokens,
                    keys,
                    values,
                    hidden_or_intermediate_states[start_pos:end_pos],
                    request_id,
                )

            logger.debug(
                f"[rank{torch.distributed.get_rank()}] Add KvCache {request_id=} to buffer DONE."
            )

    def recv_kv_caches_and_hidden_states(
        self,
        attn: torch.nn.Module,
        start_layer: int,
        end_layer: int,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
    ) -> Tuple[
        Union[torch.Tensor, IntermediateTensors],
        bool,
        "ModelInputForGPUWithSamplingMetadata",
    ]:

        # When this flag is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True

        # passed in seq_lens including all offloaded requests
        # thus we need to filter out the local requests
        # TODO: fix prepare_model_input to avoid this,
        # however it is related to sampling metadata construction
        mask = [
            idx for idx, prefill_rank in enumerate(model_input.prefill_ranks)
            if prefill_rank == get_attn_rank()
        ]
        if mask:
            start_index, end_index = mask[0], mask[-1] + 1
            token_start_index = sum(model_input.seq_lens[:start_index])
            token_end_index = sum(model_input.seq_lens[:end_index])
            input_tokens_tensor = model_input.input_tokens[
                token_start_index:token_end_index
            ]
            seq_lens = model_input.seq_lens[start_index:end_index]
        else:
            seq_lens: list[int] = []
            input_tokens_tensor = torch.zeros(0, dtype=torch.long)
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

        hidden_or_intermediate_states_for_one_req = []

        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []

        # enumerate different requests
        # FIXME(Kuntai): This impl assumes that all requests are prefill.
        assert len(seq_lens) == len(mask)
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]
            num_tokens = slen

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            if self.recv_buffer is None:
                bypass_model_exec = False
                break

            request_id = model_input.request_ids[idx]
            ret = self.recv_buffer.drop_select(current_tokens, request_id)
            if ret[0] is None:
                # didn't find any match.
                bypass_model_exec = False
                num_computed_tokens_list.append(0)
                assert ret[0] is not None, "no match"
                continue

            input_tokens: torch.Tensor = ret[0]
            keys: torch.Tensor = ret[1]
            values: torch.Tensor = ret[2]
            hidden: torch.Tensor = ret[3]

            num_computed_tokens = input_tokens.shape[0]
            num_computed_tokens_list.append(num_computed_tokens)

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == num_tokens), hidden is not None]):
                bypass_model_exec = False
                assert num_computed_tokens == num_tokens, f"computed_tokens ({num_computed_tokens}) != num_tokens ({num_tokens})"

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # put received KV caches into paged memory
            for i in range(start_layer, end_layer):
                kv_cache = kv_caches[i - start_layer]

                # FIXME: assert using flashinfer here
                # key_cache, value_cache = kv_cache[0], kv_cache[1]
                key_cache, value_cache = kv_cache[:, 0], kv_cache[:, 1]
                ops.reshape_and_cache_flash(
                    keys[i - start_layer].to(key_cache.device),
                    values[i - start_layer].to(value_cache.device),
                    key_cache,
                    value_cache,
                    slot_mapping[start_pos:end_pos],
                    attn.kv_cache_dtype,
                    attn._k_scale,
                    attn._v_scale,
                )

            hidden_or_intermediate_states_for_one_req.append(hidden)

        logger.debug(
            "[rank%d]: Successfully received all KVs and hidden "
            "states, skip model forwarding.", torch.distributed.get_rank())
        if len(hidden_or_intermediate_states_for_one_req) > 0:
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0
            )
        else:
            hidden_or_intermediate_states = None

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def async_recv_kv_caches_and_hidden_states(
        self,
        attn: torch.nn.Module,
        start_layer: int,
        end_layer: int,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        sync_groups: list[torch.distributed.ProcessGroup],
        offload_group: torch.distributed.ProcessGroup,
    ) -> Tuple[
        Union[torch.Tensor, IntermediateTensors],
        bool,
        "ModelInputForGPUWithSamplingMetadata",
    ]:

        with self.recv_condition:
            self.recv_batch_queue.append(
                (attn, start_layer, end_layer, model_input, kv_caches)
            )
            self.recv_condition.notify()

        f = self.recv_handler_thread_executor.submit(self._recv_handler, sync_groups, offload_group)
        f.add_done_callback(handle_error)
        return torch.tensor([0]), True, model_input

    def _recv_handler(
        self,
        sync_groups: list[torch.distributed.ProcessGroup],
        offload_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        """独立线程, 循环接收 KV Cache"""
        with self.recv_condition:
            self.recv_condition.wait_for(lambda: len(self.recv_batch_queue) > 0)
            attn, start_layer, end_layer, model_input, kv_caches = (
                self.recv_batch_queue.pop(0)
            )

        with torch.cuda.stream(self.recv_stream):
            self._recv_one_batch(
                attn, start_layer, end_layer, model_input, kv_caches, sync_groups, offload_group
            )

    def _recv_and_save_kvcache(
        self,
        attn: torch.nn.Module,
        start_layer: int,
        end_layer: int,
        kv_caches: list[torch.Tensor],
        current_tokens: torch.Tensor,
        slot_mapping: torch.Tensor,
        request_id: str,
    ):
        assert self.recv_buffer is not None, "self.recv_buffer is None"
        ret = self.recv_buffer.drop_select(current_tokens, request_id)
        assert ret[0] is not None, "no match kvcache"
        input_tokens: torch.Tensor = ret[0]
        keys: torch.Tensor = ret[1]
        values: torch.Tensor = ret[2]
        hidden: torch.Tensor = ret[3]

        num_recvd_tokens = input_tokens.shape[0]
        assert current_tokens.size(0) == num_recvd_tokens, f"{current_tokens.size(0)=} != {num_recvd_tokens=}"
        assert hidden is not None

        # put received KV caches into paged memory
        for i in range(start_layer, end_layer):
            kv_cache = kv_caches[i - start_layer]
            # FIXME: assert using flashinfer
            # key_cache, value_cache = kv_cache[0], kv_cache[1]
            key_cache, value_cache = kv_cache[:, 0], kv_cache[:, 1]
            ops.reshape_and_cache_flash(
                keys[i - start_layer],
                values[i - start_layer],
                key_cache,
                value_cache,
                slot_mapping,
                attn.kv_cache_dtype,
                attn._k_scale,
                attn._v_scale,
            )

        return hidden

    def _recv_one_batch(
        self,
        attn: torch.nn.Module,
        start_layer: int,
        end_layer: int,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
        sync_groups: list[torch.distributed.ProcessGroup],
        offload_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Tuple[
        Union[torch.Tensor, IntermediateTensors],
        bool,
        "ModelInputForGPUWithSamplingMetadata",
    ]:
        # passed in seq_lens including all offloaded requests
        # thus we need to filter out the local requests
        # TODO: fix prepare_model_input to avoid this,
        # however it is related to sampling metadata construction
        taken_reqs = [
            idx for idx, rank in enumerate(model_input.prefill_ranks)
            if rank == get_attn_rank()
        ]

        req_start_idx, req_end_idx = None, None
        rank_start_pos = None
        if len(taken_reqs) > 0:
            req_start_idx, req_end_idx = taken_reqs[0], taken_reqs[-1]
            rank_start_pos = sum(model_input.seq_lens[:req_start_idx])

        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
        # enumerate different requests
        # FIXME(Kuntai): This impl assumes that all requests are prefill.
        num_reqs = len(model_input.request_ids)
        assert ENABLE_KV_OFFLOAD
        assert offload_group is not None
        for i in range(num_reqs):
            if len(taken_reqs) == 0 or not (i >= req_start_idx and i <= req_end_idx):
                torch.distributed.barrier(offload_group)
                continue

            seqlen = model_input.seq_lens[i]
            start_pos = sum(model_input.seq_lens[:i])
            end_pos = start_pos + seqlen
            current_tokens = model_input.input_tokens[start_pos:end_pos]
            req_slot_mapping = slot_mapping[
                start_pos - rank_start_pos : end_pos - rank_start_pos
            ]
            request_id = model_input.request_ids[i]
            hidden = self._recv_and_save_kvcache(
                attn,
                start_layer,
                end_layer,
                kv_caches,
                current_tokens,
                req_slot_mapping,
                request_id,
            )

            current_stream = torch.cuda.current_stream()
            assert current_stream != torch.cuda.default_stream()
            current_stream.synchronize()

            with self.received_requests_lock:
                self.received_requests[request_id] = hidden
                for sync_group in sync_groups:
                    torch.distributed.barrier(sync_group)
                torch.distributed.barrier(offload_group)

        logger.debug(
            f"[rank{torch.distributed.get_rank()}] Successfully received all KVs and hidden states, skip model forwarding."
        )
        # bypass hidden_or_intermediate_states in prefill schedule iteration
        # rebuilid hidden_or_intermediate_states in prefill ready iteration

        return torch.tensor([0]), True, model_input

    def extract_hidden_for_ready_requests(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
    ):
        hidden_for_one_request: list[Union[torch.Tensor, IntermediateTensors]] = []
        for request_id, prefill_rank in zip(model_input.request_ids, model_input.prefill_ranks):
            if prefill_rank != get_attn_rank():
                continue
            with self.received_requests_lock:
                hidden = self.received_requests.pop(request_id)
            hidden_for_one_request.append(hidden)

        hidden_or_intermediate_states = None
        if len(hidden_for_one_request) > 0:
            hidden_or_intermediate_states: torch.Tensor = torch.cat(hidden_for_one_request, dim=0)
        return hidden_or_intermediate_states, True, model_input
