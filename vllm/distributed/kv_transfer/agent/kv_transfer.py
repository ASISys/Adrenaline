"""vLLM distributed KV cache transfer API.
These APIs are used in `vllm/worker/model_runner.py`.

Currently supporting TP. The TP between prefill and decode instance needs to be 
the same.

Workflow (disaggregated prefill)
- In prefill instance
    - After prefill, vLLM `insert` its KV caches into a lookup buffer.
    - The prefill instance will also open up a thread that listens to 
      `drop_select` request.
- In decode instance
    - vLLM first runs `drop_select` to send input tokens and a mask on input 
      tokens (we call it roi, region of interest) to prefill instance
    - The prefill instance then respond to `drop_select` request by
        - Finding a match in current lookup buffer.
        - Clone and send the matched item out
        - Delete the matched item in the lookup buffer to free up GPU memory.
    - The decode vLLM then store the KV cache into paged memory.
"""
import threading
import time
import torch
import traceback

from concurrent.futures import ThreadPoolExecutor, Future
from copy import deepcopy

from torch.distributed import Backend

from vllm import _custom_ops as ops
from vllm.distributed.utils import TransferUtils
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer import SimpleKVLookupBuffer
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import KVLookupBufferBase
from vllm.distributed.kv_transfer.kv_pipe.torch_distributed_pipe import (
    TorchDistributedPipe,
)
from vllm.distributed.kv_transfer.kv_pipe.py_pipe import PyPipe
from vllm.distributed.kv_transfer.vllm_adapter import *
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Dict

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


def build_partial_prefill_input(
    model_input: "ModelInputForGPUWithSamplingMetadata",
    input_tokens_list: List[torch.Tensor],
    num_computed_tokens_list: List[int],
    start_pos_list: List[int],
    slot_mapping_flat: torch.Tensor,
    device: torch.device,
) -> "ModelInputForGPUWithSamplingMetadata":
    """
    Helper function to rebuild the model input for the current request.
    Goal: avoid running redundant prefill on those tokens that already has KV 
    caches received.
    """
    rebuilt_input_tokens = []
    rebuilt_input_positions = []
    rebuilt_query_lens = []

    rebuilt_num_prefills = 0
    rebuilt_num_prefill_tokens = 0
    rebuilt_slot_mapping = []
    rebuilt_max_query_len = 0

    rebuilt_block_tables = []

    rebuilt_query_start_loc = [0]
    rebuilt_context_lens_tensor = []
    rebuilt_selected_token_indices = []

    # recounting query and context lengths
    for idx in range(len(input_tokens_list)):
        token_tensor = input_tokens_list[idx]
        num_token = len(token_tensor)
        num_computed_token = num_computed_tokens_list[idx]
        # currently attention kernel cannot handle the case where there is 0
        # query token.
        if num_computed_token == num_token:
            num_computed_token -= 1
        start_pos = start_pos_list[idx]

        rebuilt_input_tokens.append(token_tensor[num_computed_token:])
        # TODO(Jiayi): please check the correctness of next line
        rebuilt_input_positions.append(
            model_input.input_positions[start_pos +
                                        num_computed_token:start_pos +
                                        num_token])
        q_len = num_token - num_computed_token
        rebuilt_query_lens.append(q_len)

        # Attn metadata-related
        rebuilt_num_prefills += 1
        rebuilt_num_prefill_tokens += q_len
        new_slot_mapping = slot_mapping_flat[start_pos +
                                             num_computed_token:start_pos +
                                             num_token]
        rebuilt_slot_mapping.append(new_slot_mapping)
        rebuilt_max_query_len = max(q_len, rebuilt_max_query_len)
        # TODO(Jiayi): remove hard-code (block_size=16)
        blk_size = 16
        temp_block_table = [
            slot_mapping_flat[i] // blk_size
            for i in range(start_pos, start_pos + num_token, blk_size)
        ]
        rebuilt_block_tables.append(temp_block_table)
        rebuilt_query_start_loc.append(
            rebuilt_num_prefill_tokens)  #start with 0
        rebuilt_context_lens_tensor.append(num_computed_token)

        # Sampling metadata related
        #seq_groups (use rebuilt query lens)
        rebuilt_selected_token_indices.append(rebuilt_num_prefill_tokens - 1)

    # rebuilt attn_metadata
    rebuilt_attn_metadata = deepcopy(model_input.attn_metadata)
    rebuilt_attn_metadata.num_prefills = rebuilt_num_prefills
    rebuilt_attn_metadata.num_prefill_tokens = rebuilt_num_prefill_tokens
    rebuilt_attn_metadata.slot_mapping = torch.cat(rebuilt_slot_mapping).to(
        device)
    rebuilt_attn_metadata.max_query_len = rebuilt_max_query_len

    rebuilt_attn_metadata.block_tables = torch.tensor(
        rebuilt_block_tables,
        dtype=model_input.attn_metadata.block_tables.dtype).to(device)

    rebuilt_attn_metadata.query_start_loc = torch.tensor(
        rebuilt_query_start_loc,
        dtype=model_input.attn_metadata.query_start_loc.dtype).to(device)
    rebuilt_attn_metadata.context_lens_tensor = torch.tensor(
        rebuilt_context_lens_tensor,
        dtype=model_input.attn_metadata.context_lens_tensor.dtype,
    ).to(device)

    rebuilt_attn_metadata._cached_prefill_metadata = None

    # rebuilt sampling_metadata
    rebuilt_sampling_metadata = deepcopy(model_input.sampling_metadata)
    for idx, q_len in enumerate(rebuilt_query_lens):
        if rebuilt_sampling_metadata.seq_groups is not None:
            rebuilt_sampling_metadata.seq_groups[idx].query_len = q_len

    rebuilt_sampling_metadata.selected_token_indices = torch.tensor(
        rebuilt_selected_token_indices,
        dtype=model_input.sampling_metadata.selected_token_indices.dtype,
    ).to(device)

    # import here to avoid circular import.
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
    rebuilt_model_input = ModelInputForGPUWithSamplingMetadata(
        input_tokens=torch.cat(rebuilt_input_tokens).to(device),
        input_positions=torch.cat(rebuilt_input_positions).to(device),
        seq_lens=model_input.seq_lens,
        query_lens=rebuilt_query_lens,
        lora_mapping=model_input.lora_mapping,
        lora_requests=model_input.lora_requests,
        attn_metadata=rebuilt_attn_metadata,
        prompt_adapter_mapping=model_input.prompt_adapter_mapping,
        prompt_adapter_requests=model_input.prompt_adapter_requests,
        multi_modal_kwargs=model_input.multi_modal_kwargs,
        request_ids_to_seq_ids=model_input.request_ids_to_seq_ids,
        finished_requests_ids=model_input.finished_requests_ids,
        virtual_engine=model_input.virtual_engine,
        sampling_metadata=rebuilt_sampling_metadata,
        is_prompt=model_input.is_prompt,
    )

    return rebuilt_model_input


def handle_error(f: Future):
    exception = f.exception()
    if exception:
        logger.error(f"[rank{torch.distributed.get_rank()}] {exception}")
        traceback.print_exception(exception)


class KV_transfer_agent:
    """
    A class designated for distributed KV transfer
    
    Target use cases:
        1. Disaggregated prefill
        2. Remote KV cache storage
    """

    def __init__(
        self,
        group_ranks: list[list[list[int]]],
        local_rank: int,
        dist_backend: Union[str, Backend] = DISTRIBUTED_BACKEND,
        # FIXME(Kuntai): remove this hardcoding
        lookup_buffer_size: int = int(4.5e9),
        transfer_utils: Optional[TransferUtils] = None,
    ):

        self.lookup_buffer_size = lookup_buffer_size

        self.send_buffer: Optional[KVLookupBufferBase] = None
        self.recv_buffer: Optional[KVLookupBufferBase] = None

        producer_send_pipes = [
            PyPipe(group_rank_per_instance, local_rank, dist_backend)
            for group_rank_per_instance in group_ranks
        ]
        consumer_send_pipes = [
            PyPipe(group_rank_per_instance, local_rank, dist_backend)
            for group_rank_per_instance in group_ranks
        ]
        producer_send_signal_pipes = [
            TorchDistributedPipe(group_rank_per_instance, local_rank, "gloo")
            for group_rank_per_instance in group_ranks
        ]
        consumer_send_signal_pipes = [
            TorchDistributedPipe(group_rank_per_instance, local_rank, "gloo")
            for group_rank_per_instance in group_ranks
        ]

        self.recv_condition = threading.Condition()
        self.recv_batch_queue: list[tuple[
            torch.nn.Module,
            ModelInputForGPUWithSamplingMetadata,
            list[torch.Tensor],
        ]] = []
        self.recv_handler_thread_executor = ThreadPoolExecutor(max_workers=1)

        self.received_requests_lock = threading.Lock()
        self.received_requests: Dict[str, torch.Tensor] = {}

        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        if IS_KV_PRODUCER:
            self.send_pipes: list[PyPipe] = []
            for pipe in producer_send_pipes:
                if pipe.device_group is not None:
                    self.send_pipes.append(pipe)
            self.send_pipe = self.send_pipes[0] 

            self.send_signal_pipes: list[TorchDistributedPipe] = []
            for pipe in producer_send_signal_pipes:
                if pipe.device_group is not None:
                    self.send_signal_pipes.append(pipe)
            self.send_signal_pipe = self.send_signal_pipes[0]

            self.recv_pipes: list[PyPipe] = []
            for pipe in consumer_send_pipes:
                if pipe.device_group is not None:
                    self.recv_pipes.append(pipe)
            self.recv_pipe = self.recv_pipes[0]

            self.recv_signal_pipes: list[TorchDistributedPipe] = []
            for pipe in consumer_send_signal_pipes:
                if pipe.device_group is not None:
                    self.recv_signal_pipes.append(pipe)
            self.recv_signal_pipe = self.recv_signal_pipes[0]

            self.send_buffer = SimpleKVLookupBuffer(
                self.send_signal_pipe,
                self.send_pipe,
                self.lookup_buffer_size,
                self.send_signal_pipes,
                self.send_pipes,
                self.send_stream,
                transfer_utils=transfer_utils,
            )
        elif IS_KV_CONSUMER:
            self.recv_pipes = []
            for pipe in producer_send_pipes:
                if pipe.device_group is not None:
                    self.recv_pipes.append(pipe)
            self.recv_pipe = self.recv_pipes[0]

            self.recv_signal_pipes = []
            for pipe in producer_send_signal_pipes:
                if pipe.device_group is not None:
                    self.recv_signal_pipes.append(pipe)
            self.recv_signal_pipe = self.recv_signal_pipes[0]

            self.send_pipes = []
            for pipe in consumer_send_pipes:
                if pipe.device_group is not None:
                    self.send_pipes.append(pipe)
            self.send_pipe = self.send_pipes[0]

            self.send_signal_pipes = []
            for pipe in consumer_send_signal_pipes:
                if pipe.device_group is not None:
                    self.send_signal_pipes.append(pipe)
            self.send_signal_pipe = self.send_signal_pipes[0]

            self.recv_buffer = SimpleKVLookupBuffer(
                self.recv_signal_pipe,
                self.recv_pipe,
                self.lookup_buffer_size,
                self.recv_signal_pipes,
                self.recv_pipes,
                transfer_utils=transfer_utils,
            )

    def add_kv_cache_and_hidden_states_to_buffer(
        self,
        model_executable: torch.nn.Module,
        kv_caches: List[torch.Tensor],
        slot_mapping: torch.Tensor,
        input_tokens: torch.Tensor,
        hidden_or_intermediate_states: torch.Tensor,
        request_id: str,
    ):
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer

        keys, values = [], []
        attn = model_executable.model.layers[0].self_attn.attn
        from vllm.attention.backends.flashinfer import FlashInferImpl
        from adrenaline.attention.backends.flashinfer import AdrenalineFlashInferImpl
        for layer_id in range(start_layer, end_layer):
            kv_cache = kv_caches[layer_id - start_layer]

            if isinstance(attn.impl, AdrenalineFlashInferImpl) or isinstance(
                attn.impl, FlashInferImpl
            ):
                _, _, num_heads, head_size = kv_cache[:, 0].shape
                key_cache = kv_cache[:, 0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[:, 1].reshape(-1, num_heads, head_size)
            else:
                _, _, num_heads, head_size = kv_cache[0].shape
                key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

            keys.append(key_cache[slot_mapping].unsqueeze(0))
            values.append(value_cache[slot_mapping].unsqueeze(0))

        keys = torch.cat(keys, dim=0)
        values = torch.cat(values, dim=0)
        if self.send_buffer is not None:
            self.send_buffer.insert(
                input_tokens,
                keys,
                values,
                hidden_or_intermediate_states,
                request_id,
            )

    def add_hidden_states_to_buffer(
        self,
        input_tokens: torch.Tensor,
        hidden_or_intermediate_states: torch.Tensor,
        request_id: str,
    ):
        if self.send_buffer is not None:
            self.send_buffer.insert(
                input_tokens,
                None,
                None,
                hidden_or_intermediate_states,
                request_id,
            )

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        seq_lens = model_input.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        input_tokens_tensor = model_input.input_tokens

        # query_lens contains new KV caches that are added to vLLM.
        # so we will send them to decode instance
        # FIXME(Kuntai): This assume that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            request_id = model_input.request_ids[idx]
            slot_mapping = slot_mapping_flat[start_pos:end_pos]
            input_tokens = input_tokens_tensor[start_pos:end_pos]
            hidden_or_intermediate_state = hidden_or_intermediate_states[start_pos:end_pos]

            if ENABLE_KV_OFFLOAD and idx >= model_input.attn_metadata.num_locals:
                self.add_hidden_states_to_buffer(
                    input_tokens, hidden_or_intermediate_state, request_id
                )
            else:
                self.add_kv_cache_and_hidden_states_to_buffer(
                    model_executable, kv_caches, slot_mapping, input_tokens, hidden_or_intermediate_state, request_id
                )

            logger.debug(
                f"[rank{torch.distributed.get_rank()}] Add KvCache {request_id=} to buffer DONE."
            )

    def destroy(self) -> None:
        if self.send_buffer is not None:
            self.send_buffer.close()
        if self.recv_buffer is not None:
            self.recv_buffer.close()
        self.recv_handler_thread_executor.shutdown()

    def async_recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        sync_groups: list[torch.distributed.ProcessGroup],
        offload_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Tuple[
        Union[torch.Tensor, IntermediateTensors],
        bool,
        "ModelInputForGPUWithSamplingMetadata",
    ]:
        with self.recv_condition:
            self.recv_batch_queue.append((model_executable, model_input, kv_caches))
            self.recv_condition.notify()

        f = self.recv_handler_thread_executor.submit(self._recv_handler, sync_groups, offload_group)
        f.add_done_callback(handle_error)
        return torch.tensor([0]), True, model_input

    def _recv_handler(
        self,
        sync_groups: list[torch.distributed.ProcessGroup],
        offload_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        with self.recv_condition:
            self.recv_condition.wait_for(lambda: len(self.recv_batch_queue) > 0)
            model_executable, model_input, kv_caches = self.recv_batch_queue.pop(0)

        with torch.cuda.stream(self.recv_stream):
            self._recv_one_batch(model_executable, model_input, kv_caches, sync_groups, offload_group)

    def _recv_one_batch(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        sync_groups: list[torch.distributed.ProcessGroup],
        offload_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Tuple[
        Union[torch.Tensor, IntermediateTensors],
        bool,
        "ModelInputForGPUWithSamplingMetadata",
    ]:

        current_stream = torch.cuda.current_stream()
        assert current_stream != torch.cuda.default_stream()

        # When this flag is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()


        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []

        # enumerate different requests
        # FIXME(Kuntai): This impl assumes that all requests are prefill.
        for idx, slen in enumerate(seq_lens):

            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]
            num_tokens = slen

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            assert self.recv_buffer is not None

            prefill_rank = model_input.prefill_ranks[idx]
            request_id = model_input.request_ids[idx]
            is_offload = ENABLE_KV_OFFLOAD and IS_KV_CONSUMER and idx >= model_input.attn_metadata.num_locals

            ret = self.recv_buffer.drop_select(
                current_tokens,
                recv_rank=prefill_rank,
                is_offload=is_offload,
            )
            assert ret[0] is not None, "no match kvcache"

            input_tokens: torch.Tensor = ret[0]
            keys: torch.Tensor = ret[1]
            values: torch.Tensor = ret[2]
            hidden: torch.Tensor = ret[3]

            num_computed_tokens = input_tokens.shape[0]
            num_computed_tokens_list.append(num_computed_tokens)

            # assert num_computed_tokens == slen
            # assert hidden is not None

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            assert num_computed_tokens == num_tokens
            assert hidden is not None

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            if not is_offload:
                # put received KV caches into paged memory
                for i in range(model_executable.model.start_layer, model_executable.model.end_layer):
                    kv_cache = kv_caches[i - model_executable.model.start_layer]
                    layer = model_executable.model.layers[i]

                    # FIXME: assert using flashinfer
                    # key_cache, value_cache = kv_cache[0], kv_cache[1]
                    key_cache, value_cache = kv_cache[:, 0], kv_cache[:, 1]
                    ops.reshape_and_cache_flash(
                        keys[i - model_executable.model.start_layer],
                        values[i - model_executable.model.start_layer],
                        key_cache,
                        value_cache,
                        slot_mapping[start_pos:end_pos],
                        layer.self_attn.attn.kv_cache_dtype,
                        layer.self_attn.attn._k_scale,
                        layer.self_attn.attn._v_scale,
                    )

            current_stream.synchronize()

            with self.received_requests_lock:
                self.received_requests[request_id] = hidden
                for sync_group in sync_groups:
                    torch.distributed.barrier(sync_group)
                is_offload = idx >= model_input.attn_metadata.num_locals
                if ENABLE_KV_OFFLOAD and offload_group is not None and is_offload:
                    torch.distributed.barrier(offload_group)

        assert bypass_model_exec, "bypass_model_exec should be True"
        logger.debug(
            f"[rank{torch.distributed.get_rank()}] Successfully received all KVs and hidden "
            "states, skip model forwarding.")
        # bypass hidden_or_intermediate_states in prefill schedule iteration
        # rebuilid hidden_or_intermediate_states in prefill ready iteration

        return torch.tensor([0]), True, model_input

    def extract_hidden_for_ready_requests(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
    ) -> Tuple[
        Union[torch.Tensor, IntermediateTensors], bool, "ModelInputForGPUWithSamplingMetadata"
    ]:
        hidden_for_one_request = []
        for request_id in model_input.request_ids:
            with self.received_requests_lock:
                hidden = self.received_requests.pop(request_id)
            hidden_for_one_request.append(hidden)
            # FIXME? send hidden maybe has multiple tokens

        hidden_or_intermediate_states = torch.cat(hidden_for_one_request, dim=0)

        return hidden_or_intermediate_states, True, model_input

    def query_recv_finished(self, request_id: str, blocking: bool = False) -> bool:
        if self.received_requests_lock.acquire(blocking=blocking):
            is_recv_fin = request_id in self.received_requests
            self.received_requests_lock.release()
        else:
            is_recv_fin = False
        return is_recv_fin
