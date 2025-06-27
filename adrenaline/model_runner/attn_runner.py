from vllm.attention.selector import global_force_attn_backend, backend_name_to_enum
from vllm.distributed.parallel_state import (
    get_offload_kv_group,
    get_offload_group,
    get_tp_group,
    get_pp_group,
)
from vllm.model_executor.layers.sampler import get_sampler, Sampler
from vllm.worker.model_runner import *
from vllm.worker.model_runner import _BATCH_SIZES_TO_CAPTURE, _NUM_WARMUP_ITERS, _BATCH_SIZE_ALIGNMENT

from adrenaline.attention.backends.flashinfer import AdrenalineFlashInferBackend, AdrenalineFlashInferState
from adrenaline.config import OFFLOAD_MAX_CAPTURE_BATCH
from adrenaline.model_loader.models.offload_attn import OffloadAttn

_OFFLOAD_BATCH_SIZES_TO_CAPTURE = _BATCH_SIZES_TO_CAPTURE

class AttentionRunner(ModelRunner):
    def load_model(self):
        assert dist_kv.ENABLE_KV_OFFLOAD
        global_force_attn_backend(
            backend_name_to_enum(AdrenalineFlashInferBackend.get_name())
        )
        self.model = OffloadAttn(vllm_config=self.vllm_config)
        self.sampler: Sampler = get_sampler()

    def capture_model(self, kv_caches):
        assert not self.model_config.enforce_eager
        logger.info("Capturing cudagraphs for decoding. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("If out-of-memory error occurs during cudagraph capture,"
                    " consider decreasing `gpu_memory_utilization` or "
                    "switching to eager mode. You can also reduce the "
                    "`max_num_seqs` as needed to decrease memory usage.")
        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Prepare dummy inputs. These will be reused for all batch sizes.
        logger.debug(f"Capuring offload cudagraph with {OFFLOAD_MAX_CAPTURE_BATCH=}")
        max_batch_size = self.max_batchsize_to_capture
        graph_batch_size = self.max_batchsize_to_capture
        batch_size_capture_list = [0] + [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
        ]
        offload_batch_size_capture_list = [
            bs for bs in _OFFLOAD_BATCH_SIZES_TO_CAPTURE if bs <= OFFLOAD_MAX_CAPTURE_BATCH + _BATCH_SIZE_ALIGNMENT - 1
        ]

        with self.attn_state.graph_capture(
            max_batch_size
        ), graph_capture() as graph_capture_context:
            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of CUDA graph.
            assert isinstance(self.attn_state, AdrenalineFlashInferState)
            for virtual_engine in range(self.parallel_config.pipeline_parallel_size):
                for num_locals, num_offloads in itertools.product(
                    reversed(batch_size_capture_list), reversed(offload_batch_size_capture_list)
                ):
                    logger.debug(
                        f"Capturing offload cudagraph for {num_locals=}, {num_offloads=}"
                    )
                    batch_size = num_offloads
                    assert batch_size > 0
                    attn_metadata = self.attn_state.graph_capture_get_metadata_for_offload(
                        batch_size=num_offloads,
                        is_encoder_decoder_model=self.model_config.is_encoder_decoder_model,
                        num_locals=num_locals,
                        num_offloads=num_offloads,
                    )

                    if self.lora_config:
                        lora_mapping = LoRAMapping(
                            **dict(index_mapping=[0] * batch_size,
                                   prompt_mapping=[0] * batch_size,
                                   is_prefill=False))
                        self.set_active_loras(set(), lora_mapping)

                    if self.prompt_adapter_config:
                        prompt_adapter_mapping = PromptAdapterMapping(
                            [-1] * batch_size,
                            [-1] * batch_size,
                        )
                        self.set_active_prompt_adapters(
                            set(), prompt_adapter_mapping)
                    graph_runner = OffloadCUDAGraphRunnerForAttnRunner(
                        self.model,
                        self.attn_backend.get_name(),
                        self.attn_state.graph_clone(batch_size),
                        self.model_config.is_encoder_decoder_model,
                    )

                    capture_inputs = {
                        "kv_caches": kv_caches[virtual_engine],
                        "attn_metadata": attn_metadata,
                        "memory_pool": self.graph_memory_pool,
                        "stream": graph_capture_context.stream,
                    }
                    if self.has_inner_state:
                        # Only used by Mamba-based models CUDA graph atm (Jamba)
                        capture_inputs.update({
                            "seqlen_agnostic_capture_inputs":
                            self.model.get_seqlen_agnostic_capture_inputs(batch_size)
                        })
                    if self.model_config.is_encoder_decoder_model:
                        # add the additional inputs to capture for
                        # encoder-decoder models.
                        self._update_inputs_to_capture_for_enc_dec_model(
                            capture_inputs)

                    with set_forward_context(attn_metadata):
                        graph_runner.capture(**capture_inputs)
                    self.graph_memory_pool = graph_runner.graph.pool()
                    if batch_size not in self.graph_runners[virtual_engine]:
                        self.graph_runners[virtual_engine][batch_size] = graph_runner
                    torch.distributed.barrier(get_offload_group().sync_comm.device_group)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes < 10 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / GiB_bytes)

    def profile_run(self):
        return

    @torch.inference_mode()
    @dump_input_when_exception(exclude_args=[0], exclude_kwargs=["self"])
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests,
                model_input.prompt_adapter_mapping)

        self.attn_state.begin_forward(model_input)

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.attn_metadata.num_offloads_padded_for_graph
            model_executable = self.graph_runners[virtual_engine][graph_batch_size]
        else:
            model_executable = self.model

        # Receive KV cache in distributed KV cache transfer setting
        # In disagg prefill setting, it will also recv hidden states and bypass
        # model forwarding
        # In KV cache database setting, it will change the model input so that
        # we can skip prefilling on tokens that successfully received KV caches
        # NOTE: The receive operation is blocking
        bypass_model_exec = False
        bypass_output_process = False
        if self.is_prefill_sche_iteration(model_input, kv_caches):
            assert isinstance(self.model, OffloadAttn)
            sync_groups = [get_tp_group().recving_sync_group, get_pp_group().recving_sync_group]
            offload_group = get_offload_group().sync_comm.device_group
            hidden_or_intermediate_states, bypass_model_exec, model_input = (
                get_offload_kv_group().async_recv_kv_caches_and_hidden_states(
                    # model is used to know which layer the current worker
                    # is working on, so that we can receive KV for only those
                    # layers.
                    self.model.model,
                    self.model.start_layer,
                    self.model.end_layer,
                    model_input,
                    kv_caches,
                    sync_groups,
                    offload_group,
                )
            )
            bypass_output_process = True
        elif self.is_prefill_ready_iteration(model_input):
            hidden_or_intermediate_states, bypass_model_exec, model_input = (
                get_offload_kv_group().extract_hidden_for_ready_requests(model_input)
            )
            bypass_output_process = False
            sync_groups = [get_tp_group().fin_recv_sync_group, get_pp_group().fin_recv_sync_group]
            for sync_group in sync_groups:
                torch.distributed.barrier(sync_group)
            torch.distributed.barrier(get_offload_group().sche_comm.device_group)

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_start = torch.cuda.Event(enable_timing=True)
            model_forward_end = torch.cuda.Event(enable_timing=True)
            model_forward_start.record()

        if not bypass_model_exec:
            assert dist_kv.IS_ATTN_EXECUTOR
            with set_forward_context(model_input.attn_metadata):
                model_executable.forward(kv_caches, model_input.attn_metadata)

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end.record()

        # Sending KV cache in distributed KV cache transfer setting
        # NOTE: the send operation is non-blocking
        if self.need_send_kv(model_input, kv_caches):
            # extract_hidden_states after recving finished
            get_disagg_group().send_kv_caches_and_hidden_states(
                # model_executable is used to know which layer the current
                # worker is working on, so that we can send KV for only those
                # layers.
                model_executable,
                model_input,
                kv_caches,
                hidden_or_intermediate_states,
            )

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            if (self.is_driver_worker
                    and hidden_or_intermediate_states is not None
                    and isinstance(hidden_or_intermediate_states,
                                   IntermediateTensors)
                    and self.observability_config is not None
                    and self.observability_config.collect_model_forward_time):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(
                    model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)).item()
                hidden_or_intermediate_states.tensors["model_forward_time"] = (
                    torch.tensor(model_forward_time + orig_model_forward_time))
            return hidden_or_intermediate_states

        if bypass_output_process:
            if model_input.async_callback is not None:
                model_input.async_callback()
            return []

        if not self.is_driver_worker:
            return []

        logits = torch.empty(
            model_input.sampling_metadata.selected_token_indices.shape[0],
            self.model_config.hf_config.vocab_size,
            dtype=self.model.model_dtype,
            device=self.device,
        )
        get_offload_group().broadcast_tensor_to_offload(logits)

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        output: SamplerOutput = self.sampler(logits, model_input.sampling_metadata)
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time
                and output is not None):
            model_forward_end.synchronize()
            model_forward_time = model_forward_start.elapsed_time(
                model_forward_end)
            orig_model_forward_time = 0.0
            if intermediate_tensors is not None:
                orig_model_forward_time = intermediate_tensors.tensors.get(
                    "model_forward_time", torch.tensor(0.0)).item()
            # If there are multiple workers, we are still tracking the latency
            # from the start time of the driver worker to the end time of the
            # driver worker. The model forward time will then end up covering
            # the communication time as well.
            output.model_forward_time = (orig_model_forward_time +
                                         model_forward_time)

        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            assert model_input.sampling_metadata is not None
            indices = model_input.sampling_metadata.selected_token_indices
            if model_input.is_prompt:
                hidden_states = hidden_or_intermediate_states.index_select(
                    0, indices)
                output.prefill_hidden_states = hidden_or_intermediate_states
            elif decode_meta.use_cuda_graph:
                hidden_states = hidden_or_intermediate_states[:len(indices)]
            else:
                hidden_states = hidden_or_intermediate_states

            output.hidden_states = hidden_states

        return [output]

    def is_prefill_sche_iteration(self, model_input, kv_caches) -> bool:
        """Check if we need to receive kv-cache from the other worker.
        We need to receive KV when
            1. current vLLM instance is KV cache consumer/decode vLLM instance
            2. this batch is not a profiling run
            3. this batch is a prefill run
            
        Args:
            model_input: input to the model executable
            kv_caches: vLLM's paged memory
        """

        # check if the current run is profiling
        is_profile_run = (kv_caches[0].numel() == 0)
        # check if the current run is prefill
        is_prefill_run = model_input.is_prompt
        is_recving_finished = self.is_prefill_ready_iteration(model_input)

        return (
            (dist_kv.IS_KV_CONSUMER or dist_kv.IS_ATTN_EXECUTOR)
            and (not is_profile_run)
            and is_prefill_run
            and (not is_recving_finished)
        )

class OffloadCUDAGraphRunnerForAttnRunner(nn.Module):

    def __init__(self, model: nn.Module, backend_name: str,
                 attn_state: AttentionState, is_encoder_decoder_model: bool):
        super().__init__()
        self.model = model
        self.backend_name = backend_name
        self.attn_state = attn_state

        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._is_encoder_decoder_model = is_encoder_decoder_model

    @property
    def graph(self):
        assert self._graph is not None
        return self._graph

    def capture(
        self,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        memory_pool: Optional[Tuple[int, int]],
        stream: torch.cuda.Stream,
        **kwargs,
    ):
        assert self._graph is None
        # Run the model a few times without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        # Note one iteration is not enough for torch.jit.script
        assert dist_kv.IS_ATTN_EXECUTOR
        for _ in range(_NUM_WARMUP_ITERS):
            self.model(kv_caches, attn_metadata)

        # Wait for the warm up operations to finish before proceeding with
        # Graph Capture.
        torch.cuda.synchronize()
        # Capture the graph.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=memory_pool, stream=stream):
            self.model(kv_caches, attn_metadata)
            gc.collect()
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "kv_caches": kv_caches,
            **self.attn_state.get_graph_input_buffers(
                attn_metadata, self._is_encoder_decoder_model),
            **kwargs,
        }

    def forward(
        self,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        if self.backend_name != "NO_ATTENTION":
            self.input_buffers["slot_mapping"].copy_(
                attn_metadata.slot_mapping, non_blocking=True)

        self.attn_state.prepare_graph_input_buffers(
            self.input_buffers, attn_metadata, self._is_encoder_decoder_model)

        if "seqlen_agnostic_capture_inputs" in self.input_buffers:
            self.model.copy_inputs_before_cuda_graphs(self.input_buffers,
                                                      **kwargs)

        if self._is_encoder_decoder_model:
            self.input_buffers["encoder_input_ids"].copy_(
                kwargs['encoder_input_ids'], non_blocking=True)
            self.input_buffers["encoder_positions"].copy_(
                kwargs['encoder_positions'], non_blocking=True)

        # Run the graph.
        self.graph.replay()

        return self.output_buffers
