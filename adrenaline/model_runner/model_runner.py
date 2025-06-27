from vllm.attention.selector import global_force_attn_backend, backend_name_to_enum
from vllm.distributed.parallel_state import get_offload_kv_group, get_offload_group, get_tp_group, get_pp_group
from vllm.distributed.kv_transfer import vllm_adapter as dist_kv
from vllm.worker.model_runner import *
from vllm.worker.model_runner import _NUM_WARMUP_ITERS, _BATCH_SIZES_TO_CAPTURE, _BATCH_SIZE_ALIGNMENT
from vllm.worker.model_runner import ModelRunner as VllmModelRunner

from adrenaline.attention.backends.flashinfer import AdrenalineFlashInferBackend, AdrenalineFlashInferMetadata, AdrenalineFlashInferState
from adrenaline.config import OFFLOAD_MAX_CAPTURE_BATCH
from adrenaline.model_loader.model_loader import get_model

_OFFLOAD_BATCH_SIZES_TO_CAPTURE = _BATCH_SIZES_TO_CAPTURE

class OffloadCUDAGraphRunnerForDecode(nn.Module):

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
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_inputs: Optional[IntermediateTensors],
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
        for _ in range(_NUM_WARMUP_ITERS):
            self.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
                intermediate_tensors=intermediate_inputs,
                **kwargs,
            )
        # Wait for the warm up operations to finish before proceeding with
        # Graph Capture.
        torch.cuda.synchronize()
        # Capture the graph.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=memory_pool, stream=stream):
            output_hidden_or_intermediate_states = self.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
                intermediate_tensors=intermediate_inputs,
                **kwargs,
            )

            if isinstance(output_hidden_or_intermediate_states, torch.Tensor):
                hidden_or_intermediate_states = weak_ref_tensor(
                    output_hidden_or_intermediate_states)
            elif isinstance(output_hidden_or_intermediate_states,
                            IntermediateTensors):
                hidden_or_intermediate_states = IntermediateTensors(
                    tensors={
                        key: weak_ref_tensor(value)
                        for key, value in
                        output_hidden_or_intermediate_states.tensors.items()
                    })

            del output_hidden_or_intermediate_states
            # make sure `output_hidden_or_intermediate_states` is deleted
            # in the graph's memory pool
            gc.collect()
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            **self.attn_state.get_graph_input_buffers(
                attn_metadata, self._is_encoder_decoder_model),
            **kwargs,
        }
        if intermediate_inputs is not None:
            self.input_buffers.update(intermediate_inputs.tensors)
        if get_pp_group().is_last_rank:
            self.output_buffers = {
                "hidden_states": hidden_or_intermediate_states
            }
        else:
            self.output_buffers = hidden_or_intermediate_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        **kwargs,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)

        if self.backend_name != "NO_ATTENTION":
            self.input_buffers["slot_mapping"].copy_(
                attn_metadata.slot_mapping, non_blocking=True)

        self.attn_state.prepare_graph_input_buffers(
            self.input_buffers, attn_metadata, self._is_encoder_decoder_model)

        if "seqlen_agnostic_capture_inputs" in self.input_buffers:
            self.model.copy_inputs_before_cuda_graphs(self.input_buffers,
                                                      **kwargs)

        if "previous_hidden_states" in self.input_buffers:
            self.input_buffers["previous_hidden_states"].copy_(
                kwargs["previous_hidden_states"], non_blocking=True)

        if intermediate_tensors is not None:
            for key in intermediate_tensors.tensors:
                if key != "model_execute_time" and key != "model_forward_time":
                    self.input_buffers[key].copy_(intermediate_tensors[key],
                                                  non_blocking=True)
        if self._is_encoder_decoder_model:
            self.input_buffers["encoder_input_ids"].copy_(
                kwargs['encoder_input_ids'], non_blocking=True)
            self.input_buffers["encoder_positions"].copy_(
                kwargs['encoder_positions'], non_blocking=True)

        # Run the graph.
        self.graph.replay()
        # Return the output tensor.
        if get_pp_group().is_last_rank:
            return self.output_buffers["hidden_states"]

        return self.output_buffers

class ModelRunner(VllmModelRunner):
    offload_graph_runners: list[dict[tuple[int, int], OffloadCUDAGraphRunnerForDecode]]

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        if dist_kv.ENABLE_KV_OFFLOAD:
            global_force_attn_backend(
                backend_name_to_enum(AdrenalineFlashInferBackend.get_name())
            )
        with DeviceMemoryProfiler() as m:
            self.model = get_model(vllm_config=self.vllm_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

        if self.lora_config:
            assert supports_lora(
                self.model
            ), f"{self.model.__class__.__name__} does not support LoRA yet."

            if supports_multimodal(self.model):
                logger.warning("Regarding multimodal models, vLLM currently "
                               "only supports adding LoRA to language model.")
            # It's necessary to distinguish between the max_position_embeddings
            # of VLMs and LLMs.
            if hasattr(self.model.config, "max_position_embeddings"):
                max_pos_embeddings = self.model.config.max_position_embeddings
            else:
                max_pos_embeddings = (
                    self.model.config.text_config.max_position_embeddings)

            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
                self.vocab_size,
                self.lora_config,
                self.device,
                self.model.embedding_modules,
                self.model.embedding_padding_modules,
                max_position_embeddings=max_pos_embeddings,
            )
            self.model = self.lora_manager.create_lora_manager(self.model)

        if self.prompt_adapter_config:
            self.prompt_adapter_manager = LRUCacheWorkerPromptAdapterManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens, self.device,
                self.prompt_adapter_config)
            self.model = (
                self.prompt_adapter_manager.create_prompt_adapter_manager(
                    self.model))

        if self.kv_cache_dtype == "fp8" and current_platform.is_rocm():
            # Currently only ROCm accepts kv-cache scaling factors
            # via quantization_param_path and this will be deprecated
            # in the future.
            if self.model_config.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    warnings.warn(
                        "Loading kv cache scaling factor from JSON is "
                        "deprecated and will be removed. Please include "
                        "kv cache scaling factors in the model checkpoint.",
                        FutureWarning,
                        stacklevel=2)
                    self.model.load_kv_cache_scales(
                        self.model_config.quantization_param_path)
                    logger.info("Loaded KV cache scaling factors from %s",
                                self.model_config.quantization_param_path)
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__)
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!")

        if envs.VLLM_TORCH_COMPILE_LEVEL == CompilationLevel.DYNAMO_AS_IS \
            and supports_dynamo():
            from vllm.plugins import get_torch_compile_backend
            backend = get_torch_compile_backend() or "eager"
            self.model = torch.compile(
                self.model,
                fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=backend)

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[List[torch.Tensor]]) -> None:
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        if dist_kv.IS_KV_PRODUCER:
            return
        super().capture_model(kv_caches)
        self.capture_offload_model(kv_caches)

    @torch.inference_mode()
    def capture_offload_model(self, kv_caches: List[List[torch.Tensor]]) -> None:
        assert dist_kv.IS_KV_CONSUMER
        assert dist_kv.ENABLE_KV_OFFLOAD, (
            "ENABLE_KV_OFFLOAD and attention instance must be set to enable cuda graph,"
            "otherwise just use the vllm's ModelRunner"
        )
        self.offload_graph_runners = [
            {} for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

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
        gemm_max_batch_size = self.max_batchsize_to_capture * (
            dist_kv.NUM_DECODE_INSTANCES + dist_kv.NUM_ATTN_INSTANCES
        )
        attn_max_batch_size = self.max_batchsize_to_capture
        input_tokens = torch.zeros(gemm_max_batch_size, dtype=torch.long).cuda()
        input_positions = torch.zeros(gemm_max_batch_size, dtype=torch.long).cuda()
        if self.model_is_mrope:
            input_positions = torch.tile(input_positions, (3, 1))
        # Prepare dummy previous_hidden_states only if needed by the model.
        # This is used by draft models such as EAGLE.
        previous_hidden_states = None
        if "previous_hidden_states" in inspect.signature(
                self.model.forward).parameters:
            previous_hidden_states = torch.empty(
                [gemm_max_batch_size, self.model_config.get_hidden_size()],
                dtype=self.model_config.dtype,
                device=self.device,
            )

        intermediate_inputs = None
        if not get_pp_group().is_first_rank:
            intermediate_inputs = self.model.make_empty_intermediate_tensors(
                batch_size=gemm_max_batch_size,
                dtype=self.model_config.dtype,
                device=self.device)

        logger.debug(f"Capuring offload cudagraph with {OFFLOAD_MAX_CAPTURE_BATCH=}")
        graph_batch_size = self.max_batchsize_to_capture
        local_batch_size_capture_list = [0] + [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
        ]
        offload_batch_size_capture_list = [
            bs for bs in _OFFLOAD_BATCH_SIZES_TO_CAPTURE
            if bs <= OFFLOAD_MAX_CAPTURE_BATCH + _BATCH_SIZE_ALIGNMENT - 1
        ]

        with self.attn_state.graph_capture(
            attn_max_batch_size
        ), graph_capture() as graph_capture_context:
            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of CUDA graph.
            assert isinstance(self.attn_state, AdrenalineFlashInferState)
            for virtual_engine in range(self.parallel_config.pipeline_parallel_size):
                for num_locals, num_offloads in itertools.product(
                    reversed(local_batch_size_capture_list), reversed(offload_batch_size_capture_list)
                ):
                    batch_size = num_locals + num_offloads * dist_kv.NUM_ATTN_INSTANCES
                    logger.debug(
                        f"Capturing offload cudagraph for {num_locals=}, {num_offloads=}x{dist_kv.NUM_ATTN_INSTANCES}"
                    )
                    attn_metadata = (
                        self.attn_state.graph_capture_get_metadata_for_offload(
                            num_locals,
                            self.model_config.is_encoder_decoder_model,
                            num_locals,
                            num_offloads,
                        )
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
                    graph_runner = OffloadCUDAGraphRunnerForDecode(
                        self.model,
                        self.attn_backend.get_name(),
                        self.attn_state.graph_clone(batch_size),
                        self.model_config.is_encoder_decoder_model,
                    )

                    capture_inputs = {
                        "input_ids": input_tokens[:batch_size],
                        "positions": input_positions[..., :batch_size],
                        "intermediate_inputs": (
                            intermediate_inputs[:batch_size]
                            if intermediate_inputs is not None
                            else None
                        ),
                        "kv_caches": kv_caches[virtual_engine],
                        "attn_metadata": attn_metadata,
                        "memory_pool": self.graph_memory_pool,
                        "stream": graph_capture_context.stream,
                    }
                    if previous_hidden_states is not None:
                        capture_inputs["previous_hidden_states"] = (
                            previous_hidden_states[:batch_size]
                        )

                    if self.has_inner_state:
                        # Only used by Mamba-based models CUDA graph atm (Jamba)
                        capture_inputs.update({
                            "seqlen_agnostic_capture_inputs":
                            self.model.get_seqlen_agnostic_capture_inputs(batch_size)
                        })
                    if self.model_config.is_encoder_decoder_model:
                        # add the additional inputs to capture for
                        # encoder-decoder models.
                        self._update_inputs_to_capture_for_enc_dec_model(capture_inputs)

                    with set_forward_context(attn_metadata):
                        graph_runner.capture(**capture_inputs)
                    self.graph_memory_pool = graph_runner.graph.pool()
                    self.offload_graph_runners[virtual_engine][(num_locals, num_offloads)] = (
                            graph_runner)
                    torch.distributed.barrier(get_offload_group().sync_comm.device_group)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes < 10 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / GiB_bytes)

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
        if prefill_meta is None and decode_meta is not None and decode_meta.use_cuda_graph:
            if (
                dist_kv.ENABLE_KV_OFFLOAD
                and dist_kv.IS_KV_CONSUMER
                and not is_profile_run(kv_caches)
                and decode_meta.num_offloads > 0
            ):
                num_locals_for_graph = decode_meta.num_locals_padded_for_graph
                num_offloads_for_graph = decode_meta.num_offloads_padded_for_graph
                model_executable = self.offload_graph_runners[virtual_engine][
                    (num_locals_for_graph, num_offloads_for_graph)
                ]
            else:
                assert model_input.input_tokens is not None
                graph_batch_size = model_input.input_tokens.shape[0]
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
        assert isinstance(model_input.attn_metadata, AdrenalineFlashInferMetadata)
        if self.is_prefill_sche_iteration(model_input, kv_caches):
            sync_groups = [get_tp_group().recving_sync_group, get_pp_group().recving_sync_group]
            if model_input.attn_metadata.num_offloads > 0:
                offload_group = get_offload_group().sync_comm.device_group
            else:
                offload_group = None
            hidden_or_intermediate_states, bypass_model_exec, model_input = \
                get_disagg_group().async_recv_kv_caches_and_hidden_states(
                    # model is used to know which layer the current worker
                    # is working on, so that we can receive KV for only those
                    # layers.
                    model_executable,
                    model_input,
                    kv_caches,
                    sync_groups,
                    offload_group,
                )
            bypass_output_process = True
        elif self.is_prefill_ready_iteration(model_input):
            hidden_or_intermediate_states, bypass_model_exec, model_input = (
                get_disagg_group().extract_hidden_for_ready_requests(model_input)
            )
            bypass_output_process = False
            sync_groups = [get_tp_group().fin_recv_sync_group, get_pp_group().fin_recv_sync_group]
            for sync_group in sync_groups:
                torch.distributed.barrier(sync_group)
            if model_input.attn_metadata.num_offloads > 0:
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
            if dist_kv.IS_KV_PRODUCER:
                with set_forward_context(model_input.attn_metadata):
                    hidden_or_intermediate_states = self.model.forward(
                        input_ids=model_input.input_tokens,
                        positions=model_input.input_positions,
                        kv_caches=kv_caches,
                        attn_metadata=model_input.attn_metadata,
                        intermediate_tensors=intermediate_tensors,
                        **MultiModalInputs.as_kwargs(
                            multi_modal_kwargs, device=self.device
                        ),
                        **seqlen_agnostic_kwargs,
                    )
            else:
                assert dist_kv.IS_KV_CONSUMER
                with set_forward_context(model_input.attn_metadata):
                    hidden_or_intermediate_states = model_executable(
                        input_ids=model_input.input_tokens,
                        positions=model_input.input_positions,
                        kv_caches=kv_caches,
                        attn_metadata=model_input.attn_metadata,
                        intermediate_tensors=intermediate_tensors,
                        **MultiModalInputs.as_kwargs(
                            multi_modal_kwargs, device=self.device
                        ),
                        **seqlen_agnostic_kwargs,
                    )

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end.record()

        # Sending KV cache in distributed KV cache transfer setting
        # NOTE: the send operation is non-blocking
        if self.need_send_kv(model_input, kv_caches):
            logger.debug("Sending KV caches and hidden states to decode instance")
            get_disagg_group().send_kv_caches_and_hidden_states(
                # model_executable is used to know which layer the current
                # worker is working on, so that we can send KV for only those
                # layers.
                model_executable,
                model_input,
                kv_caches,
                hidden_or_intermediate_states,
            )
            if dist_kv.ENABLE_KV_OFFLOAD:
                logger.debug("Sending KV caches and hidden states to offload instance")
                get_offload_kv_group().send_kv_caches_and_hidden_states(
                    model_executable.model.start_layer,
                    model_executable.model.end_layer,
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

        logits: torch.Tensor = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        if not self.is_driver_worker:
            return []

        if dist_kv.ENABLE_KV_OFFLOAD and dist_kv.IS_KV_CONSUMER and not is_profile_run(kv_caches):
            assert isinstance(model_input.attn_metadata, AdrenalineFlashInferMetadata)
            if model_input.attn_metadata.num_offloads > 0:
                offload_logits = logits[model_input.attn_metadata.num_locals :]
                get_offload_group().broadcast_tensor_to_offload(offload_logits)

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
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

def is_profile_run(kv_caches: list[torch.Tensor]):
    return kv_caches[0].numel() == 0
