# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import torch

from torch import nn
from typing import Optional
from vllm.attention import AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_offload_group
from vllm.distributed.kv_transfer import vllm_adapter as dist_kv
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.models import llama as vllmLlama
from vllm.sequence import IntermediateTensors

from adrenaline.attention.layer import Attention


def enable_offload(attn_metadata: AttentionMetadata):
    return (
        dist_kv.ENABLE_KV_OFFLOAD
        and dist_kv.IS_KV_CONSUMER
        and attn_metadata.num_offloads
        and attn_metadata.num_offloads > 0
    )

class LlamaMLP(nn.Module):
    def __init__(self, mlp: vllmLlama.LlamaMLP, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act_fn = mlp.act_fn
        self.mlp = mlp

    def forward(self, x):
        return self.mlp.forward(x)


class LlamaAttention(nn.Module):
    def __init__(self, self_attn: vllmLlama.LlamaAttention, config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attn = Attention(
            self_attn.num_heads,
            self_attn.head_dim,
            self_attn.scaling,
            self_attn.num_kv_heads,
            cache_config=config.cache_config,
            quant_config=config.quant_config,
        )
        self.rotary_emb = self_attn.rotary_emb
        self.q_size = self_attn.q_size
        self.kv_size = self_attn.kv_size
        self.self_attn = self_attn

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        if enable_offload(attn_metadata):
            return self.forward_with_offload(
                positions, hidden_states, kv_cache, attn_metadata
            )
        return self.origin_forward(positions, hidden_states, kv_cache, attn_metadata)

    def origin_forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        qkv, _ = self.self_attn.qkv_proj.forward(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn.forward(q, k, v, kv_cache, attn_metadata)
        output, _ = self.self_attn.o_proj.forward(attn_output)
        return output

    def forward_with_offload(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        qkv, _ = self.self_attn.qkv_proj.forward(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        split_sizes = [attn_metadata.num_locals] + [attn_metadata.num_offloads] * dist_kv.NUM_ATTN_INSTANCES
        split_qkv = qkv.split(split_sizes)
        work = get_offload_group().scatter_tensor_to_offload(
            input_tensors=list(split_qkv), async_op=True
        )
        local_q = q[:attn_metadata.num_locals]
        local_k = k[:attn_metadata.num_locals]
        local_v = v[:attn_metadata.num_locals]
        attn_output = torch.empty_like(q)
        if attn_metadata.num_decode_tokens > 0:
            attn_output[:attn_metadata.num_locals] = self.attn.forward(
                local_q, local_k, local_v, kv_cache, attn_metadata
            )
        split_attn_output = attn_output.split(split_sizes)
        if work is not None:
            work.wait()
        work = get_offload_group().gather_tensor_to_decode(
            output_tensors=list(split_attn_output), async_op=False
        )
        output, _ = self.self_attn.o_proj.forward(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, layer: vllmLlama.LlamaDecoderLayer, config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.self_attn = LlamaAttention(layer.self_attn, config)
        self.mlp = LlamaMLP(layer.mlp)
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ):
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions,
            hidden_states,
            kv_cache,
            attn_metadata,
        )
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class LlamaModel(nn.Module):
    def __init__(self, model: vllmLlama.LlamaModel, config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embed_tokens = model.embed_tokens
        self.start_layer, self.end_layer = model.start_layer, model.end_layer
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(layer, config) for layer in model.layers]
        )
        self.norm = model.norm

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer.forward(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

class LlamaForCausalLM(nn.Module):
    def __init__(self, model: vllmLlama.LlamaForCausalLM, config: VllmConfig) -> None:
        super().__init__()
        self.model = LlamaModel(model.model, config)

        self.lm_head = model.lm_head
        self.sampler = model.sampler
        self.logits_processor = model.logits_processor

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
    ):
        model_output = self.model.forward(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
        )
        return model_output

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ):
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata):
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
