import torch

from torch import nn
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    get_offload_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.utils import get_pp_indices

from adrenaline.attention.backends.flashinfer import AdrenalineFlashInferMetadata
from adrenaline.attention.layer import Attention
from adrenaline.model_loader.model_loader import get_attn_module


class OffloadAttn(nn.Module):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        total_num_heads: int = hf_config.num_attention_heads
        hidden_size: int = hf_config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        head_dim: int = getattr(hf_config, "head_dim", hidden_size // total_num_heads)
        num_kv_heads: int = getattr(hf_config, "num_key_value_heads", total_num_heads)
        num_hidden_layers: int = hf_config.num_hidden_layers

        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.model: Attention = get_attn_module(vllm_config=vllm_config)
        self.start_layer, self.end_layer = get_pp_indices(
            num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size
        )
        self.model_dtype = vllm_config.model_config.dtype
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        self.qkv_size = self.q_size + self.kv_size + self.kv_size

    def forward(self, kv_caches, attn_metadata):
        for layer_idx in range(self.start_layer, self.end_layer):
            self.forward_with_offload(
                kv_caches[layer_idx - self.start_layer],
                attn_metadata,
            )

    def forward_with_offload(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        assert isinstance(attn_metadata, AdrenalineFlashInferMetadata), f"{type(attn_metadata)}"
        batch = attn_metadata.num_offloads
        qkv = torch.empty(
            batch, self.qkv_size, dtype=self.model_dtype, device=attn_metadata.device
        )
        get_offload_group().scatter_tensor_to_offload(qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_out = self.model.forward_with_blk(
            q,
            k,
            v,
            kv_cache,
            attn_metadata,
        )
        get_offload_group().gather_tensor_to_decode(attn_out)
