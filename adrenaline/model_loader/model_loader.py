from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.model_loader import get_model as vllm_get_model
from vllm.model_executor.models import llama

from adrenaline.attention.layer import Attention
from adrenaline.model_loader.models import llama as adrenalineLlama


def get_model(*, vllm_config: VllmConfig) -> nn.Module:
    vllm_model = vllm_get_model(vllm_config=vllm_config)
    assert isinstance(vllm_model, llama.LlamaForCausalLM)
    model = adrenalineLlama.LlamaForCausalLM(vllm_model, vllm_config)
    return model

def build_model_attn_construct_args(config: VllmConfig):
    model_config = config.model_config
    parallel_config = config.parallel_config

    hf_config = model_config.hf_config
    hidden_size = model_config.get_hidden_size()
    num_heads = model_config.get_num_attention_heads(parallel_config)
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    rope_theta = getattr(config, "rope_theta", 10000)
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None and getattr(
        config, "original_max_position_embeddings", None
    ):
        rope_scaling["original_max_position_embeddings"] = (
            config.original_max_position_embeddings
        )
    max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
    attention_bias = getattr(config, "attention_bias", False) or getattr(
        config, "bias", False
    )
    return {
        "config": hf_config,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "rope_theta": rope_theta,
        "rope_scaling": rope_scaling,
        "max_position_embeddings": max_position_embeddings,
        "bias": attention_bias,
        "cache_config": config.cache_config,
    }

def build_attn_construct_args(config: VllmConfig):
    model_config = config.model_config
    parallel_config = config.parallel_config

    num_heads = model_config.get_num_attention_heads(parallel_config)
    head_dim = model_config.get_head_size()
    scale: float = head_dim**-0.5
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    return num_heads, head_dim, scale, num_kv_heads

def get_attn_module(*, vllm_config: VllmConfig) -> Attention:
    attn_args = build_attn_construct_args(vllm_config)
    attn = Attention(
        *attn_args,
        cache_config=vllm_config.cache_config,
        quant_config=vllm_config.quant_config,
    )
    return attn
