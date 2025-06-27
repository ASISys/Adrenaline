import os
import torch

from vllm.attention.layer import *
from vllm.attention.layer import Attention as VllmAttention

class Attention(VllmAttention):
    def forward_with_blk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" in os.environ.keys()
        return self.impl.forward_with_blk(
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            self._k_scale,
            self._v_scale,
            attn_type,
        )
