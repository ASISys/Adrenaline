import vllm.envs as envs

# check VLLM_DISTRIBUTERD_KV_ROLE and set corresponding flags
kv_roles = ["producer", "consumer", "both", "offload"]
assert (
    envs.VLLM_DISTRIBUTED_KV_ROLE in [None] + kv_roles
), "VLLM_DISTRIBUTERD_KV_ROLE can only be producer, consumer, both or offload."
IS_DISTRIBUTED_KV_INSTANCE: bool = envs.VLLM_DISTRIBUTED_KV_ROLE in kv_roles
IS_KV_PRODUCER: bool = envs.VLLM_DISTRIBUTED_KV_ROLE in ["producer", "both"]
IS_KV_CONSUMER: bool = envs.VLLM_DISTRIBUTED_KV_ROLE in ["consumer", "both"]
IS_ATTN_EXECUTOR: bool = envs.VLLM_DISTRIBUTED_KV_ROLE == "offload"
ENABLE_KV_OFFLOAD: bool = envs.VLLM_ENABLE_KV_OFFLOAD

NUM_PREFILL_INSTANCES: int = envs.VLLM_PREFILL_INSTANCES
NUM_DECODE_INSTANCES: int = envs.VLLM_DECODE_INSTANCES
if ENABLE_KV_OFFLOAD:
    NUM_ATTN_INSTANCES: int = envs.VLLM_PREFILL_INSTANCES
else:
    NUM_ATTN_INSTANCES: int = 0

if envs.VLLM_RANK_IN_INSTANCES != -1:
    INSTANCE_RANK: int = envs.VLLM_RANK_IN_INSTANCES
elif IS_DISTRIBUTED_KV_INSTANCE:
    assert NUM_PREFILL_INSTANCES == 1
    assert NUM_DECODE_INSTANCES == 1
    if ENABLE_KV_OFFLOAD:
        assert NUM_ATTN_INSTANCES == 1
    if IS_KV_PRODUCER:
        INSTANCE_RANK = 0
    elif IS_KV_CONSUMER:
        INSTANCE_RANK = 1
    elif IS_ATTN_EXECUTOR:
        INSTANCE_RANK = 2
else:
    INSTANCE_RANK: int = envs.VLLM_RANK_IN_INSTANCES

# When the current instance is both KV producer and KV consumer,
# it is likely connected to a KV storage service on CPU/disk
# so the communication backend needs to be "gloo" for that case.
DISTRIBUTED_BACKEND: str = "gloo" if (IS_KV_PRODUCER
                                      and IS_KV_CONSUMER) else "nccl"
# corresponding device
DISTRIBUTED_DEVICE: str = "cpu" if (IS_KV_PRODUCER
                                    and IS_KV_CONSUMER) else "cuda"

def get_attn_rank():
    assert IS_ATTN_EXECUTOR
    return INSTANCE_RANK - NUM_PREFILL_INSTANCES - NUM_DECODE_INSTANCES
