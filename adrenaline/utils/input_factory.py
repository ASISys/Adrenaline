import random

from typing import Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import vllm_adapter as dist_kv
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.utils import get_kv_cache_torch_dtype, cdiv
from vllm.worker.model_runner import ModelRunner

from tests.kernels.utils import make_kv_cache


def _adjust_random_nums_by_max(nums: list[int], max_num: int):
    exceed_len = sum([max(seq_len - max_num, 0) for seq_len in nums])
    for i in range(len(nums)):
        if nums[i] > max_num:
            nums[i] = max_num

        if exceed_len > 0 and nums[i] < max_num:
            delta = min(max_num - nums[i], exceed_len)
            nums[i] += delta
            exceed_len -= delta

    return nums


def _create_random_intergers_by_sum(
    sum_value: int, num: int, start: int = 1
) -> list[int]:
    """
    generate (num - 1) points from range [0, sum_value - start * num],
    so the mean value of intervals between adjacent points is (sum_value / num - start).
    And we add start later to get list with mean value of avg_value.
    """
    if num == 0:
        return []
    partition_length = sum_value - start * num
    partitions = [random.randint(0, partition_length) for _ in range(num - 1)]
    partitions.sort()
    partitions = [0] + partitions + [partition_length]
    samples = [partitions[i + 1] - partitions[i] for i in range(len(partitions) - 1)]
    nums = [sample + start for sample in samples]
    return nums


def _create_random_integers_by_max(num_ints: int, max_int: int, min_int: int = 1) -> list[int]:
    assert max_int > 0 and num_ints > 0
    seq_lens = [random.randint(min_int, max_int) for _ in range(num_ints)]
    seq_lens[-1] = max_int
    return seq_lens


def _create_random_integers_by_avg(
    avg_seq_len: int, batch: int, max_seq_len: Optional[int] = None
) -> list[int]:
    seq_lens = _create_random_intergers_by_sum(avg_seq_len * batch, batch)
    # generate random intergers by average only ensure the average value,
    # i.e. the sum of intergers is (avg_seq_len x batch), however,
    # there may be some intergers larger then max_seq_len, there we need to
    # adjust seq_lens by max_seq_len
    if max_seq_len is not None:
        if isinstance(max_seq_len, int):
            assert avg_seq_len <= max_seq_len
            seq_lens = _adjust_random_nums_by_max(seq_lens, max_seq_len)

    return seq_lens


def create_seq_lens(
    batch: int,
    avg_seq_len: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    config: Optional[VllmConfig] = None,
) -> list[int]:
    if max_seq_len is None:
        assert config is not None
        max_seq_len = config.model_config.max_model_len
    if avg_seq_len is None:
        seq_lens = _create_random_integers_by_max(batch, max_seq_len)
    else:
        seq_lens = _create_random_integers_by_avg(avg_seq_len, batch, max_seq_len)
    return seq_lens


def create_seq_lens_by_block_nums(
    batch: int,
    total_block_nums: int,
    block_size: int,
    max_model_len: Optional[int] = None,
):
    block_nums = _create_random_intergers_by_sum(total_block_nums, batch)
    if max_model_len is not None:
        max_block_num = cdiv(max_model_len, block_size)
        assert total_block_nums // batch <= max_block_num
        block_nums = _adjust_random_nums_by_max(block_nums, max_block_num)
    # normal seq_len range: [(block_num - 1) * block_size + 1, block_num * block_size]
    # however, we create minimal seq_len for sepecified block_nums,
    seq_lens = [(block_num - 1) * block_size for block_num in block_nums]
    return seq_lens


def build_seq_lens_with_offload(
    input_lens: list[int],
    max_output_lens: list[int],
    num_offloads: int = 0,
):
    assert len(input_lens) == len(max_output_lens)
    batch = len(input_lens)
    assert num_offloads >= 0 and num_offloads <= batch
    seq_lens = build_seq_lens(input_lens, max_output_lens)
    num_locals = batch - num_offloads
    offload_mask = [False for _ in range(num_locals)] + [
        True for _ in range(num_offloads)
    ]
    return seq_lens, offload_mask


def build_seq_lens(input_lens: list[int], max_output_lens: Optional[list[int]] = None):
    if max_output_lens is None:
        max_output_lens = [0 for _ in range(len(input_lens))]
    assert len(input_lens) == len(max_output_lens)
    assert all([input_len > 0 for input_len in input_lens])
    assert all([max_output_len >= 0 for max_output_len in max_output_lens])
    output_lens = [
        random.randint(1, max_output_len) if max_output_len > 0 else 0
        for max_output_len in max_output_lens
    ]
    seq_lens = [
        input_len + output_len
        for input_len, output_len in zip(input_lens, output_lens)
    ]
    return seq_lens


def build_prefill_seq_group_metadatas(
    seq_lens: list[int],
    offload_mask: list[bool],
    block_size: int = 16,
    block_offset: int = 0,
):
    assert len(offload_mask) == len(seq_lens)
    batch = len(seq_lens)
    seq_datas = [SequenceData.from_seqs(range(seq_len)) for seq_len in seq_lens]
    block_lens = [cdiv(seq_len, block_size) for seq_len in seq_lens]
    prefix_block_lens = [sum(block_lens[:i]) for i in range(len(block_lens))]
    block_tables = [
        list(range(block_offset + prefix_len, block_offset + prefix_len + block_len))
        for block_len, prefix_len in zip(block_lens, prefix_block_lens)
    ]
    seq_group_metadata_list = [
        SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=True,
            seq_data={0: seq_datas[i]},
            sampling_params=SamplingParams(temperature=0, offload=offload_mask[i]),
            block_tables={0: block_tables[i]},
        )
        for i in range(batch)
    ]
    return seq_group_metadata_list


def build_decode_seq_group_metadatas_for_d_instance(
    seq_lens: list[int],
    offload_mask: list[bool],
    block_size: int = 16,
    block_offset: int = 0,
):
    assert len(offload_mask) == len(seq_lens)
    batch = len(seq_lens)
    seq_datas = [SequenceData.from_seqs(range(seq_len - 1)) for seq_len in seq_lens]
    for seq_data, seq_len in zip(seq_datas, seq_lens):
        seq_data.update_num_computed_tokens(seq_len - 1)
        seq_data.append_token_id(1, 0)
    block_lens = [
        cdiv(seq_len, block_size) if not offload else 0
        for seq_len, offload in zip(seq_lens, offload_mask)
    ]
    prefix_block_lens = [sum(block_lens[:i]) for i in range(len(block_lens))]
    block_tables = [
        list(range(block_offset + prefix_len, block_offset + prefix_len + block_len))
        for block_len, prefix_len in zip(block_lens, prefix_block_lens)
    ]
    seq_group_metadata_list = [
        SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=False,
            seq_data={0: seq_datas[i]},
            sampling_params=SamplingParams(temperature=0, offload=offload_mask[i]),
            block_tables={0: block_tables[i]},
        )
        for i in range(batch)
    ]
    return seq_group_metadata_list


def build_decode_seq_group_metadatas_for_attn_instance(
    seq_lens: list[int],
    offload_mask: list[bool],
    block_size: int = 16,
    block_offset: int = 0,
):
    assert len(offload_mask) == len(seq_lens)
    assert all(offload_mask)
    batch = len(seq_lens)
    seq_datas = [SequenceData.from_seqs(range(seq_len)) for seq_len in seq_lens]
    for seq_data, seq_len in zip(seq_datas, seq_lens):
        seq_data.update_num_computed_tokens(seq_len)
        seq_data.append_token_id(1, 0)
    block_lens = [
        cdiv(seq_len + 1, block_size) if offload else 0
        for seq_len, offload in zip(seq_lens, offload_mask)
    ]
    prefix_block_lens = [sum(block_lens[:i]) for i in range(len(block_lens))]
    block_tables = [
        list(range(block_offset + prefix_len, block_offset + prefix_len + block_len))
        for block_len, prefix_len in zip(block_lens, prefix_block_lens)
    ]
    seq_group_metadata_list = [
        SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=False,
            seq_data={0: seq_datas[i]},
            sampling_params=SamplingParams(temperature=0, offload=offload_mask[i]),
            block_tables={0: block_tables[i]},
        )
        for i in range(batch)
    ]
    return seq_group_metadata_list


def build_kv_caches(config: VllmConfig, num_blocks: int, backend: str):
    model_dtype = config.model_config.dtype
    kv_dtype = config.cache_config.cache_dtype
    device = config.device_config.device
    num_kv_heads = config.model_config.get_num_kv_heads(config.parallel_config)
    head_size = config.model_config.get_head_size()
    num_layers = config.model_config.get_num_layers(config.parallel_config)
    block_size = config.cache_config.block_size

    with set_default_torch_dtype(get_kv_cache_torch_dtype(kv_dtype, model_dtype)):
        kv_caches = [
            make_kv_cache(
                num_blocks, num_kv_heads, head_size, block_size, device, backend
            )
            for _ in range(num_layers)
        ]
    return kv_caches


def prepare_prefill_seq_group_metadatas(
    config: VllmConfig,
    seq_lens: list[int],
    offload_mask: Optional[list[bool]] = None,
    block_offset: int = 0,
):
    if offload_mask is None:
        offload_mask = [False for _ in range(len(seq_lens))]
    block_size = config.cache_config.block_size
    seq_group_metadata_list = build_prefill_seq_group_metadatas(
        seq_lens, offload_mask, block_size, block_offset
    )
    return seq_group_metadata_list


def prepare_decode_seq_group_metadatas_for_d_instance(
    config: VllmConfig,
    seq_lens: list[int],
    offload_mask: Optional[list[bool]] = None,
    block_offset: int = 0,
):
    if offload_mask is None:
        offload_mask = [False for _ in range(len(seq_lens))]
    block_size = config.cache_config.block_size
    seq_group_metadata_list = build_decode_seq_group_metadatas_for_d_instance(
        seq_lens, offload_mask, block_size, block_offset
    )
    for seq_group_metadata in seq_group_metadata_list:
        assert seq_group_metadata.token_chunk_size == 1
    return seq_group_metadata_list


def prepare_decode_seq_group_metadatas_for_attn_instance(
    config: VllmConfig,
    seq_lens: list[int],
    block_size: int = 16,
    block_offset: int = 0,
):
    block_size = config.cache_config.block_size
    offload_mask = [True for _ in range(len(seq_lens))]
    seq_group_metadata_list = build_decode_seq_group_metadatas_for_attn_instance(
        seq_lens, offload_mask, block_size, block_offset
    )
    for seq_group_metadata in seq_group_metadata_list:
        assert seq_group_metadata.token_chunk_size == 1
    return seq_group_metadata_list


def prepare_prefill_model_input_and_num_blocks(
    config: VllmConfig,
    model_runner: ModelRunner,
    batch_size: int = 1,
    input_lens: Optional[list[int]] = None,
    input_len: int = 1024,
    offload_mask: Optional[list[bool]] = None,
    block_offset: int = 0,
):
    if input_lens is None:
        input_lens = [input_len] * batch_size
    seq_lens = build_seq_lens(input_lens)
    seq_group_metadata_list = prepare_prefill_seq_group_metadatas(
        config, seq_lens, offload_mask, block_offset
    )
    model_input = model_runner.prepare_model_input(seq_group_metadata_list)
    num_seq_blocks = sum([len(sg.block_tables[0]) for sg in seq_group_metadata_list])
    return model_input, num_seq_blocks


def prepare_prefill_model_input(
    config: VllmConfig,
    model_runner: ModelRunner,
    batch_size: int = 1,
    input_lens: Optional[list[int]] = None,
    input_len: int = 1024,
    offload_mask: Optional[list[bool]] = None,
    num_gpu_blocks: Optional[int] = None,
    block_offset: int = 0,
):
    if input_lens is None:
        input_lens = [input_len] * batch_size
    seq_lens = build_seq_lens(input_lens)
    seq_group_metadata_list = prepare_prefill_seq_group_metadatas(
        config, seq_lens, offload_mask, block_offset
    )
    model_input = model_runner.prepare_model_input(seq_group_metadata_list)
    if num_gpu_blocks is not None:
        num_seq_blocks = sum([len(sg.block_tables[0]) for sg in seq_group_metadata_list])
        assert num_seq_blocks <= num_gpu_blocks
    return model_input


def prepare_decode_model_runner_input_and_num_blocks(
    config: VllmConfig,
    model_runner: ModelRunner,
    decode_batch: int,
    input_lens: Optional[list[int]] = None,
    max_output_lens: Optional[list[int]] = None,
    input_len: int = 1024,
    max_output_len: int = 128,
    num_offloads: int = 0,
    block_offset: int = 0,
):
    assert dist_kv.IS_DISTRIBUTED_KV_INSTANCE
    assert dist_kv.IS_KV_CONSUMER or dist_kv.IS_ATTN_EXECUTOR
    if input_lens is None:
        input_lens = [input_len] * decode_batch
    if max_output_lens is None:
        max_output_lens = [max_output_len] * decode_batch
    seq_lens, offload_mask = build_seq_lens_with_offload(
        input_lens, max_output_lens, num_offloads
    )
    if dist_kv.IS_KV_CONSUMER:
        seq_group_metadata_list = prepare_decode_seq_group_metadatas_for_d_instance(
            config, seq_lens, offload_mask, block_offset
        )
        model_input = model_runner.prepare_model_input(seq_group_metadata_list)
    else:
        assert dist_kv.IS_ATTN_EXECUTOR
        if num_offloads > 0:
            seq_lens = seq_lens[-num_offloads:]
            seq_group_metadata_list = prepare_decode_seq_group_metadatas_for_attn_instance(
                config, seq_lens, block_offset
            )
            model_input = model_runner.prepare_model_input(seq_group_metadata_list)
        else:
            model_input = None

    if model_input is not None:
        num_seq_blocks = sum([len(sg.block_tables[0]) for sg in seq_group_metadata_list])
    else:
        num_seq_blocks = None
    return model_input, num_seq_blocks


def prepare_decode_model_runner_input(
    config: VllmConfig,
    model_runner: ModelRunner,
    decode_batch: int,
    input_lens: Optional[list[int]] = None,
    max_output_lens: Optional[list[int]] = None,
    input_len: int = 1024,
    max_output_len: int = 128,
    num_gpu_blocks: Optional[int] = None,
    num_offloads: int = 0,
    block_offset: int = 0,
):
    assert dist_kv.IS_DISTRIBUTED_KV_INSTANCE
    assert dist_kv.IS_KV_CONSUMER or dist_kv.IS_ATTN_EXECUTOR
    if input_lens is None:
        input_lens = [input_len] * decode_batch
    if max_output_lens is None:
        max_output_lens = [max_output_len] * decode_batch
    assert len(input_lens) == decode_batch
    assert len(max_output_lens) == decode_batch
    seq_lens, offload_mask = build_seq_lens_with_offload(
        input_lens, max_output_lens, num_offloads
    )
    if dist_kv.IS_KV_CONSUMER:
        seq_group_metadata_list = prepare_decode_seq_group_metadatas_for_d_instance(
            config, seq_lens, offload_mask, block_offset
        )
        model_input = model_runner.prepare_model_input(seq_group_metadata_list)
    else:
        assert dist_kv.IS_ATTN_EXECUTOR
        if num_offloads > 0:
            seq_lens = seq_lens[-num_offloads:]
            seq_group_metadata_list = prepare_decode_seq_group_metadatas_for_attn_instance(
                config, seq_lens, block_offset
            )
            model_input = model_runner.prepare_model_input(seq_group_metadata_list)
        else:
            model_input, num_seq_blocks = None, None

    if num_gpu_blocks is not None and model_input is not None:
        num_seq_blocks = sum([len(sg.block_tables[0]) for sg in seq_group_metadata_list])
        assert num_seq_blocks <= num_gpu_blocks, f"{num_seq_blocks=} <= {num_gpu_blocks=}"
    return model_input
