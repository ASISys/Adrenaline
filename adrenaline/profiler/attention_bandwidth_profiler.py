# borrowed from flashinfer
# https://docs.flashinfer.ai/api/decode.html#batch-decoding

import argparse
import torch
import flashinfer
import os

from torch import multiprocessing
from multiprocessing.managers import DictProxy
from vllm.config import LoadFormat, ModelConfig, ParallelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import cdiv
from vllm.utils import update_environment_variables

from adrenaline.utils import input_factory, csv_utils
from adrenaline.utils.utils import get_token_size


def profile_bandwidth(model: str, batch_size: int, avg_seq_len: int, tensor_parallel_size: int = 1):
    engine_args = EngineArgs(
        model, load_format=LoadFormat.DUMMY, tensor_parallel_size=tensor_parallel_size
    )
    config = engine_args.create_engine_config()

    num_layers = config.model_config.get_num_layers(config.parallel_config)
    num_qo_heads = config.model_config.get_num_attention_heads(config.parallel_config)
    num_kv_heads = config.model_config.get_num_kv_heads(config.parallel_config)
    head_dim = config.model_config.get_head_size()
    block_size = config.cache_config.block_size

    seq_lens = input_factory.create_seq_lens(batch_size, avg_seq_len, config=config)
    block_lens = [cdiv(seq_len, block_size) for seq_len in seq_lens]

    max_num_pages = sum(block_lens)
    prefix_block_lens = [sum(block_lens[:i]) for i in range(len(block_lens) + 1)]
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    kv_page_indices = torch.arange(max_num_pages, dtype=torch.int32, device="cuda")
    kv_page_indptr = torch.tensor(
        prefix_block_lens, dtype=torch.int32, device="cuda"
    )
    # 1 <= kv_last_page_len <= page_size
    kv_last_page_len = torch.randint(
        1, block_size + 1, (batch_size,), dtype=torch.int32, device="cuda"
    )
    decode_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        block_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16
    )
    kv_cache_for_layers = [
        torch.randn(
            max_num_pages,
            2,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float16,
            device="cuda",
        )
        for _ in range(num_layers)
    ]
    q_for_layers = [
        torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
        for _ in range(num_layers)
    ]
    for q, kv_cache in zip(q_for_layers[-3:], kv_cache_for_layers[-3:]):
        o = decode_wrapper.run(q, kv_cache)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(int(1e7))
    start.record()
    for q, kv_cache in zip(q_for_layers, kv_cache_for_layers):
        o = decode_wrapper.run(q, kv_cache)
    end.record()
    torch.cuda.synchronize()
    attn_time = start.elapsed_time(end)  # ms
    token_size = get_token_size(config.model_config, config.parallel_config, torch.float16)
    size = avg_seq_len * batch_size * token_size  # MB
    bandwidth = size / attn_time  # GB/s
    return bandwidth


def worker(model: str, batch_size: int, avg_seq_len: int, sm_pct: int, bandwidths: DictProxy):
    print(f"profiling bandwidth using {sm_pct}% SM")
    bandwidth = profile_bandwidth(model, batch_size, avg_seq_len)
    bandwidths[sm_pct] = bandwidth


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--avg-seq-len", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sm-pcts", type=int, nargs="+", default=[100])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-file", type=str, default="attention_bandwidth.csv")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model: str = args.model
    batch_size: int = args.batch_size
    avg_seq_len: int = args.avg_seq_len
    sm_pcts: list[int] = args.sm_pcts
    output_file: str = args.output_file
    manager = multiprocessing.Manager()
    bandwidths = manager.dict()
    ctx = multiprocessing.get_context("spawn")
    for sm_pct in sm_pcts:
        process_args = (model, batch_size, avg_seq_len, sm_pct, bandwidths)
        p = ctx.Process(target=worker, args=process_args)
        env_dicts = {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(sm_pct)}
        update_environment_variables(env_dicts)
        p.start()
        p.join()

    csv_utils.save_to_csv(
        items=[
            {"sm_pct": sm_pct, "bandwidth": bandwidth}
            for sm_pct, bandwidth in bandwidths.items()
        ],
        headers=["sm_pct", "bandwidth"],
        output_path=output_file,
    )
