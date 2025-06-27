import argparse
import os
import statistics
import torch

from torch import multiprocessing
from torch import distributed as dist
from typing import Optional
from vllm.attention.selector import backend_name_to_enum, global_force_attn_backend
from vllm.config import VllmConfig, LoadFormat
from vllm.distributed import parallel_state
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.utils import set_random_seed
from vllm.worker.model_runner import ModelRunner, _BATCH_SIZES_TO_CAPTURE, _BATCH_SIZE_ALIGNMENT
from vllm.utils import (
    update_environment_variables,
    get_distributed_init_method,
    get_ip,
    get_open_port,
    cdiv,
)

from adrenaline.attention.backends import flashinfer
from adrenaline.utils import csv_utils, profile_utils, input_factory


def init_parallel_groups(rank: int, world_size: int, init_method: str):
    dist.init_process_group(
        backend=dist.Backend.GLOO,
        rank=rank,
        world_size=world_size,
        init_method=init_method,
    )
    parallel_state._PP = parallel_state.init_model_parallel_group(
        [[0]], 0, dist.Backend.NCCL, use_custom_allreduce=False, group_name="pp"
    )
    parallel_state._TP = parallel_state.init_model_parallel_group(
        [[0]], 0, dist.Backend.NCCL, use_message_queue_broadcaster=True, group_name="tp"
    )


def destroy_parallel_groups():
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def prepare_request_lens(batch: int, input_len: int, max_output_len: int):
    requests = [("", input_len, max_output_len, None) for _ in range(batch)]
    return requests


def post_process_for_cuda_graph(
    gpu_run_times: dict[int, float], batch_size_capture_list: list[int]
):
    ret: dict[int, float] = {}
    for batch_size_begin, batch_size_end in zip(batch_size_capture_list[:-1], batch_size_capture_list[1:]):
        avg_gpu_run_time = statistics.fmean(
            gpu_run_times[batch_size]
            for batch_size in range(batch_size_begin, batch_size_end)
        )
        ret.update(
            {batch_size: avg_gpu_run_time for batch_size in range(batch_size_begin, batch_size_end)}
        )
    return ret


def worker_task(
    config: VllmConfig,
    max_batch_size: int,
    input_len: Optional[int] = None,
    max_output_len: Optional[int] = None,
    num_warmups: int = 2,
    num_execs: int = 10,
    output_file: str = "gemm_run_time.csv",
    seed: int = 0,
):
    attn_backend_name = flashinfer.AdrenalineFlashInferBackend.get_name()
    global_force_attn_backend(backend_name_to_enum(attn_backend_name))
    block_size = config.cache_config.block_size
    set_random_seed(seed)
    requests = prepare_request_lens(max_batch_size, input_len, max_output_len)
    input_lens = [req[1] for req in requests]
    max_output_lens = [req[2] for req in requests]
    model_runner = ModelRunner(vllm_config=config, is_driver_worker=True)
    req_need_blocks = [cdiv(req[1] + req[2], block_size) for req in requests]
    num_req_needed_blocks = sum(req_need_blocks)
    batch_size_capture_list = [1, 2, 4] + [
        _BATCH_SIZE_ALIGNMENT * i
        for i in range(1, cdiv(max_batch_size, _BATCH_SIZE_ALIGNMENT) + 1)
    ]
    batch_size_to_capture = max(batch_size_capture_list)
    model_runner.max_batchsize_to_capture = batch_size_to_capture
    num_capture_need_blocks = batch_size_to_capture
    num_gpu_blocks = max(num_req_needed_blocks, num_capture_need_blocks)

    model_runner.load_model()
    config.cache_config.num_gpu_blocks = num_gpu_blocks
    kv_caches = input_factory.build_kv_caches(config, num_gpu_blocks, "FLASHINFER")
    if not config.model_config.enforce_eager:
        kv_caches_for_capture = [kv_caches]
        model_runner.capture_model(kv_caches_for_capture, batch_size_capture_list)
    set_random_seed(seed)

    model_input = input_factory.prepare_decode_model_runner_input(
        config,
        model_runner,
        max_batch_size,
        input_lens,
        max_output_lens,
        num_gpu_blocks=num_gpu_blocks,
    )

    for _ in range(num_warmups):
        model_runner.profile_model_execute(model_input, kv_caches)

    import gc
    gc.disable()
    gpu_run_times: dict[int, float] = {}
    with profile_utils.profile():
        for batch_size in range(1, max_batch_size):
            model_input = input_factory.prepare_decode_model_runner_input(
                config,
                model_runner,
                batch_size,
                input_lens[:batch_size],
                max_output_lens[:batch_size],
                num_gpu_blocks=num_gpu_blocks,
            )
            torch.cuda._sleep(20000000)
            profile_results = [
                model_runner.profile_model_execute(model_input, kv_caches)
                for _ in range(num_execs)
            ]
            _, profile_gpu_times, _ = zip(*profile_results)
            avg_gpu_time = torch.tensor(profile_gpu_times).mean().item()
            gpu_run_times[batch_size] = avg_gpu_time

    gc.enable()
    gpu_run_times = post_process_for_cuda_graph(gpu_run_times, batch_size_capture_list)
    csv_utils.save_to_csv(
        items=[
            {"batch_size": batch_size, "gemm_time": gemm_time}
            for batch_size, gemm_time in gpu_run_times.items()
        ],
        headers=["batch_size", "gemm_time"],
        output_path = output_file,
    )


def worker_fn(
    rank: int,
    world_size: int,
    dist_init_method: str,
    config: VllmConfig,
    max_batch_size: int,
    input_len: Optional[int] = None,
    max_output_len: Optional[int] = None,
    num_warmups: int = 2,
    num_execs: int = 2,
    output_file: str = "gemm_run_time.csv",
    seed: int = 0,
):
    init_parallel_groups(rank, world_size, dist_init_method)
    worker_task(
        config,
        max_batch_size,
        input_len,
        max_output_len,
        num_warmups,
        num_execs,
        output_file,
        seed,
    )
    # destroy_parallel_groups()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-len", type=int, default=1)
    parser.add_argument("--max-output-len", type=int, default=15)
    parser.add_argument(
        "--max-batch-size", type=int, default=max(_BATCH_SIZES_TO_CAPTURE) * 2
    )
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument("--num-execs", type=int, default=5)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-file", type=str, default="gemm_run_time.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    enforce_eager: bool = args.enforce_eager
    max_batch_size: int = args.max_batch_size
    engine_args = EngineArgs(
        "meta-llama/Llama-2-7b-hf",
        load_format=LoadFormat.DUMMY,
        enforce_eager=enforce_eager,
    )
    config = engine_args.create_engine_config()
    ctx = multiprocessing.get_context("spawn")
    roles = ["consumer"]
    world_size = len(roles)
    default_devices = ",".join(str(i) for i in range(world_size))
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", default_devices).split(",")
    assert len(devices) == world_size
    dist_init_method = get_distributed_init_method(get_ip(), get_open_port())
    processes_args = [
        (
            rank,
            world_size,
            dist_init_method,
            config,
            args.max_batch_size,
            args.input_len,
            args.max_output_len,
            args.num_warmups,
            args.num_execs,
            args.output_file,
            args.seed,
        )
        for rank in range(world_size)
    ]
    processes = [ctx.Process(target=worker_fn, args=args) for args in processes_args]
    assert (
        "CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING"
        in os.environ.keys()
    )
    for p, role, device in zip(processes, roles, devices):
        env_dicts = {
            "CUDA_VISIBLE_DEVICES": device,
            "VLLM_ATTENTION_BACKEND": "ADRENALINE_FLASHINFER",
            "VLLM_DISTRIBUTED_KV_ROLE": role,
            "VLLM_LOGGING_LEVEL": "DEBUG",
            "VLLM_MAX_GRAPH_BATCH": str(max_batch_size),
        }
        update_environment_variables(env_dicts)
        p.start()
    [p.join(timeout=90) for p in processes]
