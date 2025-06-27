import argparse
import asyncio
import statistics

from quart import Quart, make_response, request
from typing import AsyncGenerator, Any, Optional
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.model_executor.utils import set_random_seed

from adrenaline.proxy.offload_manager.offload_manager import get_manager
from adrenaline.proxy.offload_manager.backends import backend
from adrenaline.proxy.offload_manager.backends.backend import Backend
from adrenaline.proxy.storage_manager import StorageManager
from adrenaline.proxy.request_dispatcher import RequestDispatcher
from adrenaline.proxy.request_tracer import RequestTracer
from adrenaline.proxy.load_estimator import LoadEstimator
from adrenaline.proxy.utils import dummy_exec

app = Quart(__name__)
logger = init_logger("vllm.adrenaline_proxy")


async def dispatch_offload_request(request_data: dict):
    attn_gens: list[AsyncGenerator[bytes, Any]] = (
        request_dispatcher.boardcast_attn_request(request_data)
    )
    # ensure the P-Attn has added request to engine, then it send request to decode instance
    # TODO: only work for stream mode, add support for non-stream response
    for attn_gen in attn_gens:
        first_chunk = await attn_gen.__anext__()
        chunk = first_chunk.strip().decode().removeprefix("data: ")
        assert chunk == "[Add Offload]", f"{chunk}"
        asyncio.create_task(dummy_exec(attn_gen))
    request_id = request_data["request_id"]
    generator, _ = request_dispatcher.dispatch_decode_request(request_data)
    prefill_phase = True
    async for chunk_bytes in generator:
        chunk = chunk_bytes.strip()
        if not chunk:
            continue
        content = chunk.decode().removeprefix("data: ")
        is_finished = content == "[DONE]"
        if is_finished:
            logger.info(f"finished request: {request_id}")
            request_tracer.release_request(request_id)
            storage_manager.release_request(request_id)
        elif not prefill_phase:
            request_tracer.update_request(request_id)
            seq_len = request_tracer.requests[request_id].seq_len
            storage_manager.update_blocks_for_decode_request(request_id, seq_len)
        prefill_phase = False
        yield chunk_bytes


async def dispatch_decode_request(request_data: dict):
    generator, _ = request_dispatcher.dispatch_decode_request(request_data)
    request_id = request_data["request_id"]
    prefill_phase = True
    async for chunk_bytes in generator:
        chunk = chunk_bytes.strip()
        if not chunk:
            continue
        content = chunk.decode().removeprefix("data: ")
        is_finished = content == "[DONE]"
        if is_finished:
            logger.info(f"finished request: {request_id}")
            storage_manager.release_request(request_id)
            request_tracer.release_request(request_id)
        elif not prefill_phase:
            request_tracer.update_request(request_id)
            seq_len = request_tracer.requests[request_id].seq_len
            storage_manager.update_blocks_for_decode_request(request_id, seq_len)
        prefill_phase = False
        yield chunk_bytes


@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    try:
        original_request: dict = await request.get_json()

        # maintain request_id in proxy for consistency among Prefill, Decode and Attn instance
        request_id = f"cmpl-{random_uuid()}"
        original_request["request_id"] = request_id
        prefill_rank = request_dispatcher.target_prefill_instance_rank
        decode_rank = request_dispatcher.target_decode_instance_rank
        original_request["prefill_rank"] = prefill_rank
        request_tracer.init_request(request_id, original_request)
        req_data = request_tracer.requests[request_id]

        is_offload = offload_manager.make_offload_decision(request_id, req_data)
        if predict_topt and decode_slo is not None:
            assert load_estimator is not None
            decode_time = load_estimator.estimate_decode_time(
                req_data, is_offload, prefill_rank, decode_rank
            )
            if estimate_delta:
                itls = [req.itl for req in request_tracer.requests.values() if req.itl is not None]
                recent_decode_time = statistics.fmean(itls) * 1000 if len(itls) > 0 else None
                num_requests = sum(storage_manager.num_requests)
                logger.info(f"estimate {decode_time=}, recent decode_time={recent_decode_time}, {num_requests=}")
            if decode_time > decode_slo:
                # abort request
                return await make_response()
        storage_manager.setup_location_for_new_request(request_id, prefill_rank, decode_rank, is_offload)
        storage_manager.update_blocks_for_new_request(request_id, req_data)
        original_request["offload"] = is_offload
        prefill_request = original_request.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request['max_tokens'] = 1

        prefill_generator, target_prefill_rank = (
            request_dispatcher.dispatch_prefill_request(prefill_request)
        )
        logger.info(f"Forwarding request {request_id} to p{target_prefill_rank} ({is_offload=})")
        # finish prefill
        await dummy_exec(prefill_generator)

        if is_offload:
            generator = dispatch_offload_request(original_request)
        else:
            generator = dispatch_decode_request(original_request)

        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--num-decode-blocks", nargs="+", type=int, required=True)
    parser.add_argument("--num-attention-blocks", nargs="+", type=int, required=True)
    parser.add_argument(
        "--offload-backend", choices=backend.backend_names, default="loadaware"
    )
    parser.add_argument(
        "--end-points",
        nargs="+",
        default=["localhost:8100", "localhost:8200", "localhost:8300"],
    )
    parser.add_argument("--offload-ratio", type=float)
    parser.add_argument("--offload-sm-pct", type=int, default=20)
    parser.add_argument("--estimate-delta", action="store_true")
    parser.add_argument("--predict-tpot", action="store_true")
    parser.add_argument("--decode-slo", type=float, default=float("inf"))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    logger.info(args)
    estimate_delta: bool = args.estimate_delta
    predict_topt: bool = args.predict_tpot
    num_decodes = len(args.num_decode_blocks)
    num_attentions = len(args.num_attention_blocks)
    num_prefills = num_attentions
    request_dispatcher = RequestDispatcher(
        args.end_points, num_prefills, num_decodes, num_attentions
    )
    request_tracer = RequestTracer(args.tokenizer)
    storage_manager = StorageManager(
        num_decode_blocks=args.num_decode_blocks,
        num_attn_blocks=args.num_attention_blocks,
        block_size=args.block_size,
    )
    decode_slo: Optional[float] = args.decode_slo
    offload_sm_pct: int = args.offload_sm_pct
    offload_backend = backend.name_to_enum(args.offload_backend)
    if predict_topt or offload_backend == Backend.LOADAWARE:
        load_estimator = LoadEstimator(
            storage_manager, offload_sm_pct, model_name=args.tokenizer
        )
    else:
        load_estimator = None
    offload_ratio: Optional[float] = args.offload_ratio
    offload_manager = get_manager(
        offload_backend,
        offload_ratio=offload_ratio,
        storage_manager=storage_manager,
        request_dispatcher=request_dispatcher,
        load_estimator=load_estimator,
    )
    set_random_seed(args.seed)
    app.run(host=args.host, port=args.port)
