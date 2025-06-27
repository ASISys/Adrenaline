import argparse
import aiohttp

from quart import Quart, make_response, request
from vllm.logger import init_logger
from vllm.utils import random_uuid
from adrenaline.proxy.request_dispatcher import RequestDispatcher

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)
logger = init_logger("vllm.disagg_proxy")


@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        request_id = f"cmpl-{random_uuid()}"
        original_request_data["request_id"] = request_id
        original_request_data["prefill_rank"] = (
            request_dispatcher.target_prefill_instance_rank
        )

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request['max_tokens'] = 1

        prefill_generator, _ = request_dispatcher.dispatch_prefill_request(prefill_request)
        # finish prefill
        async for _ in prefill_generator:
            continue

        # return decode
        generator, _ = request_dispatcher.dispatch_decode_request(original_request_data)
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
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--end-points", nargs="+", default=["localhost:8100", "localhost:8200"])
    parser.add_argument("--num-prefills", type=int, default=1)
    parser.add_argument("--num-decodes", type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    request_dispatcher = RequestDispatcher(args.end_points, args.num_prefills, args.num_decodes)
    app.run(host=args.host, port=args.port)
