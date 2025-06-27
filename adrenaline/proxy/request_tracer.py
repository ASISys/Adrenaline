from vllm.transformers_utils.tokenizer import get_tokenizer
from adrenaline.proxy.utils import RequestRuntimeData

class RequestTracer:

    def __init__(self, tokenizer_name: str):
        self.requests: dict[str, RequestRuntimeData] = {}
        self.tokenizer = get_tokenizer(tokenizer_name)

    def init_request(self, request_id: str, request_data: dict):
        prompt = request_data.get("prompt")
        prompt_len = len(self.tokenizer(prompt).input_ids)
        max_gen_tokens = request_data.get("max_tokens")
        self.requests[request_id] = RequestRuntimeData(
            prompt_len, prompt_len + max_gen_tokens
        )

    def release_request(self, request_id: str):
        self.requests.pop(request_id)

    def update_request(self, request_id: str):
        self.requests[request_id].update_for_decode()
