import itertools
from typing import Any
from adrenaline.proxy.utils import forward_request

class RequestDispatcher:

    def __init__(
        self,
        end_points: list[str],
        num_prefills: int,
        num_decodes: int,
        num_attentions: int = 0,
    ):
        assert len(end_points) == num_prefills + num_decodes + num_attentions
        assert num_decodes == 1, "Only support num_decodes = 1"
        self.num_prefills = num_prefills
        self.num_decodes = num_decodes
        self.num_attentions = num_attentions

        num_instances = [num_prefills, num_decodes, num_attentions]
        self.instance_prefix_sum = list(itertools.accumulate(num_instances))
        self.prefill_end_points = end_points[:self.instance_prefix_sum[0]]
        self.decode_end_points = end_points[self.instance_prefix_sum[0] : self.instance_prefix_sum[1]]
        self.attention_end_points = end_points[self.instance_prefix_sum[1] : self.instance_prefix_sum[2]]

        # [0, #prefills - 1]
        self._target_prefill_instance_rank = 0
        # [0, #decodes - 1]
        self._target_decode_instance_rank = 0

    @property
    def target_prefill_instance_rank(self):
        return self._target_prefill_instance_rank

    @property
    def target_decode_instance_rank(self):
        return self._target_decode_instance_rank

    def _update_prefill_instance_rank(self):
        self._target_prefill_instance_rank = (
            self._target_prefill_instance_rank + 1
        ) % self.num_prefills

    def _update_decode_instance_rank(self):
        self._target_decode_instance_rank = (
            self._target_decode_instance_rank + 1
        ) % self.num_decodes

    def dispatch_prefill_request(self, prefill_request_data: dict[str, Any]):
        target_prefill_instance_rank = self._target_prefill_instance_rank
        end_point = self.prefill_end_points[target_prefill_instance_rank]
        generator = forward_request(
            f"http://{end_point}/v1/completions", prefill_request_data
        )
        self._update_prefill_instance_rank()
        return generator, target_prefill_instance_rank

    def dispatch_decode_request(self, request_data: dict[str, Any]):
        target_decode_instance_rank = self._target_decode_instance_rank
        end_point = self.decode_end_points[target_decode_instance_rank]
        generator = forward_request(
            f"http://{end_point}/v1/completions", request_data
        )
        self._update_decode_instance_rank()
        return generator, target_decode_instance_rank   

    def boardcast_attn_request(self, request_data: dict[str, Any]):
        end_points = [end_point for end_point in self.attention_end_points]
        generators = [
            forward_request(f"http://{end_point}/v1/completions", request_data)
            for end_point in end_points
        ]
        return generators
