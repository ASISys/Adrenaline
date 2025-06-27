from vllm.logger import init_logger
from typing import Optional

from adrenaline.proxy.offload_manager.backends.backend import OffloadManager
from adrenaline.proxy.request_dispatcher import RequestDispatcher
from adrenaline.proxy.storage_manager import StorageManager
from adrenaline.proxy.load_estimator import LoadEstimator
from adrenaline.proxy.utils import RequestRuntimeData

logger = init_logger("vllm.proxy.offload.backends.loadaware")


class LoadawareOffloadManager(OffloadManager):
    def __init__(
        self,
        storage_manager: StorageManager,
        request_dispatcher: RequestDispatcher,
        load_estimator: LoadEstimator,
    ):
        self.storage_manager = storage_manager
        self.request_dispatcher = request_dispatcher
        self.load_estimator = load_estimator

    def make_offload_decision(
        self, req_id: Optional[str], req_data: Optional[RequestRuntimeData]
    ):
        prefill_rank = self.request_dispatcher.target_prefill_instance_rank
        decode_rank = self.request_dispatcher.target_decode_instance_rank
        offload_attn_time = self.load_estimator.estimate_offload_attn_time(
            req_data, prefill_rank
        )
        decode_attn_time = self.load_estimator.estimate_decode_attn_time(
            req_data, decode_rank
        )
        if offload_attn_time < decode_attn_time:
            return True
        else:
            return False
