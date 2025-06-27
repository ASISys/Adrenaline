from typing import Optional
from adrenaline.proxy.utils import RequestRuntimeData
from adrenaline.proxy.offload_manager.backends.backend import OffloadManager

class PercentageOffloadManager(OffloadManager):
    def __init__(self, offload_ratio: float):
        self.offload_ratio = offload_ratio

        self.local_cnt = 0
        self.offload_cnt = 0

    def make_offload_decision(
        self,
        request_id: str,
        req_data: Optional[RequestRuntimeData] = None,
    ):
        if self.offload_cnt + 1 < int(self.local_cnt * self.offload_ratio):
            self.offload_cnt += 1
            return True
        else:
            self.local_cnt += 1
            return False
