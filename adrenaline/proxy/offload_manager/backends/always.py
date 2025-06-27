from adrenaline.proxy.offload_manager.backends.backend import OffloadManager

class AlwaysOffloadManager(OffloadManager):
    def __init__(self):
        return

    def make_offload_decision(self, request_id: str):
        return True
