from vllm.logger import init_logger
from typing import Optional

from adrenaline.proxy.offload_manager.backends.backend import Backend, OffloadManager
from adrenaline.proxy.offload_manager.backends import loadaware, percentage, never, always
from adrenaline.proxy.request_dispatcher import RequestDispatcher
from adrenaline.proxy.storage_manager import StorageManager
from adrenaline.proxy.load_estimator import LoadEstimator

logger = init_logger("vllm.proxy.offload_manager")


def get_manager(
    backend: Backend,
    offload_ratio: Optional[float] = None,
    storage_manager: Optional[StorageManager] = None,
    request_dispatcher: Optional[RequestDispatcher] = None,
    load_estimator: Optional[LoadEstimator] = None,
) -> OffloadManager:
    if backend == Backend.LOADAWARE:
        assert load_estimator is not None
        assert storage_manager is not None
        assert request_dispatcher is not None
        manager = loadaware.LoadawareOffloadManager(
            storage_manager, request_dispatcher, load_estimator
        )
    elif backend == Backend.ALWAYS:
        manager = always.AlwaysOffloadManager()
    elif backend == Backend.NEVER:
        manager = never.NeverOffloadManager()
    elif backend == Backend.PERCENTAGE:
        assert offload_ratio is not None
        manager = percentage.PercentageOffloadManager(offload_ratio)
    else:
        raise ValueError(f"Invalid offload backend")
    assert isinstance(manager, OffloadManager)
    return manager
