import contextlib
import nvtx
import torch

from vllm.logger import init_logger

logger = init_logger("vllm.utils.profile_utils")


@contextlib.contextmanager
def nvtx_profile():
    p = nvtx.Profile()
    p.enable()
    yield
    p.disable()


@contextlib.contextmanager
def cuda_profile():
    torch.cuda.profiler.start()
    yield
    torch.cuda.profiler.stop()


@contextlib.contextmanager
def profile(enable_nvtx: bool = True):
    if enable_nvtx:
        with nvtx_profile(), cuda_profile():
            yield
    else:
        with cuda_profile():
            yield
