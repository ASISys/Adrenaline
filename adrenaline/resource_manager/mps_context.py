import contextlib
import torch
import os

from cuda.bindings import driver
from typing import Optional
from vllm.logger import init_logger

logger = init_logger(__name__)

_cuda_initialized: bool = False


def error_handler(err):
    assert err == driver.CUresult.CUDA_SUCCESS, f"{err}: {driver.cuGetErrorString(err)}"


def init():
    global _cuda_initialized
    if _cuda_initialized:
        logger.warning("cuda already initialized")
        return
    err = driver.cuInit(0)[0]
    error_handler(err)
    _cuda_initialized = True
    torch.zeros(1, device="cuda")


def _create_sm_affinity(sm_count: int) -> driver.CUexecAffinityParam:
    affinity = driver.CUexecAffinityParam()
    affinity.type = driver.CUexecAffinityType.CU_EXEC_AFFINITY_TYPE_SM_COUNT
    affinity.param.smCount.val = sm_count
    return affinity


def get_sm_count() -> int:
    global _cuda_initialized
    assert _cuda_initialized
    err, affinity = driver.cuCtxGetExecAffinity(
        driver.CUexecAffinityType.CU_EXEC_AFFINITY_TYPE_SM_COUNT
    )
    error_handler(err)
    sm_count = affinity.param.smCount.val
    return sm_count


def create_cu_ctx(sm_count: Optional[int] = None, device: int = 0):
    global _cuda_initialized
    assert _cuda_initialized
    if sm_count is None:
        logger.info(f"create default ctx on device {device}")
        err, ctx = driver.cuCtxCreate(driver.CUctx_flags.CU_CTX_SCHED_AUTO, device)
        error_handler(err)
        return ctx

    assert (
        os.environ["CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING"] == "1"
    )
    affinity = _create_sm_affinity(sm_count)
    affinities = [affinity]
    err, ctx = driver.cuCtxCreate_v3(
        affinities, len(affinities), driver.CUctx_flags.CU_CTX_SCHED_AUTO, device
    )
    error_handler(err)
    return ctx


def pop_cu_ctx():
    global _cuda_initialized
    assert _cuda_initialized
    err, ctx = driver.cuCtxPopCurrent()
    error_handler(err)
    return ctx


@contextlib.contextmanager
def cuda_context(ctx: driver.CUcontext):
    global _cuda_initialized
    assert _cuda_initialized
    err, = driver.cuCtxPushCurrent(ctx)
    error_handler(err)
    yield
    err, _ = driver.cuCtxPopCurrent()
    error_handler(err)


def get_device_name(dev: int, length: int = 32) -> str:
    global _cuda_initialized
    assert _cuda_initialized
    err, bs = driver.cuDeviceGetName(length, dev)
    error_handler(err)
    dev_name = bs.decode("ascii").rstrip("\x00")
    return dev_name
