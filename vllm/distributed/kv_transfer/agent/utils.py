import torch
import traceback

from concurrent.futures import Future

from vllm.logger import init_logger


logger = init_logger(__name__)

def handle_error(f: Future):
    exception = f.exception()
    if exception:
        logger.error(f"[rank{torch.distributed.get_rank()}] {exception}")
        traceback.print_exception(exception)
