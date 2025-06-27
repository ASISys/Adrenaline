import time
import aiohttp
import os

from typing import AsyncGenerator, Optional
from vllm.logger import init_logger

logger = init_logger("vllm.proxy.utils")

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


class RequestRuntimeData:

    def __init__(self, seq_len: int, max_seq_len: int):
        self.seq_len = seq_len
        self.max_seq_len = max_seq_len

        self.timestamp: Optional[float] = None
        self.itl: Optional[float] = None

    def update_for_decode(self):
        self.seq_len += 1
        new_timestamp = time.perf_counter()
        if self.timestamp is not None:
            self.itl = new_timestamp - self.timestamp
        self.timestamp = new_timestamp


async def forward_request(url, data):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        logger.debug(f"Forwarding request {data["request_id"]} to {url}")
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content:
                    yield chunk_bytes
            else:
                logger.error(f"Error occurred: {response.status}")
                raise Exception(f"Error occurred: {response.status}")


async def dummy_exec(generator: AsyncGenerator):
    async for _ in generator:
        continue
