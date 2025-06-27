import enum
from typing import Optional
from adrenaline.proxy.utils import RequestRuntimeData


class Backend(enum.Enum):
    ALWAYS = enum.auto()
    NEVER = enum.auto()
    LOADAWARE = enum.auto()
    PERCENTAGE = enum.auto()


backend_names = [backend.name.lower() for backend in Backend]


def name_to_enum(backend_name: str):
    assert backend_name is not None
    backend_members = Backend.__members__
    backend_name = backend_name.upper()
    assert backend_name in backend_members
    return Backend[backend_name]


class OffloadManager:

    def make_offload_decision(
        self,
        req_id: Optional[str] = None,
        req_data: Optional[RequestRuntimeData] = None,
    ) -> bool:
        raise NotImplementedError
