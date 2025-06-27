# This file is the wrapper for the adrenaline_ops module created by pybind11,
# so as to provide a more user-friendly interface, such as type notation.
import torch
import adrenaline_ops
from typing import Any

type IpcHandle = list[int]
type DevPtr = Any


def get_ipc_mem_handle(tensor: torch.Tensor) -> IpcHandle:
    ipc_handle = adrenaline_ops.get_ipc_mem_handle(tensor)
    return ipc_handle


def open_ipc_mem_handle(ipc_handle: IpcHandle) -> DevPtr:
    return adrenaline_ops.open_ipc_mem_handle(ipc_handle)


def close_ipc_mem_handle(addr: DevPtr):
    adrenaline_ops.close_ipc_mem_handle(addr)


def copy_tensor_to_buffer(
    buffer: torch.Tensor, tensor: torch.Tensor, async_op: bool = False
):
    adrenaline_ops.copy_tensor_to_buffer(buffer, tensor, async_op)


def copy_tensor_from_buffer_addr(
    tensor: torch.Tensor, addr: DevPtr, async_op: bool = False
):
    adrenaline_ops.copy_tensor_from_buffer_addr(tensor, addr, async_op)
