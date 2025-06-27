#include "ipc_utils.h"

#include <iostream>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAStream.h>

#include "cuda_utils.h"

static std::vector<int64_t> cudaIpcMemHandle2Bytes(const cudaIpcMemHandle_t &handle) {
	std::vector<int64_t> result;
	for (size_t i = 0; i < sizeof(handle); i++) {
		result.push_back(((uint8_t*) &handle)[i]);
	}
	return result;
}

static cudaIpcMemHandle_t bytes2cudaIpcMemHandle(const std::vector<int64_t> &bytes) {
    assert(bytes.size() == sizeof(cudaIpcMemHandle_t));
    cudaIpcMemHandle_t handle;
    for (size_t i = 0; i < sizeof(handle); i++) {
        ((uint8_t*) &handle)[i] = bytes[i];
    }
    return handle;
}

std::vector<int64_t> get_ipc_mem_handle(torch::Tensor t) {
    cudaIpcMemHandle_t handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, t.data_ptr()));
    return cudaIpcMemHandle2Bytes(handle);
}

void* open_ipc_mem_handle(std::vector<int64_t> handle_vec) {
    const cudaIpcMemHandle_t handle = bytes2cudaIpcMemHandle(handle_vec);
    void* addr = nullptr;
    cudaError_t err = cudaIpcOpenMemHandle(&addr, handle, cudaIpcMemLazyEnablePeerAccess);
    if (err == cudaErrorPeerAccessUnsupported) {
        std::cerr << "Warning: Peer-to-peer access is unsupported on this platform." << '\n';
    } else {
        CUDA_CHECK(err);
    }
    return addr;
}

void close_ipc_mem_handle(void *addr) {
    CUDA_CHECK(cudaIpcCloseMemHandle(addr));
}

void copy_tensor_to_buffer(
    torch::Tensor buffer, torch::Tensor in, bool async_op
) {
    const void* src = in.const_data_ptr();
    void* dst = buffer.data_ptr();
    size_t count = in.numel() * in.dtype().itemsize();
    if (async_op) {
        auto stream = at::cuda::getCurrentCUDAStream();
        cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream);
    } else {
        cudaMemcpy(dst, src, count, cudaMemcpyDefault);
    }
}

void copy_tensor_from_buffer_addr(
    torch::Tensor out, void* addr, bool async_op
) {
    assert(out.is_contiguous());
    void* dst = out.data_ptr();
    size_t count = out.numel() * out.element_size();
    if (async_op) {
        auto stream = at::cuda::getCurrentCUDAStream();
        cudaMemcpyAsync(dst, addr, count, cudaMemcpyDefault, stream);
    } else {
        cudaMemcpy(dst, addr, count, cudaMemcpyDefault);
    }
}
