#include <vector>
#include <cstdint>
#include <torch/extension.h>

std::vector<int64_t> get_ipc_mem_handle(torch::Tensor t);
void* open_ipc_mem_handle(std::vector<int64_t> handle_vec);
void close_ipc_mem_handle(void *addr);
void copy_tensor_to_buffer(torch::Tensor buffer, torch::Tensor in, bool async);
void copy_tensor_from_buffer_addr(torch::Tensor out, void *addr, bool async_op = true);
