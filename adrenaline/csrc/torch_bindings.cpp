#include <pybind11/pybind11.h>

#include "ipc_utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "adrenaline custom operators";
    m.def("get_ipc_mem_handle", &get_ipc_mem_handle);
    m.def("open_ipc_mem_handle", &open_ipc_mem_handle);
    m.def("close_ipc_mem_handle", &close_ipc_mem_handle);
    m.def("copy_tensor_to_buffer", &copy_tensor_to_buffer);
    m.def("copy_tensor_from_buffer_addr", &copy_tensor_from_buffer_addr);
}
