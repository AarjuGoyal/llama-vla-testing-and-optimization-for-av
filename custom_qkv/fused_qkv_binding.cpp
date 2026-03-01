#include<torch/extension.h>
#include<pybind11/pybind11.h>
// #include<pybind/stl.h>

torch::Tensor fused_qkv_cuda(torch::Tensor input, torch::Tensor weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_qkv_cuda, "Fused QKV GEMM (CUDA)");

}