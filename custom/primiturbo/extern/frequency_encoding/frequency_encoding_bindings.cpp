#include <torch/extension.h>
#include <vector>

// Forward declarations
torch::Tensor frequency_encoding_cuda_forward(
    torch::Tensor inputs,
    torch::Tensor frequencies,
    int n_output_dims
);

torch::Tensor frequency_encoding_cuda_backward(
    torch::Tensor grad_outputs,
    torch::Tensor inputs,
    torch::Tensor frequencies
);

// Remove the C++ autograd::Function class definition
// class FrequencyEncodingCudaFunction : public torch::autograd::Function<FrequencyEncodingCudaFunction> { ... };

// Bind the forward and backward functions directly
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("freq_encode_forward_cuda", &frequency_encoding_cuda_forward,
          "Frequency Encoding forward (CUDA)");
    m.def("freq_encode_backward_cuda", &frequency_encoding_cuda_backward,
          "Frequency Encoding backward (CUDA)");
} 