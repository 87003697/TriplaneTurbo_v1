#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For M_PI, sinf, cosf

// CUDA kernel for frequency encoding
__global__ void frequency_encoding_kernel(
    const float* __restrict__ inputs, // Shape: (N, n_input_dims)
    const float* __restrict__ freqs,  // Shape: (n_frequencies)
    float* __restrict__ outputs,      // Shape: (N, n_output_dims)
    const int N,                      // Total number of elements (batch_size * ...)
    const int n_input_dims,
    const int n_frequencies,
    const int n_output_dims            // Should be n_input_dims * n_frequencies * 2
) {
    // Calculate the global thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds: each thread processes one element (across all input dims)
    if (idx >= N) {
        return;
    }

    // Iterate over each input dimension for the current element
    for (int dim_idx = 0; dim_idx < n_input_dims; ++dim_idx) {
        // Calculate the index in the flattened input tensor
        const int input_idx = idx * n_input_dims + dim_idx;
        const float x = inputs[input_idx];

        // Iterate over each frequency
        for (int freq_idx = 0; freq_idx < n_frequencies; ++freq_idx) {
            const float freq = freqs[freq_idx];
            const float val = freq * x;

            // Calculate the base index in the output tensor for this input dim and freq
            // Output layout: [dim0_freq0_sin, dim0_freq0_cos, dim0_freq1_sin, dim0_freq1_cos, ..., dim1_freq0_sin, ...]
            const int output_base_idx = idx * n_output_dims + dim_idx * (n_frequencies * 2) + freq_idx * 2;

            // Calculate sin and cos and write to output
            outputs[output_base_idx + 0] = sinf(val);
            outputs[output_base_idx + 1] = cosf(val);
        }
    }
}

// C++ wrapper function callable from Python
torch::Tensor frequency_encoding_cuda_forward(
    torch::Tensor inputs,         // Input tensor (..., n_input_dims)
    torch::Tensor frequencies,    // Frequencies tensor (n_frequencies)
    int n_output_dims            // Pre-calculated output dimension
) {
    // Input validation
    TORCH_CHECK(inputs.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(frequencies.is_cuda(), "Frequencies tensor must be a CUDA tensor");
    TORCH_CHECK(inputs.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(frequencies.scalar_type() == torch::kFloat32, "Frequencies tensor must be float32");
    TORCH_CHECK(frequencies.dim() == 1, "Frequencies tensor must be 1D");

    inputs = inputs.contiguous(); // Ensure contiguous memory layout

    const auto input_dims = inputs.dim();
    const int n_input_dims = inputs.size(input_dims - 1);
    const int n_frequencies = frequencies.size(0);

    // Calculate total number of elements to process
    long N = 1;
    for (int i = 0; i < input_dims - 1; ++i) {
        N *= inputs.size(i);
    }

    // Create output tensor shape
    auto output_shape = inputs.sizes().vec();
    output_shape[input_dims - 1] = n_output_dims;
    auto outputs = torch::empty(output_shape, inputs.options());

    // Check consistency
    TORCH_CHECK(n_output_dims == n_input_dims * n_frequencies * 2,
                "n_output_dims mismatch: expected ", n_input_dims * n_frequencies * 2,
                ", got ", n_output_dims);

    if (N == 0) { // Handle empty input
        return outputs;
    }

    // Kernel launch configuration
    const int threads_per_block = 256; // Common block size
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    frequency_encoding_kernel<<<blocks_per_grid, threads_per_block>>>(
        inputs.data_ptr<float>(),
        frequencies.data_ptr<float>(),
        outputs.data_ptr<float>(),
        static_cast<int>(N), // Pass N as int
        n_input_dims,
        n_frequencies,
        n_output_dims
    );

    // Check for kernel launch errors (optional but good practice)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return outputs;
}

// --- Backward Pass ---

// CUDA kernel for frequency encoding backward pass
__global__ void frequency_encoding_kernel_backward(
    const float* __restrict__ grad_outputs, // Shape: (N, n_output_dims)
    const float* __restrict__ inputs,       // Shape: (N, n_input_dims) (Saved from forward)
    const float* __restrict__ freqs,        // Shape: (n_frequencies) (Saved from forward)
    float* __restrict__ grad_inputs,        // Shape: (N, n_input_dims) (Output: Gradients wrt inputs)
    const int N,                            // Total number of elements
    const int n_input_dims,
    const int n_frequencies,
    const int n_output_dims
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    // Accumulate gradients for each input dimension
    for (int dim_idx = 0; dim_idx < n_input_dims; ++dim_idx) {
        float grad_acc = 0.0f;
        const int input_idx = idx * n_input_dims + dim_idx;
        const float x = inputs[input_idx];

        for (int freq_idx = 0; freq_idx < n_frequencies; ++freq_idx) {
            const float freq = freqs[freq_idx];
            const float val = freq * x;
            const float sin_val = sinf(val);
            const float cos_val = cosf(val);

            // Index for sin(freq*x) and cos(freq*x) in grad_outputs
            const int output_base_idx = idx * n_output_dims + dim_idx * (n_frequencies * 2) + freq_idx * 2;
            const float grad_sin = grad_outputs[output_base_idx + 0];
            const float grad_cos = grad_outputs[output_base_idx + 1];

            // Chain rule contribution:
            // dL/dx += dL/d(sin) * d(sin)/dx + dL/d(cos) * d(cos)/dx
            // dL/dx += grad_sin * (f * cos(f*x)) + grad_cos * (-f * sin(f*x))
            grad_acc += freq * (grad_sin * cos_val - grad_cos * sin_val);
        }
        // Write the accumulated gradient for this input dimension
        // Note: If multiple threads wrote to the same grad_inputs location,
        // atomicAdd would be needed. Here, each thread calculates the full grad
        // for its corresponding input elements, so direct write is fine.
        grad_inputs[input_idx] = grad_acc;
    }
}

// C++ wrapper for backward pass
torch::Tensor frequency_encoding_cuda_backward(
    torch::Tensor grad_outputs,   // Gradient wrt outputs (dL/dy)
    torch::Tensor inputs,         // Original inputs (saved from forward)
    torch::Tensor frequencies    // Original frequencies (saved from forward)
) {
    // Input validation
    TORCH_CHECK(grad_outputs.is_cuda(), "grad_outputs must be a CUDA tensor");
    TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
    TORCH_CHECK(frequencies.is_cuda(), "frequencies must be a CUDA tensor");
    TORCH_CHECK(grad_outputs.scalar_type() == torch::kFloat32, "grad_outputs must be float32");
    TORCH_CHECK(inputs.scalar_type() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(frequencies.scalar_type() == torch::kFloat32, "frequencies must be float32");
    TORCH_CHECK(frequencies.dim() == 1, "Frequencies tensor must be 1D");

    grad_outputs = grad_outputs.contiguous();
    inputs = inputs.contiguous();

    const auto grad_output_dims = grad_outputs.dim();
    const int n_output_dims = grad_outputs.size(grad_output_dims - 1);
    const auto input_dims_vec = inputs.sizes(); // Get original input shape

    const int n_input_dims = inputs.size(inputs.dim() - 1);
    const int n_frequencies = frequencies.size(0);

    // Calculate N
    long N = 1;
    for (int i = 0; i < inputs.dim() - 1; ++i) {
        N *= inputs.size(i);
    }

    // Create gradient tensor for inputs, matching original input shape
    auto grad_inputs = torch::zeros_like(inputs);

    TORCH_CHECK(n_output_dims == n_input_dims * n_frequencies * 2, "Backward n_output_dims mismatch");

    if (N == 0) {
        return grad_inputs;
    }

    // Kernel launch configuration (same as forward)
    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Launch the backward kernel
    frequency_encoding_kernel_backward<<<blocks_per_grid, threads_per_block>>>(
        grad_outputs.data_ptr<float>(),
        inputs.data_ptr<float>(),
        frequencies.data_ptr<float>(),
        grad_inputs.data_ptr<float>(),
        static_cast<int>(N),
        n_input_dims,
        n_frequencies,
        n_output_dims
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA backward kernel launch failed: ", cudaGetErrorString(err));

    return grad_inputs;
} 