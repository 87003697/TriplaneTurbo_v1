import torch
import torch.nn as nn
from torch.autograd import Function
import os
import warnings
import importlib.util
import sys

# --- Load CUDA extension via absolute path ---
ext_loaded = False
_frequency_encoding_cuda_ext = None

# Construct the expected path to the .so file within the build directory
_build_path = os.path.join(
    os.path.dirname(__file__), # Current file directory
    'build',
    f"lib.{os.uname().sysname.lower()}-{os.uname().machine}-cpython-{sys.version_info.major}{sys.version_info.minor}" # Platform specific build dir
)
_so_file_name = f"_frequency_encoding_cuda_ext.cpython-{sys.version_info.major}{sys.version_info.minor}-{os.uname().machine}-linux-gnu.so"
_so_file_path = os.path.join(_build_path, _so_file_name)

# Check if the .so file exists and try to load it
if os.path.exists(_so_file_path):
    try:
        spec = importlib.util.spec_from_file_location("_frequency_encoding_cuda_ext", _so_file_path)
        if spec and spec.loader:
             _frequency_encoding_cuda_ext = importlib.util.module_from_spec(spec)
             spec.loader.exec_module(_frequency_encoding_cuda_ext)
             # Verify if the functions are actually loaded
             if hasattr(_frequency_encoding_cuda_ext, 'freq_encode_forward_cuda') and \
                hasattr(_frequency_encoding_cuda_ext, 'freq_encode_backward_cuda'):
                 ext_loaded = True
                 print("CUDA extension loaded successfully via absolute path.")
             else:
                 warnings.warn("CUDA extension loaded via path, but functions are missing!")
                 _frequency_encoding_cuda_ext = None # Reset if functions missing
        else:
            warnings.warn(f"Could not create spec for CUDA extension at {_so_file_path}")
    except ImportError as e:
        warnings.warn(f"ImportError loading CUDA extension from {_so_file_path}: {e}. Check LD_LIBRARY_PATH.")
    except Exception as e:
        warnings.warn(f"Error loading CUDA extension from {_so_file_path}: {e}")
else:
    warnings.warn(f"CUDA extension .so file not found at expected build path: {_so_file_path}. Falling back.")

if not ext_loaded:
    warnings.warn("FrequencyEncoding: CUDA extension failed to load. Using PyTorch fallback.")

# Define the Python autograd Function to bridge PyTorch and CUDA C++
class _FrequencyEncodingCUDAFunc(Function):
    @staticmethod
    def forward(ctx, inputs, frequencies, n_output_dims):
        """Call the C++ forward function and save tensors for backward."""
        # Ensure inputs are contiguous
        inputs = inputs.contiguous()
        frequencies = frequencies.contiguous()

        # Call C++ CUDA forward function with new name
        outputs = _frequency_encoding_cuda_ext.freq_encode_forward_cuda(inputs, frequencies, n_output_dims)

        # Save inputs and frequencies for backward pass
        ctx.save_for_backward(inputs, frequencies)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """Call the C++ backward function."""
        # Ensure grad_outputs is contiguous
        grad_outputs = grad_outputs.contiguous()

        # Retrieve saved tensors
        inputs, frequencies = ctx.saved_tensors

        # Call C++ CUDA backward function with new name
        grad_inputs = _frequency_encoding_cuda_ext.freq_encode_backward_cuda(grad_outputs, inputs, frequencies)

        # Gradients must be returned for each input argument to forward:
        # inputs, frequencies, n_output_dims
        return grad_inputs, None, None # No gradient for frequencies or n_output_dims

class FrequencyEncoding(nn.Module):
    """Frequency Encoding (NeRF-style Positional Encoding) with CUDA acceleration.

    Args:
        in_channels (int): Number of input channels/dimensions.
        n_frequencies (int): Number of frequency bands.
        use_cuda (bool): Whether to attempt using the compiled CUDA extension. Defaults to True.
    """
    def __init__(self, in_channels: int, n_frequencies: int, use_cuda: bool = True):
        super().__init__()
        self.n_input_dims = in_channels
        self.n_frequencies = n_frequencies
        self.n_output_dims = self.n_input_dims * self.n_frequencies * 2
        # Use the globally determined ext_loaded status
        self.use_cuda = use_cuda and ext_loaded and torch.cuda.is_available()

        # Create frequency bands: 2^0, 2^1, ..., 2^(N_freqs-1)
        freq_bands = 2.0 ** torch.linspace(0.0, self.n_frequencies - 1, self.n_frequencies)
        self.register_buffer('freq_bands', freq_bands, persistent=False)

        if not self.use_cuda:
            # Print reason only if CUDA was requested but unavailable/unloaded
            if use_cuda:
                if not ext_loaded:
                     print("FrequencyEncoding Info: CUDA extension not loaded/found. Using PyTorch fallback.")
                elif not torch.cuda.is_available():
                     print("FrequencyEncoding Info: CUDA not available. Using PyTorch fallback.")

    def get_output_dim(self) -> int:
        return self.n_output_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_channels).

        Returns:
            torch.Tensor: Encoded tensor of shape (..., n_output_dims).
        """
        if self.use_cuda and x.is_cuda:
            # Use the custom CUDA kernel via Python autograd function
            # Assume _frequency_encoding_cuda_ext is loaded if self.use_cuda is True
            freq_bands_dev = self.freq_bands.to(x.device)
            return _FrequencyEncodingCUDAFunc.apply(x, freq_bands_dev, self.n_output_dims)
        else:
            # Fallback to PyTorch implementation
            return self._pytorch_forward(x)

    def _pytorch_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reference PyTorch implementation (matches the CUDA kernel output order)."""
        # x: (..., C)
        # freq_bands: (F)
        # Output: (..., C*F*2)
        # Order: [sin(f0*x_ch0), cos(f0*x_ch0), sin(f1*x_ch0), cos(f1*x_ch0), ..., sin(f0*x_ch1), cos(f0*x_ch1), ...]

        output_list = []
        freq_bands_dev = self.freq_bands.to(x.device)

        # Unsqueeze x and freq_bands for broadcasting
        # x_unsqueezed: (..., C, 1)
        # freq_bands_unsqueezed: (F,)
        x_unsqueezed = x.unsqueeze(-1)

        # Calculate all angles: shape (..., C, F)
        angles = x_unsqueezed * freq_bands_dev # Broadcasting happens here

        # Calculate sin and cos: shape (..., C, F)
        sines = torch.sin(angles)
        cosines = torch.cos(angles)

        # Stack sin and cos: shape (..., C, F, 2)
        stacked = torch.stack([sines, cosines], dim=-1)

        # Reshape to desired output: shape (..., C*F*2)
        output = stacked.reshape(list(x.shape[:-1]) + [self.n_output_dims])
        return output

# Example Usage (after building the extension):
if __name__ == '__main__':
    in_dim = 3
    n_freq = 10
    batch_size = 5

    # Import might change depending on how you structure imports
    # Assuming frequency_encoding is installed or in PYTHONPATH
    # from frequency_encoding import FrequencyEncoding

    encoder = FrequencyEncoding(in_dim, n_freq)
    print(f"Output dimension: {encoder.get_output_dim()}")

    # CPU Input (will use PyTorch fallback)
    print("\nTesting with CPU input...")
    cpu_input = torch.randn(batch_size, in_dim)
    cpu_output = encoder(cpu_input)
    print(f"Input shape: {cpu_input.shape}")
    print(f"Output shape: {cpu_output.shape}")

    if torch.cuda.is_available() and ext_loaded:
        print("\nTesting with LARGE CUDA input...")
        encoder = encoder.cuda()
        # Input for CUDA backward needs grad
        # Significantly larger input size (N = 500k)
        cuda_input = torch.randn(500000, in_dim, device='cuda', requires_grad=True)
        print(f"Input shape: {cuda_input.shape}")

        # Time the CUDA version
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        cuda_output = encoder(cuda_input)
        end_event.record()
        torch.cuda.synchronize()
        cuda_time = start_event.elapsed_time(end_event)
        print(f"CUDA Output shape: {cuda_output.shape}")
        print(f"CUDA Execution time: {cuda_time:.4f} ms")

        # Time the PyTorch fallback on GPU
        # Needs gradients for backward check
        cuda_input_torch = cuda_input.detach().clone().requires_grad_(True)
        encoder_fallback = FrequencyEncoding(in_dim, n_freq, use_cuda=False).cuda()
        start_event.record()
        pytorch_output_gpu = encoder_fallback(cuda_input_torch)
        end_event.record()
        torch.cuda.synchronize()
        pytorch_gpu_time = start_event.elapsed_time(end_event)
        print(f"PyTorch GPU Fallback Output shape: {pytorch_output_gpu.shape}")
        print(f"PyTorch GPU Fallback Execution time: {pytorch_gpu_time:.4f} ms")

        # == Forward Verification ==
        diff_fwd = torch.abs(cuda_output - pytorch_output_gpu).mean()
        print(f"\nMean absolute difference (Forward): {diff_fwd.item():.6f}")
        assert torch.allclose(cuda_output, pytorch_output_gpu, atol=1e-5), "Forward outputs do not match!"
        print("Forward outputs match.")

        # == Backward Verification ==
        print("\nVerifying gradients...")
        # Use the same random gradient for both
        grad_output = torch.randn_like(cuda_output)

        # Backward pass for CUDA version
        cuda_output.backward(gradient=grad_output)
        cuda_grad_input = cuda_input.grad.clone()
        cuda_input.grad.zero_() # Clear grad for next backward

        # Backward pass for PyTorch version
        pytorch_output_gpu.backward(gradient=grad_output)
        pytorch_grad_input = cuda_input_torch.grad.clone()

        # Compare gradients
        diff_bwd = torch.abs(cuda_grad_input - pytorch_grad_input).mean()
        print(f"Mean absolute difference (Backward - Input Grad): {diff_bwd.item():.6f}")
        # Increase tolerance slightly for backward pass due to potential minor float differences
        assert torch.allclose(cuda_grad_input, pytorch_grad_input, rtol=1e-4, atol=1e-4), "Backward gradients do not match!"
        print("Backward gradients match.")

    else:
        print("\nCUDA not available or extension not loaded. Skipping CUDA tests.") 