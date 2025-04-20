# CUDA Accelerated Frequency Encoding Module

This directory contains a custom PyTorch module `FrequencyEncoding` that implements NeRF-style positional encoding (frequency encoding) using a custom CUDA kernel for acceleration.

It provides significant speedups compared to pure PyTorch implementations, especially for large input tensors.

## Dependencies

*   **PyTorch:** Developed and tested with PyTorch (ensure it's installed with CUDA support).
*   **CUDA Toolkit:** Required for compiling the CUDA kernel. Ensure `nvcc` is in your system PATH.
*   **C++ Compiler:** A compatible C++ compiler (like g++) is needed.
*   **Python:** Python environment (e.g., Conda) where PyTorch is installed.

## Installation

1.  **Navigate to this Directory:**
    Open your terminal and change the directory to where this README file is located:
    ```bash
    cd path/to/your/project/custom/primiturbo/extern/frequency_encoding
    ```

2.  **Activate Your Python Environment:**
    Make sure the correct Python environment (e.g., conda environment) where you intend to use this module is activated.
    ```bash
    conda activate your_environment_name
    ```

3.  **Compile the Extension:**
    You have two main options:

    *   **Option A: Build In-Place (Recommended for Development)**
        This compiles the extension (`.so` file) directly into the current directory within a `build` folder. The Python code in this directory will be able to find and use it directly (provided runtime dependencies are met).
        ```bash
        python setup.py build_ext --inplace
        ```

    *   **Option B: Install into Environment**
        This compiles the extension and installs it into your Python environment's `site-packages` directory. This makes it importable like any other installed package, but might require root/administrator privileges depending on your environment setup.
        ```bash
        # Clean previous builds first (optional but recommended)
        # rm -rf build dist *.egg-info 
        python setup.py install
        ```

4.  **Runtime Dependency (IMPORTANT): `LD_LIBRARY_PATH`**
    The compiled extension links against PyTorch's C++ libraries (like `libc10.so`). At runtime, the dynamic linker needs to find these libraries. Often, simply activating the conda environment is *not* enough.

    Before running any Python script that imports this module, you **must** ensure PyTorch's library path is in the `LD_LIBRARY_PATH` environment variable. You can do this temporarily in your terminal session:

    ```bash
    # Activate your environment first
    conda activate your_environment_name

    # Find PyTorch lib path and export LD_LIBRARY_PATH
    export PYTORCH_LIB_PATH=$(python -c 'import torch; import os; print(os.path.dirname(torch.__file__))')/lib
    export LD_LIBRARY_PATH=$PYTORCH_LIB_PATH:$LD_LIBRARY_PATH

    # Now run your python script
    # python your_script.py
    ```

    For a more permanent solution, consider adding the export lines to your shell configuration file (e.g., `~/.bashrc`, `~/.zshrc`) or your conda environment's activation hooks.

## Usage

Once compiled and with `LD_LIBRARY_PATH` set correctly, you can import and use the module in your Python code:

```python
import torch
from frequency_encoding import FrequencyEncoding # Assuming this dir is in PYTHONPATH or installed

# Parameters
in_channels = 3       # Number of input dimensions (e.g., 3 for XYZ coordinates)
n_frequencies = 10   # Number of frequency bands (like in NeRF)

# Instantiate the encoder
# It will automatically try to use CUDA if available and compiled.
# Set use_cuda=False to force PyTorch fallback.
encoder = FrequencyEncoding(in_channels=in_channels, n_frequencies=n_frequencies)

# Move encoder to GPU if using CUDA
if encoder.use_cuda:
    encoder = encoder.cuda()
    print("Using CUDA backend.")
else:
    print("Using PyTorch fallback.")

# Create some input data
if encoder.use_cuda:
    # Large batch on GPU
    input_tensor = torch.randn(100000, in_channels, device='cuda')
else:
    # Smaller batch on CPU
    input_tensor = torch.randn(10, in_channels, device='cpu')

# Apply the encoding
encoded_tensor = encoder(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output dimension: {encoder.get_output_dim()}")
print(f"Encoded shape: {encoded_tensor.shape}")
# Expected output shape: (batch_size, ..., in_channels * n_frequencies * 2)
```

## Troubleshooting

*   **`ImportError: libc10.so: cannot open shared object file...`**: This almost certainly means `LD_LIBRARY_PATH` is not set correctly at runtime. See Step 4 in Installation.
*   **`AttributeError: module '_frequency_encoding_cuda_ext' has no attribute ...`**: This might happen if Python loads an older/cached version of the compiled module, or if the installation/compilation failed silently. Try cleaning build artifacts (`rm -rf build dist *.egg-info`) and recompiling/reinstalling. Ensure you restart your Python kernel/interpreter after reinstalling.
*   **Compilation Errors:** Ensure `nvcc` and a compatible `g++` are installed and in the PATH. Check CUDA/PyTorch version compatibility if errors mention specific CUDA API mismatches. 