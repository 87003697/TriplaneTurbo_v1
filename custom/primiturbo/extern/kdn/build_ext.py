import torch
from torch.utils.cpp_extension import load

print("Building CUDA KNN extension...")
cuda_knn = load(
    name="cuda_knn",
    sources=["ext.cpp", "knn.cu", "knn_cpu.cpp"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    with_cuda=torch.cuda.is_available()
)

print("Extension loaded successfully!") 