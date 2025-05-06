# Center Point Depth Rasterization CUDA Extension

## 项目目标

此 CUDA 扩展旨在实现一个自定义的 PyTorch 算子 `rasterize_gaussians_center_depth`，用于计算输入的一组 3D 点（例如高斯分布的中心点 `means3D`）投影到指定相机视角后的中心点深度图 (`center_depth_map`) 和可选的不透明度图 (`center_opacity_map`)。

核心要求是对于输出深度图的每个像素，只记录投影到该像素的所有 3D 点中，具有最近深度（即视图空间 Z 坐标最大，最接近 0 的负数）的那个点的深度值。

## 初始方法与挑战

最初的尝试是实现一个单一的 CUDA Kernel，该 Kernel 会为每个输入的 3D 点执行以下操作：
1.  坐标变换：将 3D 点从世界空间变换到视图空间，再到裁剪空间，最后到 NDC 和屏幕像素坐标 (`px`, `py`)。
2.  深度记录：计算视图空间的深度 (`p_view_z`)。
3.  原子写入：如果点有效（例如在视锥体内，像素坐标在图像边界内），则使用浮点原子操作 (`atomicMaxFloat`，通过 CAS 实现）将 `p_view_z` 写入全局输出深度缓冲区 `out_depth[py * W + px]` 的对应位置。

然而，这种直接的方法遇到了**极其顽固**的问题：
*   **浮点原子操作挂起/崩溃**: 只要 Kernel 中包含任何形式的浮点原子操作（无论是 `atomicMinFloat` 还是 `atomicMaxFloat`，无论是哪种 CAS 实现，甚至尝试作用于独立分配的 `cudaMalloc` 内存），在 PyTorch C++ 扩展环境中调用该 Kernel 就会导致 Python 测试脚本**无声地失败**（没有任何错误信息或输出）。
*   **独立测试成功**: 与之形成对比的是，包含相同 `atomicMaxFloat` (CAS 实现) 的独立 CUDA C 程序（不依赖 PyTorch 扩展）能够成功编译和运行，证明原子操作本身在当前 CUDA 环境/硬件下是可行的。
*   **其他调试困难**: 在此之前，还遇到了大量的编译链接错误、坐标变换逻辑错误（矩阵乘法约定、投影矩阵计算）、测试数据生成错误（点云大部分在视锥外）等问题，这些都通过分步调试和验证逐一解决。

核心症结在于 **PyTorch C++ 扩展环境与浮点原子操作（CAS 实现）之间存在某种未知的冲突或不兼容**，导致 Kernel 无法正常执行。

## 多阶段 GPU 解决方案 (最终实现)

为了绕开无法解决的浮点原子操作挂起问题，最终采用了多阶段的 GPU 计算方案，避免了在 Kernel 中直接进行全局原子最大值写入：

1.  **Kernel 1: 投影 (`projectPointsKernel`)**: 
    *   并行处理所有输入的 3D 点。
    *   执行完整的坐标变换 (World -> View -> Clip -> NDC -> Screen -> Pixel)。
    *   计算每个点的 1D 像素索引 (`pix_id = py * W + px`) 和视图深度 (`p_view_z`)。
    *   进行有效性检查（是否在边界内、深度是否有效）。
    *   将有效的 `(pix_id, p_view_z)` 对写入两个临时的 GPU 缓冲区（大小为 P）。无效点写入特殊标记（如 pix_id = -1）。

2.  **GPU 排序 (Thrust)**:
    *   使用 NVIDIA Thrust 库的高性能并行排序算法。
    *   对临时缓冲区进行两次排序：
        a.  首先，按 `view_depths` **降序**排序 (`thrust::sort_by_key`，主键是深度，值是像素索引)。这使得具有较大深度值（更接近 0 的负数）的点排在前面。
        b.  然后，对上一步的结果，按 `pixel_indices` **升序**进行**稳定排序** (`thrust::stable_sort_by_key`，主键是像素索引，值是深度)。稳定排序保证了对于相同的像素索引，之前按深度降序的顺序得以保留。
    *   排序后，对于每个像素索引 `pix_id`，所有投影到该像素的点在缓冲区中是连续存放的，并且具有最大深度（最近）的点位于该连续段的第一个。 

3.  **Kernel 2: 选择 (`selectDepthKernel`)**: 
    *   并行处理排序后的 P 个条目。
    *   每个线程 `idx` 检查其 `sorted_pixel_indices[idx]` 是否与前一个线程 `idx-1` 的不同（或 `idx == 0`）。
    *   如果不同，则表明这是该像素索引 `pix_id` 在排序列表中**首次出现**。
    *   由于之前的排序策略，首次出现的条目对应的 `sorted_view_depths[idx]` 即为该像素 `pix_id` 的最近深度。
    *   该线程**直接 (非原子地)** 将 `sorted_view_depths[idx]` 写入最终输出深度图 `out_depth[pix_id]` 的对应位置。因为每个像素只由一个线程（第一个遇到的线程）写入，所以不存在数据竞争。

这种多阶段方法成功地在 GPU 上完成了最近点深度的计算，避免了原子操作问题，并通过了平面点云的精确性验证。

## 已克服的关键问题

*   **编译与链接**: 解决了大量头文件包含、库依赖 (glm)、符号未定义等编译链接问题。
*   **C++/CUDA 语法与接口**: 修正了函数签名不匹配、返回值数量不一致、指针类型错误等问题。
*   **坐标变换**: 诊断并修复了由于矩阵乘法约定（行向量 vs 列向量）和投影矩阵计算方式不一致导致的坐标错误。
*   **测试数据**: 识别并修正了原始测试脚本中点云生成逻辑的错误，该错误导致大部分点落在视锥体外。
*   **浮点原子操作挂起**: 通过切换到多阶段 GPU 实现，绕开了在 PyTorch 扩展环境中调用浮点原子操作（CAS 实现）导致程序挂起的根本性障碍。
*   **实际流程集成问题**: 解决了在集成到完整渲染管线时，算子输出为空的问题（详见调试过程）。

## 最终实现方案

*   **核心文件**: `rasterize_points.cu`, `rasterize_points.h`, `ext.cpp`
*   **主要组件**: `projectPointsKernel` (CUDA), `Thrust` 库排序 (`sort_by_key`, `stable_sort_by_key`), `selectDepthKernel` (CUDA)。
*   **接口**: Python 通过 `center_depth_rasterization.rasterize_gaussians_center_depth` 调用 C++ 绑定，最终执行 CUDA 实现。
*   **功能**: 计算输入点云 (`means3D`) 投影到相机视锥后的中心点最近正深度图 (`center_depth_map`) 和对应的二值不透明度图 (`center_opacity_map`)。
*   **测试**: `test_new_center_depth.py` 使用平面点云、重叠点和视图外点测试验证了算子的核心逻辑（投影、裁剪、排序、选择）。

## 调试与修复过程 (详细)

在 `test_new_center_depth.py` 中通过基本测试后，将算子集成到实际渲染流程 (`generative_space_3dgs_renderer_v5.py`) 时遇到了输出为空的问题。通过一系列详细的调试步骤最终定位并解决了问题：

1.  **问题现象**: 自定义算子在实际流程中始终输出空的深度图和不透明度图（无有限深度值，Opacity Coverage 0%），尽管输入检查显示点云和矩阵均有效。但在独立的 `test_new_center_depth.py` 中，相同的算子逻辑（使用精心构造的数据）可以正确工作。

2.  **调试方向 1: 矩阵约定与传递**:
    *   检查 Python 端传递给 CUDA 的 View/MVP 矩阵的约定（Row-Major/Col-Major, W2C/W2C.T, MVP/MVP.T）与 CUDA 核函数 `transformPoint4x4` 的数学实现（行向量左乘 `v @ M`）是否匹配。
    *   **确认**: `transformPoint4x4` 执行 `v @ M`。标准图形矩阵 M 通常用于 `M @ v`。因此，Python 端需要传递标准矩阵的转置 `M.T` 给 CUDA。
    *   检查 `gaussian_utils.py` 中的 `Camera` 类和 `generative_space_3dgs_renderer_v5.py` 中的调用，确认传递给自定义算子的 `viewmatrix` 参数确实是 `W2C.T`，`mvp_matrix_T` 参数确实是 `(P @ W2C).T`。

3.  **调试方向 2: 投影/逆投影一致性 (像素映射)**:
    *   对比了 `test_new_center_depth.py` 中用于生成测试数据的逆投影函数 `unproject_pixels_to_world`（基于像素中心 `px+0.5`）和 CUDA 核函数 `projectPointsKernel` 中将屏幕坐标映射到像素索引的逻辑。
    *   最初 CUDA 使用 `roundf(coord - 0.5f)`（等价于 `floor`），后尝试改为 `roundf(coord)`（四舍五入到最近）。
    *   通过在 `test_new_center_depth.py` 中简化 Overlap 测试（只用一个中心像素）并在 CUDA 中打印投影结果，发现无论使用 `floor` 还是 `roundf`，从 `px+0.5` 逆投影再正向投影回来的像素索引都与初始像素索引存在偏差。
    *   最终定位到是 `unproject_pixels_to_world` 中对 NDC Y 坐标的翻转 `ndc_y = -((...) * 2.0 - 1.0)` 与 CUDA 端从 NDC 到屏幕坐标的转换（没有翻转）之间存在**约定冲突**。**移除 `unproject_pixels_to_world` 中的负号**，并保持 CUDA 端使用 `floor` 映射 (`roundf(coord - 0.5f)`)，解决了单元测试中的 Overlap 问题。

4.  **调试方向 3: 裁剪逻辑**: 
    *   在 CUDA 核函数 `projectPointsKernel` 中逐步添加裁剪逻辑：先是 View Z near/far 裁剪，然后是 NDC X/Y 裁剪。
    *   发现使用 `gaussian_utils.py` 中定义的投影矩阵计算出的 NDC Z 坐标范围超出了 `[-1, 1]`，导致 `abs(ndc.z) <= 1` 会错误地过滤掉有效点。因此**移除了 NDC Z 裁剪**，只保留 View Z near/far 和 NDC X/Y 裁剪。
    *   修复了 Clipping 测试用例的设计，确保 "good" 点的深度小于任何可能通过裁剪的 "bad" 点，验证了裁剪逻辑能正确过滤无效点且不影响有效点的深度选择。

5.  **调试方向 4: 函数签名与参数传递 (最终根源)**:
    *   在所有上述修正后，实际流程 (`exp1_DF415_debug_v3.sh`) 中算子输出仍然为空。
    *   通过在 CUDA 核函数中打印传入的 `w2c_matrix` 的特定元素（用于计算正确 View Z），发现其值与 Python 端传入的原始 W2C 矩阵**不符**。
    *   进一步检查发现，在 C++/CUDA 函数签名、Pybind11 绑定 (`m.def`) 以及 CUDA Kernel 内部的参数列表中，新添加的 `w2c_matrix` 参数被错误地放在了 `mvp_matrix_T` **之前**，而 Python 调用时是按 `viewmatrix`, `mvp_matrix_T`, `w2c_matrix` 的顺序传入。这导致 CUDA Kernel 内部计算 View Z 时，实际上使用了错误的 `mvp_matrix_T` 的内存地址！
    *   **最终解决方案**: 修正 `rasterize_points.h`, `ext.cpp`, `rasterize_points.cu` 中的所有相关函数签名和核函数调用，确保 `w2c_matrix` 参数始终位于 `mvp_matrix_T` **之后**。

**最终状态**: 修正参数顺序后，自定义算子在实际流程中可以正确计算并输出非空的中心点深度图和不透明度图。

## 使用注意事项

调用 Python 函数 `center_depth_rasterization.rasterize_gaussians_center_depth` 时，必须注意参数的意义和约定：

*   **`means3D`**: `[N, 3]` Float Tensor，世界坐标系下的点云。
*   **`viewmatrix`**: `[4, 4]` Float Tensor，**必须是 World-to-Camera (W2C) 矩阵的转置 (`W2C.T`)**。确保它是 C-contiguous。
*   **`mvp_matrix_T`**: `[4, 4]` Float Tensor，**必须是标准 MVP 矩阵 (P @ W2C) 的转置 (`(P @ W2C).T`)**。确保它是 C-contiguous。
*   **`w2c_matrix`**: `[4, 4]` Float Tensor，**必须是原始的 World-to-Camera (W2C) 矩阵** (即 `viewmatrix` 的转置)。**必须是 Row-Major 且 C-contiguous** (`.contiguous()`)。此矩阵仅用于在 CUDA 内部正确计算 View Z 进行裁剪。
*   **`tan_fovx`, `tan_fovy`**: `float`，分别为相机水平和垂直半视场角 (`fov/2`) 的正切值。
*   **`image_height`, `image_width`**: `int`，输出图像的高和宽。
*   **`near_plane`, `far_plane`**: `float`，相机的近裁剪面和远裁剪面距离。**必须与用于计算 `mvp_matrix_T` 的值一致**。
*   **`scale_modifier`, `kernel_size`, `prefiltered`, `debug`**: 这些参数当前在 CUDA 核函数中未使用，可以传递默认值（1.0, 0.0, False, False）。

**返回值**:
*   `tuple`: 包含两个 Tensor `(opacity_map, depth_map)`。
    *   `opacity_map`: `[H, W]` Float Tensor，值为 0.0 或 1.0。
    *   `depth_map`: `[H, W]` Float Tensor，值为正数深度（距离相机距离，近似 `-view_z`）或 `inf`。

## 如何构建和测试

1.  确保您的环境已安装 PyTorch 和 CUDA Toolkit。
2.  进入 `custom/primiturbo/extern/diff-gaussian-rasterization-center-depth` 目录。
3.  运行编译命令: `pip install -e .`
4.  运行测试脚本: `python test_new_center_depth.py` (确保当前 conda 环境已激活)
5.  检查 `test_output_accuracy` 目录下的输出图像和终端打印的比较结果。 