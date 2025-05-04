# Distance Center Gaussian Rasterization Extension (Modified)

This directory contains a modified version of the original 3D Gaussian Splatting rasterization CUDA extension.

## Modifications

The primary goal of this modification is to add a new rendering mode focused on extracting geometric information rather than rendering final colors. Specifically, we added a new function (`rasterize_gaussians_center_depth`) to the Python bindings, leveraging internal CUDA kernels.

This new function aims to provide guidance maps for subsequent rendering stages:

1.  **Render Center Opacity Map (`center_point_opacity_map`):**
    *   Internally simulates rendering only the center point of each Gaussian (conceptually zero scale, full opacity).
    *   Outputs a 2D map where pixel `(u, v)` indicates if any Gaussian center projects onto it, considering only the Gaussian closest to the camera (based on view-space Z depth).
    *   This map can potentially guide pixel sampling.

2.  **Render Center Depth Map (`center_point_depth_map`):**
    *   Generated under the same zero-scale conditions.
    *   Outputs a 2D map where pixel `(u, v)` stores the **view-space Z depth** (distance from the Gaussian center to the camera plane along the view direction) of the closest Gaussian center projecting onto it.
    *   This map is intended to guide depth range sampling (`gs_depth`) in subsequent volume rendering stages, as it estimates the distance `t` along the ray where the surface is expected.

**Key Characteristics:**

*   This center-point rendering mode **does not compute gradients** and is optimized for speed when only these geometric maps are needed.
*   It relies on the existing preprocessing and sorting infrastructure but uses a modified or new rendering kernel (`compute_center_depth_kernel`).
*   The output depth is **view-space Z depth**, not Euclidean distance to the camera center.

**Intended Use:**

To provide guidance maps (`center_point_opacity_map`, `center_point_depth_map`) for a dual-renderer setup (e.g., `DualRenderers`), where these maps inform a secondary, possibly lower-resolution or sparser renderer (like a volume renderer using the depth map for sampling guidance). 