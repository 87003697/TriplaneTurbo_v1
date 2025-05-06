import sys
print(">>> SCRIPT STARTING...")

# print("--- sys.path ---")
# for path in sys.path:
#     print(path)
# print("--- end sys.path ---")

import torch
import torch.nn.functional as F
import numpy as np
import math
import os
from pathlib import Path
from torchvision.utils import save_image
import warnings
import sys
# import os
# from torchvision.utils import save_image 
# import warnings
# import sys 

try:
    from center_depth_rasterization import (
        rasterize_gaussians_center_depth 
    )
    CURRENT_EXT_LOADED = True
    print("Successfully imported current center_depth_rasterization extension.")
except ImportError as e:
    print("Could not import the current center_depth_rasterization extension.")
    print(f"Import error: {e}")
    print("Please make sure the extension in the current directory is compiled.")
    CURRENT_EXT_LOADED = False
    exit(1)


def setup_camera(w, h, fov_x_rad, fov_y_rad, near, far, cam_pos_vec, target_vec, up_vec, device):
    """Sets up projection and view matrices and related camera parameters."""
    tan_fovx = math.tan(fov_x_rad * 0.5)
    tan_fovy = math.tan(fov_y_rad * 0.5)

    # View matrix (lookAt)
    Z = F.normalize(cam_pos_vec - target_vec, dim=0)
    X = F.normalize(torch.cross(up_vec, Z, dim=0), dim=0)
    Y = F.normalize(torch.cross(Z, X, dim=0), dim=0) # Recalculate Y for orthogonality
    view = torch.zeros(4, 4, dtype=torch.float32, device=device)
    view[0, :3] = X
    view[1, :3] = Y
    view[2, :3] = Z
    view[:3, 3] = -torch.tensor([torch.dot(X, cam_pos_vec), torch.dot(Y, cam_pos_vec), torch.dot(Z, cam_pos_vec)], device=device)
    view[3, 3] = 1.0
    campos = cam_pos_vec.clone().detach()

    # Projection matrix (using gaussian_utils logic)
    fov_y_tensor = torch.tensor(fov_y_rad, device=device)
    fov_x_tensor = torch.tensor(fov_x_rad, device=device) 
    tanHalfFovY = torch.tan(fov_y_tensor * 0.5)
    tanHalfFovX = torch.tan(fov_x_tensor * 0.5) 
    top = tanHalfFovY * near
    bottom = -top
    right = tanHalfFovX * near 
    left = -right
    P_mat = torch.zeros(4, 4, device=device)
    z_sign = 1.0 
    P_mat[0, 0] = 2.0 * near / (right - left)
    P_mat[1, 1] = 2.0 * near / (top - bottom)
    P_mat[0, 2] = (right + left) / (right - left) 
    P_mat[1, 2] = (top + bottom) / (top - bottom) 
    P_mat[2, 2] = z_sign * (far + near) / (far - near) 
    P_mat[2, 3] = -2.0 * z_sign * far * near / (far - near)
    P_mat[3, 2] = z_sign 
    projmatrix = P_mat.T.contiguous() 
    
    # Calculate MVP matrix
    mvp_matrix = torch.matmul(projmatrix, view).contiguous() 
    print("MVP Matrix (Recalculated):\n", mvp_matrix)
    print("------------------------------------------")

    return view, projmatrix, mvp_matrix, campos, tan_fovy, tan_fovx


def generate_rays(w, h, viewmatrix, tan_fovx, tan_fovy, campos, device):
    """Generates camera rays (origin and direction) in world space."""
    fx = w / (2 * tan_fovx)
    fy = h / (2 * tan_fovy)
    cx = w / 2.0
    cy = h / 2.0

    # Create pixel grid
    px, py = torch.meshgrid(
        torch.arange(w, device=device, dtype=torch.float32),
        torch.arange(h, device=device, dtype=torch.float32),
        indexing='xy'
    )

    # Camera space directions (points on virtual image plane at z=-1)
    dirs_x = (px + 0.5 - cx) / fx
    dirs_y = -(py + 0.5 - cy) / fy # Flip Y to match common convention where +Y is down in image space
    dirs_z = -torch.ones_like(px)
    dirs_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1) # H, W, 3

    # Normalize directions (important!)
    dirs_cam = F.normalize(dirs_cam, dim=-1)

    # Get world space directions by rotating camera space directions
    # viewmatrix[:3, :3] maps world to camera, so its transpose maps camera to world
    cam_rot = viewmatrix[:3, :3].T
    rays_d = torch.matmul(dirs_cam.reshape(-1, 3), cam_rot).reshape(h, w, 3) # H, W, 3

    # Ray origins are the camera position, expanded to match rays_d shape
    rays_o = campos.reshape(1, 1, 3).expand(h, w, 3) # H, W, 3

    return rays_o.contiguous(), rays_d.contiguous()


def normalize_depth_for_vis(depth_map, min_val=None, max_val=None, background_value=None):
    """Normalizes a depth map to [0, 1] for visualization, handling background."""
    depth_map_vis = depth_map.squeeze().cpu().clone() # H, W

    valid_mask = torch.isfinite(depth_map_vis)
    if background_value is not None:
        valid_mask &= (depth_map_vis != background_value)

    if not valid_mask.any():
        # print("  Warning: No valid depth values found for normalization. Returning black image.")
        return torch.zeros_like(depth_map_vis).unsqueeze(0) # 1, H, W

    valid_depths = depth_map_vis[valid_mask]

    if min_val is None:
        min_val = torch.min(valid_depths)
    if max_val is None:
        max_val = torch.max(valid_depths)

    if max_val <= min_val: # Handle constant depth case
        normalized_depths = torch.full_like(valid_depths, 0.5) # Assign mid-gray
    else:
        normalized_depths = (valid_depths - min_val) / (max_val - min_val + 1e-8) # Add epsilon

    # Create normalized image, setting invalid/background pixels to 0 (black)
    normalized_image = torch.zeros_like(depth_map_vis)
    normalized_image[valid_mask] = normalized_depths
    normalized_image = torch.clamp(normalized_image, 0.0, 1.0)

    return normalized_image.unsqueeze(0) # Add channel dim: 1, H, W


# Replaced original save_depth_map and save_opacity_map
# Use torchvision.utils.save_image with normalize_depth_for_vis

# Original save_depth_map (kept for reference or potential reuse if needed)
# def save_depth_map(depth_map, filename, background_value=torch.finfo(torch.float32).max):
#     """Saves a depth map using imageio, handling potential background values (e.g., float max)."""
#     h, w = depth_map.shape
#     depth_map_cpu = depth_map.cpu()
#     valid_mask = depth_map_cpu != background_value
#     num_valid = valid_mask.sum().item()
#     if num_valid == 0:
#         img_normalized_uint8 = np.zeros((h, w), dtype=np.uint8)
#     else:
#         valid_depths = depth_map_cpu[valid_mask]
#         min_depth = valid_depths.min().item()
#         max_depth = valid_depths.max().item()
#         if max_depth <= min_depth:
#             normalized_depths = torch.full_like(valid_depths, 0.5)
#         else:
#             normalized_depths = (valid_depths - min_depth) / (max_depth - min_depth)
#         img_normalized_float = torch.zeros_like(depth_map_cpu, dtype=torch.float32)
#         img_normalized_float[valid_mask] = normalized_depths
#         img_normalized_uint8 = (img_normalized_float.clamp(0.0, 1.0) * 255).byte().numpy()
#     imageio.imwrite(str(filename), img_normalized_uint8)

# def save_opacity_map(opacity_map, filename):
#     """Saves a byte opacity map using imageio."""
#     img_uint8 = (opacity_map * 255).cpu().numpy()
#     imageio.imwrite(str(filename), img_uint8)


if __name__ == "__main__":
    # --- Parameters ---
    img_height = 128
    img_width = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path("./test_output_accuracy")
    output_dir.mkdir(exist_ok=True)
    print(f"Output images will be saved to: {output_dir.resolve()}")

    # --- Camera Base Settings ---
    cam_pos_vec = torch.tensor([0.0, 0.0, 3.0], device=device)
    target_vec = torch.tensor([0.0, 0.0, 0.0], device=device)
    up_vec = torch.tensor([0.0, 1.0, 0.0], device=device)
    fov_x_deg = 60.0
    fov_y_deg = 60.0
    near_plane = 0.1
    far_plane = 5.0
    fov_x_rad = math.radians(fov_x_deg)
    fov_y_rad = math.radians(fov_y_deg)

    # --- Setup Camera ---
    print("Setting up camera...")
    viewmatrix, projmatrix, mvp_matrix, campos, tan_fovy, tan_fovx = setup_camera(
        img_width, img_height,
        fov_x_rad, fov_y_rad,
        near_plane, far_plane,
        cam_pos_vec, target_vec, up_vec, device
    )
    print("Camera setup complete.")
    # <<< Add print statements for camera matrices and params >>>
    print("--- Camera Matrix & Params Check ---")
    # Correct printing for tensors
    print("View Matrix (W2C):\n", viewmatrix)
    print("Projection Matrix:\n", projmatrix)
    print(f"tanfovx: {tan_fovx:.4f}, tanfovy: {tan_fovy:.4f}")
    print(f"Cam Pos: {campos.cpu().numpy()}") # Print numpy array for clarity
    print("----------------------------------")

    # --- Test Case 1: Plane Point Cloud ---
    print("\n--- Test Case 1: Plane Point Cloud ---") 
    print("Creating 3D means (simple centered cube)...")
    P = 128 * 128 
    side_len = 1.0 
    z_center = -2.0 
    expected_view_depth = -5.0 
    means3D_flat = torch.zeros(P, 3, device=device)
    grid_size = int(math.sqrt(P))
    x_coords = torch.linspace(-side_len/2, side_len/2, grid_size, device=device)
    y_coords = torch.linspace(-side_len/2, side_len/2, grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
    means3D_flat[:, 0] = grid_x.flatten()
    means3D_flat[:, 1] = grid_y.flatten()
    means3D_flat[:, 2] = z_center 
    print(f"Generated {P} 3D means for plane test.")
    
    # ... (Manual Transform Check - Optional for this run) ...

    # --- Call Rasterizer (Plane Test) ---
    print("Calling rasterize_gaussians_center_depth for Plane...")
    scale_modifier = 1.0 
    kernel_size = 0.0 
    prefiltered = False 
    mvp_matrix_T = mvp_matrix.T.contiguous() 

    try:
        # <<< Recalculate MVP.T to pass to kernel >>>
        mvp_matrix_T = mvp_matrix.T.contiguous()
        
        # --- Expect TWO return values (Final GPU implementation) --- 
        center_opacity_map, final_depth_map = rasterize_gaussians_center_depth(
            means3D_flat,
            viewmatrix.T, 
            mvp_matrix_T, 
            tan_fovx,
            tan_fovy,
            img_height,
            img_width,
            scale_modifier,
            kernel_size,
            prefiltered,
            False
        )
        print("GPU Rasterization (Projection, Sorting, Selection) complete.")

        # --- Compare Depths (Using final_depth_map from GPU) ---
        print("Comparing GPU-selected depth with expected plane depth...")
        gt_depth_plane = torch.full((img_height, img_width), float('inf'), device=device, dtype=torch.float32)
        valid_mask_rendered = torch.isfinite(final_depth_map)
        gt_depth_plane[valid_mask_rendered] = expected_view_depth
        valid_mask_comparison = torch.isfinite(final_depth_map)

        if not valid_mask_comparison.any():
             print("Error: No finite pixels rendered for plane test!")
        else:
             rendered_depth_valid = final_depth_map[valid_mask_comparison]
             gt_depth_valid = gt_depth_plane[valid_mask_comparison]
             diff = torch.abs(rendered_depth_valid - gt_depth_valid)
             mean_abs_diff = torch.mean(diff)
             max_abs_diff = torch.max(diff)
             num_valid_pixels = valid_mask_comparison.sum().item()
             total_pixels = img_height * img_width
             valid_percentage = (num_valid_pixels / total_pixels) * 100
             print(f"Plane Depth Comparison Results (vs Expected Plane Depth {expected_view_depth}):")
             print(f"  Total Pixels: {total_pixels}")
             print(f"  Valid (Finite) Rendered Pixels: {num_valid_pixels} ({valid_percentage:.2f}%)")
             print(f"  Mean Absolute Difference: {mean_abs_diff:.6f}")
             print(f"  Max Absolute Difference:  {max_abs_diff:.6f}")
        
        # --- Save Plane Visualizations ---
        print("Saving visualizations for Plane Test...")
        vis_min_plane = torch.min(rendered_depth_valid).item() if valid_mask_comparison.any() else near_plane
        vis_max_plane = torch.max(rendered_depth_valid).item() if valid_mask_comparison.any() else far_plane
        gt_plane_vis = normalize_depth_for_vis(gt_depth_plane, vis_min_plane, vis_max_plane, background_value=torch.inf)
        save_image(gt_plane_vis, output_dir / "depth_ground_truth_plane.png")
        depth_rendered_plane_vis = normalize_depth_for_vis(final_depth_map, vis_min_plane, vis_max_plane, background_value=torch.inf)
        save_image(depth_rendered_plane_vis, output_dir / "depth_rendered_plane.png")
        opacity_vis_plane = center_opacity_map.float().unsqueeze(0)
        save_image(opacity_vis_plane, output_dir / "opacity_rendered_plane.png")
        print(f"Plane test visualizations saved to {output_dir.resolve()}")

        # --- Test Case 2: Tilted Plane (Temporarily Disabled) ---
        # <<< This section is commented out to ensure base functionality >>>

    except Exception as e:
        print(f"An error occurred during rasterization or processing: {e}")
        import traceback
        traceback.print_exc()

    print("Test script finished.")

# Remove multi-frame rendering and video creation logic
# ... (Removed code related to frame loop, video paths, imageio.mimwrite) ... 

# print(">>> SCRIPT FINISHED (basic imports only)") # <<< REMOVE THIS LINE 