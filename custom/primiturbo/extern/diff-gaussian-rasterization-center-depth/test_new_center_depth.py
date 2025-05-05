import sys
print("--- sys.path ---")
for path in sys.path:
    print(path)
print("--- end sys.path ---")

import torch
import torch.nn.functional as F
import numpy as np
import math
import os
from pathlib import Path
# from PIL import Image # No longer needed
from torchvision.utils import save_image # Use torchvision for simpler saving
import warnings
# import imageio # No longer needed for video
# import time # No longer needed
import sys # Import sys for float_info

# --- Import Custom Extension ---
try:
    # Note: We are importing from the *current* directory's extension
    from diff_gaussian_rasterization import (
        # GaussianRasterizationSettings, # REMOVE THIS LINE (Commented out)
        # GaussianRasterizer,            # REMOVE THIS LINE (Commented out)
        rasterize_gaussians_center_depth # Import the new function
    )
    CURRENT_EXT_LOADED = True
    print("Successfully imported current diff_gaussian_rasterization extension.")
except ImportError as e:
    print("Could not import the current diff_gaussian_rasterization extension.")
    print(f"Import error: {e}")
    print("Please make sure the extension in the current directory (custom/primiturbo/extern/diff-gaussian-rasterization) is compiled (e.g., run 'pip install -e .' in this directory).")
    CURRENT_EXT_LOADED = False
    exit(1)


def setup_camera(w, h, fov_x_rad, fov_y_rad, near, far, cam_pos_vec, target_vec, up_vec, device):
    """Sets up projection and view matrices and related camera parameters."""
    # Projection matrix
    tan_fovx = math.tan(fov_x_rad * 0.5)
    tan_fovy = math.tan(fov_y_rad * 0.5)

    proj = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj[0, 0] = 1.0 / tan_fovx
    proj[1, 1] = 1.0 / tan_fovy
    proj[2, 2] = (far + near) / (far - near)
    proj[2, 3] = - (2.0 * far * near) / (far - near)
    proj[3, 2] = 1.0
    # proj = proj.cuda() # Use device argument

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
    # view = view.cuda() # Use device argument

    # Camera center tensor
    campos = cam_pos_vec.clone().detach()

    # Full projection (optional, not needed by our function)
    # full_proj = torch.matmul(proj.T, view.T).T

    return view, proj, campos, tan_fovy, tan_fovx


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
    # num_gaussians no longer needed, determined by image size
    img_height = 128
    img_width = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # num_frames = 120 # Single frame test
    # video_fps = 30

    # --- Output Directories ---
    # video_output_dir = Path("./test_output_video") # No video output
    output_dir = Path("./test_output_accuracy") # New output dir name
    output_dir.mkdir(exist_ok=True)
    # frame_output_dir = video_output_dir / "frames"
    # video_output_dir.mkdir(exist_ok=True)
    # frame_output_dir.mkdir(exist_ok=True)
    # print(f"Output frames will be saved to: {frame_output_dir}")
    # print(f"Output videos will be saved to: {video_output_dir}")
    print(f"Output images will be saved to: {output_dir.resolve()}")


    # --- Camera Base Settings ---
    cam_pos_vec = torch.tensor([0.0, 0.0, 3.0], device=device) # Closer camera
    target_vec = torch.tensor([0.0, 0.0, 0.0], device=device)
    up_vec = torch.tensor([0.0, 1.0, 0.0], device=device)
    fov_x_deg = 60.0
    fov_y_deg = 60.0
    near_plane = 0.1
    far_plane = 5.0 # Closer far plane for better depth ramp vis
    fov_x_rad = math.radians(fov_x_deg)
    fov_y_rad = math.radians(fov_y_deg)

    # --- Setup Camera (Single Pose) ---
    print("Setting up camera...")
    viewmatrix, projmatrix, campos, tan_fovy, tan_fovx = setup_camera(
        img_width, img_height,
        fov_x_rad, fov_y_rad,
        near_plane, far_plane,
        cam_pos_vec, target_vec, up_vec, device
    )
    print("Camera setup complete.")

    # --- Generate Rays ---
    print("Generating rays...")
    rays_o, rays_d = generate_rays(
        img_width, img_height, viewmatrix, tan_fovx, tan_fovy, campos, device
    ) # Shape: (H, W, 3)
    print("Rays generated.")

    # --- Define Ground Truth Depths (t) ---
    print("Defining ground truth depth map (depth ramp)...")
    # Create a depth ramp from near+0.2 to far-0.2 across width
    t_row = torch.linspace(near_plane + 0.2, far_plane - 0.2, img_width, device=device)
    # Expand to H, W, 1 shape
    t_gt = t_row.view(1, img_width, 1).expand(img_height, img_width, 1).contiguous() # H, W, 1
    print(f"Ground truth depth range: {t_gt.min():.3f} to {t_gt.max():.3f}")

    # --- Create Point Cloud on Rays at Depth t ---
    print("Creating 3D means based on rays and ground truth depth...")
    means3D = rays_o + rays_d * t_gt # H, W, 3
    P = img_height * img_width
    means3D_flat = means3D.view(P, 3).contiguous()
    print(f"Generated {P} 3D means.")

    # Gaussians are point-like, no need for scales/rots for our function
    # scales = torch.full((P, 3), 1e-6, device=device, dtype=torch.float32).contiguous()
    # rotations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32).view(1, 4).expand(P, 4).contiguous()

    # --- Call the Center Depth Function ---
    print("Calling rasterize_gaussians_center_depth...")
    # --- Debug Print Arguments ---
    print("  Arguments for rasterize_gaussians_center_depth:")
    print(f"    means3D_flat: shape={means3D_flat.shape}, dtype={means3D_flat.dtype}, device={means3D_flat.device}")
    # Print min/max to check range
    if means3D_flat.numel() > 0:
        print(f"      min={means3D_flat.min():.3f}, max={means3D_flat.max():.3f}, mean={means3D_flat.mean():.3f}")
    print(f"    viewmatrix.T: shape={viewmatrix.T.shape}, dtype={viewmatrix.T.dtype}, device={viewmatrix.T.device}")
    print(f"    projmatrix: shape={projmatrix.shape}, dtype={projmatrix.dtype}, device={projmatrix.device}")
    print(f"    tan_fovx: {tan_fovx:.4f} (type: {type(tan_fovx)})")
    print(f"    tan_fovy: {tan_fovy:.4f} (type: {type(tan_fovy)})")
    print(f"    image_height: {img_height} (type: {type(img_height)})")
    print(f"    image_width: {img_width} (type: {type(img_width)})")
    # Add missing parameters to print
    scale_modifier = 1.0 # Example value, adjust if needed
    kernel_size = 0.0 # Example value, adjust if needed
    prefiltered = False # Example value
    print(f"    scale_modifier: {scale_modifier:.4f} (type: {type(scale_modifier)})")
    print(f"    kernel_size: {kernel_size:.4f} (type: {type(kernel_size)})")
    print(f"    prefiltered: {prefiltered} (type: {type(prefiltered)})")
    print(f"    debug: False")
    print("  --- End Arguments ---")
    # --- End Debug Print ---
    try:
        # --- Modify to expect TWO return values --- 
        center_opacity_map, center_depth_map = rasterize_gaussians_center_depth(
            means3D_flat,          # arg0
            viewmatrix.T,          # arg1
            projmatrix,            # arg2
            tan_fovx,              # arg3
            tan_fovy,              # arg4
            img_height,            # arg5
            img_width,             # arg6
            scale_modifier,        # arg7
            kernel_size,           # arg8
            prefiltered,           # arg9
            False                  # arg10 (debug)
        )
        print("Rasterization complete.")

        # --- Compare Depths (Expect inf difference initially in Step 2) ---
        print("Comparing rendered depth with ground truth...")
        t_gt_compare = t_gt.squeeze(-1) # H, W

        # Check if depth map is all infinite (expected initially)
        if torch.isinf(center_depth_map).all():
            print("  Result: Rendered depth map is all infinite. OK for start of Step 2.")
        else:
            # If not all inf, calculate diff (might happen if printf accidentally writes)
            valid_mask = torch.isfinite(center_depth_map)
            if valid_mask.any():
                rendered_depth_valid = center_depth_map[valid_mask]
                gt_depth_valid = t_gt_compare[valid_mask]
                diff = torch.abs(rendered_depth_valid - gt_depth_valid)
                mean_abs_diff = torch.mean(diff)
                max_abs_diff = torch.max(diff)
                print(f"  Warning: Found finite values! Mean Abs Diff: {mean_abs_diff}, Max Abs Diff: {max_abs_diff}")
            else:
                 print("  Result: Rendered depth map is not infinite, but contains no finite values (e.g., all NaN?).")

        # Save visualizations regardless
        print("Saving visualizations...")
        # ... (saving logic - maybe rename output files slightly)
        vis_min = t_gt_compare.min().item()
        vis_max = t_gt_compare.max().item()
        t_gt_vis = normalize_depth_for_vis(t_gt_compare, vis_min, vis_max)
        save_image(t_gt_vis, output_dir / "depth_ground_truth.png")
        depth_rendered_vis = normalize_depth_for_vis(center_depth_map, vis_min, vis_max, background_value=torch.inf)
        save_image(depth_rendered_vis, output_dir / "depth_rendered_step2_start.png") # Rename output
        opacity_vis = center_opacity_map.float().unsqueeze(0)
        save_image(opacity_vis, output_dir / "opacity_rendered_step2_start.png") # Rename output
        print(f"Visualizations saved to {output_dir.resolve()}")

    except Exception as e:
        print(f"An error occurred during rasterization or processing: {e}")
        import traceback
        traceback.print_exc()

    print("Test script finished.")

# Remove multi-frame rendering and video creation logic
# ... (Removed code related to frame loop, video paths, imageio.mimwrite) ... 