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

    # <<< Use getProjectionMatrix from gaussian_utils.py logic >>>
    # Convert fov to tensor before using torch.tan
    fov_y_tensor = torch.tensor(fov_y_rad, device=device)
    fov_x_tensor = torch.tensor(fov_x_rad, device=device) 
    tanHalfFovY = torch.tan(fov_y_tensor * 0.5)
    tanHalfFovX = torch.tan(fov_x_tensor * 0.5) # Use fov_x_rad from input
    # <<< End conversion >>>

    top = tanHalfFovY * near
    bottom = -top
    # Use consistent tanHalfFovX for right/left calculation
    right = tanHalfFovX * near 
    left = -right

    P = torch.zeros(4, 4, device=device)
    z_sign = 1.0 

    P[0, 0] = 2.0 * near / (right - left)
    P[1, 1] = 2.0 * near / (top - bottom)
    P[0, 2] = (right + left) / (right - left) 
    P[1, 2] = (top + bottom) / (top - bottom) 
    P[2, 2] = z_sign * (far + near) / (far - near) 
    P[2, 3] = -2.0 * z_sign * far * near / (far - near)
    P[3, 2] = z_sign 
    projmatrix = P.T.contiguous() 
    # <<< End replacement >>>

    # Calculate MVP matrix using the correct variable name 'view'
    mvp_matrix = torch.matmul(projmatrix, view).contiguous() 
    print("MVP Matrix (Recalculated):\n", mvp_matrix)
    print("------------------------------------------")

    # <<< Return mvp_matrix >>>
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

    # --- Create 3D means (Simple Cube/Plane instead of Ray-based) ---
    print("Creating 3D means (simple centered cube)...")
    P = 128 * 128 # Keep the number of points
    side_len = 1.0 # Cube side length in world space
    z_center = -2.0 # Center cube in front of camera
    z_depth = 1.0   # Depth of the cube

    # Create a grid of points for the cube faces (or just a plane)
    # Example: A plane at z = z_center
    means3D_flat = torch.zeros(P, 3, device=device)
    grid_size = int(math.sqrt(P))
    x_coords = torch.linspace(-side_len/2, side_len/2, grid_size, device=device)
    y_coords = torch.linspace(-side_len/2, side_len/2, grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')

    means3D_flat[:, 0] = grid_x.flatten()
    means3D_flat[:, 1] = grid_y.flatten()
    means3D_flat[:, 2] = z_center # Place all points on a plane at z=-2
    # <<< End Simple Point Cloud Generation >>>

    print(f"Generated {P} 3D means.")
    print(f"  World X range: [{means3D_flat[:, 0].min():.2f}, {means3D_flat[:, 0].max():.2f}]")
    print(f"  World Y range: [{means3D_flat[:, 1].min():.2f}, {means3D_flat[:, 1].max():.2f}]")
    print(f"  World Z range: [{means3D_flat[:, 2].min():.2f}, {means3D_flat[:, 2].max():.2f}]")

    # --- Manual Transformation Check (Should now be mostly in bounds) ---
    print("--- Manual Transformation Check (Python) ---")
    indices_to_check = [0, P // 2, P - 1]
    points_to_check = means3D_flat[indices_to_check].cpu()
    view_T_py = viewmatrix.T.cpu() # W2C Transposed
    proj_py = projmatrix.cpu()
    W_py = img_width
    H_py = img_height
    for i, p_world in enumerate(points_to_check):
        idx_check = indices_to_check[i]
        print(f"-- Point {idx_check} --")
        print(f"  World: {p_world.numpy()}")
        p_world_h = torch.cat([p_world, torch.tensor([1.0])])
        # World to View (using W2C.T - expecting vector result after multiplication)
        # Note: PyTorch matmul is typically vector @ matrix
        p_view_h = p_world_h @ view_T_py 
        print(f"  View (Homogeneous, from W2C.T): {p_view_h.numpy()}")
        p_view = p_view_h[:3]
        p_view_z_py = p_view_h[2] # Z depth in view space
        print(f"  View: {p_view.numpy()}, Depth (z): {p_view_z_py.item():.3f}")
        # View to Clip
        p_view_h_for_proj = torch.cat([p_view, torch.tensor([1.0])]) # Need homogeneous coord for projection
        p_clip_h = p_view_h_for_proj @ proj_py
        print(f"  Clip (Homogeneous): {p_clip_h.numpy()}")
        # Clip to NDC
        w_py = p_clip_h[3]
        if abs(w_py.item()) < 1e-5: 
            print("  NDC: w is too small!")
            continue
        ndc = p_clip_h[:3] / w_py
        print(f"  NDC: {ndc.numpy()}")
        # NDC to Screen
        screen_x_py = (ndc[0] + 1.0) * W_py * 0.5
        screen_y_py = (ndc[1] + 1.0) * H_py * 0.5 # Try standard first, maybe flip later
        print(f"  Screen Coords (float): ({screen_x_py.item():.2f}, {screen_y_py.item():.2f})")
        # Screen to Pixel
        px_py = int(round(screen_x_py.item() - 0.5))
        py_py = int(round(screen_y_py.item() - 0.5))
        print(f"  Pixel Coords (int): ({px_py}, {py_py})")
        in_bounds = (px_py >= 0 and px_py < W_py and py_py >= 0 and py_py < H_py)
        print(f"  In Bounds [0, {W_py-1}] x [0, {H_py-1}]: {in_bounds}")
    print("------------------------------------------")

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
    print(f"    mvp_matrix: shape={mvp_matrix.shape}, dtype={mvp_matrix.dtype}, device={mvp_matrix.device}")
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
        # <<< Recalculate MVP.T to pass to kernel >>>
        mvp_matrix_T = mvp_matrix.T.contiguous()
        
        # --- Expect TWO return values (Final GPU implementation) --- 
        center_opacity_map, final_depth_map = rasterize_gaussians_center_depth(
            means3D_flat,
            viewmatrix.T, 
            mvp_matrix_T, # Pass MVP Transposed
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
        print("Comparing rendered depth with ground truth...")
        t_gt_compare = t_gt.squeeze(-1) # H, W
        # Use the map returned directly from the C++ extension
        valid_mask = torch.isfinite(final_depth_map) 

        if not valid_mask.any():
             print("Error: No finite pixels rendered!")
        else:
             rendered_depth_valid = final_depth_map[valid_mask]
             gt_depth_valid = t_gt_compare[valid_mask]
             diff = torch.abs(rendered_depth_valid - gt_depth_valid)
             mean_abs_diff = torch.mean(diff)
             max_abs_diff = torch.max(diff)
             num_valid_pixels = valid_mask.sum().item()
             total_pixels = img_height * img_width
             valid_percentage = (num_valid_pixels / total_pixels) * 100

             print(f"Depth Comparison Results:")
             print(f"  Total Pixels: {total_pixels}")
             print(f"  Valid (Finite) Rendered Pixels: {num_valid_pixels} ({valid_percentage:.2f}%)")
             print(f"  Mean Absolute Difference (Valid Pixels): {mean_abs_diff:.6f}")
             print(f"  Max Absolute Difference (Valid Pixels):  {max_abs_diff:.6f}")
        
        # Save visualizations using final_depth_map
        print("Saving visualizations...")
        vis_min = torch.min(rendered_depth_valid).item() if valid_mask.any() else near_plane
        vis_max = torch.max(rendered_depth_valid).item() if valid_mask.any() else far_plane
        t_gt_vis = normalize_depth_for_vis(t_gt_compare, vis_min, vis_max)
        save_image(t_gt_vis, output_dir / "depth_ground_truth.png")
        depth_rendered_vis = normalize_depth_for_vis(final_depth_map, vis_min, vis_max, background_value=torch.inf)
        save_image(depth_rendered_vis, output_dir / "depth_rendered_multi_stage.png") 
        opacity_vis = center_opacity_map.float().unsqueeze(0) # Opacity is still just zeros
        save_image(opacity_vis, output_dir / "opacity_rendered_multi_stage.png")
        print(f"Visualizations saved to {output_dir.resolve()}")

    except Exception as e:
        print(f"An error occurred during rasterization or processing: {e}")
        import traceback
        traceback.print_exc()

    print("Test script finished.")

# Remove multi-frame rendering and video creation logic
# ... (Removed code related to frame loop, video paths, imageio.mimwrite) ... 

print(">>> SCRIPT FINISHED (basic imports only)") 