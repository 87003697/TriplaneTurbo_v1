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
    import center_depth_rasterization # Keep the original import
    from center_depth_rasterization import (
        rasterize_gaussians_center_depth 
    )
    # ADDED: Print module path information
    print(f"DEBUG Python: Imported module center_depth_rasterization: {center_depth_rasterization}")
    try:
        print(f"DEBUG Python: Module file attribute: {center_depth_rasterization.__file__}")
    except AttributeError:
        print("DEBUG Python: Module does not have __file__ attribute.")
    print(f"DEBUG Python: Imported function: {rasterize_gaussians_center_depth}")
    
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

    # Projection matrix (using gaussian_utils logic - this calculates P_mat suitable for M@v)
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
    # Original projmatrix was P_mat.T - we need P_mat itself for MVP calculation
    # projmatrix = P_mat.T.contiguous()
    
    # Calculate CORRECT MVP matrix = P @ W2C (where view is W2C)
    mvp = torch.matmul(P_mat, view).contiguous()
    print("Correct MVP Matrix (P @ W2C):\n", mvp)
    # The old calculation was P_mat.T @ W2C - incorrect
    # mvp_matrix_old = torch.matmul(projmatrix, view).contiguous()
    # print("Old MVP Matrix (P.T @ W2C):\n", mvp_matrix_old)
    print("------------------------------------------")

    # Return view (W2C), P_mat, and correct mvp
    return view, P_mat, mvp, campos, tan_fovy, tan_fovx


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


# Helper function for inverse projection (needs careful implementation)
def unproject_pixels_to_world(px, py, view_z, W, H, viewmatrix, tan_fovx, tan_fovy, device):
    """Unprojects pixel coordinates (px, py) at a given view space Z to world coordinates."""
    # 1. Pixel center to NDC
    ndc_x = (px + 0.5) / W * 2.0 - 1.0
    ndc_y = (py + 0.5) / H * 2.0 - 1.0
    # Flip Y back from image convention (+Y down) to NDC convention (+Y up)
    # ndc_y = -ndc_y # REMOVE THE FLIP? Assume +Y up in pixel and NDC?

    # 2. NDC to View Space at given Z
    # Assumes standard perspective projection where:
    # ndc_x = (view_x / view_z) * (1 / tan_fovx) => view_x = ndc_x * view_z * tan_fovx
    # ndc_y = (view_y / view_z) * (1 / tan_fovy) => view_y = ndc_y * view_z * tan_fovy
    # Need absolute view_z here if view_z is negative
    # abs_view_z = torch.abs(view_z) # Incorrect: view_z is typically negative
    view_x = ndc_x * view_z * tan_fovx # Use actual view_z
    view_y = ndc_y * view_z * tan_fovy # Use actual view_z
    # Combine into view space points (N, 3) where N is number of points
    # Ensure view_z has the same shape for stacking
    if view_z.numel() == 1:
        view_z = view_z.expand_as(view_x)
    points_view = torch.stack([view_x, view_y, view_z], dim=-1)

    # 3. View Space to World Space
    inv_viewmatrix = torch.inverse(viewmatrix)

    # Make points_view homogeneous (N, 4)
    points_view_h = torch.cat([points_view, torch.ones(points_view.shape[0], 1, device=device)], dim=-1)

    # Transform (N, 4) @ (4, 4) -> (N, 4)
    points_world_h = points_view_h @ inv_viewmatrix.T

    # Normalize homogeneous coords (though W should be 1)
    points_world = points_world_h[:, :3] / points_world_h[:, 3, None]

    return points_world


if __name__ == "__main__":
    # --- Parameters ---
    img_height = 128
    img_width = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path("./test_output_accuracy")
    output_dir.mkdir(exist_ok=True)
    print(f"Output images will be saved to: {output_dir.resolve()}")
    total_pixels = img_height * img_width # Define total_pixels early

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
    viewmatrix, projmatrix_orig, mvp_matrix_correct, campos, tan_fovy, tan_fovx = setup_camera(
        img_width, img_height,
        fov_x_rad, fov_y_rad,
        near_plane, far_plane,
        cam_pos_vec, target_vec, up_vec, device
    )
    print("Camera setup complete.")
    print("--- Camera Matrix & Params Check ---")
    print("View Matrix (W2C):\n", viewmatrix)
    print("Projection Matrix (P):\n", projmatrix_orig) # Print original P
    print(f"tanfovx: {tan_fovx:.4f}, tanfovy: {tan_fovy:.4f}")
    print(f"Cam Pos: {campos.cpu().numpy()}")
    print("----------------------------------")

    # Prepare matrices for CUDA kernel (expects row-vector pre-multiplication, needs transposed matrices)
    viewmatrix_T_for_cuda = viewmatrix.T.contiguous()
    mvp_matrix_T_for_cuda = mvp_matrix_correct.T.contiguous() # Pass transpose of correct MVP

    # --- Test Case 1: Plane Point Cloud ---
    print("\n--- Test Case 1: Plane Point Cloud ---") 
    print("Creating 3D means (simple centered cube)...")
    P = 128 * 128 
    side_len = 1.0 # World space size
    z_center = -1.9 # Move slightly inside far plane
    # Recalculate expected depth: view_z = dot([0,0,1], [x,y,-1.9] - [0,0,3]) = dot([0,0,1], [x,y,-4.9]) = -4.9
    expected_depth = 4.9
    means3D_flat = torch.zeros(P, 3, device=device)
    grid_size = int(math.sqrt(P))
    x_coords = torch.linspace(-side_len/2, side_len/2, grid_size, device=device)
    y_coords = torch.linspace(-side_len/2, side_len/2, grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
    means3D_flat[:, 0] = grid_x.flatten()
    means3D_flat[:, 1] = grid_y.flatten()
    means3D_flat[:, 2] = z_center 
    print(f"Generated {P} 3D means for plane test at Z={z_center:.1f}.")
    
    # ... (Manual Transform Check - Optional for this run) ...

    # --- Call Rasterizer (Plane Test) ---
    print("Calling rasterize_gaussians_center_depth for Plane...")
    scale_modifier = 1.0 
    kernel_size = 0.0 
    prefiltered = False 

    try:
        # --- Expect TWO return values (Final GPU implementation) --- 
        center_opacity_map, final_depth_map = rasterize_gaussians_center_depth(
            means3D_flat,
            viewmatrix_T_for_cuda, # Pass View.T
            mvp_matrix_T_for_cuda, # Pass (P @ W2C).T
            tan_fovx,
            tan_fovy,
            img_height,
            img_width,
            near_plane,
            far_plane,
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
        gt_depth_plane[valid_mask_rendered] = expected_depth # Use the positive expected depth
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
             print(f"Plane Depth Comparison Results (vs Expected Depth {expected_depth}):")
             print(f"  Total Pixels: {total_pixels}")
             print(f"  Valid (Finite) Rendered Pixels: {num_valid_pixels} ({valid_percentage:.2f}%)")
             print(f"  Mean Absolute Difference: {mean_abs_diff:.6f}")
             print(f"  Max Absolute Difference:  {max_abs_diff:.6f}")
        
        # --- Compare Opacity ---
        print("Comparing rendered opacity map with expectation...")
        # Expected opacity: 1.0 where depth is valid, 0.0 otherwise
        gt_opacity_plane = torch.zeros_like(center_opacity_map, dtype=torch.float32)
        gt_opacity_plane[valid_mask_rendered] = 1.0

        opacity_diff = torch.abs(center_opacity_map - gt_opacity_plane)
        matching_pixels = torch.sum(opacity_diff < 1e-5).item() # Count pixels where opacity matches expectation
        non_matching_pixels = total_pixels - matching_pixels

        print(f"Plane Opacity Comparison Results:")
        print(f"  Total Pixels: {total_pixels}")
        print(f"  Pixels with Matching Opacity (Rendered vs Expected): {matching_pixels} ({ (matching_pixels/total_pixels)*100:.2f}%)")
        print(f"  Pixels with Non-Matching Opacity: {non_matching_pixels}")
        
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

    except Exception as e:
        print(f"An error occurred during rasterization or processing: {e}")
        import traceback
        traceback.print_exc()

    # --- Test Case 2: Controlled Overlapping Points ---
    print("\n--- Test Case 2: Controlled Overlapping Points ---")
    # Select ONLY the center pixel
    center_h, center_w = img_height // 2, img_width // 2
    selected_px = torch.tensor([center_w], device=device, dtype=torch.float32)
    selected_py = torch.tensor([center_h], device=device, dtype=torch.float32)
    num_selected_pixels = selected_px.shape[0]
    print(f"Selected ONLY center pixel ({center_w}, {center_h}) for overlap test.")

    # Define near and far depths
    target_depth_near = 2.0 # Make it closer than the bad point at z=0 (depth 3.0)
    target_depth_far = 4.9 # Keep this inside far plane
    view_z_near = -target_depth_near
    view_z_far = -target_depth_far
    print(f"Overlap Test using target depths: Near={target_depth_near:.1f}, Far={target_depth_far:.1f}")

    # Generate near points by unprojecting
    print(f"Generating 'near' points at depth {target_depth_near:.2f} (view_z {view_z_near:.2f})")
    means3D_near = unproject_pixels_to_world(
        selected_px, selected_py,
        torch.full_like(selected_px, view_z_near),
        img_width, img_height, viewmatrix, tan_fovx, tan_fovy, device
    )

    # Generate far points by unprojecting at the same pixel locations
    print(f"Generating 'far' points at depth {target_depth_far:.2f} (view_z {view_z_far:.2f})")
    means3D_far = unproject_pixels_to_world(
        selected_px, selected_py,
        torch.full_like(selected_px, view_z_far),
        img_width, img_height, viewmatrix, tan_fovx, tan_fovy, device
    )

    # Combine points
    means3D_overlap = torch.cat([means3D_near, means3D_far], dim=0)
    print(f"Total points for overlap test: {means3D_overlap.shape[0]}")

    # Create expected depth and opacity maps
    expected_depth_map_overlap = torch.full((img_height, img_width), float('inf'), device=device, dtype=torch.float32)
    expected_opacity_map_overlap = torch.zeros((img_height, img_width), device=device, dtype=torch.float32)

    # Fill expected values at selected pixel locations
    selected_px_int = selected_px.long()
    selected_py_int = selected_py.long()
    expected_depth_map_overlap[selected_py_int, selected_px_int] = target_depth_near # Expect the NEW NEAR depth
    expected_opacity_map_overlap[selected_py_int, selected_px_int] = 1.0

    # --- Call Rasterizer (Overlap Test) ---
    print("Calling rasterize_gaussians_center_depth for Overlap Test...")
    try:
        center_opacity_map_overlap, final_depth_map_overlap = rasterize_gaussians_center_depth(
            means3D_overlap,
            viewmatrix_T_for_cuda, # Pass View.T
            mvp_matrix_T_for_cuda, # Pass (P @ W2C).T
            tan_fovx,
            tan_fovy,
            img_height,
            img_width,
            near_plane,
            far_plane,
            scale_modifier,
            kernel_size,
            prefiltered,
            False
        )
        print("GPU Rasterization (Overlap) complete.")

        # --- Compare Depths (Overlap) ---
        print("Comparing GPU-selected depth with expected overlap depth...")
        # Only compare at the selected pixels where we expect valid depth
        rendered_depth_selected = final_depth_map_overlap[selected_py_int, selected_px_int]
        expected_depth_selected = expected_depth_map_overlap[selected_py_int, selected_px_int]

        diff_overlap = torch.abs(rendered_depth_selected - expected_depth_selected)
        mean_abs_diff_overlap = torch.mean(diff_overlap)
        max_abs_diff_overlap = torch.max(diff_overlap)
        # Check if all selected pixels were rendered correctly
        num_correct_depth_pixels = torch.sum(diff_overlap < 1e-5).item()
        coverage_percentage_overlap = (num_correct_depth_pixels / num_selected_pixels) * 100

        print(f"Overlap Depth Comparison Results (Expected Depth {target_depth_near:.2f}):")
        print(f"  Number of Target Pixels: {num_selected_pixels}")
        print(f"  Pixels with Correct Depth Rendered: {num_correct_depth_pixels} ({coverage_percentage_overlap:.2f}%)")
        print(f"  Mean Absolute Difference (at target pixels): {mean_abs_diff_overlap:.6f}")
        print(f"  Max Absolute Difference (at target pixels):  {max_abs_diff_overlap:.6f}")

        # --- Compare Opacity (Overlap) ---
        print("Comparing rendered opacity map with expectation (Overlap)...")
        # Check full map against expected map
        opacity_diff_overlap = torch.abs(center_opacity_map_overlap - expected_opacity_map_overlap)
        matching_pixels_overlap = torch.sum(opacity_diff_overlap < 1e-5).item()
        non_matching_pixels_overlap = total_pixels - matching_pixels_overlap

        print(f"Overlap Opacity Comparison Results:")
        print(f"  Total Pixels: {total_pixels}")
        print(f"  Pixels with Matching Opacity: {matching_pixels_overlap} ({ (matching_pixels_overlap/total_pixels)*100:.2f}%)")
        print(f"  Pixels with Non-Matching Opacity: {non_matching_pixels_overlap}")

        # --- Save Overlap Visualizations ---
        print("Saving visualizations for Overlap Test...")
        vis_min_overlap = target_depth_near - 0.5 # Set fixed range around expected depth
        vis_max_overlap = target_depth_near + 0.5

        gt_overlap_vis = normalize_depth_for_vis(expected_depth_map_overlap, vis_min_overlap, vis_max_overlap, background_value=torch.inf)
        save_image(gt_overlap_vis, output_dir / "depth_ground_truth_overlap.png")

        depth_rendered_overlap_vis = normalize_depth_for_vis(final_depth_map_overlap, vis_min_overlap, vis_max_overlap, background_value=torch.inf)
        save_image(depth_rendered_overlap_vis, output_dir / "depth_rendered_overlap.png")

        opacity_vis_overlap = center_opacity_map_overlap.float().unsqueeze(0)
        save_image(opacity_vis_overlap, output_dir / "opacity_rendered_overlap.png")
        print(f"Overlap test visualizations saved to {output_dir.resolve()}")

    except Exception as e:
        print(f"An error occurred during overlap rasterization or processing: {e}")
        import traceback
        traceback.print_exc()

    # --- Test Case 3: Points Outside View ---
    print("\n--- Test Case 3: Points Outside View ---")
    # Use the near points from the overlap test as the 'good' points
    # Need to ensure means3D_near exists (e.g., run overlap test first or define it)
    if 'means3D_near' not in locals():
         print("Skipping Clipping Test: means3D_near not defined (Overlap test might have failed or been skipped).")
    else:
        means3D_good = means3D_near.clone()
        num_good = means3D_good.shape[0]

        # Create some 'bad' points known to be outside the frustum
        # Ensure inv_viewmatrix is calculated if not already
        if 'inv_viewmatrix' not in locals():
            inv_viewmatrix = torch.inverse(viewmatrix)

        bad_view_z_near = -target_depth_near # Use the same depth as good points for simplicity

        means3D_bad_list = [
            torch.tensor([[0.0, 0.0, 100.0]], device=device), # Too far (World Z)
            torch.tensor([[0.0, 0.0, 0.0]], device=device),   # Too near (World Z, behind camera at +3)
            torch.tensor([[10.0, 0.0, -2.0]], device=device), # Too far left/right (World X)
            torch.tensor([[0.0, 10.0, -2.0]], device=device)  # Too far up/down (World Y)
        ]

        # Add a point that projects outside W/H after projection
        bad_ndc_x = 2.0
        bad_ndc_y = 0.0
        bad_view_x = bad_ndc_x * abs(bad_view_z_near) * tan_fovx
        bad_view_y = bad_ndc_y * abs(bad_view_z_near) * tan_fovy
        bad_view = torch.tensor([[bad_view_x, bad_view_y, bad_view_z_near]], device=device)
        bad_view_h = torch.cat([bad_view, torch.ones(1, 1, device=device)], dim=-1)
        bad_world_h = bad_view_h @ inv_viewmatrix.T
        bad_world_clip = bad_world_h[:, :3] / bad_world_h[:, 3, None]
        means3D_bad_list.append(bad_world_clip)

        means3D_bad = torch.cat(means3D_bad_list, dim=0)

        num_bad = means3D_bad.shape[0]
        print(f"Created {num_bad} points explicitly outside the view frustum.")

        # Combine good and bad points
        means3D_clip_test = torch.cat([means3D_good, means3D_bad], dim=0)
        print(f"Total points for clipping test: {means3D_clip_test.shape[0]}")

        # --- Call Rasterizer (Clipping Test) ---
        print("Calling rasterize_gaussians_center_depth for Clipping Test...")
        try:
            center_opacity_map_clip, final_depth_map_clip = rasterize_gaussians_center_depth(
                means3D_clip_test, # Use combined points
                viewmatrix_T_for_cuda, # Pass View.T
                mvp_matrix_T_for_cuda, # Pass (P @ W2C).T
                tan_fovx,
                tan_fovy,
                img_height,
                img_width,
                near_plane,
                far_plane,
                scale_modifier,
                kernel_size,
                prefiltered,
                False
            )
            print("GPU Rasterization (Clipping) complete.")

            # --- Compare Clipping Results ---
            # Expected result should match overlap's expected map (depth 2.0 at center)
            print("Comparing clipping test output with expected result (ignoring bad points)...")
            # ... (Debug print now compares against expected 2.0)
            expected_center_depth_overlap_val = expected_depth_map_overlap[center_h, center_w].item()
            print(f"DEBUG CLIP: Rendered center depth = {final_depth_map_clip[center_h, center_w].item():.6f}")
            print(f"DEBUG CLIP: Expected center depth = {expected_center_depth_overlap_val:.6f}")
            # ... (Comparison logic compares against expected_depth_map_overlap)
            depth_diff_clip = torch.abs(final_depth_map_clip - expected_depth_map_overlap)
            # Need to handle inf == inf as zero difference
            inf_mask_clip = torch.isinf(final_depth_map_clip) & torch.isinf(expected_depth_map_overlap)
            depth_diff_clip[inf_mask_clip] = 0.0
            # Check non-inf differences only for max
            valid_diff_mask = ~torch.isinf(depth_diff_clip)
            max_depth_diff_clip = torch.max(depth_diff_clip[valid_diff_mask]) if valid_diff_mask.any() else torch.tensor(0.0)

            opacity_diff_clip = torch.abs(center_opacity_map_clip - expected_opacity_map_overlap)
            max_opacity_diff_clip = torch.max(opacity_diff_clip)

            if max_depth_diff_clip < 1e-6 and max_opacity_diff_clip < 1e-6:
                print("Clipping Test Passed: Output matches expectation (bad points correctly ignored).")
            else:
                print(f"Clipping Test Failed! Max depth diff: {max_depth_diff_clip:.6f}, Max opacity diff: {max_opacity_diff_clip:.6f}")

        except Exception as e:
            print(f"An error occurred during clipping test: {e}")
        import traceback
        traceback.print_exc()

    # --- Test Case 4: Single Point Depth Verification ---
    print("\n--- Test Case 4: Single Point Depth Verification ---")
    # Define a single test point in world space
    P_test_world = torch.tensor([[0.1, -0.2, -1.0]], device=device) # Example point
    print(f"Test point world coord: {P_test_world.cpu().numpy()}")
    print(f"Camera world coord:   {campos.cpu().numpy()}")

    # 1. Calculate expected distance (t)
    t_expected = torch.linalg.norm(P_test_world - campos, dim=1)
    print(f"Expected distance (t_expected): {t_expected.item():.6f}")

    # 2. Calculate expected projection pixel (px, py) AND view-space depth
    P_test_h = torch.cat([P_test_world, torch.ones(1, 1, device=device)], dim=1).T # Shape (4, 1)
    
    # Calculate view space coordinate
    P_test_view_h = torch.matmul(viewmatrix, P_test_h) # viewmatrix is W2C
    z_view = P_test_view_h[2, 0].item() # Get the Z coordinate in view space
    abs_z_view = abs(z_view)
    # print(f"DEBUG Python: z_view = {z_view:.6f}, abs(z_view) = {abs_z_view:.6f}") # REMOVED DEBUG PRINT

    # Check clipping based on view space Z (using abs value is equivalent to CUDA kernel logic)
    if abs_z_view < near_plane or abs_z_view > far_plane:
        print(f"Error: Test point clipped by depth. abs(z_view)={abs_z_view:.4f}, near={near_plane:.4f}, far={far_plane:.4f}")
        px_expected, py_expected = -1, -1 # Indicate error
    else:
        # If not depth clipped, proceed to calculate NDC and check NDC/bounds clipping
        # Transform to clip space using the CORRECT MVP matrix
        P_clip_h = torch.matmul(mvp_matrix_correct, P_test_h) # Shape (4, 1)
        P_clip_h = P_clip_h.T # Shape (1, 4)
        w_clip = P_clip_h[:, 3]
        # print(f"DEBUG Python: w_clip = {w_clip.item()}") # Optional: verify w_clip == z_view if needed
        
        if torch.abs(w_clip) < 1e-8: # Division safety check
            print("Error: w_clip is near zero, cannot calculate NDC.")
            px_expected, py_expected = -1, -1
        else:
            # NDC coordinates
            ndc_coords = P_clip_h[:, :3] / w_clip[:, None]
            ndc_x, ndc_y = ndc_coords[:, 0], ndc_coords[:, 1]
            # Check NDC clipping
            if torch.abs(ndc_x) > 1.0 or torch.abs(ndc_y) > 1.0:
                print(f"Error: Test point clipped by NDC bounds. ndc=({ndc_x.item():.4f}, {ndc_y.item():.4f})")
                px_expected, py_expected = -1, -1
            else:
                # Screen coordinates
                screen_x = (ndc_x + 1.0) * img_width * 0.5
                screen_y = (ndc_y + 1.0) * img_height * 0.5
                px_expected = int(round(screen_x.item() - 0.5))
                py_expected = int(round(screen_y.item() - 0.5))
                print(f"Expected projection pixel (px, py): ({px_expected}, {py_expected})")

    # 3. Call Rasterizer with the single point
    if px_expected != -1: # Only proceed if point is expected to be visible
        print("Calling rasterize_gaussians_center_depth for Single Point Test...")
        try:
            _, final_depth_map_single = rasterize_gaussians_center_depth(
                P_test_world,
                viewmatrix_T_for_cuda, # Pass View.T
                mvp_matrix_T_for_cuda, # Pass (P @ W2C).T
                tan_fovx,
                tan_fovy,
                img_height,
                img_width,
                near_plane,
                far_plane,
                scale_modifier, # Using plane test values
                kernel_size,
                prefiltered,
                False # debug
            )
            print("GPU Rasterization (Single Point) complete.")

            # 4. Compare output depth with expected distance AND abs(z_view)
            if 0 <= py_expected < img_height and 0 <= px_expected < img_width:
                depth_output = final_depth_map_single[py_expected, px_expected].item()
                print(f"Output depth at ({px_expected}, {py_expected}): {depth_output:.6f}")
                print(f"Compare with abs(z_view):      {abs_z_view:.6f}")
                print(f"Compare with t_expected:   {t_expected.item():.6f}")
                
                # Simplified Check: Just report the differences
                depth_diff_vs_zview = abs(depth_output - abs_z_view)
                print(f"Difference vs abs(z_view): {depth_diff_vs_zview:.6e}")
                depth_diff_vs_t = abs(depth_output - t_expected.item())
                print(f"Difference vs t_expected:  {depth_diff_vs_t:.6e} (Small diff expected)")

            else:
                print("Error: Expected pixel is outside image bounds, cannot verify output depth.")

        except Exception as e:
            print(f"An error occurred during single point rasterization or processing: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping rasterization call as test point is expected to be clipped.")

    print("Test script finished.")

# Remove multi-frame rendering and video creation logic
# ... (Removed code related to frame loop, video paths, imageio.mimwrite) ... 

# print(">>> SCRIPT FINISHED (basic imports only)") # <<< REMOVE THIS LINE 