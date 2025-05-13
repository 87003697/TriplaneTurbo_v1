import torch
import torch.nn.functional as F
import math
from pathlib import Path
from torchvision.utils import save_image
import numpy as np # For L2 diff if needed, or torch.norm
import imageio
from typing import Union, Tuple, Optional, Dict, Any
import sys

# Imports for diff_gaussian_rasterization
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# Imports for gsplat
import gsplat

# === COPIED FROM threestudio.utils.ops BEGIN ===
def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> torch.Tensor:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        if principal is None:
            cx, cy = W / 2, H / 2
        else:
            cx, cy = principal
    else:
        fx, fy = focal
        if principal is None:
            cx, cy = W / 2, H / 2
        else:
            cx, cy = principal       

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)
    return directions

def get_rays(
    directions: torch.Tensor,
    c2w: torch.Tensor,
    keepdim=False,
    noise_scale=0.0,
    normalize=True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Rotate ray directions from camera coordinate to the world coordinate
    # directions: [N_rays, 3] or [B, N_rays, 3] or [B, H, W, 3]
    # c2w: [4, 4] or [B, 4, 4]
    prefix = directions.shape[:-1]
    if c2w.ndim == 2:
        # treat as one camera
        rays_d = torch.matmul(directions, c2w[:3, :3].T)
        rays_o = c2w[:3, 3].expand_as(rays_d)
    elif c2w.ndim == 3:
        # treat as B cameras
        if directions.ndim == 2:
            # directions: [N_rays, 3], c2w: [B, 4, 4]
            # output: [B, N_rays, 3]
            rays_d = torch.matmul(directions.unsqueeze(0).expand(c2w.shape[0], -1, -1), c2w[:, :3, :3].transpose(1, 2))
            rays_o = c2w[:, :3, 3].unsqueeze(1).expand(-1, directions.shape[0], -1)
        elif directions.ndim == 3:
             # directions: [B, N_rays, 3], c2w: [B, 4, 4]
            # output: [B, N_rays, 3]
            assert directions.shape[0] == c2w.shape[0], f"Batch size mismatch: directions {directions.shape[0]}, c2w {c2w.shape[0]}"
            rays_d = torch.matmul(directions, c2w[:, :3, :3].transpose(1,2))
            rays_o = c2w[:, :3, 3].unsqueeze(1).expand_as(rays_d)
        elif directions.ndim == 4:
            # directions: [B, H, W, 3], c2w: [B, 4, 4]
            # output: [B, H, W, 3]
            assert directions.shape[0] == c2w.shape[0], f"Batch size mismatch: directions {directions.shape[0]}, c2w {c2w.shape[0]}"
            rays_d = torch.matmul(directions.view(directions.shape[0], -1, 3), c2w[:, :3, :3].transpose(1,2)).view(*prefix, 3)
            rays_o = c2w[:, :3, 3].unsqueeze(1).unsqueeze(1).expand(-1, directions.shape[1], directions.shape[2], -1) 
    else:
        raise ValueError(f"Invalid c2w dimension: {c2w.ndim}")

    if normalize:
        rays_d = F.normalize(rays_d, p=2, dim=-1)

    if noise_scale > 0.0:
        rays_d = rays_d + torch.randn_like(rays_d) * noise_scale
        if normalize:
            rays_d = F.normalize(rays_d, p=2, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d

def get_projection_matrix(
    znear: float,
    zfar: float,
    fovX: float,
    fovY: float,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    # Convert fovX and fovY to radians
    fovX_rad = fovX * math.pi / 180.0
    fovY_rad = fovY * math.pi / 180.0

    tanHalfFovY = math.tan(fovY_rad / 2.0)
    tanHalfFovX = math.tan(fovX_rad / 2.0)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = -1.0
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)

    return P

def get_mvp_matrix(
    c2w: torch.Tensor, P: torch.Tensor
) -> torch.Tensor:
    # c2w: camera to world matrix, [B, 4, 4] or [4, 4]
    # P: projection matrix, [B, 4, 4] or [4, 4]
    # returns: mvp matrix, [B, 4, 4] or [4, 4]
    if c2w.ndim != P.ndim:
        # unsqueeze c2w to match P if P has a batch dim and c2w does not
        if P.ndim == 3 and c2w.ndim == 2:
            c2w = c2w.unsqueeze(0)
        else:
            raise ValueError(
                f"get_mvp_matrix: c2w and P must have the same number of dimensions, or P has 3 and c2w has 2. Got {c2w.ndim} and {P.ndim}"
            )
    # multiply P and inv(c2w)
    # P: [B, 4, 4] or [4, 4]
    # c2w: [B, 4, 4] or [4, 4]
    # w2c: [B, 4, 4] or [4, 4]
    # mvp: [B, 4, 4] or [4, 4]
    w2c = torch.inverse(c2w)
    mvp = torch.matmul(P, w2c)
    return mvp

# === COPIED FROM threestudio.utils.ops END ===

# === ADDING get_projection_matrix_gaussian FROM ops.py ===
def get_projection_matrix_gaussian(znear, zfar, fovX, fovY, device="cuda"):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P
# === END ADDING get_projection_matrix_gaussian ===

# --- Global Helper Functions for Camera Logic ---
# (Moved here to be accessible by all setup functions)

# --- Helper: get K from FoV ---
def get_intrinsics_from_fov(fov_rad, W, H, device):
    """Calculates K matrix from FoV Y."""
    if isinstance(fov_rad, torch.Tensor):
        fov_rad_scalar = fov_rad.item() if fov_rad.numel() == 1 else fov_rad[0].item()
    else:
        fov_rad_scalar = fov_rad
    tan_fovy_half = math.tan(fov_rad_scalar * 0.5)
    if tan_fovy_half == 0: tan_fovy_half = 1e-6
    fy = H / (2 * tan_fovy_half)
    fx = fy # Simpler assumption (square pixels)
    cx = W / 2.0
    cy = H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32)
    return K

# --- Helper: Convert W2C GL to CV ---
def convert_w2c_gl_to_cv(w2c_gl):
    """Converts OpenGL W2C matrix to OpenCV W2C matrix."""
    conversion = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                              dtype=w2c_gl.dtype, device=w2c_gl.device)
    return torch.matmul(conversion, w2c_gl)

# --- Helper Functions (Ensure Definitions Exist) ---

# Function to calculate the LookAt matrix (OpenGL style)
def LookAt(eye, at, up):
    device = eye.device
    eye = eye.to(torch.float32)
    at = at.to(torch.float32)
    up = up.to(torch.float32)
    F_vec = (at - eye)
    if torch.norm(F_vec) < 1e-8: F_vec = torch.tensor([0.0, 0.0, -1.0], device=device)
    F_vec = F.normalize(F_vec, dim=0)
    up_world = up.to(torch.float32)
    if torch.norm(up_world) < 1e-8: up_world = torch.tensor([0.0, 1.0, 0.0], device=device)
    up_world = F.normalize(up_world, dim=0)
    R_vec = torch.linalg.cross(F_vec, up_world)
    if torch.norm(R_vec) < 1e-8:
        if abs(F_vec[1]) < 0.999:
             R_vec = torch.linalg.cross(F_vec, torch.tensor([0.0, 1.0, 0.0], device=device))
        else:
             R_vec = torch.linalg.cross(F_vec, torch.tensor([1.0, 0.0, 0.0], device=device))
    R_vec = F.normalize(R_vec, dim=0)
    U_vec = torch.linalg.cross(R_vec, F_vec)
    w2c = torch.eye(4, device=device)
    w2c[0, :3] = R_vec
    w2c[1, :3] = U_vec
    w2c[2, :3] = -F_vec
    cam_pos_world = eye.clone()
    w2c[0, 3] = -torch.dot(R_vec, cam_pos_world)
    w2c[1, 3] = -torch.dot(U_vec, cam_pos_world)
    w2c[2, 3] = torch.dot(F_vec, cam_pos_world)
    return w2c

# Function to get intrinsics from FoV
def get_camera_space_intrinsics(fov_x_rad, fov_y_rad, W, H, device):
    tan_fovx_half = math.tan(fov_x_rad / 2.0)
    tan_fovy_half = math.tan(fov_y_rad / 2.0)
    fx = W / (2 * tan_fovx_half)
    fy = H / (2 * tan_fovy_half)
    cx = W / 2.0
    cy = H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32)
    return K, tan_fovx_half, tan_fovy_half

# --- Helper: Setup Camera ---
def setup_camera_params(W, H, azimuth_deg, elevation_deg, distance, device):
    # 1. Calculate Camera Position
    phi = math.radians(90.0 - elevation_deg) # Polar angle (from +Y axis)
    theta = math.radians(azimuth_deg)        # Azimuthal angle (around Y axis, from +Z)
    cam_x = distance * math.sin(phi) * math.sin(theta)
    cam_y = distance * math.cos(phi)
    cam_z = distance * math.sin(phi) * math.cos(theta)
    cam_pos_world = torch.tensor([cam_x, cam_y, cam_z], device=device)

    # Camera looks at origin, standard up vector
    look_at_point = torch.tensor([0.0, 0.0, 0.0], device=device)
    up_vector = torch.tensor([0.0, 1.0, 0.0], device=device) # Y-up

    # Calculate World-to-Camera (View) matrix (OpenGL style)
    w2c_gl = LookAt(cam_pos_world, look_at_point, up_vector)

    # Camera settings (base)
    fov_x_deg_initial, fov_y_deg_initial = 68.0, 68.0 # Wider FoV, closer to NPZ example
    print(f"  Using Initial FoV: {fov_x_deg_initial}, {fov_y_deg_initial} degrees")
    near, far = 0.1, 1000.0 # Use consistent large far plane

    # Calculate initial FoV tangents and K for gsplat (can be overridden)
    fov_x_rad_current = math.radians(fov_x_deg_initial)
    fov_y_rad_current = math.radians(fov_y_deg_initial)
    
    # K_params for gsplat, tan_fovx_half_current, tan_fovy_half_current for DGR FoV
    K_params, tan_fovx_half_current, tan_fovy_half_current = get_camera_space_intrinsics(
        fov_x_rad_current, fov_y_rad_current, W, H, device
    )

    # --- Override K for gsplat and update FoV parameters for DGR if W,H match test_garden.npz ---
    if W == 648 and H == 420:
        print("  Overriding K with test_garden.npz values for gsplat.")
        fx_npz, fy_npz = 480.61234, 481.54453
        cx_npz, cy_npz = 324.1875, 210.0625
        K_params = torch.tensor([[fx_npz, 0, cx_npz], [0, fy_npz, cy_npz], [0, 0, 1]], device=device, dtype=torch.float32)
        
        # Update tan_fovx_half_current and tan_fovy_half_current for DGR based on overridden fx, fy
        # These will define the FoV for DGR's projection matrix
        print("  Recalculating FoV tangents for DGR based on NPZ intrinsics.")
        tan_fovx_half_current = W / (2 * fx_npz) if fx_npz != 0 else float('inf')
        tan_fovy_half_current = H / (2 * fy_npz) if fy_npz != 0 else float('inf')
        
        # Update fov_x_rad_current and fov_y_rad_current for DGR projection matrix calculation
        fov_x_rad_current = 2 * math.atan(tan_fovx_half_current)
        fov_y_rad_current = 2 * math.atan(tan_fovy_half_current)
        print(f"  Updated FoV for DGR (from NPZ): {math.degrees(fov_x_rad_current):.2f} deg (X), {math.degrees(fov_y_rad_current):.2f} deg (Y)")
    # ----------------------------------------------------------------------------

    # Camera position is 'eye'
    camera_pos = cam_pos_world.clone()

    # --- Parameters for DGR (Corrected based on ops.py interpretation) ---
    # w2c_gl is standard OpenGL W2C matrix (world-to-camera)

    # 1. Construct flip_yz matrix (applied to W2C_opengl to get DGR's W2C convention)
    flip_yz = torch.zeros(4, 4, device=device, dtype=w2c_gl.dtype)
    flip_yz[0, 0] = 1.0
    flip_yz[1, 1] = -1.0 # Flip Y axis for DGR convention
    flip_yz[2, 2] = -1.0 # Flip Z axis for DGR convention
    flip_yz[3, 3] = 1.0
    
    # 2. Transform W2C_opengl to DGR's W2C system: DGR_W2C = flip_yz @ W2C_opengl
    w2c_dgr_convention = torch.matmul(flip_yz, w2c_gl)

    # 3. Transpose for DGR's viewmatrix argument
    view_matrix_dgr_corrected = w2c_dgr_convention.transpose(0, 1).contiguous()
    
    # 4. DGR Projection Matrix: Use get_projection_matrix_gaussian from ops.py
    # It requires fovX and fovY in RADIANS (derived from current, possibly NPZ-updated, tangents)
    P_dgr_style = get_projection_matrix_gaussian(
        znear=near,
        zfar=far,
        fovX=fov_x_rad_current, # Use the (potentially updated) radians
        fovY=fov_y_rad_current, # Use the (potentially updated) radians
        device=device
    )
    # proj_matrix_for_dgr_corrected = P_dgr_style.transpose(0, 1).contiguous() # Old: P.T
    # New: DGR expects projmatrix = (W2C_dgr_convention.T @ P_dgr_style.T)
    # This is equivalent to (P_dgr_style @ W2C_dgr_convention).T
    mvp_dgr_style = torch.matmul(P_dgr_style, w2c_dgr_convention)
    proj_matrix_for_dgr_corrected = mvp_dgr_style.transpose(0,1).contiguous()

    camera_center_dgr_corrected = camera_pos # 'campos' for DGR is the camera position in world space
    
    # tanfovx/y for DGR raster settings should correspond to the FoV used in P_dgr_style
    tanfovx_dgr_corrected = tan_fovx_half_current 
    tanfovy_dgr_corrected = tan_fovy_half_current
    
    # --- Parameters for gsplat (v6 convention - OpenCV W2C + K) ---
    w2c_cv = convert_w2c_gl_to_cv(w2c_gl)
    view_matrix_gsplat_cv = w2c_cv.unsqueeze(0) # Add batch dim [1, 4, 4]
    
    K_gsplat_matrix = get_intrinsics_from_fov(fov_y_rad_current, W, H, device) 
    K_gsplat = K_gsplat_matrix.unsqueeze(0) # Add batch dim [1, 3, 3]

    return {
        # DGR Params (Corrected based on ops.py interpretation)
        "view_matrix_dgr": view_matrix_dgr_corrected,
        "full_proj_matrix_dgr": proj_matrix_for_dgr_corrected,
        "camera_center_dgr": camera_center_dgr_corrected,
        "tanfovx_dgr": tanfovx_dgr_corrected,
        "tanfovy_dgr": tanfovy_dgr_corrected,
        
        # gsplat Params (standard OpenCV convention)
        "view_matrix_gsplat_cv": view_matrix_gsplat_cv,
        "K_gsplat": K_gsplat,
        
        # Store original W2C_gl (based on gsplat target for reference) and P_gl for DGR for reference
        "w2c_gl_standard": LookAt(cam_pos_world, look_at_point, up_vector), # Original W2C OpenGL
        "W2C_dgr_convention_standard": w2c_dgr_convention, # flip_yz @ W2C_opengl
        "P_gl_standard": P_dgr_style, # The P matrix used for DGR (before transpose)
        "name": "custom_view" 
    }

# --- Helper: Generate Sample Gaussians (Modified for 1000 random points) ---
def generate_gaussians(num_points, device, radius=0.5):
    # Generate points within a sphere
    phi = torch.rand(num_points, device=device) * 2 * math.pi  # Azimuthal angle
    costheta = torch.rand(num_points, device=device) * 2 - 1   # Cosine of polar angle
    theta = torch.acos(costheta)                             # Polar angle
    u = torch.rand(num_points, device=device)                # Uniform random for radius
    r = radius * (u ** (1./3.))                              # Radius, scaled to be uniform in volume

    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    
    means = torch.stack((x, y, z), dim=-1)
    print(f"  Generated {num_points} Gaussians spherically distributed around origin with radius {radius}")
    
    colors = torch.rand(num_points, 3, device=device)
    # Make scales smaller for a denser cloud appearance if radius is small
    scale_factor = radius # Smaller radius, smaller gaussians
    scales = torch.rand(num_points, 3, device=device) * 0.05 * scale_factor + 0.01 * scale_factor 
    print(f"  Using scales for Gaussians: approx range [{0.01*scale_factor:.3f}, {0.06*scale_factor:.3f}]")
    opacities = torch.rand(num_points, 1, device=device) * 0.7 + 0.3 # Range [0.3, 1.0]
    rotations = torch.zeros(num_points, 4, device=device)
    rotations[:, 0] = 1.0 # w=1

    return {
        "means3D": means, "colors_precomp": colors, "opacities": opacities,
        "scales": scales, "rotations": rotations, "shs": None
    }
    
# --- Helper: Calculate Camera Pose from Angles/Distance ---
def get_camera_pose(azimuth_deg, elevation_deg, distance, device):
    target_world = torch.tensor([0.0, 0.0, 0.0], device=device)
    up_world = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)
    
    # Spherical to Cartesian conversion (Y-up)
    x = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    y = distance * math.sin(elevation_rad)
    z = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    
    cam_pos = torch.tensor([x, y, z], device=device)
    return cam_pos, target_world, up_world

# --- Helper: Normalize depth for visualization ---
def normalize_depth_for_vis(depth_map, min_val_vis=None, max_val_vis=None, alpha_map=None, near_clip=0.0, far_clip=float('inf')):
    if depth_map is None:
        return torch.zeros(1, 256, 256) # Default placeholder if depth is None
    
    if depth_map.ndim == 3 and depth_map.shape[0] == 1:
        depth_map_vis = depth_map.squeeze(0).cpu().clone()
    elif depth_map.ndim == 2:
        depth_map_vis = depth_map.cpu().clone()
    else:
        raise ValueError(f"Unexpected depth map shape: {depth_map.shape}")

    valid_mask = torch.isfinite(depth_map_vis) & (depth_map_vis > near_clip) & (depth_map_vis < far_clip)
    
    if alpha_map is not None:
        if alpha_map.ndim == 3 and alpha_map.shape[0] == 1:
            alpha_map_squeezed = alpha_map.squeeze(0).cpu()
        elif alpha_map.ndim == 2:
            alpha_map_squeezed = alpha_map.cpu()
        else:
            raise ValueError(f"Unexpected alpha map shape: {alpha_map.shape}")
        valid_mask &= (alpha_map_squeezed > 0.01)


    if not valid_mask.any():
        return torch.zeros_like(depth_map_vis).unsqueeze(0)

    valid_depths = depth_map_vis[valid_mask]
    if not valid_depths.numel():
         return torch.zeros_like(depth_map_vis).unsqueeze(0)

    current_min_val = torch.min(valid_depths) if min_val_vis is None else min_val_vis
    current_max_val = torch.max(valid_depths) if max_val_vis is None else max_val_vis
    
    # Ensure min_val is not greater than max_val after potential clamping by near/far
    current_min_val = min(current_min_val, current_max_val)

    if current_max_val <= current_min_val:
        # Handle cases where all valid depths are the same or range is zero
        if current_min_val == 0 and current_max_val == 0: # all black
             normalized_depths = torch.zeros_like(valid_depths)
        else: # all some constant depth, map to mid-gray
             normalized_depths = torch.full_like(valid_depths, 0.5)
    else:
        normalized_depths = (valid_depths - current_min_val) / (current_max_val - current_min_val + 1e-8)

    normalized_image = torch.zeros_like(depth_map_vis) # Background is black
    normalized_image[valid_mask] = normalized_depths
    normalized_image = torch.clamp(normalized_image, 0.0, 1.0)
    return normalized_image.unsqueeze(0) # 1, H, W

# --- Main Comparison Logic ---
# Restore to original logic comparing DGR and gsplat
def compare_renders(cam_params, gaussians, W, H, bg_color=(1.0, 1.0, 1.0), output_dir=Path("./output"), view_idx=0):
    device = gaussians["means3D"].device
    means = gaussians["means3D"]
    colors = gaussians["colors_precomp"]
    opacities = gaussians["opacities"] # Expects N,1
    scales = gaussians["scales"]
    rotations = gaussians["rotations"] # Expects N,4 WXYZ
    # Infer sh_degree based on color dimensions if needed, assume 0 for now
    sh_degree = 0 if colors.ndim == 2 or colors.shape[1]==1 else int(math.sqrt(colors.shape[1]) - 1)
    print(f"  Inferred SH degree: {sh_degree}")

    bg_color_tensor = torch.tensor(bg_color, dtype=torch.float32, device=device)

    view_dir = output_dir / f"view_{view_idx}"
    view_dir.mkdir(exist_ok=True, parents=True)

    # Initialize render results to None or error indicators
    _render_dgr, _alpha_dgr = None, None
    _render_gsplat, _alpha_gsplat = None, None
    l1_diff_dgr = torch.tensor(0.0, device=device) # Default to 0 diff if error
    l1_diff_gsplat = torch.tensor(0.0, device=device)
    img_l1, img_mse, alpha_l1, alpha_mse = torch.tensor(-1.0), torch.tensor(-1.0), torch.tensor(-1.0), torch.tensor(-1.0) # Indicate failure

    # === DGR Render ===
    view_matrix_for_dgr_call = cam_params["view_matrix_dgr"]
    proj_matrix_for_dgr_call = cam_params["full_proj_matrix_dgr"]

    # Setup DGR rasterizer settings
    dgr_raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=cam_params["tanfovx_dgr"],
        tanfovy=cam_params["tanfovy_dgr"],
        bg=bg_color_tensor,
        scale_modifier=1.0,
        viewmatrix=view_matrix_for_dgr_call,
        projmatrix=proj_matrix_for_dgr_call,
        sh_degree=0,
        campos=cam_params["camera_center_dgr"],
        prefiltered=False,
        kernel_size=0.0,
        require_depth=True,
        require_coord=False,
        debug=False
    )
    dgr_rasterizer = GaussianRasterizer(raster_settings=dgr_raster_settings)

    # DGR Rendering (with colors_precomp, shs=None)
    rendered_image_dgr, _, rendered_depth_dgr_raw, _, rendered_alpha_dgr, _, _, _ = dgr_rasterizer(
        means3D=means,
        means2D=torch.zeros_like(means, dtype=torch.float32, device=device),
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    _render_dgr = rendered_image_dgr
    _alpha_dgr = rendered_alpha_dgr

    # Reshape bg_color_tensor to be (C, 1, 1) for broadcasting with _render_dgr (C, H, W)
    l1_diff_dgr = torch.abs(_render_dgr - bg_color_tensor.view(-1, 1, 1)).mean()
    print(f"  Rendered vs Background L1 Diff (DGR): {l1_diff_dgr.item():.6f}")

    # --- DGR Projection Check (threestudio convention) ---
    if view_idx == 0: # Only print for the first view for brevity
        print("  --- DGR: Projecting first 5 Gaussian means to 2D screen space (Threestudio Convention) ---")
        # proj_matrix_for_dgr_call is DGR's projmatrix argument: (P_dgr_style @ W2C_dgr_convention).T
        # view_matrix_for_dgr_call is DGR's viewmatrix argument: W2C_dgr_convention.T
        
        # For debug projection, we need MVP_dgr = P_dgr_style @ W2C_dgr_convention
        P_dgr_style_for_check = cam_params.get("P_gl_standard") 
        W2C_dgr_convention_for_check = cam_params.get("W2C_dgr_convention_standard")

        if P_dgr_style_for_check is None or W2C_dgr_convention_for_check is None:
            print("    Skipping DGR projection check: P_gl_standard or W2C_dgr_convention_standard not found in cam_params.")
        else:
            mvp_dgr_effective = torch.matmul(P_dgr_style_for_check, W2C_dgr_convention_for_check)

            num_gaussians_to_check = min(5, means.shape[0])
            for i in range(num_gaussians_to_check):
                mean_world = means[i]
                mean_world_h = torch.cat([mean_world, torch.tensor([1.0], device=device, dtype=mean_world.dtype)])
                
                coords_clip_dgr = mvp_dgr_effective @ mean_world_h
                
                # Check if behind camera (w_clip < near_plane_threshold, e.g. a small positive epsilon or 0)
                # A simple check for w_clip > 0 is usually enough to see if it's in front of the projection center.
                if coords_clip_dgr[3] <= 1e-5: # w-component (depth factor)
                    print(f"    Gaussian {i} (World: {mean_world.cpu().numpy()}) is behind or too close to DGR camera plane (w_clip={coords_clip_dgr[3].item()})")
                    continue

                coords_ndc_dgr = coords_clip_dgr[:3] / coords_clip_dgr[3]
                
                pixel_x_dgr = (coords_ndc_dgr[0] + 1) / 2 * W
                pixel_y_dgr = (1 - coords_ndc_dgr[1]) / 2 * H # Y is inverted from NDC

                on_screen_x = 0 <= pixel_x_dgr.item() < W
                on_screen_y = 0 <= pixel_y_dgr.item() < H
                on_screen_z = -1 <= coords_ndc_dgr[2].item() <= 1 # Check if Z is within NDC depth bounds

                print(f"    Gaussian {i} (World: {mean_world.cpu().numpy()})")
                print(f"      -> DGR NDC: [{coords_ndc_dgr[0].item():.3f}, {coords_ndc_dgr[1].item():.3f}, {coords_ndc_dgr[2].item():.3f}] (w_clip={coords_clip_dgr[3].item():.3f})")
                print(f"      -> DGR Screen: x={pixel_x_dgr.item():.1f}, y={pixel_y_dgr.item():.1f} (On Screen X: {on_screen_x}, Y: {on_screen_y}, Z: {on_screen_z})")
        print("  --- End DGR Projection Check ---")

    print(f"Debug DGR render before save: shape={_render_dgr.shape}, dtype={_render_dgr.dtype}, min={_render_dgr.min()}, max={_render_dgr.max()}")
    sys.stdout.flush()
    # Save DGR image and alpha map
    save_image(_render_dgr, view_dir / f"dgr_render_view_{view_idx}.png")
    save_image(_alpha_dgr, view_dir / f"dgr_alpha_view_{view_idx}.png")

    # --- Calculate and print bounding box for DGR render ---
    bg_color_r, bg_color_g, bg_color_b = bg_color_tensor[0], bg_color_tensor[1], bg_color_tensor[2]
    # Create a mask for non-background pixels (handle potential float inaccuracies)
    # _render_dgr is (C, H, W)
    dgr_is_not_bg_r = torch.abs(_render_dgr[0] - bg_color_r) > 1e-3
    dgr_is_not_bg_g = torch.abs(_render_dgr[1] - bg_color_g) > 1e-3
    dgr_is_not_bg_b = torch.abs(_render_dgr[2] - bg_color_b) > 1e-3
    dgr_content_mask = dgr_is_not_bg_r | dgr_is_not_bg_g | dgr_is_not_bg_b # (H, W)
    
    dgr_rendered_pixels_y, dgr_rendered_pixels_x = torch.where(dgr_content_mask)
    if dgr_rendered_pixels_x.numel() > 0:
        dgr_xmin, dgr_xmax = dgr_rendered_pixels_x.min().item(), dgr_rendered_pixels_x.max().item()
        dgr_ymin, dgr_ymax = dgr_rendered_pixels_y.min().item(), dgr_rendered_pixels_y.max().item()
        print(f"  DGR Rendered Content BBox: x=[{dgr_xmin}-{dgr_xmax}], y=[{dgr_ymin}-{dgr_ymax}], w={dgr_xmax-dgr_xmin+1}, h={dgr_ymax-dgr_ymin+1}")
    else:
        print("  DGR Rendered Content BBox: No non-background pixels found.")

    # --- 2. Render with gsplat ---
    print("\n--- Rendering with gsplat ---")
    try:
        # Extract necessary parameters from cam_params for gsplat
        view_matrix_gsplat_cv_batched = cam_params["view_matrix_gsplat_cv"] # Use the pre-converted OpenCV W2C [1, 4, 4]
        K_gsplat_batched = cam_params["K_gsplat"] # Standard K [1, 3, 3]

        # For projection check, use the unbatched versions
        view_matrix_gsplat_cv_unbatched = view_matrix_gsplat_cv_batched[0] # [4, 4]
        K_gsplat_3x3_unbatched = K_gsplat_batched[0] # [3, 3]

        print("  Using OpenCV W2C matrix provided by setup_camera_params.")

        # --- BEGIN: Print projected 2D coordinates for gsplat (for view_idx == 0) ---
        if view_idx == 0:
            print("  --- gsplat: Projecting first 5 Gaussian means to 2D screen space ---")
            num_gaussians_to_check = min(5, means.shape[0])
            for i in range(num_gaussians_to_check):
                mean_world = means[i]
                mean_world_h = torch.cat([mean_world, torch.tensor([1.0], device=device, dtype=mean_world.dtype)])
                
                # Transform to OpenCV camera coordinates
                # view_matrix_gsplat_cv is W2C_cv (world to OpenCV camera)
                coords_cam_cv_h = view_matrix_gsplat_cv_unbatched @ mean_world_h
                coords_cam_cv = coords_cam_cv_h[:3]

                if coords_cam_cv[2] <= 1e-5: # Depth in camera space (Z should be positive and > near_clip for OpenCV)
                    print(f"    Gaussian {i} (World: {mean_world.cpu().numpy()}) is behind or too close to gsplat camera plane (Z_cv={coords_cam_cv[2].item()})")
                    continue

                # Project to image plane using K
                # K_gsplat_3x3 is [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
                # coords_img_h = [u*z, v*z, z]
                coords_img_h = K_gsplat_3x3_unbatched @ coords_cam_cv 
                
                pixel_x_gsplat = coords_img_h[0] / coords_img_h[2]
                pixel_y_gsplat = coords_img_h[1] / coords_img_h[2]
                depth_z_gsplat = coords_img_h[2]

                on_screen_x = 0 <= pixel_x_gsplat.item() < W
                on_screen_y = 0 <= pixel_y_gsplat.item() < H

                print(f"    Gaussian {i} (World: {mean_world.cpu().numpy()})")
                print(f"      -> gsplat CamCV: [{coords_cam_cv[0].item():.3f}, {coords_cam_cv[1].item():.3f}, {coords_cam_cv[2].item():.3f}]")
                print(f"      -> gsplat Screen: x={pixel_x_gsplat.item():.1f}, y={pixel_y_gsplat.item():.1f}, depth_Z={depth_z_gsplat.item():.3f} (On Screen X: {on_screen_x}, Y: {on_screen_y})")
            print("  --- End gsplat Projection Check ---")
        # --- END: Print projected 2D coordinates for gsplat ---

        # Extract Gaussian parameters for gsplat
        means3D_gsplat = gaussians["means3D"]
        quats_gsplat = gaussians["rotations"]
        scales_gsplat = gaussians["scales"]
        opacities_gsplat = gaussians["opacities"]
        colors_gsplat = gaussians["colors_precomp"]

        if opacities_gsplat.ndim == 2 and opacities_gsplat.shape[1] == 1:
            opacities_input_gsplat = opacities_gsplat.squeeze(-1)
        else:
            opacities_input_gsplat = opacities_gsplat
        if colors_gsplat is None:
             raise ValueError("Missing 'colors_precomp' in gaussians dict for gsplat.")

        _render_gsplat, _alpha_gsplat, _ = gsplat.rasterization(
            means=means3D_gsplat,
            quats=quats_gsplat,
            scales=scales_gsplat,
            opacities=opacities_input_gsplat,
            colors=colors_gsplat,
            viewmats=view_matrix_gsplat_cv_batched, # Pass the original batched [1, 4, 4]
            Ks=K_gsplat_batched,                   # Pass the original batched [1, 3, 3]
            width=W,
            height=H,
            render_mode='RGB',
            backgrounds=bg_color_tensor.unsqueeze(0)
        )
        l1_diff_gsplat = torch.abs(_render_gsplat - bg_color_tensor).mean()
        print(f"  Rendered vs Background L1 Diff (gsplat): {l1_diff_gsplat.item():.6f}")

        # Save gsplat image and alpha map
        save_image(_render_gsplat[0].permute(2,0,1), view_dir / f"gsplat_render_view_{view_idx}.png")
        save_image(_alpha_gsplat[0].permute(2,0,1), view_dir / f"gsplat_alpha_view_{view_idx}.png")

        # --- Calculate and print bounding box for gsplat render ---
        # _render_gsplat is (B, H, W, C), take B=0: (H, W, C)
        gsplat_render_for_bbox = _render_gsplat[0] # (H, W, C)
        gsplat_is_not_bg_r = torch.abs(gsplat_render_for_bbox[:,:,0] - bg_color_r) > 1e-3
        gsplat_is_not_bg_g = torch.abs(gsplat_render_for_bbox[:,:,1] - bg_color_g) > 1e-3
        gsplat_is_not_bg_b = torch.abs(gsplat_render_for_bbox[:,:,2] - bg_color_b) > 1e-3
        gsplat_content_mask = gsplat_is_not_bg_r | gsplat_is_not_bg_g | gsplat_is_not_bg_b # (H, W)
        
        gsplat_rendered_pixels_y, gsplat_rendered_pixels_x = torch.where(gsplat_content_mask)
        if gsplat_rendered_pixels_x.numel() > 0:
            gsplat_xmin, gsplat_xmax = gsplat_rendered_pixels_x.min().item(), gsplat_rendered_pixels_x.max().item()
            gsplat_ymin, gsplat_ymax = gsplat_rendered_pixels_y.min().item(), gsplat_rendered_pixels_y.max().item()
            print(f"  Gsplat Rendered Content BBox: x=[{gsplat_xmin}-{gsplat_xmax}], y=[{gsplat_ymin}-{gsplat_ymax}], w={gsplat_xmax-gsplat_xmin+1}, h={gsplat_ymax-gsplat_ymin+1}")
        else:
            print("  Gsplat Rendered Content BBox: No non-background pixels found.")

    except ValueError as ve:
         print(f"  ValueError during gsplat setup: {ve}")
    except Exception as e:
        print(f"  Error during gsplat rendering: {e}")
        import traceback
        traceback.print_exc()

    # --- 3. Compare Outputs & Save --- 
    if _render_dgr is not None:
        # _render_dgr is (C, H, W), save_image handles this directly.
        save_image(_render_dgr, view_dir / f"render_dgr_view{view_idx}.png")
        if _alpha_dgr is not None:
            # _alpha_dgr is (1, H, W). Repeat to (3, H, W) for saving as grayscale image.
            save_image(_alpha_dgr.repeat(3,1,1), view_dir / f"alpha_dgr_view{view_idx}.png")

    if _render_gsplat is not None:
        # _render_gsplat is [B, H, W, C], take first batch, permute to [C, H, W]
        img_gsplat_to_save = _render_gsplat[0].permute(2,0,1)
        save_image(img_gsplat_to_save, view_dir / f"render_gsplat_view{view_idx}.png")
        if _alpha_gsplat is not None:
            # _alpha_gsplat is [B, H, W, 1], take first batch [0] -> (H,W,1), permute to (1,H,W)
            alpha_gsplat_save = _alpha_gsplat[0].permute(2,0,1) # (1, H, W)
            # Repeat to (3, H, W) for saving as grayscale image.
            save_image(alpha_gsplat_save.repeat(3,1,1), view_dir / f"alpha_gsplat_view{view_idx}.png")

    if _render_dgr is not None and _render_gsplat is not None:
        # Need to make sure render_gsplat is also 3D for comparison (D, H, W)
        render_gsplat_comp = _render_gsplat[0].permute(2, 0, 1) # Convert (H,W,D) -> (D,H,W)
        alpha_gsplat_comp = _alpha_gsplat[0].permute(2, 0, 1) # Convert (H,W,D) -> (D,H,W)
        
        img_l1 = torch.mean(torch.abs(_render_dgr - render_gsplat_comp))
        img_mse = torch.mean((_render_dgr - render_gsplat_comp)**2)
        print(f"  Image L1 Diff: {img_l1.item():.6f}")
        print(f"  Image MSE: {img_mse.item():.6e}")
        
        # Save difference image (scaled for visibility)
        img_diff = torch.abs(_render_dgr - render_gsplat_comp)
        save_image(torch.clamp(img_diff.mean(dim=0, keepdim=True) * 5.0, 0, 1), view_dir / f"img_abs_diff_scaled_view{view_idx}.png") 

        if _alpha_dgr is not None and _alpha_gsplat is not None:
            alpha_l1 = torch.mean(torch.abs(_alpha_dgr - alpha_gsplat_comp))
            alpha_mse = torch.mean((_alpha_dgr - alpha_gsplat_comp)**2)
            print(f"  Alpha L1 Diff: {alpha_l1.item():.6f}")
            print(f"  Alpha MSE: {alpha_mse.item():.6e}")
            # Save alpha difference image
            alpha_diff = torch.abs(_alpha_dgr - alpha_gsplat_comp)
            save_image(torch.clamp(alpha_diff * 5.0, 0, 1), view_dir / f"alpha_abs_diff_scaled_view{view_idx}.png")
        else:
            print("  Cannot compare alpha maps (one or both missing).")
    else:
        print("  Skipping comparison due to missing render(s).")

    return l1_diff_dgr, l1_diff_gsplat, img_l1, img_mse, alpha_l1, alpha_mse

# === STEP 2: Add Adapted _create_camera_from_angle Logic ===
# (Placed somewhere before setup_camera_params_threestudio_logic)

def create_camera_from_angle_standalone(
    elevation_deg: torch.Tensor, # Shape [B]
    azimuth_deg: torch.Tensor,   # Shape [B]
    camera_distances: torch.Tensor, # Shape [B]
    fovy_deg: torch.Tensor,      # Shape [B]
    H: int,
    W: int,
    device: torch.device,
    near_plane: float,
    far_plane: float,
    rays_d_normalize: bool = True, # Match dataloader default
    relative_radius: bool = True, # Match dataloader default, though likely False needed for consistency
) -> Dict[str, Any]:
    """
    Generates camera parameters (c2w, mvp, etc.) using logic adapted from
    multiview_multiprompt_multistep_scene.py.
    Assumes batch size B.
    """
    batch_size = elevation_deg.shape[0]

    fovy = fovy_deg * math.pi / 180
    azimuth: torch.Tensor = azimuth_deg * math.pi / 180
    elevation: torch.Tensor = elevation_deg * math.pi / 180

    # Handle relative radius if needed (likely should be False for direct comparison)
    effective_camera_distances = camera_distances
    if relative_radius:
        scale = 1 / torch.tan(0.5 * fovy)
        effective_camera_distances = scale * camera_distances

    # Convert spherical coordinates to cartesian coordinates (Y-up default in original code, but let\'s check axis)
    # Original dataloader comment: \"right hand coordinate system, x back, y right, z up\"
    # Let\'s use this Z-up convention:
    camera_positions: torch.Tensor = torch.stack(
        [
            effective_camera_distances * torch.cos(elevation) * torch.cos(azimuth), # X = R * cos(el) * cos(az)
            effective_camera_distances * torch.cos(elevation) * torch.sin(azimuth), # Y = R * cos(el) * sin(az)
            effective_camera_distances * torch.sin(elevation),                     # Z = R * sin(el)
        ],
        dim=-1,
    ).to(torch.float32).to(device)

    # Default scene center at origin
    center: torch.Tensor = torch.zeros_like(camera_positions)
    # Default camera up direction as +Z
    up: torch.Tensor = torch.as_tensor([0, 0, 1], dtype=torch.float32, device=device)[
        None, :
    ].repeat(batch_size, 1)

    # Camera to world matrix (constructing C2W directly)
    lookat: torch.Tensor = F.normalize(center - camera_positions, dim=-1) # Direction camera points (world space)
    right: torch.Tensor = F.normalize(torch.linalg.cross(lookat, up, dim=-1), dim=-1) # Camera right vector (world space)
    up_actual: torch.Tensor = F.normalize(torch.linalg.cross(right, lookat, dim=-1), dim=-1) # Camera up vector (orthogonalized)

    c2w3x4: torch.Tensor = torch.cat(
        [torch.stack([right, up_actual, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w: torch.Tensor = torch.cat(
        [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
    )
    c2w[:, 3, 3] = 1.0

    # Calculate FoVx from FoVy and aspect ratio for projection matrix
    aspect_ratio = W / H
    fovx = 2 * torch.atan(torch.tan(fovy * 0.5) * aspect_ratio)

    # Calculate projection matrix (using threestudio function)
    # Note: get_projection_matrix expects fovy in radians
    proj_mtx: torch.Tensor = get_projection_matrix(
        fovy, aspect_ratio, near_plane, far_plane
    ).to(device).repeat(batch_size, 1, 1) # Make it shape [B, 4, 4]

    # Calculate Model-View-Projection matrix (using threestudio function)
    # Note: get_mvp_matrix expects proj_mtx and c2w
    mvp_mtx: torch.Tensor = get_mvp_matrix(c2w, proj_mtx) # mvp = proj @ inverse(c2w)

    # Calculate ray directions (can use for verification, not strictly needed for rasterizers)
    # Using rendering dimensions H, W
    directions_unit_focal = get_ray_directions(H=H, W=W, focal=1.0).to(device)
    focal_length_pix = 0.5 * H / torch.tan(0.5 * fovy) # Focal length in pixels
    directions = directions_unit_focal[None, :, :, :].repeat(batch_size, 1, 1, 1)
    directions[:, :, :, :2] = (
        directions[:, :, :, :2] / focal_length_pix[:, None, None, None]
    )
    # Note: get_rays expects directions [B, H, W, 3] and c2w [B, 4, 4]
    rays_o, rays_d = get_rays(directions, c2w, keepdim=True, normalize=rays_d_normalize)


    return {
        "c2w": c2w, # [B, 4, 4]
        "mvp_mtx": mvp_mtx, # [B, 4, 4] (P_gl @ W2C_gl)
        "camera_positions": camera_positions, # [B, 3]
        "fovy": fovy, # [B] radians
        "fovx": fovx, # [B] radians
        # Optional:
        # "rays_o": rays_o,
        # "rays_d": rays_d,
        # "proj_mtx": proj_mtx # OpenGL projection matrix P_gl
    }
# ==========================================================

# === STEP 3: Define setup_camera_params_threestudio_logic ===
# (Placed somewhere after the above function and before compare_renders)

def setup_camera_params_threestudio_logic(W, H, azimuth_deg, elevation_deg, distance, fovy_deg, device, near=0.1, far=100.0):
    """
    Sets up camera parameters using threestudio logic and derives parameters
    for both DGR (v5 transposed convention) and gsplat (v6 OpenCV convention).
    """
    # Prepare inputs for create_camera_from_angle_standalone (batch size 1)
    az_t = torch.tensor([azimuth_deg], dtype=torch.float32, device=device)
    el_t = torch.tensor([elevation_deg], dtype=torch.float32, device=device)
    dist_t = torch.tensor([distance], dtype=torch.float32, device=device)
    fovy_deg_t = torch.tensor([fovy_deg], dtype=torch.float32, device=device)

    # Get camera parameters from the adapted dataloader logic
    ts_cam_params = create_camera_from_angle_standalone(
        elevation_deg=el_t,
        azimuth_deg=az_t,
        camera_distances=dist_t,
        fovy_deg=fovy_deg_t,
        H=H, W=W, device=device,
        near_plane=near, far_plane=far,
        relative_radius=False # Use absolute distance
    )

    c2w = ts_cam_params["c2w"][0]                 # [4, 4]
    mvp_mtx = ts_cam_params["mvp_mtx"][0]         # [4, 4] (P_gl @ W2C_gl)
    camera_pos = ts_cam_params["camera_positions"][0] # [3]
    fovy_rad = ts_cam_params["fovy"][0]           # scalar tensor
    fovx_rad = ts_cam_params["fovx"][0]           # scalar tensor

    # Calculate W2C_gl
    w2c_gl = torch.inverse(c2w)

    # --- Parameters for DGR ---
    # Correct DGR conversion based on ops.py: viewmatrix = (flip_yz @ W2C_opengl).T
    # w2c_gl is standard OpenGL W2C matrix (world-to-camera)
    # P_gl is standard OpenGL projection matrix

    # 1. Construct flip_yz matrix (applied to W2C_opengl to get DGR\'s W2C convention)
    flip_yz = torch.zeros(4, 4, device=device, dtype=w2c_gl.dtype)
    flip_yz[0, 0] = 1.0
    flip_yz[1, 1] = -1.0 # Flip Y axis for DGR convention
    flip_yz[2, 2] = -1.0 # Flip Z axis for DGR convention
    flip_yz[3, 3] = 1.0
    
    # 2. Transform W2C_opengl to DGR\'s W2C system: DGR_W2C = flip_yz @ W2C_opengl
    w2c_dgr_convention = torch.matmul(flip_yz, w2c_gl)

    # 3. Transpose for DGR\'s viewmatrix argument
    view_matrix_dgr_corrected = w2c_dgr_convention.transpose(0, 1).contiguous()
    
    # P_gl is standard OpenGL projection. DGR expects P_gl.T for its projmatrix argument
    proj_matrix_for_dgr_corrected = P_gl.transpose(0, 1).contiguous()

    camera_center_dgr_corrected = camera_pos # \'campos\' for DGR is the camera position in world space
    
    # tanfovx/y for DGR should correspond to the FoV used in P_gl
    # tan_fovx_half and tan_fovy_half were used to construct P_gl
    tanfovx_dgr_corrected = math.tan(math.radians(fovx_rad) * 0.5)
    tanfovy_dgr_corrected = math.tan(fovy_rad * 0.5)
    
    # --- Parameters for gsplat (v6 convention - OpenCV W2C + K) ---
    w2c_cv = convert_w2c_gl_to_cv(w2c_gl)
    view_matrix_gsplat_cv = w2c_cv.unsqueeze(0) # Add batch dim [1, 4, 4]
    
    K_gsplat_matrix = get_intrinsics_from_fov(fovy_rad, W, H, device) 
    K_gsplat = K_gsplat_matrix.unsqueeze(0) # Add batch dim [1, 3, 3]

    return {
        # DGR Params (Corrected based on ops.py interpretation)
        "view_matrix_dgr": view_matrix_dgr_corrected,
        "full_proj_matrix_dgr": proj_matrix_for_dgr_corrected,
        "camera_center_dgr": camera_center_dgr_corrected,
        "tanfovx_dgr": tanfovx_dgr_corrected,
        "tanfovy_dgr": tanfovy_dgr_corrected,
        # gsplat Params (v6 convention - OpenCV W2C + K)
        "view_matrix_gsplat_cv": view_matrix_gsplat_cv,
        "K_gsplat": K_gsplat,
        
        # Store original W2C_gl (based on gsplat target) and P_gl for reference/other checks
        "w2c_gl_standard": w2c_gl, 
        "P_gl_standard": P_gl, 
        "name": "custom_view" 
    }
# =================================================================

# === NEW FUNCTION: Generate 36 Custom Viewpoints ===
# Modified to generate num_azimuth_steps (e.g., 4) views around the object at a given elevation and distance.
# Both dgr_specific_target and semantic_target_for_gsplat will be the origin [0,0,0].
def generate_custom_viewpoints_data(device: torch.device, 
                                    camera_distance: float = 5.0, 
                                    elevation_deg: float = 30.0,
                                    num_azimuth_steps: int = 4
                                   ) -> list[dict[str, Any]]:
    viewpoints = []
    azimuth_angles_deg = np.linspace(0, 360, num_azimuth_steps, endpoint=False)

    # Both DGR and gsplat will look at the origin for these standard views.
    semantic_target_origin = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
    up_world = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32) # Y-up

    for i, az_deg in enumerate(azimuth_angles_deg):
        # Calculate camera position using helper
        # Convert elevation and azimuth to radians for spherical to Cartesian
        phi = math.radians(90.0 - elevation_deg)  # Polar angle from Y+ axis
        theta = math.radians(az_deg)             # Azimuthal angle around Y from Z+ axis

        eye_x = camera_distance * math.sin(phi) * math.sin(theta)
        eye_y = camera_distance * math.cos(phi)
        eye_z = camera_distance * math.sin(phi) * math.cos(theta)
        eye_pos_tensor = torch.tensor([eye_x, eye_y, eye_z], device=device, dtype=torch.float32)
        
        view_name = f"Az_{az_deg:.0f}_El_{elevation_deg:.0f}_Dist_{camera_distance:.1f}"
        viewpoints.append({
            "name": view_name,
            "eye": eye_pos_tensor.tolist(),
            "semantic_target_for_gsplat": semantic_target_origin.tolist(), # Gsplat looks at origin
            "dgr_specific_target": semantic_target_origin.tolist(),      # DGR also looks at origin
            "up_for_lookat": up_world.tolist(),
        })
    print(f"  Generated {len(viewpoints)} semantic viewpoints: Elevation={elevation_deg}deg, Distance={camera_distance}, Azimuths={azimuth_angles_deg}")
    return viewpoints

# === NEW FUNCTION: Setup Camera Params for Custom View ===
def setup_camera_params_custom_view(
    W: int, H: int,
    eye: torch.Tensor, 
    semantic_target_for_gsplat: torch.Tensor, # Actual point of interest (e.g., origin)
    dgr_specific_target: torch.Tensor,
    up_for_lookat: torch.Tensor,
    fovy_deg: float, device: torch.device,
    near: float, far: float,
) -> Dict[str, Any]:
    """
    Sets up camera parameters for a custom view defined by semantic eye, target, and up.
    Handles DGR's reversed convention internally.
    """

    # Convert list inputs to tensors
    eye_t = torch.tensor(eye, dtype=torch.float32, device=device)
    semantic_target_for_gsplat_t = torch.tensor(semantic_target_for_gsplat, dtype=torch.float32, device=device)
    dgr_specific_target_t = torch.tensor(dgr_specific_target, dtype=torch.float32, device=device)
    up_for_lookat_t = torch.tensor(up_for_lookat, dtype=torch.float32, device=device)

    # --- Calculate FoV and standard GL matrices ---
    fovy_rad = math.radians(fovy_deg)
    aspect_ratio = W / H
    # Ensure fovx_deg is calculated using fovy_rad and aspect_ratio
    fovx_deg = math.degrees(2 * math.atan(math.tan(fovy_rad * 0.5) * aspect_ratio))
    fovx_rad = math.radians(fovx_deg) # fovx in radians

    W2C_gl = LookAt(eye_t, dgr_specific_target_t, up_for_lookat_t) 
    # P_gl = get_projection_matrix(near, far, fovx_deg, fovy_deg, device) # Standard OpenGL projection, not for DGR directly

    # --- DGR Camera Setup (Corrected based on ops.py interpretation) ---
    # W2C_gl here is LookAt(eye_t, dgr_specific_target_t, up_for_lookat_t), which is W2C_opengl

    # 1. Construct flip_yz matrix
    flip_yz = torch.zeros(4, 4, device=device, dtype=W2C_gl.dtype)
    flip_yz[0, 0] = 1.0
    flip_yz[1, 1] = -1.0 # Flip Y
    flip_yz[2, 2] = -1.0 # Flip Z
    flip_yz[3, 3] = 1.0
    
    # 2. Transform W2C_opengl to DGR's W2C system: DGR_W2C = flip_yz @ W2C_opengl
    w2c_dgr_convention = torch.matmul(flip_yz, W2C_gl)

    # 3. Transpose for DGR's viewmatrix argument
    view_matrix_dgr_corrected = w2c_dgr_convention.transpose(0, 1).contiguous()
    
    # 4. DGR Projection Matrix: Use get_projection_matrix_gaussian from ops.py
    # It requires fovX and fovY in RADIANS.
    P_dgr_style = get_projection_matrix_gaussian(
        znear=near,
        zfar=far,
        fovX=fovx_rad, # Use calculated radians
        fovY=fovy_rad, # Use input radians
        device=device
    )
    # proj_matrix_for_dgr_corrected = P_dgr_style.transpose(0, 1).contiguous() # Old: P.T
    # New: DGR expects projmatrix = (W2C_dgr_convention.T @ P_dgr_style.T)
    # This is equivalent to (P_dgr_style @ W2C_dgr_convention).T
    mvp_dgr_style = torch.matmul(P_dgr_style, w2c_dgr_convention)
    proj_matrix_for_dgr_corrected = mvp_dgr_style.transpose(0,1).contiguous()

    camera_center_dgr_corrected = eye_t.clone() # 'campos' for DGR is the camera position in world space
    
    # tanfovx/y for DGR raster settings should correspond to the FoV used in P_dgr_style
    tanfovx_dgr_corrected = math.tan(fovx_rad * 0.5)
    tanfovy_dgr_corrected = math.tan(fovy_rad * 0.5)
    
    # --- Parameters for gsplat (OpenCV W2C + K) ---
    w2c_cv_gsplat = LookAt(eye_t, semantic_target_for_gsplat_t, up_for_lookat_t)
    w2c_cv_gsplat = convert_w2c_gl_to_cv(w2c_cv_gsplat)
    view_matrix_gsplat_cv = w2c_cv_gsplat.unsqueeze(0) 
    
    K_gsplat_matrix = get_intrinsics_from_fov(fovy_rad, W, H, device) 
    K_gsplat = K_gsplat_matrix.unsqueeze(0) # Add batch dim [1, 3, 3]

    return {
        # DGR Params (Corrected based on ops.py interpretation)
        "view_matrix_dgr": view_matrix_dgr_corrected,
        "full_proj_matrix_dgr": proj_matrix_for_dgr_corrected,
        "camera_center_dgr": camera_center_dgr_corrected,
        "tanfovx_dgr": tanfovx_dgr_corrected,
        "tanfovy_dgr": tanfovy_dgr_corrected,
        
        # gsplat Params (standard OpenCV convention)
        "view_matrix_gsplat_cv": view_matrix_gsplat_cv,
        "K_gsplat": K_gsplat,
        
        # Store original W2C_gl (based on gsplat target for reference) and P_gl for DGR for reference
        "w2c_gl_standard": LookAt(eye_t, dgr_specific_target_t, up_for_lookat_t), # W2C for DGR specific target
        "W2C_dgr_convention_standard": w2c_dgr_convention, # flip_yz @ W2C_opengl
        "P_gl_standard": P_dgr_style, # The P matrix used for DGR (before transpose)
        "name": "custom_view" 
    }

# --- Main Comparison Script ---
if __name__ == "__main__":
    # --- Parameters ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Simplified Script Parameters ---
    W_orig, H_orig = 648, 420 
    print(f"  Using W={W_orig}, H={H_orig}")
    num_points_render = 100 # Fewer points for faster testing of spherical cloud
    gaussian_cloud_radius = 0.5 # Radius of the spherical cloud
    print(f"Rendering {num_points_render} Gaussian points (spherical cloud, radius {gaussian_cloud_radius}).")
    
    fovy_deg_for_setup = 60.0 # Standard FoV Y
    near_plane, far_plane = 0.1, 100.0 

    # Camera setup for the 4 views
    camera_distance_val = 5.0
    elevation_deg_val = 30.0
    num_azimuth_steps_val = 4

    # Generate Gaussians (spherical cloud)
    gaussians_data = generate_gaussians(num_points=num_points_render, device=device, radius=gaussian_cloud_radius)

    # Define viewpoints using the new custom function
    viewpoints = generate_custom_viewpoints_data(
        device=device, 
        camera_distance=camera_distance_val, 
        elevation_deg=elevation_deg_val,
        num_azimuth_steps=num_azimuth_steps_val
    )
    print(f"Generated {len(viewpoints)} semantic viewpoints.")


    # --- Output Setup ---
    output_dir_name = f"exp_gsplat_output_4views_el{elevation_deg_val:.0f}_dist{camera_distance_val:.0f}_rad{gaussian_cloud_radius:.1f}"
    output_dir = Path(f"./{output_dir_name}") 
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output images will be saved to: {output_dir.resolve()}")

    # --- Loop & Averaging --- 
    total_img_l1, total_img_mse = 0.0, 0.0
    total_alpha_l1, total_alpha_mse = 0.0, 0.0
    successful_views = 0
    total_l1_dgr, total_l1_gsplat = 0.0, 0.0

    # Loop over viewpoints
    print(f"Starting loop over {len(viewpoints)} viewpoints...") 
    for i, view_config in enumerate(viewpoints):
        view_name = view_config.get("name", f"custom_view_{i}")
        print(f"\n===== View {i}: {view_name} =====")

        print(f"  Calling setup_camera_params_custom_view for view {i}...") 
        cam_params = setup_camera_params_custom_view(
            W=W_orig, H=H_orig,
            eye=view_config['eye'],
            semantic_target_for_gsplat=view_config['semantic_target_for_gsplat'], 
            dgr_specific_target=view_config['dgr_specific_target'],
            up_for_lookat=view_config['up_for_lookat'],
            fovy_deg=fovy_deg_for_setup,
            device=device,
            near=near_plane,
            far=far_plane
        )
        cam_params["name"] = view_name
        print(f"  Finished setup_camera_params_custom_view for view {i}.")

        # Conditional printing of matrices (e.g., for the first view)
        if i == 0: # Print for the first view only for brevity
            print(f"\nDebug Matrices for DGR - View: {view_name}")
            print(f"  eye for LookAt: {view_config['eye']}")
            print(f"  gsplat semantic_target: {view_config['semantic_target_for_gsplat']}")
            print(f"  DGR specific LookAt target: {view_config['dgr_specific_target']}")
            print(f"  up for LookAt: {view_config['up_for_lookat']}")
        # --- END: Print projected 2D coordinates for gsplat ---

        print(f"  Calling compare_renders for view {i}...")
        l1_dgr, l1_gsplat, img_l1, img_mse, alpha_l1, alpha_mse = compare_renders(
            cam_params, gaussians_data, W_orig, H_orig, output_dir=output_dir, view_idx=i
        )
        print(f"  Finished compare_renders for view {i}.")
        
        if isinstance(img_l1, torch.Tensor) and img_l1.item() > -1: 
             total_l1_dgr += l1_dgr.item()
             total_l1_gsplat += l1_gsplat.item()
             total_img_l1 += img_l1.item()
             total_img_mse += img_mse.item()
             total_alpha_l1 += alpha_l1.item()
             total_alpha_mse += alpha_mse.item()
             successful_views += 1 
        elif isinstance(l1_dgr, torch.Tensor) and l1_dgr.item() > -1: 
             total_l1_dgr += l1_dgr.item()
             total_l1_gsplat += l1_gsplat.item() 
        print(f"  End of loop iteration for view {i}.")

    if successful_views > 0:
        print(f"\n===== Average Metrics Across {successful_views}/{len(viewpoints)} Fully Compared Views =====")
        # Divide by len(viewpoints) for DGR/gsplat L1 vs BG as they are always attempted
        print(f"  Average DGR L1 vs BG: {total_l1_dgr / len(viewpoints):.6f}") 
        print(f"  Average gsplat L1 vs BG: {total_l1_gsplat / len(viewpoints):.6f}")
        print(f"  Average Image L1 Diff: {total_img_l1 / successful_views:.6f}")
        print(f"  Average Image MSE: {total_img_mse / successful_views:.6e}")
        print(f"  Average Alpha L1 Diff: {total_alpha_l1 / successful_views:.6f}")
        print(f"  Average Alpha MSE: {total_alpha_mse / successful_views:.6e}")
    else:
        print("\nNo views were successfully compared, or DGR/gsplat renders were missing.")
        if len(viewpoints) > 0 :
             print(f"  Average DGR L1 vs BG (all views attempted): {total_l1_dgr / len(viewpoints):.6f}")
             print(f"  Average gsplat L1 vs BG (all views attempted): {total_l1_gsplat / len(viewpoints):.6f}")

    print(f"\nComparison finished. Check outputs in {output_dir.resolve()}")
