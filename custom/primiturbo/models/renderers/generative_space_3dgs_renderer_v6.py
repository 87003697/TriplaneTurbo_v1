import math
from dataclasses import dataclass

# import numpy as np # Not strictly needed after changes
import threestudio
import torch
import torch.nn.functional as F
# DGR imports removed
# from diff_gaussian_rasterization import (
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )
import gsplat # Added gsplat import

from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *

# get_projection_matrix_gaussian might not be needed if FoV is directly used for K
from threestudio.utils.ops import get_cam_info_gaussian, get_ray_directions, get_rays # Added get_ray_directions and get_rays
from torch.cuda.amp import autocast

from .gaussian_utils import GaussianModel, Depth2Normal # Added Depth2Normal
# Custom center depth rasterizer import removed
# from center_depth_rasterization import rasterize_gaussians_center_depth


class Camera(NamedTuple): # Keep this as it's used by get_cam_info_gaussian
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor # This is W2C_dgr_convention.T
    full_proj_transform: torch.Tensor  # This is (P_dgr_style @ W2C_dgr_convention).T
    znear: float
    zfar: float


# Register with a new name or update if this is an in-place modification
@threestudio.register("generative-space-gsplat-renderer-v6") # Changed name
class GenerativeSpaceGsplatRendererV6(Rasterizer): # Changed name
    @dataclass
    class Config(Rasterizer.Config):
        near_plane: float = 0.1 # Used by get_cam_info_gaussian
        far_plane: float = 100  # Used by get_cam_info_gaussian

        normal_direction: str = "camera"

        rgb_grad_shrink: float = 1.0
        xyz_grad_shrink: float = 1.0
        opacity_grad_shrink: float = 1.0
        scale_grad_shrink: float = 1.0
        rotation_grad_shrink: float = 1.0


        with_eval3d: bool = False # Enable Eval3D mode
        with_ut: bool = False # Enable Uncertainty (UT) computation for main pass
        depth_render_mode: str = "D"  # Options: "D" (Accumulated), "ED" (Expected); Default: "D"
        rasterize_mode: str = "classic"  # Options: “classic” and “antialiased”; Default: "classic"

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        # Message might need update if gsplat supports material/background differently
        threestudio.info(
            "[Note] Gaussian Splatting (gsplat) interaction with material/background modules."
        )
        super().configure(geometry, material, background)
        # Instantiate Depth2Normal module
        self.normal_module = Depth2Normal()

    def _forward(
        self,
        pc: GaussianModel,
        batched_W2C_gsplat: Float[Tensor, "C 4 4"],
        batched_Ks_gsplat: Float[Tensor, "C 3 3"],
        batched_bg_color: Float[Tensor, "C 3"],
        H: int, W: int,
        batched_rays_o_world: Float[Tensor, "C H W 3"],
        batched_rays_d_world: Float[Tensor, "C H W 3"],
        scaling_modifier=1.,
        **kwargs
    ) -> Dict[str, Any]:
        
        # Prepare gsplat inputs
        means_gsplat = pc.xyz
        # gsplat expects rotations as quaternions (WXYZ), pc.rotation should be this.
        quats_gsplat = pc.rotation 
        # Apply scaling_modifier to scales if necessary, or gsplat might have its own global scale.
        # For now, assume pc.scale is ready. gsplat's rasterization takes `scales`.
        scales_gsplat = pc.scale * scaling_modifier
        
        opacities_gsplat = pc.opacity.squeeze(-1) # gsplat expects [N]
        colors_gsplat = pc.rgb # Assumes pc.rgb is [N, 3] or [N, SH_FEATURES*3]

        # Derive gsplat camera parameters from viewpoint_camera
        # viewpoint_camera.world_view_transform is W2C_dgr_convention.T
        # W2C_dgr_convention is an OpenCV-style W2C matrix (X right, Y down, Z in)
        # So, W2C_gsplat IS viewpoint_camera.world_view_transform.T 
        # W2C_gsplat = viewpoint_camera.world_view_transform.T 
        # viewmats_gsplat = W2C_gsplat.unsqueeze(0) # Old: Add batch dim: [1, 4, 4]
        viewmats_gsplat = batched_W2C_gsplat # New: Already batched [C, 4, 4]

        # H = int(viewpoint_camera.image_height) # Old
        # W = int(viewpoint_camera.image_width)  # Old
        
        # FoVx and FoVy are already in radians in the Camera NamedTuple if populated by get_cam_info_gaussian
        # However, get_cam_info_gaussian itself takes fovx, fovy as potentially scalar tensors.
        # The Camera tuple stores them. Let's assume they are radians.
        # tanfovy = torch.tan(viewpoint_camera.FoVy * 0.5) # Old
        # tanfovx = torch.tan(viewpoint_camera.FoVx * 0.5) # Old

        # K_matrix related calculations are now done in the main forward loop before calling _forward
        # and passed as batched_Ks_gsplat.
        # Ks_gsplat = K_matrix.unsqueeze(0) # Old: Add batch dim: [1, 3, 3]
        Ks_gsplat = batched_Ks_gsplat # New: Already batched [C, 3, 3]

        # backgrounds_gsplat = bg_color_tensor.unsqueeze(0) # Old: Shape [1, C]
        backgrounds_gsplat = batched_bg_color # New: Shape [C, NumChannels]

        # === Debug Prints Before gsplat.rasterization (REMOVING) ===
        # print("\n--- Debug Info: Inputs to gsplat.rasterization ---")
        # print(f"  means_gsplat: shape={means_gsplat.shape}, dtype={means_gsplat.dtype}")
        # if means_gsplat.numel() > 0:
        #     print(f"    means_gsplat stats: min={torch.min(means_gsplat).item() if means_gsplat.numel() > 0 else 'N/A'}, max={torch.max(means_gsplat).item() if means_gsplat.numel() > 0 else 'N/A'}, mean={torch.mean(means_gsplat).item() if means_gsplat.numel() > 0 else 'N/A'}, isfinite={torch.all(torch.isfinite(means_gsplat)).item() if means_gsplat.numel() > 0 else 'N/A'}")
        # print(f"  quats_gsplat: shape={quats_gsplat.shape}, dtype={quats_gsplat.dtype}")
        # print(f"  scales_gsplat: shape={scales_gsplat.shape}, dtype={scales_gsplat.dtype}")
        # print(f"  opacities_gsplat: shape={opacities_gsplat.shape}, dtype={opacities_gsplat.dtype}")
        # print(f"  colors_gsplat: shape={colors_gsplat.shape}, dtype={colors_gsplat.dtype}")
        # if colors_gsplat.numel() > 0:
        #     print(f"    colors_gsplat stats: min={torch.min(colors_gsplat).item() if colors_gsplat.numel() > 0 else 'N/A'}, max={torch.max(colors_gsplat).item() if colors_gsplat.numel() > 0 else 'N/A'}, mean={torch.mean(colors_gsplat).item() if colors_gsplat.numel() > 0 else 'N/A'}, isfinite={torch.all(torch.isfinite(colors_gsplat)).item() if colors_gsplat.numel() > 0 else 'N/A'}")
        # print(f"  viewmats_gsplat: shape={viewmats_gsplat.shape}, dtype={viewmats_gsplat.dtype}")
        # print(f"  Ks_gsplat: shape={Ks_gsplat.shape}, dtype={Ks_gsplat.dtype}")
        # print(f"  width (W): {W}, height (H): {H}")
        # sh_degree_to_pass_to_gsplat = None
        # print(f"  pc.sh_degree attribute value (source): {pc.sh_degree if hasattr(pc, 'sh_degree') else 'Not Set'}")
        # print(f"  sh_degree (actually passed to gsplat.rasterization): {sh_degree_to_pass_to_gsplat}")
        # if hasattr(pc, 'sh_degree'):
        #     print(f"  pc.sh_degree attribute value: {pc.sh_degree}")
        # else:
        #     print("  pc object does not have sh_degree attribute.")
        # print(f"  backgrounds_gsplat: shape={backgrounds_gsplat.shape}, dtype={backgrounds_gsplat.dtype}")
        # === End Debug Prints ===
        
        sh_degree_to_pass_to_gsplat = None # Retain this line for clarity

        # gsplat rasterization for the main pass
        output_gsplat, rendered_alpha_gsplat, meta = gsplat.rasterization(
            means=means_gsplat,
            quats=quats_gsplat,
            scales=scales_gsplat,
            opacities=opacities_gsplat,
            colors=colors_gsplat, 
            viewmats=viewmats_gsplat,
            Ks=Ks_gsplat,
            width=W,
            height=H,
            render_mode=f"RGB+{self.cfg.depth_render_mode}", # Dynamically set based on config
            backgrounds=backgrounds_gsplat,
            sh_degree=sh_degree_to_pass_to_gsplat,
            with_eval3d=self.cfg.with_eval3d,
            with_ut=self.cfg.with_ut,
            packed=not (self.cfg.with_eval3d or self.cfg.with_ut),
            rasterize_mode=self.cfg.rasterize_mode,
        )

        # === Debug: Print meta dictionary keys (Removed as depth is not in meta for RGB+D) ===
        # print("--- Debug Info: Keys in meta dictionary from gsplat.rasterization ---")
        # if meta is not None and isinstance(meta, dict):
        #     print(f"  meta.keys(): {meta.keys()}")
        # else:
        #     print(f"  meta object is None or not a dictionary. meta: {meta}")
        # print("--- End Debug Info ---")
        # === End Debug ===

        # Process outputs: output_gsplat is [C, H, W, D+1] (e.g., [C, H, W, 4] if D=3 for RGB)
        # rendered_alpha_gsplat is [C, H, W, 1]
        # We take [0] before, now this will be a batch of C images/maps.
        rendered_image_batch = output_gsplat[:, :, :, :3] # Extract RGB: [C, H, W, 3]
        rendered_depth_batch = output_gsplat[:, :, :, 3:4] # Extract Depth: [C, H, W, 1]
        rendered_alpha_batch = rendered_alpha_gsplat   # [C, H, W, 1]

        # Normals: gsplat does not directly return normals.
        # Calculate normals from depth using Depth2Normal module.
        # CURRENT NORMAL LOGIC IS FOR SINGLE IMAGE - NEEDS BATCHING LATER
        # For now, let's compute normal for the first image in the batch as a placeholder
        # or return zeros for the batch.
        num_cameras_in_batch = viewmats_gsplat.shape[0]
        
        # --- Fully batched normal calculation using world-space XYZ coordinates ---
        # rendered_depth_batch is [C, H, W, 1]
        # Ks_gsplat is [C, 3, 3], W2C_gsplat (viewmats_gsplat) is [C, 4, 4]

        # Steps 1-4 (intrinsics extraction, get_ray_directions, C2W, get_rays) are REMOVED
        # as batched_rays_o_world and batched_rays_d_world are now passed in.

        # 1. Get intrinsics per camera (REMOVED)
        # fx_batch = Ks_gsplat[:, 0, 0]
        # ...
        # 2. Create camera-space ray directions (REMOVED)
        # camera_ray_directions_list = []
        # ...
        # camera_ray_directions_batch = torch.stack(camera_ray_directions_list, dim=0)

        # 3. Get C2W matrices (REMOVED)
        # batched_C2W_gsplat = torch.inverse(viewmats_gsplat)

        # 4. Transform to world rays (REMOVED)
        # internal_rays_o_world_batch, internal_rays_d_world_batch = get_rays(...)
        
        # 5. Calculate world-space XYZ coordinates using passed-in rays
        xyz_map_world_batch = batched_rays_o_world + rendered_depth_batch * batched_rays_d_world # [C, H, W, 3]

        # 6. Prepare input for normal_module by permuting to [C, 3, H, W]
        normal_module_input_batch = xyz_map_world_batch.permute(0, 3, 1, 2) # [C, 3, H, W]

        # 7. Call normal_module
        # Output of normal_module is [C, 3, H, W] (cross product is on dim=1 which is channel)
        output_normals_intermediate_batch = self.normal_module(normal_module_input_batch)

        # 8. Post-process: permute back to [C, H, W, 3] and normalize
        normal_map_batch = output_normals_intermediate_batch.permute(0, 2, 3, 1) # [C, H, W, 3]
        normal_map_batch = F.normalize(normal_map_batch, p=2, dim=-1)
        # --- End Fully Batched Normal Calculation ---

        # Debug: Print shapes for normal calculation (REMOVING)
        # if self.cfg.get("print_debug_info", False):
        #     print(f"  normal_map_batch shape: {normal_map_batch.shape}")

        # Center Point Depth/Opacity: Implemented by a second gsplat pass
        # CURRENT CENTER POINT LOGIC IS FOR SINGLE IMAGE - NEEDS BATCHING LATER
        # For now, compute for the first image in batch or return zeros.
        if num_cameras_in_batch > 0:
            with torch.no_grad():
                # For center point depth, we want to render each Gaussian as a tiny, fully opaque point.
                num_points = pc.xyz.shape[0]
                
                epsilon_scale = 1e-3 # Use small epsilon scale
                scales_for_depth_pass = torch.full((num_points, 3), epsilon_scale, device=pc.xyz.device, dtype=torch.float32)
                
                # Opacities should be fully opaque and have shape [N]
                opacities_input_for_depth_pass = torch.ones(num_points, device=pc.xyz.device, dtype=torch.float32)
                
                # Colors are dummy white [N, 3]
                colors_for_depth_pass_rgb = torch.ones(num_points, 3, device=pc.xyz.device, dtype=torch.float32)

                # Background for RGB+D pass is 3-channel zeros, as gsplat expects [C,3] with RGB inputs
                background_rgbd_depth_pass = torch.zeros(num_cameras_in_batch, 3, device=pc.xyz.device, dtype=torch.float32)

                # Using RGB+D mode for the second pass test
                output_center_rgbd, alpha_center_rgbd, _ = gsplat.rasterization(
                    means=pc.xyz,
                    quats=pc.rotation,
                    scales=scales_for_depth_pass, # Actual scales
                    opacities=opacities_input_for_depth_pass, # Actual opacities
                    colors=colors_for_depth_pass_rgb, # Actual colors
                    sh_degree=None, # TEST: Explicitly pass None for sh_degree
                    viewmats=viewmats_gsplat,
                    Ks=Ks_gsplat,
                    width=W,
                    height=H,
                    render_mode='RGB+D', 
                    backgrounds=background_rgbd_depth_pass 
                )

                # output_center_rgbd is [C, H, W, 4]
                center_point_depth_map_batch = output_center_rgbd[..., 3:4] # Extract Depth: [C, H, W, 1]
                center_point_opacity_map_batch = alpha_center_rgbd # Use alpha from RGB+D output [C, H, W, 1]
        else:
            center_point_depth_map_batch = torch.zeros_like(rendered_depth_batch)
            center_point_opacity_map_batch = torch.zeros_like(rendered_alpha_batch)
        # --- End Temporary Center Point Calculation ---

        outputs = {
            "comp_rgb": rendered_image_batch,    # [C, H, W, 3]
            "depth": rendered_depth_batch,        # [C, H, W, 1]
            "opacity": rendered_alpha_batch,      # [C, H, W, 1]
            "normal": normal_map_batch,         # [C, H, W, 3]
            "center_point_depth": center_point_depth_map_batch, # [C, H, W, 1]
            "center_point_opacity": center_point_opacity_map_batch, # [C, H, W, 1]
        }
        
        return outputs

    def _space_cache_to_pc(
        self,
        space_cache: Dict[str, Union[Float[Tensor, "B ..."], List[Float[Tensor, "B ..."]]]],
    ):
        pc_list = []
        batch_size_space_cache = space_cache["position"].shape[0]
        for i in range(batch_size_space_cache):
            xyz = space_cache["position"][i] 
            xyz = self.cfg.xyz_grad_shrink * xyz + (1 - self.cfg.xyz_grad_shrink) * xyz.detach()
            
            rgb_features = space_cache["color"][i]   
            rgb_features = self.cfg.rgb_grad_shrink * rgb_features + (1 - self.cfg.rgb_grad_shrink) * rgb_features.detach()
            
            current_rgb_for_pc = rgb_features
            current_sh_degree = 0 

            scale = space_cache["scale"][i] 
            scale = self.cfg.scale_grad_shrink * scale + (1 - self.cfg.scale_grad_shrink) * scale.detach()
            
            rotation = space_cache["rotation"][i] 
            rotation = self.cfg.rotation_grad_shrink * rotation + (1 - self.cfg.rotation_grad_shrink) * rotation.detach()
            
            opacity = space_cache["opacity"][i] 
            opacity = self.cfg.opacity_grad_shrink * opacity + (1 - self.cfg.opacity_grad_shrink) * opacity.detach()
            
            gaussians = GaussianModel().set_data( 
                    xyz=xyz,
                rgb=current_rgb_for_pc, # Use potentially material-processed features
                    scale=scale,
                    rotation=rotation,
                    opacity=opacity,
                )
            gaussians.sh_degree = current_sh_degree 
            pc_list.append(gaussians)
        return pc_list

    def forward(
        self, 
        c2w: Float[Tensor, "B 4 4"],
        fovy: Float[Tensor, "B"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"], # Not used by gsplat directly in _forward
        rays_o_rasterize: Float[Tensor, "B H W 3"], # Not used by gsplat _forward
        rays_d_rasterize: Float[Tensor, "B H W 3"], # Not used by gsplat _forward
        space_cache: Dict[str, Union[Float[Tensor, "B ..."], List[Float[Tensor, "B ..."]]]],
        fovx: Optional[Float[Tensor, "B"]] = None,
        camera_distances: Optional[Float[Tensor, "B"]] = None,
        text_embed: Optional[Float[Tensor, "B C"]] = None,
        text_embed_bg: Optional[Float[Tensor, "B C"]] = None,
        **kwargs 
    ):
        batch_size = c2w.shape[0]
        # Corrected assert statement to check for valid space_cache['position']
        assert ("position" in space_cache and
                hasattr(space_cache["position"], 'shape') and
                isinstance(space_cache["position"], torch.Tensor) and
                space_cache["position"].ndim > 0 and
                space_cache["position"].shape[0] > 0), \
               "space_cache['position'] is invalid or empty, leading to zero batch_size_space_cache."
        batch_size_space_cache = space_cache["position"].shape[0]

        num_views_per_batch = batch_size // batch_size_space_cache
        # Ensure width and height are integers
        width = int(rays_d_rasterize.shape[2]) 
        height = int(rays_d_rasterize.shape[1])

        pc_list = self._space_cache_to_pc(space_cache)

        if hasattr(self, 'background') and self.background is not None:
            bg_text_embed = text_embed_bg if text_embed_bg is not None else text_embed
            # Ensure dirs has the correct shape [B, H, W, 3]
            dirs_for_bg = rays_d_rasterize.view(batch_size, height, width, 3)
            if hasattr(self.background, 'enabling_hypernet') and self.background.enabling_hypernet:
                 comp_rgb_bg_all = self.background(dirs=dirs_for_bg, text_embed=bg_text_embed)
            else:
                 comp_rgb_bg_all = self.background(dirs=dirs_for_bg)
            comp_rgb_bg_all = comp_rgb_bg_all.view(batch_size, height, width, -1) # Ensure [B,H,W,C]
        else:
            threestudio.warn("No background module provided or configured. Using default black background.")
            comp_rgb_bg_all = torch.zeros(batch_size, height, width, 3, device=c2w.device)


        all_comp_rgb = []
        all_normals = []
        all_depths = []
        all_masks = []
        all_center_point_depths = []
        all_center_point_masks = []

        for pc_index, pc in enumerate(pc_list):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # ----- Batch Camera Parameter Collection for this pc_index -----
            num_views_for_this_pc = num_views_per_batch # C: number of cameras for this pc
            start_idx_in_batch = pc_index * num_views_per_batch
            end_idx_in_batch = (pc_index + 1) * num_views_per_batch

            collected_W2C_gsplat_list = []
            collected_Ks_gsplat_list = []
            collected_bg_color_list = []
            # We also need H and W, assuming they are constant for the batch
            # height and width are already defined outside this loop

            for i in range(start_idx_in_batch, end_idx_in_batch):
                current_fovx_scalar = fovx[i] if fovx is not None else fovy[i]
                current_fovy_scalar = fovy[i]

                # get_cam_info_gaussian returns DGR-style matrices
                w2c_dgr_t, _, _ = get_cam_info_gaussian(
                    c2w=c2w[i],
                    fovx=current_fovx_scalar, 
                    fovy=current_fovy_scalar,
                    znear=self.cfg.near_plane,
                    zfar=self.cfg.far_plane
                )
                # Convert DGR W2C to gsplat OpenCV W2C
                # w2c_dgr_t is W2C_dgr_convention.T
                # W2C_gsplat = W2C_dgr_convention = w2c_dgr_t.T
                W2C_gsplat_single = w2c_dgr_t.T 
                collected_W2C_gsplat_list.append(W2C_gsplat_single)

                # Calculate K_matrix for gsplat
                tanfovy_s = torch.tan(current_fovy_scalar * 0.5)
                tanfovx_s = torch.tan(current_fovx_scalar * 0.5)
                epsilon = 1e-6
                if torch.abs(tanfovx_s) < epsilon: tanfovx_s = epsilon * torch.sign(tanfovx_s) if torch.sign(tanfovx_s) !=0 else epsilon
                if torch.abs(tanfovy_s) < epsilon: tanfovy_s = epsilon * torch.sign(tanfovy_s) if torch.sign(tanfovy_s) !=0 else epsilon
                fy_s = height / (2 * tanfovy_s)
                fx_s = width / (2 * tanfovx_s)
                cx_s = width / 2.0
                cy_s = height / 2.0
                K_matrix_single = torch.tensor(
                    [[fx_s, 0, cx_s], 
                     [0, fy_s, cy_s], 
                     [0, 0, 1]], 
                    device=pc.xyz.device, 
                    dtype=torch.float32
                )
                collected_Ks_gsplat_list.append(K_matrix_single)
                
                # Background color for this view
                # comp_rgb_bg_all is [TotalBatch, H, W, 3]
                # We need [NumColorChannels] for each camera in the batch for _forward
                current_bg_color_single = comp_rgb_bg_all[i, 0, 0, :3] 
                collected_bg_color_list.append(current_bg_color_single)
            
            batched_W2C_gsplat_tensor = torch.stack(collected_W2C_gsplat_list, dim=0) # [C, 4, 4]
            batched_Ks_gsplat_tensor = torch.stack(collected_Ks_gsplat_list, dim=0)   # [C, 3, 3]
            batched_bg_color_tensor = torch.stack(collected_bg_color_list, dim=0)     # [C, 3]
            # ----- End Batch Camera Parameter Collection -----

            with autocast(enabled=False):
                render_pkg_batch = self._forward(
                    pc=pc,
                    batched_W2C_gsplat=batched_W2C_gsplat_tensor,
                    batched_Ks_gsplat=batched_Ks_gsplat_tensor,
                    batched_bg_color=batched_bg_color_tensor,
                    H=height,
                    W=width,
                    batched_rays_o_world=rays_o_rasterize[start_idx_in_batch:end_idx_in_batch, :, :, :],
                    batched_rays_d_world=rays_d_rasterize[start_idx_in_batch:end_idx_in_batch, :, :, :],
                    scaling_modifier=kwargs.get('scaling_modifier', 1.0),
                    **kwargs
                )
            
            # render_pkg_batch contains tensors of shape [C, H, W, Channels]
            all_comp_rgb.append(render_pkg_batch["comp_rgb"]) # List of [C,H,W,3]
            all_depths.append(render_pkg_batch["depth"])       # List of [C,H,W,1]
            all_masks.append(render_pkg_batch["opacity"])    # List of [C,H,W,1]
            all_normals.append(render_pkg_batch["normal"])    # List of [C,H,W,3]
            all_center_point_depths.append(render_pkg_batch["center_point_depth"])
            all_center_point_masks.append(render_pkg_batch["center_point_opacity"])

        # Concatenate results from all pc_list items along the batch dimension (dim 0)
        # Each item in all_comp_rgb is [C, H, W, 3], where C is num_views_per_batch
        # So, concatenating them along dim 0 results in [TotalBatch, H, W, 3]
        comp_rgb = torch.cat(all_comp_rgb, dim=0) 
        opacity = torch.cat(all_masks, dim=0)    
        depth = torch.cat(all_depths, dim=0)      
        comp_normal_rendered = torch.cat(all_normals, dim=0) 
        
        comp_center_point_depth = torch.cat(all_center_point_depths, dim=0) \
            if all_center_point_depths and all_center_point_depths[0] is not None else torch.zeros_like(depth)
        comp_center_point_opacity = torch.cat(all_center_point_masks, dim=0) \
            if all_center_point_masks and all_center_point_masks[0] is not None else torch.zeros_like(opacity)

        comp_rgb = self.material(features=comp_rgb)

        outputs = {
            "comp_rgb": comp_rgb,
            "opacity": opacity,
            "depth": depth,
            "comp_center_point_depth": comp_center_point_depth,
            "comp_center_point_opacity": comp_center_point_opacity,
        }

        if comp_normal_rendered.numel() > 0 : # Check if normal tensor is not empty
            comp_normal_rendered = F.normalize(comp_normal_rendered, dim=-1, p=2)
        outputs["comp_normal"] = comp_normal_rendered 

        if self.cfg.normal_direction == "camera":
            bg_normal_val = 0.5
            bg_normal = torch.full_like(comp_normal_rendered, bg_normal_val, device=comp_normal_rendered.device)
            if comp_normal_rendered.shape[-1] == 3 : bg_normal[..., 2] = 1.0 

            processed_normal = comp_normal_rendered * -1

            bg_normal_white = torch.ones_like(processed_normal, device=processed_normal.device)
            comp_normal_vis_white = (processed_normal + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
            comp_normal_vis = (processed_normal + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal
            outputs["comp_normal_cam_vis_white"] = comp_normal_vis_white
            outputs["comp_normal_cam_vis"] = comp_normal_vis

        elif self.cfg.normal_direction == "world":
            bg_normal_white = torch.ones_like(comp_normal_rendered, device=comp_normal_rendered.device)
            comp_normal_rendered_vis_white = (comp_normal_rendered + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
            outputs["comp_normal_cam_vis_white"] = comp_normal_rendered_vis_white
        elif self.cfg.normal_direction == "front":
             bg_normal_white = torch.ones_like(comp_normal_rendered, device=comp_normal_rendered.device)
             comp_normal_rendered_vis_white = (comp_normal_rendered + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
             outputs["comp_normal_cam_vis_white"] = comp_normal_rendered_vis_white
        else:
            raise ValueError(f"Unknown normal direction: {self.cfg.normal_direction}")

        if camera_distances is not None:
            cam_dist_view = camera_distances.view(-1, 1, 1, 1)
            range_offset = torch.sqrt(torch.tensor(3.0, device=cam_dist_view.device))
            
            effective_far = cam_dist_view + range_offset
            effective_near = torch.clamp(cam_dist_view - range_offset, min=1e-5)

            depth_blend = depth * opacity + (1.0 - opacity) * effective_far 
            disparity_norm = (effective_far - depth_blend) / (effective_far - effective_near + 1e-7)
            disparity_norm = torch.clamp(disparity_norm, 0.0, 1.0)
            outputs["disparity"] = disparity_norm
        else:
            outputs["disparity"] = torch.zeros_like(depth)

        return outputs
