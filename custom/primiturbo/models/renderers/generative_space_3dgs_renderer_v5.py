import math
from dataclasses import dataclass

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *


import torch
from threestudio.utils.ops import get_cam_info_gaussian, get_projection_matrix_gaussian
from torch.cuda.amp import autocast

from .gaussian_utils import GaussianModel
from center_depth_rasterization import rasterize_gaussians_center_depth




class Camera(NamedTuple):
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    znear: float
    zfar: float



@threestudio.register("generative-space-3dgs-rasterize-renderer-v5")
class GenerativeSpace3dgsRasterizeRendererV5(Rasterizer):
    @dataclass
    class Config(Rasterizer.Config):
        near_plane: float = 0.1
        far_plane: float = 100

        # for rendering the normal
        normal_direction: str =  "camera"  # "front" or "camera" or "world"

        rgb_grad_shrink: float = 1.0
        xyz_grad_shrink: float = 1.0
        opacity_grad_shrink: float = 1.0
        scale_grad_shrink: float = 1.0
        rotation_grad_shrink: float = 1.0

        covariance_forcing: bool = False


    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        threestudio.info(
            "[Note] Gaussian Splatting doesn't support material and background now."
        )
        super().configure(geometry, material, background)


    def _forward(
        self,
        viewpoint_camera,
        pc,
        bg_color: torch.Tensor,
        light_positions: Float[Tensor, "1 3"],
        rays_o_rasterize: Float[Tensor, "H W 3"],
        rays_d_rasterize: Float[Tensor, "H W 3"],
        original_w2c: torch.Tensor,
        scaling_modifier=1.,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Render the scene foreground using Gaussian Splatting for a single view, applying material.
        Background tensor (bg_color) is not used for compositing here; compositing happens in the main forward.
        """

        # kwargs.keys() = dict_keys([
        # 'prompt', 'guidance_utils', 'condition_utils',
        # 'rays_o', 'rays_d', 'mvp_mtx', 'camera_positions',
        # 'c2w', 'light_positions', 'elevation', 'azimuth', 'camera_distances',
        # 'camera_distances_relative', 'height', 'width', 'fovy', 'rays_d_rasterize',
        # 'noise', 'text_embed_bg', 'text_embed', 'space_cache'])

        # Original invert_bg_prob logic is removed as we composite later
        # Use a fixed black background for foreground rendering
        # bg_color_tensor = torch.zeros(3, dtype=torch.float32, device=pc.xyz.device)
        # Use the provided bg_color for rasterization edge cases
        bg_color_tensor = bg_color.to(pc.xyz.device)

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color_tensor, # Render foreground against provided bg (often black)
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=0, # Set explicitly to 0 as features/colors are precomputed by material
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            # Add missing arguments
            kernel_size=0.,
            require_depth=True, # Likely needed as we use rendered_depth
            require_coord=False # Match v2, likely not needed directly
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings).to(pc.xyz.device)

        # Call standard rasterizer for main outputs
        # Original v5 unpacking (4 values):
        # rendered_image, radii, rendered_depth, rendered_alpha  = rasterizer(...)
        # Modified unpacking to match v2 (8 values, expecting direct normal):
        rendered_image, _, _, _, rendered_depth, _, rendered_alpha, rendered_normal_direct  = rasterizer(
            means3D=pc.xyz,
            means2D=torch.zeros_like(pc.xyz, dtype=torch.float32, device=pc.xyz.device),
            shs=None,
            colors_precomp=pc.rgb,
            opacities=pc.opacity,
            scales=pc.scale,
            rotations=pc.rotation,
            cov3D_precomp=None,
        )

        # --- Normal Calculation (using standard depth) ---
        # The following lines will likely cause an error now because:
        # 1. self.normal_module was removed in a previous step.
        # 2. We are now getting rendered_normal_direct from rasterizer.
        # We should decide whether to use rendered_normal_direct or restore normal_module.
        # For now, I will leave the calculation code as is, but it needs attention.
        # _, H, W = rendered_depth.shape
        # xyz_map = rays_o_rasterize + rendered_depth.permute(1, 2, 0) * rays_d_rasterize
        # The next line will error if self.normal_module is not defined
        # normal_map = self.normal_module(xyz_map.permute(2, 0, 1).unsqueeze(0))[0]
        # normal_map = F.normalize(normal_map, dim=0)
        # Let's use the directly unpacked normal for now, assuming it's valid:
        # Ensure rendered_normal_direct is in the correct shape [C, H, W] before normalize
        if rendered_normal_direct.ndim == 4 and rendered_normal_direct.shape[0] == 1: # Output is likely [1, C, H, W]
            rendered_normal_direct = rendered_normal_direct.squeeze(0)
        elif rendered_normal_direct.ndim != 3:
             raise ValueError(f"Unexpected shape for direct normal from rasterizer: {rendered_normal_direct.shape}")
        normal_map = F.normalize(rendered_normal_direct, dim=0) # Use the direct normal


        # --- Custom Center Point Rasterization (Second Call - using custom operator) ---

        # Call the custom center depth rasterizer instead
        center_point_opacity_map_raw, center_point_depth_map_raw = rasterize_gaussians_center_depth(
            pc.xyz,
            viewpoint_camera.world_view_transform,
            viewpoint_camera.full_proj_transform,
            original_w2c,
            tanfovx,
            tanfovy,
            int(viewpoint_camera.image_height),
            int(viewpoint_camera.image_width),
            viewpoint_camera.znear,
            viewpoint_camera.zfar,
            1.0,
            0.0,
            False,
            True
        )

        # Note: Custom operator returns (opacity[H,W], depth[H,W]). Depth is positive.
        # Add channel dimension to match expected output shape [1, H, W]
        center_point_depth_map = center_point_depth_map_raw.unsqueeze(0)
        center_point_opacity_map = center_point_opacity_map_raw.unsqueeze(0)


        return {
            "comp_rgb": rendered_image, 
            "depth": rendered_depth,     
            "opacity": rendered_alpha,   
            "normal": normal_map,      
            "center_point_depth": center_point_depth_map, # From custom rasterizer
            "center_point_opacity": center_point_opacity_map, # From custom rasterizer
        }
    

    def _space_cache_to_pc(
        self,
        space_cache: Dict[str, Union[Float[Tensor, "B ..."], List[Float[Tensor, "B ..."]]]],
    ):
        """Converts a batch of space cache data into a list of GaussianModel objects."""
        pc_list = []
        # Assuming space_cache keys like 'position', 'rgb', 'scale', etc. hold tensors of shape [CacheBatchSize, NumGaussians, Dim]
        batch_size_space_cache = space_cache["position"].shape[0]
        for i in range(batch_size_space_cache):
            # === Get tensors for current cache element ===
            xyz = space_cache["position"][i] 
            xyz = self.cfg.xyz_grad_shrink * xyz + (1 - self.cfg.xyz_grad_shrink) * xyz.detach()
            
            # rgb = space_cache["color"][i] 
            rgb = space_cache["color"][i]   
            rgb = self.cfg.rgb_grad_shrink * rgb + (1 - self.cfg.rgb_grad_shrink) * rgb.detach()
            rgb = self.material( # TODO: for other material, we need to change this
                features=rgb,
            )

            scale = space_cache["scale"][i] 
            scale = self.cfg.scale_grad_shrink * scale + (1 - self.cfg.scale_grad_shrink) * scale.detach()
            
            
            rotation = space_cache["rotation"][i] 
            rotation = self.cfg.rotation_grad_shrink * rotation + (1 - self.cfg.rotation_grad_shrink) * rotation.detach()
            
            opacity = space_cache["opacity"][i] 
            opacity = self.cfg.opacity_grad_shrink * opacity + (1 - self.cfg.opacity_grad_shrink) * opacity.detach()
            
            pc_list.append(
                GaussianModel().set_data( 
                    xyz=xyz,
                    rgb=rgb,
                    scale=scale,
                    rotation=rotation,
                    opacity=opacity,
                )
            )
        return pc_list

    def forward(
        self, 
        c2w: Float[Tensor, "B 4 4"],
        fovy: Float[Tensor, "B"],
        # Pass light positions explicitly
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        rays_o_rasterize: Float[Tensor, "B H W 3"],
        rays_d_rasterize: Float[Tensor, "B H W 3"],
        space_cache: Dict[str, Union[Float[Tensor, "B ..."], List[Float[Tensor, "B ..."]]]],
        fovx: Optional[Float[Tensor, "B"]] = None,
        camera_distances: Optional[Float[Tensor, "B"]] = None,
        text_embed: Optional[Float[Tensor, "B C"]] = None,
        text_embed_bg: Optional[Float[Tensor, "B C"]] = None,
        **kwargs # Catch unused arguments
    ):
        # Removed space_cache structure verification

        batch_size = c2w.shape[0]
        if "position" in space_cache and hasattr(space_cache["position"], 'shape'):
             batch_size_space_cache = space_cache["position"].shape[0]
        else:
             print("[WARNING] Could not determine batch_size_space_cache from space_cache['position']. Falling back.")
             batch_size_space_cache = 0 

        if batch_size_space_cache == 0:
            print("[ERROR] batch_size_space_cache determined to be 0. Setting num_views_per_batch to 0.")
            num_views_per_batch = 0
        else:
            num_views_per_batch = batch_size // batch_size_space_cache

        width, height = rays_d_rasterize.shape[1:3]

        pc_list = self._space_cache_to_pc(space_cache)


        # --- Background Computation ---        
        if hasattr(self, 'background') and self.background is not None:
            # Determine which embedding to use for the background
            bg_text_embed = text_embed_bg if text_embed_bg is not None else text_embed

            # Check if background network requires specific inputs (adapt as needed)
            if hasattr(self.background, 'enabling_hypernet') and self.background.enabling_hypernet:
                 comp_rgb_bg = self.background(dirs=rays_d_rasterize.view(batch_size, height, width, 3), text_embed=bg_text_embed)
            else:
                 comp_rgb_bg = self.background(dirs=rays_d_rasterize.view(batch_size, height, width, 3))
            comp_rgb_bg = comp_rgb_bg.view(batch_size, height, width, -1) # Ensure correct shape [B, H, W, C]
        else:
            raise ValueError("No background module provided or configured.")


        comp_rgb = []
        normals = []
        depths = []
        masks = []
        xyz_maps = []
        center_point_depths = [] # ADDED
        center_point_masks = [] # ADDED

        w2cs = []
        projs = []
        cam_ps = []
        # 在批处理循环中添加
        for pc_index, pc in enumerate(pc_list):
            # 释放其他点云的内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 只处理当前点云
            for batch_idx in range(pc_index * num_views_per_batch, (pc_index + 1) * num_views_per_batch):
                # Ensure ray directions are normalized (do it once per batch item)
                current_rays_d = F.normalize(rays_d_rasterize[batch_idx], dim=-1)

                w2c, proj, cam_p = get_cam_info_gaussian(
                    c2w=c2w[batch_idx],
                    fovx=fovx[batch_idx] if fovx is not None else fovy[batch_idx], # Use calculated or provided fovx
                    fovy=fovy[batch_idx],
                    znear=self.cfg.near_plane, # Use config value
                    zfar=self.cfg.far_plane   # Use config value
                )

                # Need the original W2C matrix (before transpose) for correct Z calculation in CUDA
                # original_w2c = w2c.T.contiguous() # This was wrong T.T
                # Ensure w2c is C-contiguous (Row-Major)
                w2c_contiguous = w2c.contiguous()

                viewpoint_cam = Camera(
                    FoVx=fovx[batch_idx] if fovx is not None else fovy[batch_idx],
                    FoVy=fovy[batch_idx],
                    image_width=width,
                    image_height=height,
                    world_view_transform=w2c.T,
                    full_proj_transform=proj,
                    camera_center=cam_p,
                    znear=self.cfg.near_plane,
                    zfar=self.cfg.far_plane
                )

                with autocast(enabled=False):
                    render_pkg = self._forward(
                        viewpoint_cam,
                        pc,
                        comp_rgb_bg[batch_idx],
                        light_positions=light_positions[batch_idx].unsqueeze(0),
                        rays_o_rasterize=rays_o_rasterize[batch_idx],
                        rays_d_rasterize=current_rays_d,
                        # Pass the contiguous W2C matrix needed by the custom kernel
                        original_w2c=w2c_contiguous,
                        **kwargs
                    )
                    # 立即释放渲染结果的内存
                    with torch.cuda.stream(torch.cuda.Stream()):
                        # 处理渲染结果 (now collecting comp_rgb)
                        if "comp_rgb" in render_pkg:
                            comp_rgb.append(render_pkg["comp_rgb"])
                        if "depth" in render_pkg:
                            depths.append(render_pkg["depth"])
                        if "opacity" in render_pkg:
                            masks.append(render_pkg["opacity"])
                        if "normal" in render_pkg:
                            normals.append(render_pkg["normal"])
                        if "xyz_map" in render_pkg:
                            xyz_maps.append(render_pkg["xyz_map"])
                        if "center_point_depth" in render_pkg: # ADDED
                            center_point_depths.append(render_pkg["center_point_depth"])
                        if "center_point_opacity" in render_pkg: # ADDED
                            center_point_masks.append(render_pkg["center_point_opacity"])


                w2cs.append(w2c)
                projs.append(proj)
                cam_ps.append(cam_p)


        # === Stack foreground results and permute ===
        comp_rgb = torch.stack(comp_rgb, dim=0).permute(0, 2, 3, 1)
        opacity = torch.stack(masks, dim=0).permute(0, 2, 3, 1)          # [B, H, W, 1]
        depth = torch.stack(depths, dim=0).permute(0, 2, 3, 1)            # [B, H, W, 1]
        comp_normal_rendered = torch.stack(normals, dim=0).permute(0, 2, 3, 1)
        # Stack center point depths
        comp_center_point_depth = torch.stack(center_point_depths, dim=0).permute(0, 2, 3, 1) if center_point_depths else None # Shape: [B, H, W, 1]
        comp_center_point_opacity = torch.stack(center_point_masks, dim=0).permute(0, 2, 3, 1) if center_point_masks else None # Shape: [B, H, W, 1]
        if comp_center_point_depth is None:
            comp_center_point_depth = torch.zeros_like(depth) # Fallback if empty

        # --- Prepare Output Dictionary --- 
        outputs = {
            "comp_rgb": comp_rgb,           # Standard composite color
            "opacity": opacity,
            "depth": depth,                 # Standard rendered surface depth
            "comp_center_point_depth": comp_center_point_depth, # From custom rasterizer
            "comp_center_point_opacity": comp_center_point_opacity, # From custom rasterizer
        }

        # --- Normal Processing --- 
        comp_normal_rendered = F.normalize(comp_normal_rendered, dim=-1) # Normalize world normal
        outputs["comp_normal"] = comp_normal_rendered # Keep world normal

        # Apply post-processing based on normal_direction config, mirroring v2 logic
        if self.cfg.normal_direction == "camera":
            # Visualize world-space normal blended with white background (mimicking v2)
            bg_normal_val = 0.5
            bg_normal = torch.full_like(comp_normal_rendered, bg_normal_val)
            bg_normal[..., 2] = 1.0 # Blue background Z

            # Apply negation like in v2 (for RaDe-GS compatibility?)
            processed_normal = comp_normal_rendered * -1

            bg_normal_white = torch.ones_like(processed_normal)
            # Blend with white background
            comp_normal_vis_white = (processed_normal + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
            # Blend with blue background
            comp_normal_vis = (processed_normal + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal
            # Update outputs dictionary
            outputs["comp_normal_cam_vis_white"] = comp_normal_vis_white
            outputs["comp_normal_cam_vis"] = comp_normal_vis # Add the blue-background version too

        elif self.cfg.normal_direction == "world":
            # Keep world normal visualization consistent (optional, could mimic v2 error)
            bg_normal_white = torch.ones_like(comp_normal_rendered)
            comp_normal_rendered_vis_white = (comp_normal_rendered + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
            outputs["comp_normal_cam_vis_white"] = comp_normal_rendered_vis_white # Use compatible key
            # Optionally raise error like v2:
            # raise NotImplementedError("Normal direction 'world' is not implemented yet.")

        elif self.cfg.normal_direction == "front":
            # Keep front normal visualization consistent (optional, could mimic v2 error)
             bg_normal_white = torch.ones_like(comp_normal_rendered)
             comp_normal_rendered_vis_white = (comp_normal_rendered + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
             outputs["comp_normal_cam_vis_white"] = comp_normal_rendered_vis_white # Use compatible key
            # Optionally raise error like v2:
            # raise NotImplementedError("Normal direction 'front' is not implemented yet.")
        else:
            raise ValueError(f"Unknown normal direction: {self.cfg.normal_direction}")

        # --- Disparity Calculation --- 
        if camera_distances is not None:
            cam_dist_view = camera_distances.view(-1, 1, 1, 1)
            # Use a fixed range or make znear/zfar accessible here if needed
            # far = zfar # Requires zfar access
            # near = znear # Requires znear access
            # A simple approximation based on distance:
            range_offset = torch.sqrt(torch.tensor(3.0, device=cam_dist_view.device)) # Similar to original
            far = cam_dist_view + range_offset
            near = torch.clamp(cam_dist_view - range_offset, min=1e-5)

            depth_blend = depth * opacity + (1.0 - opacity) * far # Use calculated far value
            disparity_norm = (far - depth_blend) / (far - near + 1e-7) # Add epsilon for stability
            disparity_norm = torch.clamp(disparity_norm, 0.0, 1.0)
            outputs["disparity"] = disparity_norm
        else:
            # Output zero disparity if camera distances are not available
            outputs["disparity"] = torch.zeros_like(depth)

        return outputs