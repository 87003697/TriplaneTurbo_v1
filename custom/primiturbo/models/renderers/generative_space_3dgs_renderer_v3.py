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

from .gaussian_utils import GaussianModel, Depth2Normal




class Camera(NamedTuple):
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor



@threestudio.register("generative-space-3dgs-rasterize-renderer-v3")
class GenerativeSpace3dgsRasterizeRendererV3(Rasterizer):
    @dataclass
    class Config(Rasterizer.Config):
        near: float = 0.1
        far: float = 100

        # for rendering the normal
        normal_direction: str = "camera"  # "front" or "camera" or "world"

        # New option for material application timing
        material_application_mode: str = "pre" # Options: "pre", "post"

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
        # <<< HACK: 实例化 Depth2Normal
        self.normal_module = Depth2Normal()
        # >>> HACK END

        # Validate material application mode
        if self.cfg.material_application_mode not in ["pre", "post"]:
            raise ValueError(f"Invalid material_application_mode: {self.cfg.material_application_mode}. Must be 'pre' or 'post'.")

    def _forward(
        self,
        viewpoint_camera,
        pc,
        bg_color: torch.Tensor,
        light_positions: Float[Tensor, "1 3"],
        rays_o_rasterize: Float[Tensor, "H W 3"],
        rays_d_rasterize: Float[Tensor, "H W 3"],
        scaling_modifier=1.0,
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
            # # cf. RaDe-GS
            # kernel_size=0.,  # cf. Mip-Splatting; not used
            # require_depth=True,
            # require_coord=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings).to(pc.xyz.device)

        # Perform rasterization to get features, depth and alpha
        # Use standard pc.rgb for color
        rendered_output, radii, rendered_depth, rendered_alpha  = rasterizer(
            means3D=pc.xyz,
            means2D=torch.zeros_like(pc.xyz, dtype=torch.float32, device=pc.xyz.device), # Use zero means2D if not using screenspace_points
            shs=None,
            colors_precomp=pc.rgb,
            opacities=pc.opacity,
            scales=pc.scale,
            rotations=pc.rotation,
            cov3D_precomp=None,
        ) # Output is [C, H, W] where C=3 for color or C=features_dim for SH

        # Post-process based on mode
        if self.cfg.material_application_mode == 'pre':
            rendered_image = rendered_output # Output is already RGB
        elif self.cfg.material_application_mode == 'post':
            raise
            # Output contains rendered features [F, H, W]
            rendered_features = rendered_output.permute(1, 2, 0) # [H, W, F]

            # We need per-pixel view direction, normal, light position etc.
            # Calculate world coordinates from depth
            _, H, W = rendered_depth.shape
            xyz_map = rays_o_rasterize + rendered_depth.permute(1, 2, 0) * rays_d_rasterize # [H, W, 3]
            # Calculate normal from world coordinates
            normal_map_world = self.normal_module(xyz_map.permute(2, 0, 1).unsqueeze(0))[0] # [3, H, W]
            normal_map_world = F.normalize(normal_map_world, dim=0).permute(1, 2, 0) # [H, W, 3]

            # Get view directions (assuming rays_d are already normalized)
            view_dirs = -rays_d_rasterize # [H, W, 3], points from surface to camera

            # Prepare material inputs (reshape features if needed)
            # Assuming material expects features as [H*W, F]
            num_pixels = H * W
            material_input_features = rendered_features.view(num_pixels, -1)
            material_input_positions = xyz_map.view(num_pixels, 3)
            material_input_viewdirs = view_dirs.view(num_pixels, 3)
            material_input_normals = normal_map_world.view(num_pixels, 3)
            # Ensure light_positions is broadcastable [1, 3] -> [num_pixels, 3]
            material_input_lights = light_positions.expand(num_pixels, -1)

            # Call material module
            # The material module MUST be designed to handle these per-pixel inputs
            rendered_image_flat = self.material(
                features=material_input_features,
                positions=material_input_positions,
                viewdirs=material_input_viewdirs,
                normal=material_input_normals,
                light_positions=material_input_lights,
                # Add any other kwargs your material might need
            )
            rendered_image = rendered_image_flat.view(H, W, 3) # Reshape back to [H, W, 3]
            rendered_image = rendered_image.permute(2, 0, 1) # Back to [3, H, W] standard

        # Calculate normal map (shared logic, but use the one calculated in 'post' if available)
        # rays_o_rasterize, rays_d_rasterize: [H, W, 3]
        _, H, W = rendered_depth.shape
        # 重建世界坐标 xyz_map: [H, W, 3]
        xyz_map = rays_o_rasterize + rendered_depth.permute(1, 2, 0) * rays_d_rasterize
        # 计算法线 normal_map: [3, H, W] (Depth2Normal 输入要求 [B, C, H, W])
        normal_map = self.normal_module(xyz_map.permute(2, 0, 1).unsqueeze(0))[0]
        normal_map = F.normalize(normal_map, dim=0) # 归一化法线

        # <<< HACK RE-ADD: 计算高斯中心点投影的深度图 >>>
        # Initialize with background value FIRST
        final_center_point_depth_map = torch.full(
            (H, W), float('inf'), device=pc.xyz.device, dtype=torch.float32
        )
        xyz_world = pc.xyz # [N, 3]
        N = xyz_world.shape[0]
        if N > 0:
            # Transform to Camera Coordinates (Needed for depth_cam_z)
            xyz1_world = torch.cat([xyz_world, torch.ones_like(xyz_world[..., :1])], dim=-1) # [N, 4]
            world_view_transform_4x4 = viewpoint_camera.world_view_transform # This is W2C^T (View Matrix Transposed)
            # Calculate camera coordinates: World @ V = World @ (V^T)^T
            xyz1_cam = xyz1_world @ world_view_transform_4x4.T # [N, 4]
            depth_cam_z = xyz1_cam[..., 2] # [N] - Stores Z in camera space.

            # Project to Clip Coordinates using the full_proj_transform
            full_proj_transform_4x4 = viewpoint_camera.full_proj_transform # This is V^T @ P^T
            # Calculate clip coordinates: Clip = World @ (P@V)^T = World @ V^T @ P^T
            xyz1_clip = xyz1_world @ full_proj_transform_4x4 # [N, 4]

            epsilon = 1e-6
            w_clip = xyz1_clip[..., 3]
            # Filter points behind camera or too close (based on clip space w)
            valid_depth_mask = w_clip > epsilon

            xyz1_clip_valid = xyz1_clip[valid_depth_mask]
            depth_cam_z_valid = depth_cam_z[valid_depth_mask] # Use Z from *before* projection
            N_valid = xyz1_clip_valid.shape[0]

            if N_valid > 0:
                # Perspective Divide to get Normalized Device Coordinates (NDC)
                xyz_ndc = xyz1_clip_valid[..., :3] / (xyz1_clip_valid[..., 3:] + epsilon) # [N_valid, 3]

                # Convert NDC to Pixel Coordinates
                ndc_x = xyz_ndc[..., 0]
                ndc_y = xyz_ndc[..., 1]

                # pixel_x: Uses the STANDARD NDC-to-pixel formula.
                # (ndc_x + 1.0) maps [-1, 1] to [0, 2]. * 0.5 maps to [0, 1]. * W maps to [0, W].
                # This implies the NDC X-axis from get_cam_info_gaussian behaves as expected.
                pixel_x = (ndc_x + 1.0) * 0.5 * W
                pixel_y = (ndc_y + 1.0) * 0.5 * H

                pixel_ix = torch.floor(pixel_x).long()
                pixel_iy = torch.floor(pixel_y).long()

                # Bounds check
                in_bounds_mask = (pixel_ix >= 0) & (pixel_ix < W) & (pixel_iy >= 0) & (pixel_iy < H)

                pixel_ix_in = pixel_ix[in_bounds_mask]
                pixel_iy_in = pixel_iy[in_bounds_mask]
                depth_cam_z_in = depth_cam_z_valid[in_bounds_mask] # Filter depths corresponding to valid pixels

                if depth_cam_z_in.numel() > 0: # Check if any points are in bounds AND potentially valid Z
                    # --- TEST HYPOTHESIS: Filter for points with POSITIVE Z ---
                    positive_z_mask = depth_cam_z_in > 0
                    
                    if torch.any(positive_z_mask):
                        pixel_ix_in_pos = pixel_ix_in[positive_z_mask]
                        pixel_iy_in_pos = pixel_iy_in[positive_z_mask]
                        depth_cam_z_in_pos = depth_cam_z_in[positive_z_mask]
    
                        flat_indices = pixel_iy_in_pos * W + pixel_ix_in_pos
    
                        # Calculate depths for valid pixels in a temporary flat map
                        temp_depth_map_flat = torch.full(
                            (H * W,), float('inf'), device=pc.xyz.device, dtype=torch.float32
                        )
                        temp_depth_map_flat.scatter_reduce_(0, flat_indices, depth_cam_z_in_pos, reduce="amin", include_self=False)
    
                        # Update the final map ONLY where scatter_reduce produced finite values
                        final_center_point_depth_map = temp_depth_map_flat.view(H, W)
                        # NO final negation needed for this hypothesis

                    # else: (no positive Z points)
                    #   final_center_point_depth_map remains 'inf'

                # else: (no points in bounds)
                #   final_center_point_depth_map remains 'inf'
            # else: (no points pass W clip)
            #   final_center_point_depth_map remains 'inf'
        # else: (N == 0)
        #   final_center_point_depth_map remains 'inf'

        # <<< HACK END >>>
        return {
            "comp_rgb": rendered_image, # Standard RGB output
            "depth": rendered_depth,     # Standard Z-buffer depth output
            "opacity": rendered_alpha,   # Standard opacity output
            "normal": normal_map,      # Normal calculated from standard depth
            "xyz_map": xyz_map,        # World coordinates from standard depth
            "center_point_depth": final_center_point_depth_map.unsqueeze(0), # Use the initialized/updated map
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
            # rgb = space_cache["color"][i] 
            rgb = self.material( # TODO: for other material, we need to change this
                features=space_cache["color"][i]
            )
            scale = space_cache["scale"][i] 
            rotation = space_cache["rotation"][i] 
            opacity = space_cache["opacity"][i] 
            
            gauss_model = GaussianModel() 

            # check material_application_mode
            if self.cfg.material_application_mode == 'pre':
                rgb = self.material(
                    features=space_cache["color"][i] 
                ) 
            else:
                rgb = space_cache["color"][i]
            
            gauss_model.set_data(
                    xyz=xyz,
                    rgb=rgb,
                    scale=scale,
                    rotation=rotation,
                    opacity=opacity,
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
                    znear=self.cfg.near,
                    zfar=self.cfg.far
                )

                viewpoint_cam = Camera(
                    FoVx=fovx[batch_idx] if fovx is not None else fovy[batch_idx], # Use calculated or provided fovx
                    FoVy=fovy[batch_idx],
                    image_width=width,
                    image_height=height,
                    world_view_transform=w2c,
                    full_proj_transform=proj,
                    camera_center=cam_p,
                )

                with autocast(enabled=False):
                    render_pkg = self._forward(
                        viewpoint_cam, 
                        pc,
                        comp_rgb_bg[batch_idx], # Pass background color for rasterizer edges
                        light_positions=light_positions[batch_idx].unsqueeze(0), # Pass correct light pos
                        rays_o_rasterize=rays_o_rasterize[batch_idx], # Pass correct rays_o
                        rays_d_rasterize=current_rays_d, # Pass NORMALIZED rays_d
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


                w2cs.append(w2c)
                projs.append(proj)
                cam_ps.append(cam_p)


        # === Stack foreground results and permute ===
        comp_rgb = torch.stack(comp_rgb, dim=0).permute(0, 2, 3, 1)
        opacity = torch.stack(masks, dim=0).permute(0, 2, 3, 1)          # [B, H, W, 1]
        depth = torch.stack(depths, dim=0).permute(0, 2, 3, 1)            # [B, H, W, 1]
        comp_normal_rendered = torch.stack(normals, dim=0).permute(0, 2, 3, 1)
        xyz_map = torch.stack(xyz_maps, dim=0).permute(0, 2, 3, 1)
        # Stack center point depths
        comp_center_point_depth = torch.stack(center_point_depths, dim=0).permute(0, 2, 3, 1) if center_point_depths else None # Shape: [B, H, W, 1]
        if comp_center_point_depth is None:
            comp_center_point_depth = torch.zeros_like(depth) # Fallback if empty

        # --- Prepare Output Dictionary --- 
        outputs = {
            "comp_rgb": comp_rgb,           # Standard composite color
            "opacity": opacity,
            "depth": depth,                 # Standard rendered surface depth
            "comp_xyz": xyz_map,            # Rendered surface world coordinates
            "comp_center_point_depth": comp_center_point_depth, # ADDED
        }

        # --- Normal Processing --- 
        comp_normal_rendered = F.normalize(comp_normal_rendered, dim=-1) # Normalize world normal
        outputs["comp_normal"] = comp_normal_rendered # Keep world normal

        # Handle different normal visualizations based on config
        if self.cfg.normal_direction == "camera":
            # Visualize world-space normal blended with white background
            bg_normal_val = 0.5
            bg_normal = torch.full_like(comp_normal_rendered, bg_normal_val)
            bg_normal[..., 2] = 1.0 # Blue background Z

            bg_normal_white = torch.ones_like(comp_normal_rendered)

            # Transform world normal to camera space
            stacked_w2c: Float[Tensor, "B 4 4"] = torch.stack(w2cs, dim=0)
            rot: Float[Tensor, "B 3 3"] = stacked_w2c[:, :3, :3]
            comp_normal_cam = torch.bmm(comp_normal_rendered.view(batch_size, -1, 3), rot.permute(0, 2, 1))

            # Apply flip matrix for visualization convention
            flip_mat = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32, device=comp_normal_cam.device)
            comp_normal_cam = torch.bmm(comp_normal_cam, flip_mat.unsqueeze(0).expand(batch_size, -1, -1))
            comp_normal_cam = comp_normal_cam.view(batch_size, height, width, 3) # Reshape back

            # Blend with background based on opacity
            comp_normal_cam_vis = (comp_normal_cam + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal
            comp_normal_cam_vis_white = (comp_normal_cam + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal_white

            outputs.update(
                {
                    "comp_normal_cam_vis": comp_normal_cam_vis,
                    "comp_normal_cam_vis_white": comp_normal_cam_vis_white,
                }
            )
        elif self.cfg.normal_direction == "world":
            # raise NotImplementedError("Normal direction 'world' is not implemented yet.")
            # Visualize world-space normal blended with white background
            bg_normal_white = torch.ones_like(comp_normal_rendered)
            comp_normal_rendered_vis_white = (comp_normal_rendered + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
            outputs["comp_normal_cam_vis_white"] = comp_normal_rendered_vis_white # Use compatible key
        elif self.cfg.normal_direction == "front":
            # raise NotImplementedError("Normal direction 'front' is not implemented yet.")
            threestudio.warn("Normal direction 'front' is complex; using world normal visualization as fallback.")
            bg_normal_white = torch.ones_like(comp_normal_rendered)
            comp_normal_rendered_vis_white = (comp_normal_rendered + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
            outputs["comp_normal_cam_vis_white"] = comp_normal_rendered_vis_white
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