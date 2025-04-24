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
from threestudio.utils.ops import get_cam_info_gaussian
from torch.cuda.amp import autocast

from .gaussian_utils import GaussianModel

class Camera(NamedTuple):
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor

    
class Depth2Normal(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delzdelxkernel = torch.tensor(
            [
                [0.00000, 0.00000, 0.00000],
                [-1.00000, 0.00000, 1.00000],
                [0.00000, 0.00000, 0.00000],
            ]
        )
        self.delzdelykernel = torch.tensor(
            [
                [0.00000, -1.00000, 0.00000],
                [0.00000, 0.00000, 0.00000],
                [0.0000, 1.00000, 0.00000],
            ]
        )

    def forward(self, x):
        B, C, H, W = x.shape
        delzdelxkernel = self.delzdelxkernel.view(1, 1, 3, 3).to(x.device)
        delzdelx = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelxkernel, padding=1
        ).reshape(B, C, H, W)
        delzdelykernel = self.delzdelykernel.view(1, 1, 3, 3).to(x.device)
        delzdely = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelykernel, padding=1
        ).reshape(B, C, H, W)
        normal = -torch.cross(delzdelx, delzdely, dim=1)
        return normal

@threestudio.register("generative-points-based-3dgs-rasterize-renderer")
class GenerativePointsBased3dgsRasterizeRenderer(Rasterizer):
    @dataclass
    class Config(Rasterizer.Config):
        # For rendering the normal map
        normal_direction: str = "camera"  # "front" or "camera" or "world"
        # Define if material requires tangents (can be interpolated if available in GaussianModel)
        material_requires_tangent: bool = False

        near: float = 0.1
        far: float = 100.0

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        # Pass background module during configuration
        super().configure(geometry, material, background)
        # Material is now configured via super()
        # Add the normal module
        self.normal_module = Depth2Normal()

    def _forward(
        self,
        viewpoint_camera: Camera,
        pc: GaussianModel, # GaussianModel now contains features instead of rgb
        light_positions: Float[Tensor, "1 3"], # Expect single light position per view for now
        rays_o_rasterize: Float[Tensor, "H W 3"], # Camera origin rays for each pixel
        rays_d_rasterize: Float[Tensor, "H W 3"], # Camera direction rays for each pixel
        scaling_modifier=1.0,
        **kwargs # Catch unused args
    ) -> Dict[str, Any]:
        """
        Render the scene foreground using Gaussian Splatting for a single view, applying material.
        """
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # Use a fixed black background for foreground rendering
        bg_color_tensor = torch.zeros(3, dtype=torch.float32, device=pc.xyz.device)

        # --- 增加对相机参数的检查 ---
        valid_camera = True
        if not torch.isfinite(viewpoint_camera.world_view_transform).all():
            print(f"!!! 警告: viewpoint_camera.world_view_transform 包含非有限值! 形状: {viewpoint_camera.world_view_transform.shape}")
            valid_camera = False
        if not torch.isfinite(viewpoint_camera.full_proj_transform).all():
            print(f"!!! 警告: viewpoint_camera.full_proj_transform 包含非有限值! 形状: {viewpoint_camera.full_proj_transform.shape}")
            valid_camera = False
        if not torch.isfinite(viewpoint_camera.camera_center).all():
            print(f"!!! 警告: viewpoint_camera.camera_center 包含非有限值! 形状: {viewpoint_camera.camera_center.shape}")
            valid_camera = False
        # Check tanfovx and tanfovy which are Python floats derived from FoVx/FoVy
        if not np.isfinite(tanfovx) or not np.isfinite(tanfovy):
             print(f"!!! 警告: tanfovx ({tanfovx}) 或 tanfovy ({tanfovy}) 是非有限值! (来自 FoVx: {viewpoint_camera.FoVx}, FoVy: {viewpoint_camera.FoVy})")
             valid_camera = False
        # Also check the source FoVx and FoVy tensors
        if not torch.isfinite(viewpoint_camera.FoVx).all() or not torch.isfinite(viewpoint_camera.FoVy).all():
            print(f"!!! 警告: viewpoint_camera.FoVx 或 viewpoint_camera.FoVy 包含非有限值! FoVx: {viewpoint_camera.FoVx}, FoVy: {viewpoint_camera.FoVy}")
            valid_camera = False

        if not valid_camera:
             raise RuntimeError("检测到无效的相机参数输入。请检查前面的日志。")
        # --- 结束相机参数检查 ---

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color_tensor, # Render foreground against black
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=0, # Set explicitly to 0 as features/colors are precomputed
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings).to(pc.xyz.device)

        # --- 增强 Input Validation ---
        valid_inputs = True # Initialize before checks
        if (pc.scale <= 1e-8).any(): # 检查非常小或负数的 scale
            print(f"!!! 警告: pc.scale 包含接近零或负值! Min: {pc.scale.min()}")
            # valid_inputs = False # 可以考虑将此设为 False

        # 检查旋转四元数是否归一化 (允许一点误差)
        rot_norm = torch.linalg.norm(pc.rotation, dim=-1)
        if not torch.allclose(rot_norm, torch.ones_like(rot_norm), atol=1e-5):
             print(f"!!! 警告: pc.rotation 包含未归一化的四元数! Norm range: [{rot_norm.min()}, {rot_norm.max()}]")
             # valid_inputs = False # 可以考虑将此设为 False

        # 检查颜色/特征维度 (假设应为 3)
        if pc.rgb.shape[-1] != 3:
             print(f"!!! 警告: pc.rgb 最后一个维度不是 3! Shape: {pc.rgb.shape}")
             valid_inputs = False # 这很可能是个错误

        # 检查传入的光线数据
        if not torch.isfinite(rays_o_rasterize).all():
             print(f"!!! 警告: rays_o_rasterize 包含非有限值! Shape: {rays_o_rasterize.shape}")
             valid_inputs = False
        if not torch.isfinite(rays_d_rasterize).all():
             print(f"!!! 警告: rays_d_rasterize 包含非有限值! Shape: {rays_d_rasterize.shape}")
             valid_inputs = False

        if not valid_inputs:
             raise RuntimeError("检测到无效的输入数据 (增强检查)。请检查前面的日志。")
        # --- 结束增强 Input Validation ---

        # --- Clamp scales ---
        clamped_scales = torch.clamp(pc.scale, min=1e-6)
        print(f"--- DEBUG: Original min scale: {pc.scale.min()}, Clamped min scale: {clamped_scales.min()}")

        # Create screenspace points tensor (this is needed for gradients in training)
        screenspace_points = torch.zeros_like(pc.xyz, dtype=pc.xyz.dtype, requires_grad=True, device=pc.xyz.device)

        # Perform rasterization to get features, depth and alpha
        rendered_features, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=pc.xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=pc.rgb,
            opacities=pc.opacity,
            scales=clamped_scales,
            rotations=pc.rotation,
            cov3D_precomp=None,
        )

        # --- 尝试简单访问 ---
        try:
            # 尝试一个非常基本的操作，而不是 isfinite().all()
            _ = rendered_alpha.shape
            _ = rendered_depth.shape
            print(f"--- DEBUG: Rasterizer output shapes accessed successfully: alpha={rendered_alpha.shape}, depth={rendered_depth.shape}")
        except Exception as e:
            print(f"!!! FATAL: 访问 rasterizer 输出张量时立即出错: {e}")
            raise RuntimeError("无法访问 rasterizer 输出，内存可能已损坏。") from e
        # --- 结束简单访问尝试 ---

        # --- 原有的检查 (现在在简单访问之后) ---
        # if not torch.isfinite(rendered_alpha).all(): # <<-- 注释掉
        #     print(f"!!! 警告: rendered_alpha 在光栅化后包含非有限值! 形状: {rendered_alpha.shape}") # <<-- 注释掉
        #     raise RuntimeError("在 rendered_alpha 中检测到非有限值。") # <<-- 注释掉
        # if not torch.isfinite(rendered_depth).all(): # <<-- 注释掉
        #     print(f"!!! 警告: rendered_depth 在光栅化后包含非有限值! 形状: {rendered_depth.shape}") # <<-- 注释掉
        #     raise RuntimeError("在 rendered_depth 中检测到非有限值。") # <<-- 注释掉
        # --- 结束检查 ---

        # --- Calculate world coordinates and normals using ray marching ---
        # Shape info
        _, H, W = rendered_features.shape

        # Calculate world coordinates (xyz_map) for all pixels
        xyz_map = rays_o_rasterize + rendered_depth.permute(1, 2, 0) * rays_d_rasterize  # [H, W, 3]

        # Calculate normals using depth-based gradient
        normal_map = self.normal_module(xyz_map.permute(2, 0, 1).unsqueeze(0))[0]  # [3, H, W]
        normal_map = F.normalize(normal_map, dim=0)

        # Get mask of visible pixels
        mask = (rendered_alpha > 0).squeeze(0)  # [H, W]

        if mask.sum() == 0:
            # Handle case with no visible foreground pixels
            comp_rgb_fg = torch.zeros((3, viewpoint_camera.image_height, viewpoint_camera.image_width), device=pc.xyz.device)
            # Return default values, ensuring shapes match expected output for stacking
            return {
                "comp_rgb_fg": comp_rgb_fg,
                "depth": torch.zeros((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device=pc.xyz.device),
                "opacity": torch.zeros((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device=pc.xyz.device),
                "normal": torch.zeros((3, viewpoint_camera.image_height, viewpoint_camera.image_width), device=pc.xyz.device),
            }

        # Permute and extract tensors for visible pixels
        gb_pos = xyz_map[mask]  # [Nvis, 3]
        gb_normal = normal_map.permute(1, 2, 0)[mask]  # [Nvis, 3]
        gb_features = rendered_features.permute(1, 2, 0)[mask]  # [Nvis, FeatureDim]

        # Calculate view directions for visible pixels
        gb_viewdirs = F.normalize(gb_pos - viewpoint_camera.camera_center, dim=-1)  # [Nvis, 3]
        
        # Expand light positions to match visible pixels
        gb_light_positions = light_positions.expand(gb_pos.shape[0], -1)  # [Nvis, 3]

        # Prepare geometry output dict
        geo_out = {}

        # Prepare extra geo info (normals)
        extra_geo_info = {}
        if self.material.requires_normal:
            # Ensure normals are normalized
            extra_geo_info["shading_normal"] = F.normalize(gb_normal, dim=-1)

        # Call the material to compute foreground color
        rgb_fg_values = self.material(
            viewdirs=gb_viewdirs,
            positions=gb_pos,
            light_positions=gb_light_positions,
            features=gb_features,
            **extra_geo_info,
            **geo_out
        )

        # Create the final foreground image and fill in visible pixels
        comp_rgb_fg = torch.zeros((viewpoint_camera.image_height, viewpoint_camera.image_width, 3), device=pc.xyz.device)
        comp_rgb_fg[mask] = rgb_fg_values

        # Process normal map - add detachment for non-masked areas
        normal_map_processed = normal_map * 0.5 * rendered_alpha + 0.5  # Scale and bias normal
        normal_mask = rendered_alpha.squeeze(0) > 0.99
        normal_mask = normal_mask.repeat(3, 1, 1)
        normal_map_processed[~normal_mask] = normal_map_processed[~normal_mask].detach()
        
        # Apply detachment to depth values in non-visible regions
        depth_mask = rendered_alpha > 0.99 # Keep shape [1, H, W]
        rendered_depth_copy = rendered_depth.clone()
        rendered_depth_copy[~depth_mask] = rendered_depth_copy[~depth_mask].detach()

        # Permute to [C, H, W] format for returning
        comp_rgb_fg_return = comp_rgb_fg.permute(2, 0, 1)  # [3, H, W]

        # Return rendered components
        return {
            "comp_rgb_fg": comp_rgb_fg_return,  # [3, H, W]
            "depth": rendered_depth_copy,       # [1, H, W]
            "opacity": rendered_alpha,          # [1, H, W] 
            "normal": normal_map_processed,     # [3, H, W]
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
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
            pc_list.append(
                GaussianModel().set_data( # GaussianModel likely handles internal shape adjustments if needed
                    xyz=space_cache["position"][i],
                    rgb=space_cache["color"][i],
                    scale=space_cache["scale"][i],
                    rotation=space_cache["rotation"][i],
                    opacity=space_cache["opacity"][i],
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
        batch_size = c2w.shape[0]
        batch_size_space_cache = space_cache["position"].shape[0]

        width, height = rays_d_rasterize.shape[1:3]

        num_views_per_batch = batch_size // batch_size_space_cache
        gaussian_list = self._space_cache_to_pc(space_cache)

        # Prepare lists to store results from each view
        renders_fg_list = []
        normals_list = []
        depths_list = []
        masks_list = []
        w2cs_list = []

        for batch_idx in range(batch_size):
            print(f"--- Processing batch_idx: {batch_idx} ---")
            # Select the correct GaussianModel for this view
            pc = gaussian_list[batch_idx // num_views_per_batch]
            # Get camera parameters for the current view
            current_fovy = fovy[batch_idx]
            current_fovx = fovx[batch_idx] if fovx is not None else current_fovy
            current_c2w = c2w[batch_idx]
            current_light_pos = light_positions[batch_idx:batch_idx+1] # Shape [1, 3]

            # Get camera intrinsics and extrinsics
            w2c, proj, cam_p = get_cam_info_gaussian(
                c2w=current_c2w,
                fovx=current_fovx,
                fovy=current_fovy,
                znear=self.cfg.near, # Consider making these configurable
                zfar=self.cfg.far  # Consider making these configurable
            )
            w2cs_list.append(w2c) # Store for later use

            viewpoint_cam = Camera(
                FoVx=current_fovx, FoVy=current_fovy, image_width=width, image_height=height,
                world_view_transform=w2c, full_proj_transform=proj, camera_center=cam_p,
            )

            # Get rays for this batch
            current_rays_d = rays_d_rasterize[batch_idx]  # [H, W, 3]
            # Create rays_o by expanding camera position
            current_rays_o = rays_o_rasterize[batch_idx]  # [H, W, 3]

            with autocast(enabled=False):
                render_pkg = self._forward(
                    viewpoint_cam,
                    pc,
                    light_positions=current_light_pos,
                    rays_o_rasterize=current_rays_o,
                    rays_d_rasterize=current_rays_d,
                    **kwargs
                )

            renders_fg_list.append(render_pkg["comp_rgb_fg"])
            normals_list.append(render_pkg["normal"])
            depths_list.append(render_pkg["depth"])
            masks_list.append(render_pkg["opacity"])

        # === Post-loop processing ===

        # Stack results from all views
        comp_rgb_fg = torch.stack(renders_fg_list, dim=0).permute(0, 2, 3, 1) # [B, H, W, 3]
        opacity = torch.stack(masks_list, dim=0).permute(0, 2, 3, 1)       # [B, H, W, 1]
        depth = torch.stack(depths_list, dim=0).permute(0, 2, 3, 1)         # [B, H, W, 1]
        comp_normal_world = torch.stack(normals_list, dim=0).permute(0, 2, 3, 1) # [B, H, W, 3], world space

        # --- Background Computation ---        
        # Use the provided background module
        if self.background is not None:
            # Determine which embedding to use for the background
            bg_text_embed = text_embed_bg if text_embed_bg is not None else text_embed

            # Check if background network requires specific inputs (e.g., text_embed)
            # This requires knowing the signature or attributes of the background module
            if hasattr(self.background, 'enabling_hypernet') and self.background.enabling_hypernet:
                 # Example: Assumes background takes 'dirs' and 'text_embed'
                 comp_rgb_bg = self.background(dirs=rays_d_rasterize, text_embed=bg_text_embed)
            else:
                 # Example: Assumes background only takes 'dirs'
                 comp_rgb_bg = self.background(dirs=rays_d_rasterize)
            comp_rgb_bg = comp_rgb_bg.view(batch_size, height, width, -1) # Ensure correct shape
        else:
            # Default to black background if no module provided
            comp_rgb_bg = torch.zeros_like(comp_rgb_fg)

        # --- Composite Foreground and Background ---
        comp_rgb = comp_rgb_fg + comp_rgb_bg * (1.0 - opacity)

        # --- Prepare Output Dictionary ---
        outputs = {
            "comp_rgb": comp_rgb,
            "comp_rgb_fg": comp_rgb_fg,
            "comp_rgb_bg": comp_rgb_bg,
            "opacity": opacity,
            "depth": depth,
        }

        # --- Normal Processing ---
        comp_normal_world = F.normalize(comp_normal_world, dim=-1) # Normalize world normal
        outputs["comp_normal"] = comp_normal_world # Keep world normal

        # Handle different normal visualizations based on config
        if self.cfg.normal_direction == "camera":
            bg_normal_val = 0.5
            bg_normal = torch.full_like(comp_normal_world, bg_normal_val)
            bg_normal[..., 2] = 1.0 # Blue background Z
            bg_normal_white = torch.ones_like(comp_normal_world)

            # Transform world normal to camera space
            stacked_w2c: Float[Tensor, "B 4 4"] = torch.stack(w2cs_list, dim=0)
            rot: Float[Tensor, "B 3 3"] = stacked_w2c[:, :3, :3]
            comp_normal_cam = torch.bmm(comp_normal_world.view(batch_size, -1, 3), rot.permute(0, 2, 1))

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
            # Visualize world-space normal blended with white background
            bg_normal_white = torch.ones_like(comp_normal_world)
            comp_normal_world_vis_white = (comp_normal_world + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
            # Use 'comp_normal_cam_vis_white' key for potential downstream compatibility
            outputs["comp_normal_cam_vis_white"] = comp_normal_world_vis_white
        elif self.cfg.normal_direction == "front":
            # This requires specific handling (e.g., using the c2w of the first view)
            # Replicating the logic from debug.py here might be complex without more context
            threestudio.warn("Normal direction 'front' is complex for 3DGS; using world normal visualization as fallback.")
            bg_normal_white = torch.ones_like(comp_normal_world)
            comp_normal_world_vis_white = (comp_normal_world + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
            outputs["comp_normal_cam_vis_white"] = comp_normal_world_vis_white
        else:
            raise ValueError(f"Unknown normal direction: {self.cfg.normal_direction}")

        # --- Disparity Calculation ---
        if camera_distances is not None:
            cam_dist_view = camera_distances.view(-1, 1, 1, 1)
            range_offset = torch.sqrt(torch.tensor(3.0, device=cam_dist_view.device))
            far = cam_dist_view + range_offset
            near = torch.clamp(cam_dist_view - range_offset, min=1e-5)

            depth_blend = depth * opacity + (1.0 - opacity) * far
            disparity_norm = (far - depth_blend) / (far - near)
            disparity_norm = torch.clamp(disparity_norm, 0.0, 1.0)
            outputs["disparity"] = disparity_norm
        else:
            # Output zero disparity if camera distances are not available
            outputs["disparity"] = torch.zeros_like(depth)

        return outputs