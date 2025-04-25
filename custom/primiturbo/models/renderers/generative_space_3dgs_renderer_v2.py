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


@threestudio.register("generative-space-3dgs-rasterize-renderer-v2")
class GenerativeSpace3dgsRasterizeRendererV2(Rasterizer):
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False

        # for rendering the normal
        normal_direction: str = "camera"  # "front" or "camera" or "world"

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
        self.tmp_background_tensor = torch.tensor(
            (0.0, 0.0, 0.0), dtype=torch.float32
        )
        self.normal_module = Depth2Normal()

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
            # cf. RaDe-GS
            kernel_size=0.,  # cf. Mip-Splatting; not used
            require_depth=True,
            require_coord=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings).to(pc.xyz.device)

        # Create screenspace points tensor if needed for gradients (optional but good practice)
        # screenspace_points = torch.zeros_like(pc.xyz, dtype=pc.xyz.dtype, requires_grad=True, device=pc.xyz.device) if self.training else None

        # Perform rasterization to get features, depth and alpha
        # Assume pc.rgb holds features
        # print all the devils of the pc
        # rendered_features, radii, rendered_depth, rendered_alpha = rasterizer(
        rendered_image, _, _, _, rendered_depth, _, rendered_alpha, rendered_normal  = rasterizer(  # not used: radii, coord, mcoord, mdepth
            means3D=pc.xyz,
            means2D=torch.zeros_like(pc.xyz, dtype=torch.float32, device=pc.xyz.device), # Use zero means2D if not using screenspace_points
            shs=None,
            colors_precomp=pc.rgb,
            opacities=pc.opacity,
            scales=pc.scale,
            rotations=pc.rotation,
            cov3D_precomp=None,
        )

         # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "comp_rgb": rendered_image,
            "depth": rendered_depth,
            "opacity": rendered_alpha,
            "normal": rendered_normal,
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
                    znear=0.1, # TODO: Make configurable?
                    zfar=100  # TODO: Make configurable?
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

                w2cs.append(w2c)
                projs.append(proj)
                cam_ps.append(cam_p)


        # === Stack foreground results and permute ===
        comp_rgb = torch.stack(comp_rgb, dim=0).permute(0, 2, 3, 1)
        opacity = torch.stack(masks, dim=0).permute(0, 2, 3, 1)          # [B, H, W, 1]
        depth = torch.stack(depths, dim=0).permute(0, 2, 3, 1)            # [B, H, W, 1]
        comp_normal_world = torch.stack(normals, dim=0).permute(0, 2, 3, 1) # [B, H, W, 3], world space


        # --- Prepare Output Dictionary --- 
        outputs = {
            "comp_rgb": comp_rgb,           # Final composite color
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
            stacked_w2c: Float[Tensor, "B 4 4"] = torch.stack(w2cs, dim=0)
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
            outputs["comp_normal_cam_vis_white"] = comp_normal_world_vis_white # Use compatible key
        elif self.cfg.normal_direction == "front":
            threestudio.warn("Normal direction 'front' is complex; using world normal visualization as fallback.")
            bg_normal_white = torch.ones_like(comp_normal_world)
            comp_normal_world_vis_white = (comp_normal_world + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
            outputs["comp_normal_cam_vis_white"] = comp_normal_world_vis_white
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
            disparity_norm = (far - depth_blend) / (far - near)
            disparity_norm = torch.clamp(disparity_norm, 0.0, 1.0)
            outputs["disparity"] = disparity_norm
        else:
            # Output zero disparity if camera distances are not available
            outputs["disparity"] = torch.zeros_like(depth)

        return outputs