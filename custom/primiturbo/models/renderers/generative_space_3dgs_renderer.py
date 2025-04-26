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


@threestudio.register("generative-space-3dgs-rasterize-renderer")
class GenerativeSpace3dgsRasterizeRenderer(Rasterizer):
    @dataclass
    class Config(Rasterizer.Config):
        # For rendering the normal map
        normal_direction: str = "camera"  # "front" or "camera" or "world"
        # Define if material requires tangents (can be interpolated if available in GaussianModel)
        material_requires_tangent: bool = False

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

    def _forward(
        self,
        viewpoint_camera: Camera,
        pc: GaussianModel, # GaussianModel now contains features instead of rgb
        light_positions: Float[Tensor, "1 3"], # Expect single light position per view for now
        scaling_modifier=1.0,
        **kwargs # Catch unused args
    ) -> Dict[str, Any]:
        """
        Render the scene foreground using Gaussian Splatting for a single view, applying material.
        """
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        bg_color_tensor = torch.zeros(3, dtype=torch.float32, device=pc.xyz.device)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color_tensor,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=0, # SH degree might be 0 if features are passed via colors_precomp
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            kernel_size=0.,
            require_depth=True,
            require_coord=True, # Need world coordinates for material input
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings).to(pc.xyz.device)

        # --- Input Validation ---
        valid_inputs = True
        if not torch.isfinite(pc.xyz).all():
            print(f"!!! FATAL: pc.xyz contains non-finite values! Shape: {pc.xyz.shape}")
            valid_inputs = False
        if not torch.isfinite(pc.scale).all():
            print(f"!!! FATAL: pc.scale contains non-finite values! Shape: {pc.scale.shape}")
            valid_inputs = False
        if not torch.isfinite(pc.rotation).all():
            print(f"!!! FATAL: pc.rotation contains non-finite values! Shape: {pc.rotation.shape}")
            valid_inputs = False
        if not torch.isfinite(pc.opacity).all():
            print(f"!!! FATAL: pc.opacity contains non-finite values! Shape: {pc.opacity.shape}")
            valid_inputs = False
        if hasattr(pc, 'features') and pc.features is not None and not torch.isfinite(pc.features).all():
            print(f"!!! FATAL: pc.features contains non-finite values! Shape: {pc.features.shape}")
            valid_inputs = False

        if not valid_inputs:
            raise RuntimeError("Invalid input detected for GaussianRasterizer. Check preceding logs.")
        # --- End Input Validation ---


        # Perform rasterization (outputs are dense HxW tensors)
        rendered_features, _, rendered_coord, _, rendered_depth, _, rendered_alpha, rendered_normal = rasterizer(
            means3D=pc.xyz,
            means2D=torch.zeros_like(pc.xyz, dtype=torch.float32, device=pc.xyz.device),
            shs=None,
            colors_precomp=pc.features if hasattr(pc, 'features') else None,
            opacities=pc.opacity,
            scales=pc.scale,
            rotations=pc.rotation,
            cov3D_precomp=None,
        )

        # --- Prepare inputs for material shading for ALL pixels ---
        # Permute tensors to [H, W, C] format
        # Assuming rasterizer output is [C, H, W] or [1, C, H, W]
        # Squeeze the batch dim if it exists (batch size is 1 in _forward)
        h, w = viewpoint_camera.image_height, viewpoint_camera.image_width
        gb_pos = rendered_coord.squeeze(0).permute(1, 2, 0) # [H, W, 3]
        gb_normal = rendered_normal.squeeze(0).permute(1, 2, 0) # [H, W, 3]
        gb_features = rendered_features.squeeze(0).permute(1, 2, 0) # [H, W, FeatureDim]

        # Calculate view directions for all pixels
        # Need pixel coordinates or use approximation with camera center
        # Approximation: Calculate viewdirs based on world positions gb_pos
        # Note: viewdirs for background pixels using gb_pos might be inaccurate,
        #       but self.material is assumed to handle this.
        gb_viewdirs = F.normalize(gb_pos - viewpoint_camera.camera_center, dim=-1) # [H, W, 3]

        # Expand light positions for all pixels
        gb_light_positions = light_positions.unsqueeze(0).unsqueeze(0).expand(h, w, -1) # [H, W, 3]

        # Prepare geometry output dict
        geo_out = {}

        # Prepare extra geo info
        extra_geo_info = {}
        if self.material.requires_normal:
             # Normalize normals for all pixels
             extra_geo_info["shading_normal"] = F.normalize(gb_normal, dim=-1) # [H, W, 3]

        # --- Call the material for ALL pixels ---
        # Note: Material needs to handle potentially invalid inputs for background pixels
        rgb_fg_values = self.material(
            viewdirs=gb_viewdirs,      # [H, W, 3]
            positions=gb_pos,         # [H, W, 3]
            light_positions=gb_light_positions, # [H, W, 3]
            features=gb_features,     # [H, W, FeatureDim]
            **extra_geo_info,
            **geo_out
        ) # Expected output shape [H, W, 3]

        # --- Prepare return values ---
        # Permute computed foreground color to [C, H, W] for stacking
        comp_rgb_fg_return = rgb_fg_values.permute(2, 0, 1) # [3, H, W]

        return {
            "comp_rgb_fg": comp_rgb_fg_return, # [C, H, W] format (C=3)
            "depth": rendered_depth,           # Keep original shape [1, H, W]
            "opacity": rendered_alpha,         # Keep original shape [1, H, W]
            "normal": rendered_normal,         # Keep original shape [C, H, W] (C=3)
        }

    def _space_cache_to_pc(
        self,
        space_cache: Dict[str, Union[Float[Tensor, "B ..."], List[Float[Tensor, "B ..."]]]],
    ):
        """Converts a batch of space cache data into a list of GaussianModel objects."""
        pc_list = []
        batch_size_space_cache = space_cache["position"].shape[0]
        for i in range(batch_size_space_cache):
            if "features" not in space_cache:
                 raise ValueError("space_cache must contain 'features' key when using material shading.")

            gaussian_model = GaussianModel().set_data(
                xyz=space_cache["position"][i],
                scale=space_cache["scale"][i],
                rotation=space_cache["rotation"][i],
                opacity=space_cache["opacity"][i],
            )
            gaussian_model.features = space_cache["features"][i]
            pc_list.append(gaussian_model)
        return pc_list

    def forward(
        self,
        space_cache: Dict[str, Union[Float[Tensor, "B ..."], List[Float[Tensor, "B ..."]]]],
        c2w: Float[Tensor, "B 4 4"],
        fovy: Float[Tensor, "B"],
        light_positions: Float[Tensor, "B 3"],
        rays_d: Float[Tensor, "B H W 3"],
        height: int,
        width: int,
        fovx: Optional[Float[Tensor, "B"]] = None,
        camera_distances: Optional[Float[Tensor, "B"]] = None,
        text_embed: Optional[Float[Tensor, "B C"]] = None,
        text_embed_bg: Optional[Float[Tensor, "B C"]] = None,
        **kwargs # Catch unused arguments
    ):
        batch_size = c2w.shape[0]
        batch_size_space_cache = space_cache["position"].shape[0]

        if batch_size_space_cache != 1 and batch_size_space_cache != batch_size:
             raise ValueError(f"Batch size mismatch: c2w has {batch_size}, but space_cache implies {batch_size_space_cache}")

        gaussian_list = self._space_cache_to_pc(space_cache)

        renders_fg_list = []
        normals_list = []
        depths_list = []
        masks_list = []
        w2cs_list = []

        for batch_idx in range(batch_size):
            pc = gaussian_list[0] if batch_size_space_cache == 1 else gaussian_list[batch_idx]
            current_fovy = fovy[batch_idx]
            current_fovx = fovx[batch_idx] if fovx is not None else current_fovy
            current_c2w = c2w[batch_idx]
            current_light_pos = light_positions[batch_idx:batch_idx+1] # Shape [1, 3]

            w2c, proj, cam_p = get_cam_info_gaussian(
                c2w=current_c2w, fovx=current_fovx, fovy=current_fovy, znear=0.1, zfar=100
            )
            w2cs_list.append(w2c)

            viewpoint_cam = Camera(
                FoVx=current_fovx, FoVy=current_fovy, image_width=width, image_height=height,
                world_view_transform=w2c, full_proj_transform=proj, camera_center=cam_p,
            )

            with autocast(enabled=False):
                render_pkg = self._forward(
                    viewpoint_cam,
                    pc,
                    light_positions=current_light_pos, # Pass light position
                    **kwargs
                )

            renders_fg_list.append(render_pkg["comp_rgb_fg"])
            normals_list.append(render_pkg["normal"])
            depths_list.append(render_pkg["depth"])
            masks_list.append(render_pkg["opacity"])

        # Stack results
        comp_rgb_fg = torch.stack(renders_fg_list, dim=0) # [B, C, H, W]
        opacity = torch.stack(masks_list, dim=0)       # [B, 1, H, W]
        depth = torch.stack(depths_list, dim=0)         # [B, 1, H, W]
        comp_normal_world = torch.stack(normals_list, dim=0) # [B, C, H, W]

        # Permute to [B, H, W, C] for consistency in output dict
        comp_rgb_fg = comp_rgb_fg.permute(0, 2, 3, 1)
        opacity = opacity.permute(0, 2, 3, 1)
        depth = depth.permute(0, 2, 3, 1)
        comp_normal_world = comp_normal_world.permute(0, 2, 3, 1)

        # Background Computation
        if self.background is not None:
            bg_text_embed = text_embed_bg if text_embed_bg is not None else text_embed
            if hasattr(self.background, 'enabling_hypernet') and self.background.enabling_hypernet:
                 comp_rgb_bg = self.background(dirs=rays_d, text_embed=bg_text_embed)
            else:
                 comp_rgb_bg = self.background(dirs=rays_d)
            comp_rgb_bg = comp_rgb_bg.view(batch_size, height, width, -1)
        else:
            comp_rgb_bg = torch.zeros_like(comp_rgb_fg)

        # Composite
        comp_rgb = comp_rgb_fg + comp_rgb_bg * (1.0 - opacity)

        # Prepare Output Dictionary
        outputs = {
            "comp_rgb": comp_rgb,
            "comp_rgb_fg": comp_rgb_fg,
            "comp_rgb_bg": comp_rgb_bg,
            "opacity": opacity,
            "depth": depth,
        }

        # Normal Processing
        comp_normal_world = F.normalize(comp_normal_world, dim=-1)
        outputs["comp_normal"] = comp_normal_world

        if self.cfg.normal_direction == "camera":
            bg_normal_val = 0.5
            bg_normal = torch.full_like(comp_normal_world, bg_normal_val)
            bg_normal[..., 2] = 1.0
            bg_normal_white = torch.ones_like(comp_normal_world)
            stacked_w2c = torch.stack(w2cs_list, dim=0)
            rot = stacked_w2c[:, :3, :3]
            comp_normal_cam = torch.bmm(comp_normal_world.view(batch_size, -1, 3), rot.permute(0, 2, 1))
            flip_mat = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32, device=comp_normal_cam.device)
            comp_normal_cam = torch.bmm(comp_normal_cam, flip_mat.unsqueeze(0).expand(batch_size, -1, -1))
            comp_normal_cam = comp_normal_cam.view(batch_size, height, width, 3)
            comp_normal_cam_vis = (comp_normal_cam + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal
            comp_normal_cam_vis_white = (comp_normal_cam + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal_white
            outputs.update({
                "comp_normal_cam_vis": comp_normal_cam_vis,
                "comp_normal_cam_vis_white": comp_normal_cam_vis_white,
            })
        elif self.cfg.normal_direction == "world":
            bg_normal_white = torch.ones_like(comp_normal_world)
            comp_normal_world_vis_white = (comp_normal_world + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
            outputs["comp_normal_cam_vis_white"] = comp_normal_world_vis_white
        elif self.cfg.normal_direction == "front":
            threestudio.warn("Normal direction 'front' is complex for 3DGS; using world normal visualization as fallback.")
            bg_normal_white = torch.ones_like(comp_normal_world)
            comp_normal_world_vis_white = (comp_normal_world + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
            outputs["comp_normal_cam_vis_white"] = comp_normal_world_vis_white
        else:
            raise ValueError(f"Unknown normal direction: {self.cfg.normal_direction}")

        # Disparity Calculation
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
            outputs["disparity"] = torch.zeros_like(depth)

        return outputs 