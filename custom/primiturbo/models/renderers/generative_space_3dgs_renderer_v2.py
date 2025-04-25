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

    
@threestudio.register("generative-space-3dgs-rasterize-renderer-v2")
class GenerativeSpace3dgsRasterizeRendererV2(Rasterizer):
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False
        invert_bg_prob: float = 0.5
        back_ground_color: Tuple[float, float, float] = (0.6, 0.6, 0.6)

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
        self.background_tensor = torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32
        )

    def _forward(
        self,
        viewpoint_camera,
        pc,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # kwargs.keys() = dict_keys([
        # 'prompt', 'guidance_utils', 'condition_utils', 
        # 'rays_o', 'rays_d', 'mvp_mtx', 'camera_positions', 
        # 'c2w', 'light_positions', 'elevation', 'azimuth', 'camera_distances', 
        # 'camera_distances_relative', 'height', 'width', 'fovy', 'rays_d_rasterize', 
        # 'noise', 'text_embed_bg', 'text_embed', 'space_cache'])

        if self.training:
            invert_bg_color = np.random.rand() > self.cfg.invert_bg_prob
        else:
            invert_bg_color = True

        bg_color = bg_color if not invert_bg_color else (1.0 - bg_color)

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color.to(self.device),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            # cf. RaDe-GS
            kernel_size=0.,  # cf. Mip-Splatting; not used
            require_depth=True,
            require_coord=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings).to(self.device)

        # === REMOVE Debug Checks ===
        # # === Debug: Check camera settings ===
        # if self.training: # Only print during training when the error occurs
        #     print(f"[DEBUG RASTERIZER SETTINGS] pc_index={pc_index}, batch_idx={batch_idx}")
        #     print(f"  image_height={raster_settings.image_height}, image_width={raster_settings.image_width}")
        #     print(f"  tanfovx={raster_settings.tanfovx}, tanfovy={raster_settings.tanfovy}")
        #     print(f"  bg_color: shape={raster_settings.bg.shape}, dtype={raster_settings.bg.dtype}, device={raster_settings.bg.device}, has_nan={torch.isnan(raster_settings.bg).any()}, has_inf={torch.isinf(raster_settings.bg).any()}")
        #     print(f"  scale_modifier={raster_settings.scale_modifier}, sh_degree={raster_settings.sh_degree}, prefiltered={raster_settings.prefiltered}")
        #     print(f"  campos: shape={raster_settings.campos.shape}, has_nan={torch.isnan(raster_settings.campos).any()}, has_inf={torch.isinf(raster_settings.campos).any()}")
        #     print(f"  viewmatrix: shape={raster_settings.viewmatrix.shape}, has_nan={torch.isnan(raster_settings.viewmatrix).any()}, has_inf={torch.isinf(raster_settings.viewmatrix).any()}")
        #     print(f"  projmatrix: shape={raster_settings.projmatrix.shape}, has_nan={torch.isnan(raster_settings.projmatrix).any()}, has_inf={torch.isinf(raster_settings.projmatrix).any()}")
        # # === End Debug ===
        # 
        # # === Debug: Check inputs to rasterizer ===
        # if self.training: # Only print during training when the error occurs
        #     print(f"[DEBUG RASTERIZER INPUTS] pc_index={pc_index}, batch_idx={batch_idx}")
        #     print(f"  pc.xyz: shape={pc.xyz.shape}, dtype={pc.xyz.dtype}, has_nan={torch.isnan(pc.xyz).any()}, has_inf={torch.isinf(pc.xyz).any()}")
        #     print(f"  pc.rgb: shape={pc.rgb.shape}, dtype={pc.rgb.dtype}, has_nan={torch.isnan(pc.rgb).any()}, has_inf={torch.isinf(pc.rgb).any()}")
        #     # Opacity checks
        #     has_nan_opacity = torch.isnan(pc.opacity).any()
        #     has_inf_opacity = torch.isinf(pc.opacity).any()
        #     min_opacity = torch.min(pc.opacity).item() if not has_nan_opacity and not has_inf_opacity else 'N/A'
        #     max_opacity = torch.max(pc.opacity).item() if not has_nan_opacity and not has_inf_opacity else 'N/A'
        #     print(f"  pc.opacity: shape={pc.opacity.shape}, dtype={pc.opacity.dtype}, has_nan={has_nan_opacity}, has_inf={has_inf_opacity}, min={min_opacity}, max={max_opacity}")
        #     # Scale checks
        #     has_nan_scale = torch.isnan(pc.scale).any()
        #     has_inf_scale = torch.isinf(pc.scale).any()
        #     min_scale = torch.min(pc.scale).item() if not has_nan_scale and not has_inf_scale else 'N/A'
        #     max_scale = torch.max(pc.scale).item() if not has_nan_scale and not has_inf_scale else 'N/A'
        #     print(f"  pc.scale: shape={pc.scale.shape}, dtype={pc.scale.dtype}, has_nan={has_nan_scale}, has_inf={has_inf_scale}, min={min_scale}, max={max_scale}")
        #     # Rotation checks
        #     has_nan_rotation = torch.isnan(pc.rotation).any()
        #     has_inf_rotation = torch.isinf(pc.rotation).any()
        #     rotation_norms = torch.linalg.norm(pc.rotation, dim=-1) if not has_nan_rotation and not has_inf_rotation else None
        #     min_norm = torch.min(rotation_norms).item() if rotation_norms is not None else 'N/A'
        #     max_norm = torch.max(rotation_norms).item() if rotation_norms is not None else 'N/A'
        #     is_normalized = torch.allclose(rotation_norms, torch.ones_like(rotation_norms), atol=1e-5) if rotation_norms is not None else 'N/A'
        #     print(f"  pc.rotation: shape={pc.rotation.shape}, dtype={pc.rotation.dtype}, has_nan={has_nan_rotation}, has_inf={has_inf_rotation}, min_norm={min_norm}, max_norm={max_norm}, all_normalized={is_normalized}")
        # # === End Debug ===

        # print all the devils of the pc
        rendered_image, _, _, _, rendered_depth, _, rendered_alpha, rendered_normal  = rasterizer(  # not used: radii, coord, mcoord, mdepth
            means3D=pc.xyz,
            means2D=torch.zeros_like(pc.xyz, dtype=torch.float32, device=pc.xyz.device),
            shs=None,
            colors_precomp=pc.rgb,
            opacities=pc.opacity,
            scales=pc.scale,
            rotations=pc.rotation,
            cov3D_precomp=None,
        )

        # # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        #     means3D=pc.xyz,
        #     means2D=torch.zeros_like(pc.xyz, dtype=torch.float32, device=pc.xyz.device),
        #     shs=None,
        #     colors_precomp=pc.rgb,
        #     opacities=pc.opacity,
        #     scales=pc.scale,
        #     rotations=pc.rotation,
        #     cov3D_precomp=None,
        # )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "depth": rendered_depth,
            "mask": rendered_alpha,
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
            rgb = space_cache["color"][i] 
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

        renders = []
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
                # print(f"batch_idx: {batch_idx}")
                # TODO: check if this is correct
                w2c, proj, cam_p = get_cam_info_gaussian(
                    c2w=c2w[batch_idx],
                    fovx=fovx[batch_idx] if fovx is not None else fovy[batch_idx], # Use calculated or provided fovx
                    fovy=fovy[batch_idx],
                    znear=0.1,
                    zfar=100
                )
                # # TODO: check if this is correct
                # w2c = torch.inverse(c2w)
                # proj = batch["mvp_mtx"][batch_idx]
                # cam_p = batch["camera_positions"][batch_idx]

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
                        self.background_tensor,
                        **kwargs
                    )
                    # 立即释放渲染结果的内存
                    with torch.cuda.stream(torch.cuda.Stream()):
                        # 处理渲染结果
                        if "render" in render_pkg:
                            renders.append(render_pkg["render"])
                        if "normal" in render_pkg:
                            normals.append(render_pkg["normal"])
                        if "depth" in render_pkg:
                            depths.append(render_pkg["depth"])
                        if "mask" in render_pkg:
                            masks.append(render_pkg["mask"])

                w2cs.append(w2c)
                projs.append(proj)
                cam_ps.append(cam_p)



        # === Debug: Check renders list length before stacking ===
        # print(f"[DEBUG STACK] len(renders) = {len(renders)}")
        # === End Debug ===
        # === Debug: Stacking renders ===
        # if self.training: print("[DEBUG STACK] Attempting torch.stack(renders)...")
        stacked_renders = torch.stack(renders, dim=0)
        # if self.training: print(f"[DEBUG STACK] torch.stack(renders) successful. Shape: {stacked_renders.shape}")
        # === End Debug ===
        # === Debug: Permuting stacked renders ===
        # if self.training: print("[DEBUG STACK] Attempting permute...")
        comp_rgb = stacked_renders.permute(0, 2, 3, 1)
        # if self.training: print(f"[DEBUG STACK] Permute successful. Shape: {comp_rgb.shape}")
        # === End Debug ===
        outputs = {
            "comp_rgb": comp_rgb,
        }
        # Remove the previous combined debug print
        # === Debug: After stacking renders ===
        # # if self.training: print(f"[DEBUG STACK] Successfully stacked renders. Shape: {outputs['comp_rgb'].shape}") 
        # === End Debug ===
        if len(masks) > 0:
            # === Debug: Stacking masks ===
            # if self.training: print(f"[DEBUG STACK] Attempting torch.stack(masks)... len={len(masks)}")
            stacked_masks = torch.stack(masks, dim=0)
            # if self.training: print(f"[DEBUG STACK] torch.stack(masks) successful. Shape: {stacked_masks.shape}")
            opacity = stacked_masks.permute(0, 2, 3, 1)
            # if self.training: print(f"[DEBUG STACK] Permute masks successful. Shape: {opacity.shape}")
            # === End Debug ===
            outputs.update(
                {
                    "opacity": opacity,
                }
            )
        if len(normals) > 0:
            # === Debug: Stacking normals ===
            # if self.training: print(f"[DEBUG STACK] Attempting torch.stack(normals)... len={len(normals)}")
            stacked_normals = torch.stack(normals, dim=0)
            # if self.training: print(f"[DEBUG STACK] torch.stack(normals) successful. Shape: {stacked_normals.shape}")
            comp_normal = stacked_normals.permute(0, 2, 3, 1)
            # if self.training: print(f"[DEBUG STACK] Permute normals successful. Shape: {comp_normal.shape}")
            # === End Debug ===
            # comp_normal = torch.stack(normals, dim=0).permute(0, 2, 3, 1)
            comp_normal = F.normalize(comp_normal, dim=-1)
            outputs.update(
                {
                    "comp_normal": comp_normal,
                }
            )

            if self.cfg.normal_direction == "camera":
                # for compatibility with RichDreamer #############
                bg_normal = 0.5 * torch.ones_like(comp_normal)
                bg_normal[:, 2] = 1.0 # for a blue background
                bg_normal_white = torch.ones_like(comp_normal)

                # === Debug: Print shapes before view ===
                # print(f"[DEBUG] Before view: comp_normal.shape = {comp_normal.shape}")
                # print(f"[DEBUG] Before view: width variable = {width}")
                # === End Debug ===
                # Use actual tensor shape[0] instead of potentially mismatched batch_size variable
                actual_batch_size = comp_normal.shape[0]
                comp_normal_cam = comp_normal.view(actual_batch_size, -1, 3) 
                flip_x = torch.eye(3, device=comp_normal_cam.device) #  pixel space flip axis so we need built negative y-axis normal
                # flip_x[0, 0] = -1
                # flip_x[1, 1] = -1
                flip_x[2, 2] = -1
                comp_normal_cam = comp_normal_cam @ flip_x[None, :, :]
                # Use actual_batch_size for the final view as well
                comp_normal_cam = comp_normal_cam.view(actual_batch_size, height, width, 3) 
                # comp_normal_cam = comp_normal * -1
                


                # Need to get opacity shape consistent as well if it exists
                if 'opacity' in outputs:
                    opacity_view = outputs['opacity'].view(actual_batch_size, height, width, 1)
                else:
                    # Handle case where opacity might not have been computed/added yet
                    opacity_view = torch.ones((actual_batch_size, height, width, 1), device=comp_normal.device) 

                comp_normal_cam_vis = (comp_normal_cam + 1.0) / 2.0 * opacity_view + (1 - opacity_view) * bg_normal.view(actual_batch_size, height, width, 3)
                comp_normal_cam_vis_white = (comp_normal_cam + 1.0) / 2.0 * opacity_view + (1 - opacity_view) * bg_normal_white.view(actual_batch_size, height, width, 3)

                outputs.update(
                    {
                        "comp_normal_cam_vis": comp_normal_cam_vis, # Already has correct shape
                        "comp_normal_cam_vis_white": comp_normal_cam_vis_white, # Already has correct shape
                    }
                )
            else:
                raise ValueError(f"Unknown normal direction: {self.cfg.normal_direction}")


        if len(depths) > 0:
            # === Debug: Stacking depths ===
            # if self.training: print(f"[DEBUG STACK] Attempting torch.stack(depths)... len={len(depths)}")
            stacked_depths = torch.stack(depths, dim=0)
            # if self.training: print(f"[DEBUG STACK] torch.stack(depths) successful. Shape: {stacked_depths.shape}")
            depth = stacked_depths.permute(0, 2, 3, 1)
            # if self.training: print(f"[DEBUG STACK] Permute depths successful. Shape: {depth.shape}")
            # === End Debug ===
            # depth = torch.stack(depths, dim=0).permute(0, 2, 3, 1)
            # TODO: check if this is correct
            camera_distances = torch.stack(cam_ps, dim=0).norm(dim=-1, p=2)[:, None, None, None]  # 2-norm of camera_positions
            far = camera_distances + torch.sqrt(3 * torch.ones(1, 1, 1, 1, device=camera_distances.device))
            near = camera_distances - torch.sqrt(3 * torch.ones(1, 1, 1, 1, device=camera_distances.device))
            disparity_tmp = depth * opacity + (1.0 - opacity) * far
            disparity_norm = (far - disparity_tmp) / (far - near)
            disparity_norm = torch.clamp(disparity_norm, 0.0, 1.0)
            outputs.update(
                {
                    "depth": depth,
                    "disparity": disparity_norm,
                }
            )
        return outputs