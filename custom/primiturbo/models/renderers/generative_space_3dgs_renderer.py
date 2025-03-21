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
        space_cache: Float[Tensor, "B ..."],
    ):
        pc_list = []
        for i in range(len(space_cache)):
            _space_cache = space_cache[i]
            pc_list.append(
                GaussianModel().set_data(
                    xyz=_space_cache["gs_xyz"],
                    rgb=_space_cache["gs_rgb"],
                    scale=_space_cache["gs_scale"],
                    rotation=_space_cache["gs_rotation"],
                    opacity=_space_cache["gs_opacity"],
                )
            )
        return pc_list

    def forward(
        self, 
        batch
    ):
        space_cache = batch['space_cache']

        batch_size = batch["c2w"].shape[0]
        batch_size_space_cache = len(space_cache)
        num_views_per_batch = batch_size // batch_size_space_cache

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
                batch["batch_idx"] = batch_idx
                fovy = batch["fovy"][batch_idx]
                fovx = batch["fovx"][batch_idx] if "fovx" in batch else fovy
                c2w = batch["c2w"][batch_idx]           
                # TODO: check if this is correct
                w2c, proj, cam_p = get_cam_info_gaussian(
                    c2w=c2w, 
                    fovx=fovx, 
                    fovy=fovy, 
                    znear=0.1, 
                    zfar=100
                )
                # # TODO: check if this is correct
                # w2c = torch.inverse(c2w)
                # proj = batch["mvp_mtx"][batch_idx]
                # cam_p = batch["camera_positions"][batch_idx]

                viewpoint_cam = Camera(
                    FoVx=fovx,
                    FoVy=fovy,
                    image_width=batch["width"],
                    image_height=batch["height"],
                    world_view_transform=w2c,
                    full_proj_transform=proj,
                    camera_center=cam_p,
                )

                with autocast(enabled=False):
                    render_pkg = self._forward(
                        viewpoint_cam, 
                        pc,
                        self.background_tensor,
                        **batch
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

                
        height = batch["height"]
        width = batch["width"]

        outputs = {
            "comp_rgb": torch.stack(renders, dim=0).permute(0, 2, 3, 1),
        }
        if len(masks) > 0:
            opacity = torch.stack(masks, dim=0).permute(0, 2, 3, 1)
            outputs.update(
                {
                    "opacity": opacity,
                }
            )
        if len(normals) > 0:
            comp_normal = torch.stack(normals, dim=0).permute(0, 2, 3, 1)
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

                # # convert_normal_to_cam_space
                # # TODO: check if this is correct
                # w2c: Float[Tensor, "B 4 4"] = torch.stack(w2cs, dim=0)
                # rot: Float[Tensor, "B 3 3"] = w2c[:, :3, :3]
                # # TODO: check if this is correct
                # w2c: Float[Tensor, "B 4 4"] = torch.inverse(batch["c2w"])
                # rot: Float[Tensor, "B 3 3"] = w2c[:, :3, :3]

                # comp_normal_cam = comp_normal.view(batch_size, -1, 3) @ rot.permute(0, 2, 1)
                comp_normal_cam = comp_normal.view(batch_size, -1, 3)
                flip_x = torch.eye(3, device=comp_normal_cam.device) #  pixel space flip axis so we need built negative y-axis normal
                # flip_x[0, 0] = -1
                # flip_x[1, 1] = -1
                flip_x[2, 2] = -1
                comp_normal_cam = comp_normal_cam @ flip_x[None, :, :]
                comp_normal_cam = comp_normal_cam.view(batch_size, height, width, 3)
                # comp_normal_cam = comp_normal * -1
                


                comp_normal_cam_vis = (comp_normal_cam + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal
                comp_normal_cam_vis_white = (comp_normal_cam + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal_white

                outputs.update(
                    {
                        "comp_normal_cam_vis": comp_normal_cam_vis.view(batch_size, height, width, 3),
                        "comp_normal_cam_vis_white": comp_normal_cam_vis_white.view(batch_size, height, width, 3),
                    }
                )
            else:
                raise ValueError(f"Unknown normal direction: {self.cfg.normal_direction}")


        if len(depths) > 0:
            depth = torch.stack(depths, dim=0).permute(0, 2, 3, 1)
            # TODO: check if this is correct
            camera_distances = torch.stack(cam_ps, dim=0).norm(dim=-1, p=2)[:, None, None, None]  # 2-norm of camera_positions
            far = camera_distances + torch.sqrt(3 * torch.ones(1, 1, 1, 1, device=camera_distances.device))
            near = camera_distances - torch.sqrt(3 * torch.ones(1, 1, 1, 1, device=camera_distances.device))
            # # TODO: check if this is correct
            # far = camera_distances + torch.sqrt(3 * torch.ones(1, 1, 1, 1, device=batch["camera_distances"].device))
            # near = batch["camera_distances"].reshape(-1, 1, 1, 1) - torch.sqrt(3 * torch.ones(1, 1, 1, 1, device=batch["camera_distances"].device))
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