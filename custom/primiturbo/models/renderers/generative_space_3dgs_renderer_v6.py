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
@threestudio.register("generative-space-gsplat-renderer-v6") # Name can remain or be made more generic
class GenerativeSpaceGsplatRendererV6(Rasterizer):
    @dataclass
    class Config(Rasterizer.Config):
        near_plane: float = 0.1
        far_plane: float = 100

        normal_direction: str = "camera"

        rgb_grad_shrink: float = 1.0
        xyz_grad_shrink: float = 1.0
        opacity_grad_shrink: float = 1.0
        scale_grad_shrink: float = 1.0
        rotation_grad_shrink: float = 1.0

        # Main rendering backend switch
        rendering_backend: str = "3dgs"  # Options: "3dgs", "2dgs"

        # Parameters for 3DGS (gsplat.rasterization)
        with_eval3d: bool = False # Eval3D mode for 3DGS
        with_ut: bool = False # Uncertainty (UT) computation for 3DGS
        rasterize_mode: str = "classic"  # Options: "classic" and "antialiased" (for 3DGS)
        
        # Common depth render mode for both backends
        depth_render_mode: str = "D"  # Options: "D" (Accumulated), "ED" (Expected)

        # Parameters for 2DGS (gsplat.rasterization_2dgs)
        # Note: 'rasterize_mode' above is for 3DGS. 2DGS doesn't have this param directly.
        # An 'antialiased' effect in 2DGS might be via eps2d or inherent.
        distloss_2dgs: bool = True
        # depth_mode for 2DGS internal normal calculation will be derived from cfg.depth_render_mode

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
        
        means_gsplat = pc.xyz
        quats_gsplat = pc.rotation 
        scales_gsplat = pc.scale * scaling_modifier
        opacities_gsplat = pc.opacity.squeeze(-1)
        colors_gsplat = pc.rgb
        viewmats_gsplat = batched_W2C_gsplat
        Ks_gsplat = batched_Ks_gsplat
        backgrounds_gsplat = batched_bg_color
        sh_degree_to_pass_to_gsplat = None # Assuming basic RGB/SH0 for now for both

        rendered_image_batch: Optional[torch.Tensor] = None
        rendered_depth_batch: Optional[torch.Tensor] = None
        rendered_alpha_batch: Optional[torch.Tensor] = None
        normal_map_batch: Optional[torch.Tensor] = None

        if self.cfg.rendering_backend == "2dgs":
            # --- 2DGS Rendering Path ---
            gsplat_depth_mode_param_2dgs = "expected" if self.cfg.depth_render_mode == "ED" else "median"
            
            if self.cfg.rasterize_mode == "antialiased": # rasterize_mode is a 3DGS config
                threestudio.warn(
                    "Config 'rasterize_mode' is 'antialiased', but 2DGS backend does not use this parameter directly. "
                    "Antialiasing for 2DGS might be controlled by 'eps2d_2dgs' or be inherent."
                )

            (
                rendered_rgb_or_rgbd_or_depth_2dgs,
                alpha_2dgs,
                normals_direct_2dgs,
                normals_from_depth_2dgs,
                _render_distort_2dgs, # Not typically used in final output dict
                _render_median_depth_2dgs, # Specific median depth, primary depth taken from main output
                _meta_2dgs,
            ) = gsplat.rasterization_2dgs(
                means=means_gsplat,
                quats=quats_gsplat,
                scales=scales_gsplat,
                opacities=opacities_gsplat,
                colors=colors_gsplat,
                viewmats=viewmats_gsplat,
                Ks=Ks_gsplat,
                width=W,
                height=H,
                render_mode=f"RGB+{self.cfg.depth_render_mode}",
                backgrounds=backgrounds_gsplat,
                sh_degree=sh_degree_to_pass_to_gsplat,
                distloss=self.cfg.distloss_2dgs,
                depth_mode=gsplat_depth_mode_param_2dgs,
            )

            rendered_alpha_batch = alpha_2dgs # [C, H, W, 1]
            current_render_mode_2dgs = f"RGB+{self.cfg.depth_render_mode}"

            if "RGB" in current_render_mode_2dgs and ("D" in current_render_mode_2dgs or "ED" in current_render_mode_2dgs):
                rendered_image_batch = rendered_rgb_or_rgbd_or_depth_2dgs[:, :, :, :3]
                rendered_depth_batch = rendered_rgb_or_rgbd_or_depth_2dgs[:, :, :, 3:4]
            elif "RGB" in current_render_mode_2dgs: # Should not happen if mode is "RGB+D/ED"
                rendered_image_batch = rendered_rgb_or_rgbd_or_depth_2dgs
                rendered_depth_batch = torch.zeros_like(rendered_alpha_batch)
            else: # Only depth
                rendered_image_batch = torch.zeros_like(rendered_rgb_or_rgbd_or_depth_2dgs.repeat(1,1,1,3))
                rendered_depth_batch = rendered_rgb_or_rgbd_or_depth_2dgs
            
            if torch.any(normals_direct_2dgs != 0):
                normal_map_batch = normals_direct_2dgs
            elif torch.any(normals_from_depth_2dgs != 0):
                normal_map_batch = normals_from_depth_2dgs
            else:
                threestudio.warn("2DGS: Normals from rasterizer are zero. Falling back to manual calculation.")
                if torch.all(rendered_depth_batch == 0):
                    normal_map_batch = torch.zeros_like(batched_rays_o_world)
                else:
                    xyz_map_world = batched_rays_o_world + rendered_depth_batch * batched_rays_d_world
                    normal_map_batch = self.normal_module(xyz_map_world.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        elif self.cfg.rendering_backend == "3dgs":
            # --- 3DGS Rendering Path (Original Logic) ---
            output_3dgs, alpha_3dgs, _meta_3dgs = gsplat.rasterization(
                means=means_gsplat,
                quats=quats_gsplat,
                scales=scales_gsplat,
                opacities=opacities_gsplat,
                colors=colors_gsplat, 
                viewmats=viewmats_gsplat,
                Ks=Ks_gsplat,
                width=W,
                height=H,
                render_mode=f"RGB+{self.cfg.depth_render_mode}",
                backgrounds=backgrounds_gsplat,
                sh_degree=sh_degree_to_pass_to_gsplat,
                with_eval3d=self.cfg.with_eval3d,
                with_ut=self.cfg.with_ut,
                packed=not (self.cfg.with_eval3d or self.cfg.with_ut), # Default packed logic for 3DGS
                rasterize_mode=self.cfg.rasterize_mode,
            )
            rendered_image_batch = output_3dgs[:, :, :, :3]
            rendered_depth_batch = output_3dgs[:, :, :, 3:4]
            rendered_alpha_batch = alpha_3dgs

            # Normals for 3DGS (always calculated from depth)
            if torch.all(rendered_depth_batch == 0) and not ("D" in f"RGB+{self.cfg.depth_render_mode}" or "ED" in f"RGB+{self.cfg.depth_render_mode}"):
                 normal_map_batch = torch.zeros_like(batched_rays_o_world)
            else:
                xyz_map_world = batched_rays_o_world + rendered_depth_batch * batched_rays_d_world
                normal_map_batch = self.normal_module(xyz_map_world.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:
            raise ValueError(f"Unknown rendering_backend: {self.cfg.rendering_backend}")

        # Ensure all output tensors are assigned and normalized if normal_map is not None
        if normal_map_batch is not None:
            normal_map_batch = F.normalize(normal_map_batch, p=2, dim=-1)
        else: # Should not happen if logic above is correct, but as a fallback:
            normal_map_batch = torch.zeros_like(batched_rays_o_world)


        # --- Center Point Depth/Opacity Pass (Common for both backends) ---
        # This part uses gsplat.rasterization (3D) as it's simpler for point-like rendering.
        center_point_depth_map_batch: Optional[torch.Tensor] = None
        center_point_opacity_map_batch: Optional[torch.Tensor] = None
        num_cameras_in_batch = viewmats_gsplat.shape[0]

        if num_cameras_in_batch > 0:
            with torch.no_grad():
                num_points = pc.xyz.shape[0]
                epsilon_scale = 1e-3
                scales_for_depth_pass = torch.full((num_points, 3), epsilon_scale, device=pc.xyz.device, dtype=torch.float32)
                opacities_input_for_depth_pass = torch.ones(num_points, device=pc.xyz.device, dtype=torch.float32)
                colors_for_depth_pass_rgb = torch.ones(num_points, 3, device=pc.xyz.device, dtype=torch.float32)
                background_rgbd_depth_pass = torch.zeros(num_cameras_in_batch, 3, device=pc.xyz.device, dtype=torch.float32)

                output_center_rgbd, alpha_center_rgbd, _ = gsplat.rasterization(
                    means=pc.xyz,
                    quats=pc.rotation,
                    scales=scales_for_depth_pass,
                    opacities=opacities_input_for_depth_pass,
                    colors=colors_for_depth_pass_rgb,
                    sh_degree=None,
                    viewmats=viewmats_gsplat,
                    Ks=Ks_gsplat,
                    width=W,
                    height=H,
                    render_mode='RGB+D', 
                    backgrounds=background_rgbd_depth_pass 
                )
                center_point_depth_map_batch = output_center_rgbd[..., 3:4]
                center_point_opacity_map_batch = alpha_center_rgbd
        else:
            # Ensure these are initialized if num_cameras_in_batch is 0
            # Their size should match rendered_depth_batch and rendered_alpha_batch
            if rendered_depth_batch is not None:
                 center_point_depth_map_batch = torch.zeros_like(rendered_depth_batch)
            else: # Should not happen if rendered_depth_batch always exists
                 center_point_depth_map_batch = torch.zeros((0, H, W, 1), device=means_gsplat.device, dtype=means_gsplat.dtype)

            if rendered_alpha_batch is not None:
                 center_point_opacity_map_batch = torch.zeros_like(rendered_alpha_batch)
            else: # Should not happen
                 center_point_opacity_map_batch = torch.zeros((0, H, W, 1), device=means_gsplat.device, dtype=means_gsplat.dtype)


        outputs = {
            "comp_rgb": rendered_image_batch if rendered_image_batch is not None else torch.zeros((num_cameras_in_batch, H, W, 3), device=means_gsplat.device),
            "depth": rendered_depth_batch if rendered_depth_batch is not None else torch.zeros((num_cameras_in_batch, H, W, 1), device=means_gsplat.device),
            "opacity": rendered_alpha_batch if rendered_alpha_batch is not None else torch.zeros((num_cameras_in_batch, H, W, 1), device=means_gsplat.device),
            "normal": normal_map_batch, # Already handled if None
            "center_point_depth": center_point_depth_map_batch,
            "center_point_opacity": center_point_opacity_map_batch,
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
