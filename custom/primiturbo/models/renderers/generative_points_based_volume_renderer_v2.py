from dataclasses import dataclass
from functools import partial
from tqdm import tqdm
import os
import sys
import subprocess

import nerfacc
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.estimators import ImportanceEstimator
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.neus_volume_renderer import NeuSVolumeRenderer
from threestudio.utils.ops import validate_empty_rays, chunk_batch
from .utils import chunk_batch as chunk_batch_custom # a different chunk_batch from threestudio.utils.ops
from threestudio.utils.typing import *
from threestudio.utils.ops import chunk_batch as chunk_batch_original

from threestudio.models.renderers.neus_volume_renderer import volsdf_density
from threestudio.utils.misc import C

from tqdm import tqdm
import math

class LearnedVariance(nn.Module):
    def __init__(self, init_val, requires_grad=True):
        super(LearnedVariance, self).__init__()
        self.register_parameter("_inv_std", nn.Parameter(torch.tensor(init_val), requires_grad=requires_grad))

    @property
    def inv_std(self):
        val = torch.exp(self._inv_std * 10.0)
        return val

    def forward(self, x):
        return torch.ones_like(x) * self.inv_std.clamp(1.0e-6, 1.0e6)


@threestudio.register("generative_point_based_volume_renderer_v2")
class GenerativePointBasedVolumeRendererV2(NeuSVolumeRenderer):
    @dataclass
    class Config(NeuSVolumeRenderer.Config):
        # the following are from NeuS #########
        num_samples_per_ray: int = 512 # ! 定义为精细采样数量 (Fine Sample Count)
        
        randomized: bool = True
        eval_chunk_size: int = 320000
        learned_variance_init: float = 0.3401 # log(30) / 10 = 0.3401, 0.3401 is the most common variance in VolSDF # 0.03 #0.3
        
        near_plane: float = 0.0
        far_plane: float = 1e10

        trainable_variance: bool = False
        
        estimator: str = "importance" # Options: 'importance', 'depth'

        # --- 新增: 渲染模式选择 ---
        rendering_mode: str = "neus" # 'neus', 'volsdf', 或 'nerf'
        # -----------------------

        # for balancing the low-res and high-res gradients
        rgb_grad_shrink: float = 1.0

        # for rendering the normal
        normal_direction: str = "camera"  # "front" or "camera" or "world"

        # for importance / depth guide coarse samples
        num_samples_per_ray_coarse: int = 64 # ! 定义为粗采样数量 (Coarse Sample Count)
        
        # --- 新增: 深度引导区间比例 (重新添加为可配置) ---
        depth_guide_interval_ratio: float = 0.1 # 粗采样区间比例 (Used when estimator='depth')
        depth_guide_interval_ratio_fine: float = 0.01 # 精细采样区间比例 (Used when estimator='depth')
        depth_guide_interval_type: str = "add" # Options: 'add', 'mul'

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.variance = LearnedVariance(self.cfg.learned_variance_init, requires_grad=self.cfg.trainable_variance)
        
        # Updated estimator check
        if self.cfg.estimator == "importance":
            self.estimator = ImportanceEstimator()
        elif self.cfg.estimator == "depth":
            # No specific estimator object needed for depth, logic is in _forward
            # Need to ensure sampling counts are still calculated
            pass 
        else:
            raise NotImplementedError(
                f"Estimator {self.cfg.estimator} not implemented. Options are 'importance', 'depth'."
            )
        
        # --- Removed KNN configuration from Renderer ---
        # assert self.cfg.knn_accelerator in ["none", "cuda-knn"], \
        #     f"knn_accelerator must be one of ['none', 'cuda-knn'], got {self.cfg.knn_accelerator}"
        # if self.cfg.knn_accelerator == "cuda-knn" and not HAS_CUDA_KNN:
        #         print("WARNING: CUDA KNN extension not installed or cannot be loaded. Using pytorch KNN implementation.")
        #         self.knn_mode = "none"
        # else:
        #     self.knn_mode = self.cfg.knn_accelerator
        # --------------------------------------------

        # --- 新增: 校验渲染模式 ---
        assert self.cfg.rendering_mode in ["neus", "volsdf", "nerf"], \
            f"rendering_mode must be one of ['neus', 'volsdf', 'nerf'], got {self.cfg.rendering_mode}"
        # ------------------------

        # --- 新增: 计算采样数 (Used by both modes currently) ---
        self.num_samples_coarse = self.cfg.num_samples_per_ray_coarse # Coarse = importance 参数
        self.num_samples_fine = self.cfg.num_samples_per_ray             # Fine = per_ray 参数
        self.total_samples = self.num_samples_coarse + self.num_samples_fine  # Total = Coarse + Fine
        threestudio.debug(f"Sampling Config: Coarse={self.num_samples_coarse}, Fine={self.num_samples_fine}, Total={self.total_samples}")
        # -----------------------------

        assert self.cfg.depth_guide_interval_type in ["add", "mul"], \
            f"depth_guide_interval_type must be one of ['add', 'mul'], got {self.cfg.depth_guide_interval_type}"

    def forward(
        self,
        rays_o: Union[Float[Tensor, "B H W 3"], Float[Tensor, "B N 3"]],
        rays_d: Union[Float[Tensor, "B H W 3"], Float[Tensor, "B N 3"]],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        noise: Optional[Float[Tensor, "B C"]] = None,
        space_cache: Optional[Union[List[Float[Tensor, "B ..."]], Dict[str, Float[Tensor, "B ..."]]]] = None,
        text_embed: Optional[Float[Tensor, "B C"]] = None,
        gs_depth: Optional[Union[Float[Tensor, "B H W 1"], Float[Tensor, "B N 1"]]] = None,
        camera_distances: Optional[Float[Tensor, "B"]] = None,
        c2w: Optional[Float[Tensor, "B 4 4"]] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:

        if type(space_cache) == list:
            assert len(space_cache) == len(text_embed), "space_cache and text_embed must have the same length"
        elif type(space_cache) == dict:
            assert "position" in space_cache, "space_cache must contain the key 'position'"
            assert len(space_cache["position"]) == len(text_embed), "space_cache['position'] and text_embed must have the same length"
        else:
            raise NotImplementedError("the provided space_cache is not supported for generative_point_based_sdf_volume_renderer")
        
        if self.training:
            out = self._forward(
                rays_o=rays_o,
                rays_d=rays_d,
                light_positions=light_positions,
                bg_color=bg_color,
                noise=noise,
                space_cache=space_cache,
                text_embed=text_embed,
                gs_depth=gs_depth,
                camera_distances=camera_distances,
                c2w=c2w,
                **kwargs
            )
        else:
            # Prepare partial function with remaining kwargs
            func = partial(
                self._forward,
                space_cache=space_cache,
                text_embed=text_embed,
                # Pass other kwargs received by forward
                # (gs_depth and camera_distances will be passed by chunk_batch)
                **kwargs
            )

            # Pass gs_depth and camera_distances explicitly to chunk_batch_original
            out = chunk_batch_original(
                func,
                chunk_size=1,
                rays_o=rays_o,
                rays_d=rays_d,
                light_positions=light_positions,
                bg_color=bg_color,
                noise=noise,
                gs_depth=gs_depth,
                camera_distances=camera_distances,
                c2w=c2w # Explicitly pass c2w here
            )
            # === END MODIFICATION ===
        return out

    def _forward(
        self,
        rays_o: Union[Float[Tensor, "B H W 3"], Float[Tensor, "B N 3"]],
        rays_d: Union[Float[Tensor, "B H W 3"], Float[Tensor, "B N 3"]],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        noise: Optional[Float[Tensor, "B C"]] = None,
        space_cache: Optional[Union[Float[Tensor, "B ..."], Dict, List]] = None,
        text_embed: Optional[Float[Tensor, "B C"]] = None,
        camera_distances: Optional[Float[Tensor, "B"]] = None,
        c2w: Optional[Float[Tensor, "B 4 4"]] = None,
        gs_depth: Optional[Union[Float[Tensor, "B H W 1"], Float[Tensor, "B N 1"]]] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        

        if rays_o.ndim == 4:
            batch_size, height, width = rays_o.shape[:3]
            rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
            rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
            light_positions_flatten: Float[Tensor, "Nr 3"] = (
                light_positions.reshape(-1, 1, 1, 3)
                .expand(-1, height, width, -1)
                .reshape(-1, 3)
            )
            render_shape = (height, width)
        elif rays_o.ndim == 3:
            raise NotImplementedError("Not debuged yet")
            batch_size, num_rays, _ = rays_o.shape[:3]
            rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
            rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
            light_positions_flatten: Float[Tensor, "Nr 3"] = (
                light_positions.reshape(-1, 1, 3)
                .expand(-1, num_rays, -1)
                .reshape(-1, 3)
            )
            render_shape = (num_rays,)
        else:
            raise NotImplementedError(f"rays_o.ndim must be 3 or 4, got {rays_o.ndim}")

        n_rays = rays_o_flatten.shape[0]

        batch_size_space_cache = text_embed.shape[0] if text_embed is not None else batch_size
        # num_views_per_batch = batch_size // batch_size_space_cache # Likely not needed here

        # --- 采样逻辑 --- Updated condition ---
        # Use estimator config instead of use_depth_guide
        if self.cfg.estimator == 'depth':
            assert gs_depth is not None, "gs_depth must not be None when estimator='depth'"
            gs_depth_flatten = gs_depth.reshape(-1, 1)
            device = rays_o.device # Get device
            stratified = self.cfg.randomized if self.training else False # Define stratified flag
            # ========== 使用深度引导 + 均匀采样 Fallback ==========
            threestudio.debug("Using depth guided sampling with uniform fallback (estimator='depth').")

            # 1. 识别有效和无效光线
            valid_depth_mask = torch.isfinite(gs_depth_flatten) & (gs_depth_flatten > 1e-5)
            valid_depth_mask_flat = valid_depth_mask.squeeze() # [Nr]
            invalid_depth_mask_flat = ~valid_depth_mask_flat

            all_ray_indices = torch.arange(n_rays, device=device)
            valid_ray_indices = all_ray_indices[valid_depth_mask_flat]
            invalid_ray_indices = all_ray_indices[invalid_depth_mask_flat]

            num_valid_rays = valid_ray_indices.shape[0]
            num_invalid_rays = invalid_ray_indices.shape[0]

            # --- 2. 处理有效光线 (深度引导采样) ---
            if num_valid_rays > 0:
                valid_depth = gs_depth_flatten[valid_depth_mask] # [N_valid, 1]
                # Ensure valid_depth has the correct shape [N_valid, 1]
                if valid_depth.ndim == 1:
                    valid_depth = valid_depth.unsqueeze(1) # Add dimension if missing

                # 计算采样区间
                if self.cfg.depth_guide_interval_type == "add":
                    interval_coarse = valid_depth * self.cfg.depth_guide_interval_ratio # 使用配置值
                    t_min_coarse = (valid_depth - interval_coarse).clamp(min=self.cfg.near_plane)
                    t_max_coarse = (valid_depth + interval_coarse).clamp(max=self.cfg.far_plane)

                    interval_fine = valid_depth * self.cfg.depth_guide_interval_ratio_fine # 使用配置值
                    t_min_fine = (valid_depth - interval_fine).clamp(min=self.cfg.near_plane)
                    t_max_fine = (valid_depth + interval_fine).clamp(max=self.cfg.far_plane)
                
                elif self.cfg.depth_guide_interval_type == "mul":
                    t_min_coarse = ((1 - self.cfg.depth_guide_interval_ratio) * valid_depth).clamp(min=self.cfg.near_plane)
                    t_max_coarse = ((1 + self.cfg.depth_guide_interval_ratio) * valid_depth).clamp(max=self.cfg.far_plane)

                    t_min_fine = ((1 - self.cfg.depth_guide_interval_ratio_fine) * valid_depth).clamp(min=self.cfg.near_plane)
                    t_max_fine = ((1 + self.cfg.depth_guide_interval_ratio_fine) * valid_depth).clamp(max=self.cfg.far_plane)

                # 生成采样点 (t 值)
                t_samples_coarse = torch.empty(num_valid_rays, self.num_samples_coarse, device=device)
                t_samples_fine = torch.empty(num_valid_rays, self.num_samples_fine, device=device)

                if stratified:
                    if self.num_samples_coarse > 0:
                        u_coarse = torch.linspace(0., 1., steps=self.num_samples_coarse, device=device)
                        u_coarse = u_coarse.expand(num_valid_rays, self.num_samples_coarse)
                        u_coarse = u_coarse + torch.rand(num_valid_rays, self.num_samples_coarse, device=device) / self.num_samples_coarse
                        t_samples_coarse = t_min_coarse + (t_max_coarse - t_min_coarse) * u_coarse
                    if self.num_samples_fine > 0:
                        u_fine = torch.linspace(0., 1., steps=self.num_samples_fine, device=device)
                        u_fine = u_fine.expand(num_valid_rays, self.num_samples_fine)
                        u_fine = u_fine + torch.rand(num_valid_rays, self.num_samples_fine, device=device) / self.num_samples_fine
                        t_samples_fine = t_min_fine + (t_max_fine - t_min_fine) * u_fine
                else:
                    if self.num_samples_coarse > 0:
                        t_samples_coarse_base = torch.linspace(0., 1., steps=self.num_samples_coarse, device=device)
                        # MODIFIED: Explicitly expand both tensors before multiplication
                        interval = (t_max_coarse - t_min_coarse).expand(-1, self.num_samples_coarse) # Shape [N_valid, N_coarse]
                        t_samples_coarse_expanded = t_samples_coarse_base.expand(num_valid_rays, -1) # Shape [N_valid, N_coarse]
                        t_samples_coarse = t_min_coarse.expand(-1, self.num_samples_coarse) + interval * t_samples_coarse_expanded
                    if self.num_samples_fine > 0:
                        t_samples_fine_base = torch.linspace(0., 1., steps=self.num_samples_fine, device=device)
                        # MODIFIED: Explicitly expand both tensors before multiplication
                        interval_fine = (t_max_fine - t_min_fine).expand(-1, self.num_samples_fine) # Shape [N_valid, N_fine]
                        t_samples_fine_expanded = t_samples_fine_base.expand(num_valid_rays, -1) # Shape [N_valid, N_fine]
                        t_samples_fine = t_min_fine.expand(-1, self.num_samples_fine) + interval_fine * t_samples_fine_expanded

                t_samples_guided = torch.cat([t_samples_coarse, t_samples_fine], dim=-1) # [N_valid, total_samples]
                t_samples_guided, _ = torch.sort(t_samples_guided, dim=-1)

                # 计算 t_starts, t_ends
                t_starts_guided = t_samples_guided[..., :-1].reshape(-1) # [N_valid * (total_samples-1)]
                t_ends_guided = t_samples_guided[..., 1:].reshape(-1)   # [N_valid * (total_samples-1)]

                # 生成光线索引 (指向原始索引)
                ray_indices_guided = valid_ray_indices.unsqueeze(-1).expand(-1, self.total_samples - 1).reshape(-1)

                # 清理内存
                del t_samples_coarse, t_samples_fine, t_samples_guided, valid_depth
                del t_min_coarse, t_max_coarse, t_min_fine, t_max_fine
                if stratified: del u_coarse, u_fine
                torch.cuda.empty_cache()
            else: # 没有有效的深度引导光线
                t_starts_guided = torch.empty((0,), dtype=torch.float32, device=device)
                t_ends_guided = torch.empty((0,), dtype=torch.float32, device=device)
                ray_indices_guided = torch.empty((0,), dtype=torch.int64, device=device)

            # --- 3. 处理无效光线 (均匀随机采样) ---
            if num_invalid_rays > 0:
                # 生成 [0, 1) 之间的随机数
                t_rand = torch.rand(num_invalid_rays, self.total_samples, device=device)
                if stratified:
                    # 分层采样：在每个区间内随机
                    t_rand = (t_rand + torch.arange(self.total_samples, device=device)[None, :]) / self.total_samples
                # else: # 均匀采样 - 可以直接从linspace生成样本点
                #    pass # t_rand can be linspace below

                # 映射到 [near, far]
                if stratified:
                    t_samples_uniform = self.cfg.near_plane + (self.cfg.far_plane - self.cfg.near_plane) * t_rand
                else:
                    # For uniform, generate points directly using linspace
                    t_samples_uniform = torch.linspace(self.cfg.near_plane, self.cfg.far_plane, steps=self.total_samples, device=device)
                    t_samples_uniform = t_samples_uniform.expand(num_invalid_rays, self.total_samples)

                t_samples_uniform, _ = torch.sort(t_samples_uniform, dim=-1) # Sort needed even for linspace if batch > 1

                # 计算 t_starts, t_ends
                t_starts_uniform = t_samples_uniform[..., :-1].reshape(-1) # [N_invalid * (total_samples-1)]
                t_ends_uniform = t_samples_uniform[..., 1:].reshape(-1)   # [N_invalid * (total_samples-1)]

                # 生成光线索引 (指向原始索引)
                ray_indices_uniform = invalid_ray_indices.unsqueeze(-1).expand(-1, self.total_samples - 1).reshape(-1)

                del t_rand, t_samples_uniform
                torch.cuda.empty_cache()
            else: # 没有无效深度光线
                t_starts_uniform = torch.empty((0,), dtype=torch.float32, device=device)
                t_ends_uniform = torch.empty((0,), dtype=torch.float32, device=device)
                ray_indices_uniform = torch.empty((0,), dtype=torch.int64, device=device)

            # --- 4. 合并结果 ---
            ray_indices = torch.cat([ray_indices_guided, ray_indices_uniform], dim=0)
            t_starts_ = torch.cat([t_starts_guided, t_starts_uniform], dim=0)
            t_ends_ = torch.cat([t_ends_guided, t_ends_uniform], dim=0)

            # 清理中间变量
            del ray_indices_guided, ray_indices_uniform, t_starts_guided, t_starts_uniform, t_ends_guided, t_ends_uniform
            torch.cuda.empty_cache()

        elif self.cfg.estimator == "importance":
            # ========== 使用 nerfacc ImportanceEstimator 采样 ==========
            threestudio.debug("Using importance sampling (estimator='importance').")

            def prop_sigma_fn(
                t_starts: Float[Tensor, "Nr Ns"],
                t_ends: Float[Tensor, "Nr Ns"],
                proposal_network,
                space_cache: Float[Tensor, "B ..."],
            ):
                
                t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                positions: Float[Tensor, "Nr Ns 3"] = (
                    t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                )
                with torch.no_grad():
                    geo_out = proposal_network(
                        positions.view(batch_size_space_cache, -1, 3),
                        space_cache=space_cache,
                        output_normal=False,
                    )
                    
                    if self.cfg.rendering_mode == 'neus':
                        inv_std = self.variance(geo_out["sdf"])
                        # Calculate density using NeuS formulation (sigmoid difference)
                        sdf = geo_out["sdf"]
                        estimated_next_sdf = sdf - self.render_step_size * 0.5
                        estimated_prev_sdf = sdf + self.render_step_size * 0.5
                        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
                        next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)
                        p = prev_cdf - next_cdf
                        c = prev_cdf
                        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
                        # Convert alpha to density for consistency in nerfacc sampling?
                        # Let's stick to the original logic first if possible
                        density = (alpha / self.render_step_size).reshape(positions.shape[:2])
                    elif self.cfg.rendering_mode == 'volsdf':
                         inv_std = self.variance(geo_out["sdf"])
                         # Calculate density using VolSDF formulation
                         density:  Float[Tensor, "B Ns"] = volsdf_density(geo_out["sdf"], inv_std).reshape(positions.shape[:2])
                    elif self.cfg.rendering_mode == 'nerf':
                        if "density" not in geo_out:
                             raise ValueError("Geometry network did not output 'density' required for NeRF rendering mode in proposal network.")
                        # 确保密度非负, 并且调整为正确的形状
                        density = F.relu(geo_out["density"]).reshape(positions.shape[:2]) # 修改: 添加 reshape
                    else:
                        raise NotImplementedError(f"Rendering mode {self.cfg.rendering_mode} not implemented in prop_sigma_fn.")

                return density

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[
                        partial(
                            prop_sigma_fn, 
                            proposal_network=self.geometry, 
                            space_cache=space_cache,
                        )
                    ],
                prop_samples=[self.cfg.num_samples_per_ray_coarse],
                num_samples=self.cfg.num_samples_per_ray,
                n_rays=n_rays,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                sampling_type="uniform",
                stratified=self.cfg.randomized if self.training else False
            )
            ray_indices = (
                torch.arange(n_rays, device=rays_o_flatten.device)
                .unsqueeze(-1)
                .expand(-1, t_starts_.shape[1])
            )
            ray_indices = ray_indices.flatten()
            t_starts_ = t_starts_.flatten()
            t_ends_ = t_ends_.flatten()
        else:
            raise NotImplementedError
        
        # the following are from NeuS #########
        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        # perform the rendering
        geo_out = self.geometry(
            positions.view(batch_size_space_cache, -1, 3),
            space_cache=space_cache,
            output_normal=True,
        )
        rgb_fg_all: Float[Tensor, "B Ns Nc"] = self.material(
            viewdirs=t_dirs,
            positions=positions,
            light_positions=t_light_positions,
            **geo_out,
            **kwargs
        )

        # background
        comp_rgb_bg: Float[Tensor, "B H W Nc"]
        if hasattr(self.background, "enabling_hypernet") and self.background.enabling_hypernet:
            text_embed_bg = text_embed if "text_embed_bg" not in kwargs else kwargs["text_embed_bg"]
            comp_rgb_bg: Float[Tensor, "B H W Nc"] = self.background(
                dirs=rays_d,
                text_embed=text_embed_bg,
            )
        else:
            comp_rgb_bg: Float[Tensor, "B H W Nc"] = self.background(dirs=rays_d)

        if self.rgb_grad_shrink != 1.0:
            # shrink the gradient of rgb_fg_all
            # this is to balance the low-res and high-res gradients
            rgb_fg_all: Float[Tensor, "Nr Nc"] = self.rgb_grad_shrink * rgb_fg_all + (1.0 - self.rgb_grad_shrink) * rgb_fg_all.detach()
        
        
        weights: Float[Tensor, "Nr 1"]
        if self.cfg.rendering_mode == 'neus':
            # NeuS uses alpha calculated via get_alpha (sigmoid difference)
            if "sdf" not in geo_out or "normal" not in geo_out:
                 raise ValueError("Geometry network did not output 'sdf' and 'normal' required for NeuS rendering mode.")
            alpha: Float[Tensor, "Nr 1"] = self.get_alpha(
                geo_out["sdf"], geo_out["normal"], t_dirs, t_intervals
            )
            weights_, _ = nerfacc.render_weight_from_alpha(
                alphas=alpha[..., 0],
                ray_indices=ray_indices,
                n_rays=n_rays,
            )
            weights = weights_[..., None]
        elif self.cfg.rendering_mode == 'volsdf':
            # Following original NeuSVolumeRenderer logic as requested:
            # Use get_alpha (NeuS alpha) and render_weight_from_alpha even for volsdf mode
            if "sdf" not in geo_out or "normal" not in geo_out:
                 raise ValueError("Geometry network did not output 'sdf' and 'normal' required for NeuS/VolSDF rendering mode (following original NeuS logic).")
            alpha: Float[Tensor, "Nr 1"] = self.get_alpha(
                geo_out["sdf"], geo_out["normal"], t_dirs, t_intervals
            )
            weights_, _ = nerfacc.render_weight_from_alpha(
                alphas=alpha[..., 0],
                ray_indices=ray_indices,
                n_rays=n_rays,
            )
            weights = weights_[..., None]
        elif self.cfg.rendering_mode == 'nerf':
            if "density" not in geo_out:
                raise ValueError("Geometry network did not output 'density' required for NeRF rendering mode.")
            # 确保密度非负, 并调整形状以匹配 nerfacc 要求
            density = F.relu(geo_out["density"][..., 0]) # [N_pts]
            weights_, trans_, _ = nerfacc.render_weight_from_density(
                t_starts=t_starts_, # 使用未加 [..., None] 维度的 t_starts
                t_ends=t_ends_,     # 使用未加 [..., None] 维度的 t_ends
                sigmas=density,
                ray_indices=ray_indices,
                n_rays=n_rays,
            )
            weights = weights_[..., None] # [N_pts, 1]
        else:
             raise NotImplementedError(f"Rendering mode {self.cfg.rendering_mode} not implemented.")

        # The following nerfacc calls use weights directly, compatible with both modes
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )

        t_depth = depth[ray_indices]
        z_variance: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=(t_positions - t_depth) ** 2,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

        if bg_color is None:
            bg_color = comp_rgb_bg
            bg_color = bg_color.reshape(math.prod([batch_size, *render_shape]), -1)
        else:
            bg_color = bg_color.reshape(n_rays, -1)

        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)

        out = {
            "comp_rgb": comp_rgb.view(batch_size, *render_shape, -1),
            "comp_rgb_fg": comp_rgb_fg.view(batch_size, *render_shape, -1),
            "comp_rgb_bg": comp_rgb_bg.view(batch_size, *render_shape, -1),
            "opacity": opacity.view(batch_size, *render_shape, 1),
            "depth": depth.view(batch_size, *render_shape, 1),
            "z_variance": z_variance.view(batch_size, *render_shape, 1),
        }

        if camera_distances is not None:
            cam_dist_shape = (-1,) + (1,) * len(render_shape) + (1,)
            cam_dist_view = camera_distances.view(cam_dist_shape)

            range_offset = torch.sqrt(torch.tensor(3.0, device=cam_dist_view.device))
            far = cam_dist_view + range_offset
            near = torch.clamp(cam_dist_view - range_offset, min=1e-5)

            depth_blend = out["depth"] * out["opacity"] + (1.0 - out["opacity"]) * far
            disparity_norm = (far - depth_blend) / (far - near)
            disparity_norm = torch.clamp(disparity_norm, 0.0, 1.0)

            out.update(
                {
                    "disparity": disparity_norm.view(batch_size, *render_shape, 1),
                }
            )
        else:
             out["disparity"] = torch.zeros_like(out["depth"])

        if "normal" in geo_out:
            comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                weights[..., 0],
                values=geo_out["normal"],
                ray_indices=ray_indices,
                n_rays=n_rays,
            )

            comp_normal = F.normalize(comp_normal, dim=-1)
            out.update(
                {
                    "comp_normal": comp_normal.view(batch_size, *render_shape, 3),
                }
            )

            if self.cfg.normal_direction == "camera":
                # for compatibility with RichDreamer #############
                bg_normal = 0.5 * torch.ones_like(comp_normal)
                bg_normal[:, 2] = 1.0
                bg_normal_white = torch.ones_like(comp_normal)

                w2c: Float[Tensor, "B 4 4"] = torch.inverse(c2w)
                rot: Float[Tensor, "B 3 3"] = w2c[:, :3, :3]
                comp_normal_world_reshaped = comp_normal.view(batch_size, -1, 3)
                comp_normal_cam = torch.bmm(comp_normal_world_reshaped, rot.permute(0, 2, 1))

                # flip_mat = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32, device=comp_normal_cam.device)
                flip_mat = torch.eye(3, dtype=torch.float32, device=comp_normal_cam.device)
                flip_mat[0, 0] = -1 # Flip X axis, consistent with generative_space_sdf_volume_renderer
                comp_normal_cam = torch.bmm(comp_normal_cam, flip_mat.unsqueeze(0).expand(batch_size, -1, -1))
                comp_normal_cam = comp_normal_cam.view(-1, 3)

                comp_normal_cam_vis = (comp_normal_cam + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal
                comp_normal_cam_vis_white = (comp_normal_cam + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal_white

                out.update(
                    {
                        "comp_normal_cam_vis": comp_normal_cam_vis.view(batch_size, *render_shape, 3),
                        "comp_normal_cam_vis_white": comp_normal_cam_vis_white.view(batch_size, *render_shape, 3),
                    }
                )
            elif self.cfg.normal_direction == "front":
                raise NotImplementedError
                # # for compatibility with Wonder3D and Era3D #############
                # bg_normal_white = torch.ones_like(comp_normal)

                # # convert_normal_to_cam_space of the front view
                # c2w_front = c2w[0::num_views_per_batch].repeat_interleave(num_views_per_batch, dim=0)
                # w2c_front: Float[Tensor, "B 4 4"] = torch.inverse(c2w_front)                
                # rot: Float[Tensor, "B 3 3"] = w2c_front[:, :3, :3]
                # comp_normal_front = comp_normal.view(batch_size, -1, 3) @ rot.permute(0, 2, 1)

                # # the following is not necessary for Wonder3D and Era3D
                # # flip_x = torch.eye(3, device=comp_normal_front.device) #  pixel space flip axis so we need built negative y-axis normal
                # # flip_x[0, 0] = -1
                # # comp_normal_front = comp_normal_front @ flip_x[None, :, :]
                
                # comp_normal_front = comp_normal_front.view(-1, 3) # reshape back to (Nr, 3)
                # comp_normal_front_vis_white = (comp_normal_front + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal_white

                # out.update(
                #     {
                #         "comp_normal_cam_vis_white": comp_normal_front_vis_white.view(batch_size, *render_shape, 3),
                #     }
                # )
            elif self.cfg.normal_direction == "world":
                bg_normal_white = torch.ones_like(comp_normal)
                comp_normal_world_vis_white = (comp_normal + 1.0) / 2.0 * opacity + (1.0 - opacity) * bg_normal_white
                out["comp_normal_cam_vis_white"] = comp_normal_world_vis_white.view(batch_size, *render_shape, 3)

        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "points": positions,
                    **geo_out,
                }
            )

            # --- 修改: 条件化输出 inv_std ---
            # inv_std is relevant for both neus and volsdf modes
            if self.cfg.rendering_mode in ['neus', 'volsdf']:
                out.update({"inv_std": self.variance.inv_std})
            # -----------------------------
        
        return out
    
    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        self.rgb_grad_shrink = C(
            self.cfg.rgb_grad_shrink, epoch, global_step
        )

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        if hasattr(self.geometry, "train"):
            self.geometry.train(mode)
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        if hasattr(self.geometry, "eval"):
            self.geometry.eval()
        return super().eval()
