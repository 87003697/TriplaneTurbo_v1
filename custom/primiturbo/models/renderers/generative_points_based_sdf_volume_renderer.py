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

# # 添加KNN扩展路径
# knn_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "extern")
# if knn_path not in sys.path:
#     sys.path.append(knn_path)
# 
# # 尝试导入CUDA KNN扩展
# try:
#     from knn import knn_search
#     print("成功导入CUDA KNN扩展")
#     HAS_CUDA_KNN = True
# except ImportError:
#     print("CUDA KNN扩展未找到或无法加载，尝试即时编译...")
#     try:
#         import torch.utils.cpp_extension
#         from torch.utils.cpp_extension import load
#         
#         # 获取KNN扩展目录
#         knn_dir = os.path.join(knn_path, "knn")
#         if os.path.exists(knn_dir):
#             # 尝试即时编译
#             sources = [
#                 os.path.join(knn_dir, "ext.cpp"),
#                 os.path.join(knn_dir, "knn.cu"),
#                 os.path.join(knn_dir, "knn_cpu.cpp")
#             ]
#             
#             if all(os.path.exists(s) for s in sources):
#                 cuda_knn = load(
#                     name="cuda_knn",
#                     sources=sources,
#                     verbose=True,
#                     extra_cflags=["-O3"],
#                     extra_cuda_cflags=["-O3"],
#                     with_cuda=torch.cuda.is_available()
#                 )
#                 knn_search = cuda_knn.knn_search
#                 print("成功即时编译并加载CUDA KNN扩展")
#                 HAS_CUDA_KNN = True
#             else:
#                 print("KNN扩展源文件不完整，无法编译")
#                 HAS_CUDA_KNN = False
#         else:
#             print(f"KNN扩展目录不存在: {knn_dir}")
#             HAS_CUDA_KNN = False
#     except Exception as e:
#         print(f"无法加载CUDA KNN扩展: {e}")
#         HAS_CUDA_KNN = False
# 
# 
# # 添加CudaKNNIndex类
# class CudaKNNIndex(object):
#     """使用我们实现的CUDA KNN扩展的KNN索引"""
#     
#     def __init__(self):
#         self.points = None
#         self.points_length = None
#         self.batch_size = 0
#         self.device = None
#         
#     def add(self, x, lengths=None):
#         """添加参考点到索引中
#         
#         Args:
#             x: 形状为[batch_size, num_points, dim]的参考点
#             lengths: 每个batch中有效的点数量，形状为[batch_size]
#         """
#         self.points = x
#         self.batch_size = x.shape[0]
#         self.device = x.device
#         
#         if lengths is None:
#             # 如果未提供长度，则所有点都有效
#             self.points_length = torch.full(
#                 (self.batch_size,), x.shape[1], 
#                 dtype=torch.int64, device=self.device
#             )
#         else:
#             self.points_length = lengths
#             
#     def search(self, query, k, norm = 2, lengths=None):
#         """搜索最近邻
#         
#         Args:
#             query: 形状为[batch_size, num_queries, dim]的查询点
#             k: 要返回的最近邻数量
#             lengths: 每个batch中有效的查询点数量，形状为[batch_size]
#             
#         Returns:
#             distances: 形状为[batch_size, num_queries, k]的距离
#             indices: 形状为[batch_size, num_queries, k]的索引
#         """
#         if not HAS_CUDA_KNN:
#             raise ImportError("CUDA KNN扩展未安装或无法加载")
#             
#         if self.points is None:
#             raise ValueError("必须先调用add方法添加参考点")
#             
#         # 确保查询点和参考点在相同设备上
#         if query.device != self.device:
#             query = query.to(self.device)
#             
#         # 创建查询长度张量
#         if lengths is None:
#             query_lengths = torch.full(
#                 (query.shape[0],), query.shape[1], 
#                 dtype=torch.int64, device=query.device
#             )
#         else:
#             query_lengths = lengths
#             
#         # 使用CUDA KNN扩展
#         distances, indices = knn_search(
#             query,
#             self.points,
#             query_lengths,
#             self.points_length,
#             k,
#             norm
#         )
#         
#         return distances, indices

# # --- Import the official KNN/KDN interfaces --- 
# from ..knn import HAS_CUDA_KNN, CudaKNNIndex
# # If KDN is also needed:
# from ..kdn import HAS_CUDA_KDN, CudaKDNIndex as CudaKDNIndexMahalanobis
# # -----------------------------------------------

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


@threestudio.register("generative_point_based_sdf_volume_renderer")
class GenerativePointBasedSDFVolumeRenderer(NeuSVolumeRenderer):
    @dataclass
    class Config(NeuSVolumeRenderer.Config):
        # the following are from NeuS #########
        num_samples_per_ray: int = 512 # ! 定义为精细采样数量 (Fine Sample Count)
        randomized: bool = True
        eval_chunk_size: int = 320000
        learned_variance_init: float = 0.3
        cos_anneal_end_steps: int = 0
        use_volsdf: bool = False

        near_plane: float = 0.0
        far_plane: float = 1e10

        trainable_variance: bool = False
        # in ['occgrid', 'importance']
        estimator: str = "importance"
        knn_accelerator: Optional[str] = "cuda-knn"

        # for occgrid
        grid_prune: bool = True
        prune_alpha_threshold: bool = True

        # for importance / depth guide coarse samples
        num_samples_per_ray_importance: int = 64 # ! 定义为粗采样数量 (Coarse Sample Count)

        # for balancing the low-res and high-res gradients
        rgb_grad_shrink: float = 1.0

        # for rendering the normal
        normal_direction: str = "camera"  # "front" or "camera" or "world"


    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.variance = LearnedVariance(self.cfg.learned_variance_init, requires_grad=self.cfg.trainable_variance)
        if self.cfg.estimator == "occgrid":
            threestudio.error("Occgrid estimator not supported for generative-space-volsdf-volume-renderer")
            raise NotImplementedError
        elif self.cfg.estimator == "importance":
            self.estimator = ImportanceEstimator()
        else:
            raise NotImplementedError(
                f"Estimator {self.cfg.estimator} not implemented"
            )
        
        assert self.cfg.knn_accelerator in ["none", "cuda-knn"], \
            f"knn_accelerator must be one of ['none', 'cuda-knn'], got {self.cfg.knn_accelerator}"
        if self.cfg.knn_accelerator == "cuda-knn" and not HAS_CUDA_KNN:
                print("WARNING: CUDA KNN extension not installed or cannot be loaded. Using pytorch KNN implementation.")
                self.knn_mode = "none"
        else:
            self.knn_mode = self.cfg.knn_accelerator

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
                space_cache=self._space_cache_acc(space_cache),
                text_embed=text_embed,
                gs_depth=gs_depth,
                **kwargs
            )
        else:
            func = partial(
                self._forward,
                space_cache=self._space_cache_acc(space_cache),
                text_embed=text_embed,
                text_embed_bg = kwargs.pop("text_embed_bg", None),
                gs_depth=gs_depth,
            )
            out = chunk_batch_original(
                func,
                chunk_size=1,
                rays_o=rays_o,
                rays_d=rays_d,
                light_positions=light_positions,
                bg_color=bg_color,
                noise=noise,
                **kwargs
            )
        return out


    def _space_cache_acc(
            self, 
            space_cache_idx: Dict[str, Float[Tensor, "B ..."]]
        ):

        position = space_cache_idx["position"].detach()
        if position.ndim == 2:
            position = position.unsqueeze(0)
            
        # get a cached index
        if self.knn_mode == "cuda-knn":
            index = CudaKNNIndex()
            index.add(position)
        else:
            raise NotImplementedError(f"KNN mode {self.knn_mode} not implemented")
        
        # update the index
        space_cache_idx.update(
            {
                "index": index
            }
        )
        return space_cache_idx
        
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
            gs_depth_flatten = gs_depth.reshape(-1, 1) if gs_depth is not None else None
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
            gs_depth_flatten = gs_depth.reshape(-1, 1) if gs_depth is not None else None
        else:
            raise NotImplementedError(f"rays_o.ndim must be 3 or 4, got {rays_o.ndim}")

        n_rays = rays_o_flatten.shape[0]

        batch_size_space_cache = text_embed.shape[0] if text_embed is not None else batch_size
        
        # num_views_per_batch = batch_size // batch_size_space_cache # Likely not needed here
        if self.cfg.estimator == "importance":
            # ========== 使用 nerfacc ImportanceEstimator 采样 ==========
            threestudio.debug("Using importance sampling.")

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
                    inv_std = self.variance(geo_out["sdf"])

                    if self.cfg.use_volsdf:
                        density:  Float[Tensor, "B Ns"] = volsdf_density(geo_out["sdf"], inv_std).reshape(positions.shape[:2])
                    else:
                        sdf = geo_out["sdf"]
                        estimated_next_sdf = sdf - self.render_step_size * 0.5
                        estimated_prev_sdf = sdf + self.render_step_size * 0.5
                        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
                        next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)
                        p = prev_cdf - next_cdf
                        c = prev_cdf
                        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
                        density = (alpha / self.render_step_size).reshape(positions.shape[:2])
                        
                return density

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[
                        partial(
                            prop_sigma_fn, 
                            proposal_network=self.geometry, 
                            space_cache=space_cache,
                        )
                    ],
                prop_samples=[self.cfg.num_samples_per_ray_importance],
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
        
        
        alpha: Float[Tensor, "Nr 1"] = self.get_alpha(
            geo_out["sdf"], geo_out["normal"], t_dirs, t_intervals
        )

        weights: Float[Tensor, "Nr 1"]
        weights_, _ = nerfacc.render_weight_from_alpha(
            alpha[..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        weights = weights_[..., None]
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
                rai 
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

            out.update({"inv_std": self.variance.inv_std})
        
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
