import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.mesh import Mesh
from threestudio.utils.misc import broadcast, get_rank, C
from threestudio.utils.typing import *

from threestudio.utils.ops import get_activation
from threestudio.models.networks import get_encoding, get_mlp

from custom.triplaneturbo.models.geometry.utils import contract_to_unisphere_custom, sample_from_planes
from einops import rearrange

# Custom autograd function for differentiable indexing
class DifferentiableIndexing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, indices):
        # Save tensors needed for backward
        ctx.save_for_backward(indices)
        ctx.input_size = input_tensor.size(0)
        ctx.feat_size = input_tensor.size(1) if input_tensor.dim() > 1 else 1
        
        # Simple indexing in forward pass
        return input_tensor[indices]
    
    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        input_grad = torch.zeros(ctx.input_size, ctx.feat_size, device=grad_output.device, dtype=grad_output.dtype)
        
        # 计算每个索引的出现次数
        index_counts = torch.bincount(indices, minlength=ctx.input_size)
        index_counts = torch.clamp(index_counts, min=1).unsqueeze(1)  # 避免除零
        
        # 梯度累加
        input_grad.index_add_(0, indices, grad_output)
        
        # 按出现次数归一化
        input_grad /= index_counts
        
        return input_grad, None
            
@threestudio.register("few-step-one-plane-stable-diffusion-v2")
class FewStepOnePlaneStableDiffusionV2(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_feature_dims: int = 3
        space_generator_config: dict = field(
            default_factory=lambda: {
                "pretrained_model_name_or_path": "stable-diffusion-2-1-base",
                "training_type": "lora",
                "output_dim": 35, # 3 + 32
                "gradient_checkpoint": False,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 2, 
            }
        )
        backbone: str = "few_step_one_plane_stable_diffusion" #TODO: change to few_step_few_plane_stable_diffusion
        normal_type: Optional[
            str
        ] = "analytic"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']

        scaling_activation: str = "exp-0.1" # in ["exp-0.1", "sigmoid", "exp", "softplus"]
        opacity_activation: str = "sigmoid-0.1" # in ["sigmoid-0.1", "sigmoid", "sigmoid-mipnerf", "softplus"]
        rotation_activation: str = "normalize" # in ["normalize"]
        color_activation: str = "sigmoid-mipnerf" # in ["scale_-11_01", "sigmoid-mipnerf"]
        position_activation: str = "none" # in ["none"]
        
        point_grad_shrink_avarage: bool = False # whether average the gradient w.r.t. the points given the retrieved times
        

    def configure(self) -> None:
        super().configure()

        print("The current device is: ", self.device)
        
        # set up the space generator
        if self.cfg.backbone == "few_step_one_plane_stable_diffusion":
            from ...extern.few_step_one_plane_sd_modules import FewStepOnePlaneStableDiffusion as Generator
            self.space_generator = Generator(self.cfg.space_generator_config)
        else:
            raise ValueError(f"Unknown backbone {self.cfg.backbone}")
        
        
        input_dim = self.space_generator.output_dim - 3 # 3 for position, 
        self.sdf_network = get_mlp(
            n_input_dims=input_dim + 3, # 3 for direction
            n_output_dims=1,
            config = self.cfg.mlp_network_config
        )
        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                n_input_dims=input_dim + 3, # 3 for direction
                n_output_dims=self.cfg.n_feature_dims,
                config = self.cfg.mlp_network_config
            )
        
        # self.scaling_activation = get_activation(self.cfg.scaling_activation)
        # self.opacity_activation = get_activation(self.cfg.opacity_activation)
        # self.rotation_activation = get_activation(self.cfg.rotation_activation)
        self.color_activation = get_activation(self.cfg.color_activation)
        self.position_activation = get_activation(self.cfg.position_activation)

    def initialize_shape(self) -> None:
        # not used
        pass

    def denoise(
        self,
        noisy_input: Any,
        text_embed: Float[Tensor, "B C"],
        timestep
    ) -> Any:
        output = self.space_generator.forward_denoise(
            text_embed = text_embed,
            noisy_input = noisy_input,
            t = timestep
        )
        return output
    
    def decode(
        self,
        latents: Any,
    ) -> Any:
        triplane = self.space_generator.forward_decode(
            latents = latents
        )
        return triplane

    def parse(
        self,
        triplane: Float[Tensor, "B 3 C//3 H W"],
    ) -> List[Dict[str, Float[Tensor, "..."]]]:
        B, _, C, H, W = triplane.shape
        pc_list = []
        for i in range(B):
            pc_list.append(
                {
                    "position": self.position_activation(
                        rearrange(
                            triplane[i, :, 0:3, :, :],
                            "N C H W -> (N H W) C"
                            )
                        ), # plus center
                    "feature": rearrange(
                            triplane[i, :, 3:, :, :], 
                            "N C H W -> (N H W) C"
                        ),
                }
            )
        return pc_list


    def _interpolate_encodings(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache_position: Float[Tensor, "*N 3"],
        space_cache_feature_geo: Float[Tensor, "*N C"],
        space_cache_feature_tex: Float[Tensor, "*N C"],
        index: Optional[Callable] = None,
        only_geo: bool = False,
    ):
        assert "index" is not None, "Index is not found in space_cache"
        distances, indices = index.search(
            points.detach(),
            k=8
        )
        indexing_func = DifferentiableIndexing.apply if self.cfg.point_grad_shrink_avarage else lambda x, y: x[y]
        if True:
            if space_cache_position.ndim == 2:  # [N, 3]
                indices_flat = indices.squeeze(0)  # Shape: [num_queries, k]
                num_queries, k = indices_flat.shape

                # differentiable indexing, 
                neighbor_position = indexing_func(
                    space_cache_position, 
                    indices_flat
                ).reshape(num_queries, k, 3)  # [num_queries, k, 3]
                neighbor_feature_geo = indexing_func(
                    space_cache_feature_geo, 
                    indices_flat
                ).reshape(num_queries, k, -1)  # [num_queries, k, feature_dim]
            else:
                raise NotImplementedError("Only support [N, 3] for now.")
        
        if only_geo:
            return neighbor_position, neighbor_feature_geo, None
        else:
            if space_cache_feature_tex.ndim == 2:
                neighbor_feature_tex = indexing_func(
                    space_cache_feature_tex, 
                    indices_flat
                ).reshape(num_queries, k, -1)  # [num_queries, k, feature_dim] 
            else:
                raise NotImplementedError("Only support [N, C] for now.")
            return neighbor_position, neighbor_feature_geo, neighbor_feature_tex
        

    def interpolate_encodings(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Dict[str, Any],
        only_geo: bool = False,
    ):
        
        # get the neighbor position, feature, and tex
        neighbor_position: Float[Tensor, "*N K 3"]
        neighbor_feature_geo: Float[Tensor, "*N K C"]
        neighbor_feature_tex: Float[Tensor, "*N K C"]
        neighbor_position, neighbor_feature_geo, neighbor_feature_tex = self._interpolate_encodings(
            points,
            space_cache["position"],
            space_cache["feature_geo"] if "feature_geo" in space_cache else space_cache["feature"],
            space_cache["feature_tex"] if "feature_tex" in space_cache else space_cache["feature"],
            index = space_cache["index"] if "index" in space_cache else None,
            only_geo = only_geo
        )

        # Recalculate the distances to enable gradient flow
        # This is crucial for gradient to flow back to input points
        distances: Float[Tensor, "*N K"] = torch.norm(
            neighbor_position - points.view(-1, 1, 3),
            dim=-1,
            p=2
        )

        # Now perform interpolation based on distances
        # Convert distances to weights using inverse distance weighting
        weights: Float[Tensor, "*N K"] = 1.0 / (distances + 1e-8)  # Add small epsilon to prevent division by zero
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize weights
        
        if True:
            # Apply weighted average to get interpolated features and positions
            if neighbor_position.ndim == 3:  # [num_queries, k, dim]
                # interpolated_position: Float[Tensor, "*N 3"] = torch.sum(neighbor_position * weights.unsqueeze(-1), dim=1)
                interpolated_feature_geo: Float[Tensor, "*N C"] = torch.sum(neighbor_feature_geo * weights.unsqueeze(-1), dim=1)
                interpolated_feature_geo = torch.cat(
                    [
                        interpolated_feature_geo, 
                        torch.sum(
                            (points.view(-1, 1, 3) - neighbor_position) * weights.unsqueeze(-1), 
                            dim=1
                        )
                    ], 
                    dim=-1
                ) # [num_queries, k, feature_dim + 3]
            else:
                raise NotImplementedError("Only support [num_queries, k, dim] for now.")
                # [B, num_queries, k, dim]
                # # interpolated_position = torch.sum(neighbor_position * weights.unsqueeze(-1), dim=2)  # [B, num_queries, 3]
                # interpolated_feature_geo = torch.sum(neighbor_feature_geo * weights.unsqueeze(-1), dim=2)  # [B, num_queries, feature_dim]
                # if not only_geo:
                #     interpolated_feature_tex = torch.sum(neighbor_feature_tex * weights.unsqueeze(-1), dim=2)  # [B, num_queries, feature_dim]

        if only_geo:
            return interpolated_feature_geo, None
        else:
            # Apply weighted average to get interpolated features and positions
            if neighbor_position.ndim == 3:
                interpolated_feature_tex: Float[Tensor, "*N C"] = torch.sum(neighbor_feature_tex * weights.unsqueeze(-1), dim=1)
                interpolated_feature_tex = torch.cat(
                    [
                        interpolated_feature_tex, 
                        torch.sum(
                            (points.view(-1, 1, 3) - neighbor_position) * weights.unsqueeze(-1), 
                            dim=1
                        )
                    ], 
                    dim=-1
                ) # [num_queries, k, feature_dim + 3]
            else:
                raise NotImplementedError("Only support [num_queries, k, dim] for now.")
            return interpolated_feature_geo, interpolated_feature_tex

    
    def forward(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Any,
        output_normal: bool = False,
    ) -> Dict[str, Float[Tensor, "..."]]:
        
        batch_size, n_points, n_dims = points.shape
        points = points.reshape(batch_size*n_points, n_dims)

        grad_enabled = torch.is_grad_enabled()
        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        # Get interpolated positions and features
        enc_geo, enc_tex = self.interpolate_encodings(
            points[None, :, :],
            space_cache
        )

        # Process features through the MLP networks
        sdf = self.sdf_network(
            enc_geo # [B*N, 1]
        )

        
        result = {
            "sdf": sdf, #[B*N, 1]
        }
        
        # Add features if needed
        if self.cfg.n_feature_dims > 0:
            result.update(
                {
                    "features": self.feature_network(
                        enc_tex
                    ), #[B*N, C]
                }
            )

        if grad_enabled: # grad_enabled is True if we are in training
            pad_zeros = lambda x: torch.cat(
                [
                    x, 
                    torch.zeros(*x.shape[:-1], 3, device=x.device)
                ], 
                dim=-1
            )
            sdf_points = self.sdf_network(
                pad_zeros(
                    space_cache["feature_geo"] if "feature_geo" in space_cache else space_cache["feature"],
                )
            )
            result.update(
                {
                    "sdf_points": sdf_points, #[B*N, 1]
                }
            )

        # Compute normal if requested
        if output_normal:
            if self.cfg.normal_type == "analytic":
                # QUESTION: the sdf is >0 for points outside the shape in the original space
                # so its normal should point outwards, but the normal is pointing inwards if we put a negative sign
                # so we need to flip the normal by multiplying it by -1
                sdf_grad = torch.autograd.grad(
                    sdf,
                    points,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=grad_enabled, # not implemented in the test
                )[0]
                normal = F.normalize(sdf_grad, dim=-1)
                if not grad_enabled:
                    torch.set_grad_enabled(False)
                    sdf_grad = sdf_grad.detach()
                    normal = normal.detach()
                    # the following items also has grad, so we need to detach them
                    result["features"] = result["features"].detach()
            else:
                raise NotImplementedError(
                    f"normal_type == {self.cfg.normal_type} is not implemented yet."
                )
            
            result.update(
                {
                    "normal": normal, #[B*N, 3]
                    "sdf_grad": sdf_grad, #[B*N, 3]
                }
            )

        return result

    def forward_sdf(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    ) -> Float[Tensor, "*N 1"]:
        raise NotImplementedError("forward_sdf is not implemented yet.")

    def forward_field(
        self, 
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        raise NotImplementedError("forward_field is not implemented yet.")

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        # TODO: is this function correct?
        raise NotImplementedError("forward_level is not implemented yet.")

    def export(
        self, 
        points: Float[Tensor, "*N Di"], 
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("export is not implemented yet.")
    

    def train(self, mode=True):
        super().train(mode)
        self.space_generator.train(mode)

    def eval(self):
        super().eval()
        self.space_generator.eval()
