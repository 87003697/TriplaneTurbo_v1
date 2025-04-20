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
        
        # Ensure indices is 1D for bincount
        indices_flat = indices.view(-1)
        # Ensure grad_output corresponds to the flattened indices
        grad_output_flat = grad_output.view(-1, ctx.feat_size)

        # Calculate counts for each index
        index_counts = torch.bincount(indices_flat, minlength=ctx.input_size)
        # Avoid division by zero and ensure correct shape for broadcasting
        # Shape should become (input_size, 1)
        index_counts = torch.clamp(index_counts, min=1).unsqueeze(1).to(grad_output.dtype)
        
        # Accumulate gradients using flattened indices and grad_output
        input_grad.index_add_(0, indices_flat, grad_output_flat)
        
        # Normalize by counts - Ensure index_counts is (N, 1) for broadcasting
        input_grad /= index_counts.view(-1, 1) # Explicitly ensure shape for division
        
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
        point_grad_shrink_point: bool = False # whether shrink the gradient w.r.t. the points
        point_grad_shrink_geo: bool = False # whether shrink the gradient w.r.t. the geometry
        point_grad_shrink_tex: bool = False # whether shrink the gradient w.r.t. the texture

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
        pc_dict = {
            "position": self.position_activation(
                rearrange(
                    triplane[:, :, 0:3, :, :],
                    "B N C H W -> B (N H W) C"
                )
            ),
            "feature": rearrange(
                triplane[:, :, 3:, :, :], 
                "B N C H W -> B (N H W) C"
            ),
        }
        return pc_dict

        #     pc_list.append(
        #         {
        #             "position": self.position_activation(
        #                 rearrange(
        #                     triplane[i, :, 0:3, :, :],
        #                     "N C H W -> (N H W) C"
        #                     )
        #                 ), # plus center
        #             "feature": rearrange(
        #                     triplane[i, :, 3:, :, :], 
        #                     "N C H W -> (N H W) C"
        #                 ),
        #         }
        #     )
        # return pc_list


    def _knn_interpolate_encodings(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache_position: Float[Tensor, "*N 3"],
        space_cache_feature_geo: Float[Tensor, "*N C"],
        space_cache_feature_tex: Float[Tensor, "*N C"],
        index: Optional[Callable] = None,
        only_geo: bool = False,
        debug: bool = False,
    ):
        assert index != None, "Index is not found in space_cache"
        k = 8

        # --- Always run CUDA KNN --- 
        if debug:
            print("DEBUG: Running CUDA KNN on all points...")
        cuda_distances, cuda_indices = index.search(
            points.detach(),
            k=k
        )
        cuda_indices_flat = cuda_indices.squeeze(0)

        # --- Optional Debug Verification --- 
        if debug:
            debug_points: int = 10

            print(f"DEBUG: Running manual PyTorch KNN verification for first {debug_points} points...")
            # Select the first N points for verification
            verify_points = points[:, :debug_points]
            
            # Calculate squared distances directly for these points
            dist_dtype = torch.float32 if verify_points.dtype == torch.float16 else verify_points.dtype
            dist_verify = torch.sum((verify_points.to(dist_dtype).view(-1, 1, 3) - space_cache_position.to(dist_dtype).view(1, -1, 3)) ** 2, dim=-1)
            
            # Get top-k smallest distances using PyTorch for the verification points
            _, pytorch_indices_verify = torch.topk(dist_verify, k=k, dim=-1, largest=False)
            print("DEBUG: Manual PyTorch KNN for verification finished.")

            # Compare Results for the first N points
            cuda_indices_verify = cuda_indices_flat[:debug_points]
            print(f"DEBUG: CUDA indices[:{debug_points}, :3]:\n{cuda_indices_verify[:, :3]}")
            print(f"DEBUG: PyTorch indices[:{debug_points}, :3]:\n{pytorch_indices_verify[:, :3]}")
            
            # Check consistency
            are_indices_same = torch.all(cuda_indices_verify == pytorch_indices_verify)
            print(f"DEBUG: Are CUDA and PyTorch indices identical for first {debug_points} points? {are_indices_same}")
            assert are_indices_same, f"CUDA KNN indices do not match manual PyTorch KNN indices for first {debug_points} points!"
            print(f"DEBUG: Indices match for first {debug_points} points! CUDA KNN fix is successful.")

            # Exit after verification if in debug mode
            print("DEBUG: Exiting after verification.")
            import os; os._exit(0)

        # --- Main Function Logic --- 
        indexing_func = DifferentiableIndexing.apply if self.cfg.point_grad_shrink_avarage else lambda x, y: x[y]
        indices_flat = cuda_indices_flat # Use the results from CUDA KNN
        num_queries, k_actual = indices_flat.shape 

        # Get the neighbor position
        neighbor_position: Float[Tensor, "*N K 3"] = indexing_func(
            space_cache_position, 
            indices_flat
        ).reshape(num_queries, k_actual, 3) 
        if self.cfg.point_grad_shrink_point: # shrink the gradient w.r.t. the points
            shrink_ratio_point: Float[Tensor, "*N K 1"] = cuda_distances.min()/ (cuda_distances.view(num_queries, k_actual, 1) + 1e-8)
            neighbor_position = shrink_ratio_point * neighbor_position + (1 - shrink_ratio_point) * neighbor_position.detach()
        
        # Get the neighbor feature
        neighbor_feature_geo: Float[Tensor, "*N K C"] = indexing_func(
            space_cache_feature_geo, 
            indices_flat
        ).reshape(num_queries, k_actual, -1) 
        if self.cfg.point_grad_shrink_geo: # shrink the gradient w.r.t. the geometry
            shrink_ratio_geo: Float[Tensor, "*N K 1"] = cuda_distances.min() / (cuda_distances.view(num_queries, k_actual, 1) + 1e-8)
            neighbor_feature_geo = shrink_ratio_geo * neighbor_feature_geo + (1 - shrink_ratio_geo) * neighbor_feature_geo.detach()
        
        if only_geo:
            return neighbor_position, neighbor_feature_geo, None
        else:
            # Get the neighbor feature
            neighbor_feature_tex: Float[Tensor, "*N K C"] = indexing_func(
                space_cache_feature_tex, 
                indices_flat
            ).reshape(num_queries, k_actual, -1)
            if self.cfg.point_grad_shrink_tex: # shrink the gradient w.r.t. the texture
                shrink_ratio_tex: Float[Tensor, "*N K 1"] = cuda_distances.min() / (cuda_distances.view(num_queries, k_actual, 1) + 1e-8)
                neighbor_feature_tex = shrink_ratio_tex * neighbor_feature_tex + (1 - shrink_ratio_tex) * neighbor_feature_tex.detach()
            return neighbor_position, neighbor_feature_geo, neighbor_feature_tex

    def interpolate_encodings(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Dict[str, Any],
        only_geo: bool = False,
        debug: bool = False,
    ):
        
        # get the neighbor position, feature, and tex
        neighbor_position: Float[Tensor, "*N K 3"]
        neighbor_feature_geo: Float[Tensor, "*N K C"]
        neighbor_feature_tex: Float[Tensor, "*N K C"]
        neighbor_position, neighbor_feature_geo, neighbor_feature_tex = self._knn_interpolate_encodings(
            points,
            space_cache["position"],
            space_cache["feature_geo"] if "feature_geo" in space_cache else space_cache["feature"],
            space_cache["feature_tex"] if "feature_tex" in space_cache else space_cache["feature"],
            index = space_cache["index"] if "index" in space_cache else None,
            only_geo = only_geo,
            debug = debug
        )

        # Recalculate the distances to enable gradient flow
        # This is crucial for gradient to flow back to input points
        # Use float32 for distance calculation stability
        dist_dtype = torch.float32 if points.dtype == torch.float16 else points.dtype
        distances: Float[Tensor, "*N K"] = torch.norm(
            neighbor_position.to(dist_dtype) - points.to(dist_dtype).view(-1, 1, 3),
            dim=-1,
            p=2
        )

        # Now perform interpolation based on distances
        # Convert distances to weights using inverse distance weighting
        inv_distances: Float[Tensor, "*N K"] = 1.0 / (distances + 1e-8)  # Add small epsilon to prevent division by zero
        weights: Float[Tensor, "*N K"] = inv_distances / inv_distances.sum(dim=-1, keepdim=True)  # Normalize weights
        
        # Apply weighted average to get interpolated features and positions
        if neighbor_position.ndim == 3:  # [num_queries, k, dim]
            interpolated_feature_geo: Float[Tensor, "*N C"]
            interpolated_feature_tex: Optional[Float[Tensor, "*N C"]] = None

            # Interpolate Geo features
            interp_geo = torch.sum(neighbor_feature_geo * weights.unsqueeze(-1), dim=1)
            pos_diff_interp = torch.sum(
                (points.view(-1, 1, 3) - neighbor_position) * weights.unsqueeze(-1), 
                dim=1
            )
            interpolated_feature_geo = torch.cat([interp_geo, pos_diff_interp], dim=-1)

            if not only_geo:
                # Interpolate Tex features
                interp_tex = torch.sum(neighbor_feature_tex * weights.unsqueeze(-1), dim=1)
                # Re-use pos_diff_interp calculated above
                interpolated_feature_tex = torch.cat([interp_tex, pos_diff_interp], dim=-1)

        else:
            raise NotImplementedError("Only support [num_queries, k, dim] for now.")

        if only_geo:
            return interpolated_feature_geo, None
        else:
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
