import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.utils.typing import *

from threestudio.utils.ops import get_activation

from einops import rearrange

from .utils import (
    HAS_CUDA_KDN, HAS_CUDA_KDON, HAS_CUDA_KNN,
    CudaKDNIndex, CudaKNNIndex, CudaKDONIndex,
    gather_gaussian_params, build_inverse_covariance, quat_to_rot_matrix,
    verify_kdon_cuda_vs_pytorch,
    verify_kdn_cuda_vs_pytorch
)


@threestudio.register("few-step-one-plane-stable-diffusion")
class FewStepOnePlaneStableDiffusion(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_feature_dims: int = 3
        space_generator_config: dict = field(
            default_factory=lambda: {
                "pretrained_model_name_or_path": "stable-diffusion-2-1-base",
                "training_type": "lora",
                "output_dim": 14,
                "gradient_checkpoint": False,
            }
        )

        backbone: str = "few_step_one_plane_stable_diffusion" #TODO: change to few_step_few_plane_stable_diffusion

        scaling_activation: str = "exp-0.1" # in ["exp-0.1", "sigmoid", "exp", "softplus"]
        opacity_activation: str = "sigmoid-0.1" # in ["sigmoid-0.1", "sigmoid", "sigmoid-mipnerf", "softplus"]
        rotation_activation: str = "normalize" # in ["normalize"]
        color_activation: str = "none" # in ["scale_-11_01", "sigmoid-mipnerf"]
        position_activation: str = "none" # in ["none"]
        
        xyz_center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
        xyz_scale: float = 1.0
        
        top_K: int = 8 # Number of nearest neighbors to consider
        knn_backend: str = 'cuda-knn' # Changed default to cuda-knn
        # forward_internal_chunk_size: Optional[int] = None # Removed chunking
        sdf_type: str = "normal_projection" # Options: "normal_projection", "mahalanobis", "mahalanobis_squared", "signed_mahalanobis", "signed_mahalanobis_squared", "none"

        # Neighbor search configuration
        neighbor_search_metric: str = 'l2' # 'l2' for KNN, 'mahalanobis' for KDN, 'density-opacity' for KDON

    def configure(self) -> None:
        super().configure()

        print("The current device is: ", self.device)
        
        # set up the space generator
        if self.cfg.backbone == "few_step_one_plane_stable_diffusion":
            from ...extern.few_step_one_plane_sd_modules import FewStepOnePlaneStableDiffusion as Generator
            self.space_generator = Generator(self.cfg.space_generator_config)
        else:
            raise ValueError(f"Unknown backbone {self.cfg.backbone}")
        
        self.scaling_activation = get_activation(self.cfg.scaling_activation)
        self.opacity_activation = get_activation(self.cfg.opacity_activation)
        self.rotation_activation = get_activation(self.cfg.rotation_activation)
        self.color_activation = get_activation(self.cfg.color_activation)
        self.position_activation = get_activation(self.cfg.position_activation)

        self.xyz_center = lambda x: torch.tensor(self.cfg.xyz_center, device=x.device)

        # --- Configure Neighbor Search --- 
        self.search_mode = None
        metric = self.cfg.neighbor_search_metric.lower()
        backend = self.cfg.knn_backend.lower()

        if metric == 'l2':
            if backend == 'cuda-knn':
                if HAS_CUDA_KNN:
                    threestudio.info("Using CUDA KNN backend (L2 distance).")
                    self.search_mode = 'knn-cuda'
                else:
                    threestudio.warning("CUDA KNN extension requested but not available/compiled. Falling back to torch KNN.")
                    self.search_mode = 'knn-torch' 
            elif backend == 'torch':
                 threestudio.info("Using PyTorch KNN backend (L2 distance, might be slow).")
                 self.search_mode = 'knn-torch'
            else:
                 raise ValueError(f"Unknown knn_backend for L2: {self.cfg.knn_backend}")
        
        elif metric == 'mahalanobis':
            if backend == 'cuda-knn': # Assuming cuda-knn backend enables custom kernels too
                 if HAS_CUDA_KDN:
                     threestudio.info("Using CUDA KDN backend (Mahalanobis distance).")
                     self.search_mode = 'kdn-cuda'
                 else:
                     threestudio.error("CUDA KDN extension requested (for Mahalanobis) but not available/compiled. Cannot proceed.")
                     raise ImportError("CUDA KDN extension failed to load.")
            else:
                 raise ValueError(f"Unsupported backend '{backend}' for Mahalanobis distance. Use 'cuda-knn'.")
        
        elif metric == 'density-opacity':
            if backend == 'cuda-knn': # Assuming cuda-knn backend enables custom kernels too
                if HAS_CUDA_KDON:
                    threestudio.info("Using CUDA KDON backend (Density-Opacity weighted Mahalanobis).")
                    self.search_mode = 'kdon-cuda'
                else:
                    threestudio.error("CUDA KDON extension requested (for Density-Opacity) but not available/compiled. Cannot proceed.")
                    raise ImportError("CUDA KDON extension failed to load.")
            else:
                 raise ValueError(f"Unsupported backend '{backend}' for Density-Opacity search. Use 'cuda-knn'.")
        else:
            raise ValueError(f"Unknown neighbor_search_metric: {self.cfg.neighbor_search_metric}")
        
        if self.search_mode is None:
             raise RuntimeError("Neighbor search mode could not be determined.") 
        
        threestudio.debug(f"Determined search mode: {self.search_mode}") # Add debug log

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

    def _build_neighbor_index(self, 
                              points: Float[Tensor, "B M 3"], 
                              inv_covariances: Optional[Float[Tensor, "B M 3 3"]], 
                              reference_opacities: Optional[Float[Tensor, "B M 1"]], # Added opacities
                              reference_lengths: Float[Tensor, "B"]):
        """Builds a KNN or KDN index based on the configured search_mode."""
        B, M, D = points.shape
        assert B == 1, "Index building currently only supports batch size 1"

        index = None
        # Use the pre-determined search_mode from configure()
        search_mode = self.search_mode 

        threestudio.debug(f"Building index for search mode: {search_mode}") # Add debug log

        if search_mode == 'knn-cuda': 
            index = CudaKNNIndex() 
            index.add(points)
            # threestudio.debug(f"Built CUDA KNN index (L2) with {M} points.") 
        elif search_mode == 'kdn-cuda': 
            assert inv_covariances is not None, "Inverse covariances needed for KDN index."
            index = CudaKDNIndex() 
            index.add(points, inv_covariances, reference_lengths)
            # threestudio.debug(f"Built CUDA KDN index (Mahalanobis) with {M} points.")
        elif search_mode == 'kdon-cuda': # Changed from 'density-opacity' to match self.search_mode
             assert inv_covariances is not None, "Inverse covariances needed for KDON index."
             assert reference_opacities is not None, "Reference opacities needed for KDON index."
             index = CudaKDONIndex()
             index.add(points, inv_covariances, reference_opacities, reference_lengths)
             # threestudio.debug(f"Built CUDA KDON index (Density-Opacity) with {M} points.")
        elif search_mode == 'knn-torch':
            threestudio.debug(f"Building Torch KNN index (L2) with {M} points.") 
            points_flat = points.squeeze(0).contiguous() 
            
            class TorchKNNIndex:
                def __init__(self, data):
                    self.data = data 
                    self.M = data.shape[0]
                def search(self, query: Float[Tensor, "Nq 3"], k: int):
                    query = query.to(self.data.device)
                    Nq = query.shape[0]
                    dist_sq = torch.sum((query.unsqueeze(1) - self.data.unsqueeze(0))**2, dim=-1) 
                    safe_k = min(k, self.M)
                    if safe_k <= 0: # Handle M=0 case
                         distances = torch.full((Nq, k), float('inf'), device=query.device, dtype=self.data.dtype)
                         indices = torch.full((Nq, k), -1, device=query.device, dtype=torch.long)
                         return distances, indices
                    distances, indices = torch.topk(dist_sq, k=safe_k, dim=-1, largest=False) 
                    if k > safe_k:
                        pad_size = k - safe_k
                        pad_dist = torch.full((Nq, pad_size), float('inf'), device=query.device, dtype=distances.dtype)
                        pad_idx = torch.full((Nq, pad_size), -1, device=query.device, dtype=indices.dtype)
                        distances = torch.cat([distances, pad_dist], dim=-1)
                        indices = torch.cat([indices, pad_idx], dim=-1)
                    return distances, indices 
            index = TorchKNNIndex(points_flat)
        else:
            # This case should be prevented by configure(), but raise error just in case
            raise NotImplementedError(f"Index building not implemented for determined search mode: {search_mode}")
            
        return index

    def parse(
        self,
        triplane: Float[Tensor, "B 3 C//3 H W"],
    ) -> Dict[str, Any]:
        pc_dict = {
            "color": self.color_activation(
                rearrange(
                    triplane[:, :, 0:3, :, :], 
                    "B N C H W -> B (N H W) C"
                )
            ),
            "position": self.position_activation(
                rearrange(
                    triplane[:, :, 3:6, :, :],
                    "B N C H W -> B (N H W) C"
                    )
                ) * self.cfg.xyz_scale + self.xyz_center(triplane),
            "scale": self.scaling_activation(
                rearrange(
                    triplane[:, :, 6:9, :, :],
                    "B N C H W -> B (N H W) C"
                )
            ),
            "rotation": self.rotation_activation(
                rearrange(
                    triplane[:, :, 9:13, :, :], 
                    "B N C H W -> B (N H W) C"
                )
            ),
            "opacity": self.opacity_activation(
                rearrange(
                    triplane[:, :, 13:14, :, :], 
                    "B N C H W -> B (N H W) C"
                )
            )
        }

        # Pre-calculate inverse covariance matrix
        pc_dict['inv_cov'] = build_inverse_covariance(pc_dict['scale'], pc_dict['rotation'])

        # --- Pre-calculate normals based on smallest scale axis ---
        scales = pc_dict['scale'] # (B, M, 3)
        quats = pc_dict['rotation'] # (B, M, 4)
        B, M, _ = scales.shape

        # Find the index of the smallest scale for each Gaussian
        min_scale_indices = torch.argmin(scales, dim=-1) # (B, M)

        # Get the rotation matrices
        rot_mats = quat_to_rot_matrix(quats) # (B, M, 3, 3)

        # Gather the corresponding column (normal vector) from rotation matrices
        min_scale_indices_exp = min_scale_indices.view(B, M, 1, 1).expand(B, M, 3, 1)
        est_normals = torch.gather(rot_mats, 3, min_scale_indices_exp).squeeze(-1) # (B, M, 3)
        pc_dict['normal'] = est_normals # Store estimated normals
        # ---------------------------------------------------------

        # Build Neighbor index
        if pc_dict['position'].shape[0] == 1:
             ref_lengths = torch.tensor([pc_dict['position'].shape[1]], dtype=torch.int64, device=pc_dict['position'].device)
             # Pass opacity to _build_neighbor_index
             pc_dict['index'] = self._build_neighbor_index(
                 pc_dict['position'], 
                 pc_dict['inv_cov'], 
                 pc_dict['opacity'], # Pass opacity here
                 ref_lengths
             )
        else:
             # print("[Warning] Neighbor index building for batch size > 1 not implemented in parse.")
             pc_dict['index'] = None # Keep as None for B > 1

        return pc_dict

    def interpolate_encodings(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
        only_geo: bool = False,
    ):
        raise NotImplementedError("interpolate_encodings is not implemented yet.")


    def rescale_points(
        self,
        points: Float[Tensor, "*N Di"],
    ):
        raise NotImplementedError("rescale_points is not implemented yet.")

    def forward(
        self,
        points: Float[Tensor, "*N Di"], # Expecting (B, N, 3)
        space_cache: Dict[str, Any],
        output_normal: bool = False,
        debug: bool = False, #False, # Added debug flag
    ) -> Dict[str, Float[Tensor, "..."]]:
        """
        Computes weighted mixture properties (color, density) and estimates SDF using
        the specified method (cfg.sdf_type). Uses KNN for efficiency.
        """
        calculate_avg_normal_output = output_normal # Decide if we output 'normal' key
        B, N, _ = points.shape
        top_K = self.cfg.top_K

        # Retrieve parameters and precomputed data from cache
        gauss_pos = space_cache['position'] # (B, M, 3)
        gauss_col = space_cache['color']    # (B, M, 3)
        gauss_opa = space_cache['opacity']  # (B, M, 1)
        inv_cov = space_cache['inv_cov']    # (B, M, 3, 3)
        index = space_cache['index']        # KNN index object
        est_normals = space_cache['normal']   # (B, M, 3) Precomputed normals for SDF
        M = gauss_pos.shape[1]
        ref_lengths = torch.tensor([M], dtype=torch.int64, device=points.device) # Assuming B=1

        if index is None:
             raise ValueError("KNN index not found in space_cache.")
        if B > 1:
            raise NotImplementedError("Forward pass currently assumes B=1 due to gather implementation.")

        # Perform KNN/KDN/KDON search
        points_flat = points.view(-1, 3) 
        query_lengths_flat = torch.full((B,), N, dtype=torch.int64, device=points.device) 
        
        search_k = min(top_K, M) 
        
        # --- Start Timing Search --- 
        # Removed timing logic
        # ---------------------------    
        
        if self.search_mode == 'knn-torch':
             _, indices = index.search(points_flat, k=search_k)
        elif self.search_mode == 'knn-cuda':
             _, indices = index.search(points_flat, k=search_k) 
        elif self.search_mode == 'kdn-cuda':
             _, indices = index.search(points, query_lengths_flat, k=search_k) 
        elif self.search_mode == 'kdon-cuda': 
             _, indices = index.search(points, query_lengths_flat, k=search_k)
        else:
             raise RuntimeError(f"Search not implemented for mode {self.search_mode}")
            
        indices = indices.view(B, N, -1) 
        K_ret = indices.shape[-1] 

        # --- End Timing Search --- 
        # Removed timing logic
        # -------------------------   

        # Gather parameters of the K neighbors
        gathered_pos = gather_gaussian_params(gauss_pos, indices)      # (B, N, K_ret, 3)
        gathered_col = gather_gaussian_params(gauss_col, indices)      # (B, N, K_ret, 3)
        gathered_opa = gather_gaussian_params(gauss_opa, indices)      # (B, N, K_ret, 1)
        gathered_inv_cov = gather_gaussian_params(inv_cov, indices) # (B, N, K_ret, 3, 3)
        gathered_normal = gather_gaussian_params(est_normals, indices)  # (B, N, K_ret, 3)

        # Calculate difference vector relative to K neighbors
        diff = points.unsqueeze(2) - gathered_pos # (B, N, K_ret, 3)

        # Calculate Mahalanobis distance squared for K neighbors
        # IMPORTANT: If using KDN/KDON CUDA, we might already have this distance.
        # For consistency or if using KNN, recalculate/use the returned distances.
        mahalanobis_sq = torch.einsum("bnki,bnkij,bnkj->bnk", diff, gathered_inv_cov, diff)
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0.0) 


        # ----- KDN Debug Check ----- 
        if debug and self.search_mode == 'kdn-cuda' and B == 1:
            # Note: KDN CUDA returns Mahalanobis sq directly
            verify_kdn_cuda_vs_pytorch(
                points=points,
                indices_cuda=indices,
                mahalanobis_sq_cuda=mahalanobis_sq,
                gauss_pos=gauss_pos,
                inv_cov=inv_cov,
                query_lengths=query_lengths_flat,
                ref_lengths=ref_lengths,
                K_ret=K_ret, 
                K_cfg=top_K, # Pass original K from config
                debug_limit_N=100
            )
        # ----- End KDN Debug Check -----

        # ----- KDON Debug Check ----- 
        if debug and self.search_mode == 'kdon-cuda' and B == 1:
            # Note: KDON CUDA returns Mahalanobis sq directly
            verify_kdon_cuda_vs_pytorch(
                points=points,
                indices_cuda=indices, 
                mahalanobis_sq_cuda=mahalanobis_sq, # Use the distance from KDON kernel
                gauss_pos=gauss_pos,
                inv_cov=inv_cov,
                gauss_opa=gauss_opa,
                K_ret=K_ret,
                debug_limit_N=100 
            )
        # ----- End KDON Debug Check -----

        # Calculate Gaussian density contribution (unnormalized) for K neighbors
        exponent = torch.clamp(-0.5 * mahalanobis_sq, max=20.0) # Clamp exponent to avoid inf/nan
        gauss_density = torch.exp(exponent) # (B, N, K_ret)

        # Calculate weights: density * opacity for K neighbors
        weights = gauss_density * gathered_opa.squeeze(-1) # (B, N, K_ret)

        # Calculate sum of weights over K neighbors for normalization and density
        sum_weights = weights.sum(dim=-1, keepdim=True) + 1e-8 # (B, N, 1)

        # Normalize weights over K neighbors
        norm_weights = weights / sum_weights # (B, N, K_ret)

        # Calculate interpolated color using K neighbors
        interpolated_color = torch.einsum("bnk,bnkc->bnc", norm_weights, gathered_col) # (B, N, 3)

        # Density is the sum of weights before normalization
        density = sum_weights # (B, N, 1)

        # --- Calculate SDF based on cfg.sdf_type ---
        sdf_type = self.cfg.sdf_type.lower()
        if sdf_type == "normal_projection":
            signed_dist_k = torch.einsum("bnki,bnki->bnk", diff, gathered_normal) # (B, N, K_ret)
            sdf = torch.sum(norm_weights * signed_dist_k, dim=-1, keepdim=True) # (B, N, 1)
        elif sdf_type == "mahalanobis":
            mahalanobis_dist_k = torch.sqrt(mahalanobis_sq + 1e-8) # (B, N, K_ret)
            sdf_k = mahalanobis_dist_k - 1.0 # Original version without sign
            sdf = torch.sum(norm_weights * sdf_k, dim=-1, keepdim=True) # (B, N, 1)
        elif sdf_type == "mahalanobis_squared": 
            sdf = torch.sum(norm_weights * mahalanobis_sq, dim=-1, keepdim=True) # Original version without sign
        elif sdf_type == "signed_mahalanobis": # New directed version
            mahalanobis_dist_k = torch.sqrt(mahalanobis_sq + 1e-8) # (B, N, K_ret)
            # Calculate direction sign based on normal projection
            signed_dist_k = torch.einsum("bnki,bnki->bnk", diff, gathered_normal)
            direction_sign = torch.sign(signed_dist_k + 1e-9)
            # Apply sign to the Mahalanobis distance
            sdf_k = direction_sign * mahalanobis_dist_k 
            sdf = torch.sum(norm_weights * sdf_k, dim=-1, keepdim=True) # (B, N, 1)
        elif sdf_type == "signed_mahalanobis_squared": # New directed version
            # Calculate direction sign based on normal projection
            signed_dist_k = torch.einsum("bnki,bnki->bnk", diff, gathered_normal)
            direction_sign = torch.sign(signed_dist_k + 1e-9)
            # Apply sign to the squared Mahalanobis distance
            sdf_k = direction_sign * mahalanobis_sq 
            sdf = torch.sum(norm_weights * sdf_k, dim=-1, keepdim=True) # (B, N, 1)
        elif sdf_type == "none":
            sdf = torch.zeros_like(density) # density has shape (B, N, 1) after sum_weights
        else:
            raise ValueError(f"Unknown sdf_type: {self.cfg.sdf_type}")
        # ---------------------------------------------

        # Calculate weighted average normal (used as sdf_grad proxy)
        avg_normal = torch.einsum("bnk,bnkc->bnc", norm_weights, gathered_normal) # (B, N, 3)
        avg_normal = F.normalize(avg_normal, p=2, dim=-1)

        # Prepare output dictionary
        num_points_total = B * N
        out = {
            "features": interpolated_color.view(num_points_total, -1),
            "density": density.view(num_points_total, 1),
            "sdf": sdf.view(num_points_total, 1) 
        }

        # Assign sdf_grad based on avg_normal (unless type is none)
        if self.cfg.sdf_type != "none":
            out["sdf_grad"] = avg_normal.view(num_points_total, 3)
        else:
            # Provide zero gradient if no SDF is calculated
            out["sdf_grad"] = torch.zeros_like(points.view(num_points_total, 3))

        # Optionally add the average normal itself to the output
        if calculate_avg_normal_output:
            out["normal"] = avg_normal.view(num_points_total, 3)

        return out

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
