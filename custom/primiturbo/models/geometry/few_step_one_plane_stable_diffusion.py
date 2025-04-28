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

# --- Import KNN related stuff ---
from ...knn import HAS_CUDA_KNN, CudaKNNIndex
# -----------------------------

# Helper function to convert quaternion to rotation matrix
def quat_to_rot_matrix(quat: Float[Tensor, "B M 4"]) -> Float[Tensor, "B M 3 3"]:
    """
    Converts a batch of unit quaternions to rotation matrices.

    Args:
        quat: Input tensor of quaternions with shape (B, M, 4).
              Assumes quaternions are in (w, x, y, z) format and normalized.
              B is batch size, M is the number of quaternions per batch item.

    Returns:
        Rotation matrices tensor with shape (B, M, 3, 3).

    Mathematical Principle:
        A unit quaternion q = (w, x, y, z) represents a rotation.
        The corresponding 3x3 rotation matrix R can be computed using the formula:
            R = [
                [1 - 2(y^2 + z^2),   2(xy - wz),         2(xz + wy)],
                [2(xy + wz),         1 - 2(x^2 + z^2),   2(yz - wx)],
                [2(xz - wy),         2(yz + wx),         1 - 2(x^2 + y^2)]
            ]
        This function implements this formula directly for batched input.
    """
    w, x, y, z = torch.unbind(quat, -1)
    B, M = w.shape
    
    # Reshape components for broadcasting during intermediate calculations
    # These keep 4 dimensions: (B, M, 1, 1)
    w_exp = w.view(B, M, 1, 1)
    x_exp = x.view(B, M, 1, 1)
    y_exp = y.view(B, M, 1, 1)
    z_exp = z.view(B, M, 1, 1)

    rot_mat = torch.zeros(B, M, 3, 3, device=quat.device, dtype=quat.dtype)

    # Compute rotation matrix elements and ensure result shape is (B, M) before assignment
    rot_mat[:, :, 0, 0] = (1 - 2*y_exp*y_exp - 2*z_exp*z_exp).view(B, M)
    rot_mat[:, :, 0, 1] = (2*x_exp*y_exp - 2*z_exp*w_exp).view(B, M)
    rot_mat[:, :, 0, 2] = (2*x_exp*z_exp + 2*y_exp*w_exp).view(B, M)

    rot_mat[:, :, 1, 0] = (2*x_exp*y_exp + 2*z_exp*w_exp).view(B, M)
    rot_mat[:, :, 1, 1] = (1 - 2*x_exp*x_exp - 2*z_exp*z_exp).view(B, M)
    rot_mat[:, :, 1, 2] = (2*y_exp*z_exp - 2*x_exp*w_exp).view(B, M)

    rot_mat[:, :, 2, 0] = (2*x_exp*z_exp - 2*y_exp*w_exp).view(B, M)
    rot_mat[:, :, 2, 1] = (2*y_exp*z_exp + 2*x_exp*w_exp).view(B, M)
    rot_mat[:, :, 2, 2] = (1 - 2*x_exp*x_exp - 2*y_exp*y_exp).view(B, M)

    return rot_mat

# Helper function to build inverse covariance matrix from scale and rotation
def build_inverse_covariance(
    scale: Float[Tensor, "B M 3"],
    rotation_quat: Float[Tensor, "B M 4"],
    eps: float = 1e-8
) -> Float[Tensor, "B M 3 3"]:
    """
    Builds the inverse covariance matrix for 3D Gaussians from scale and rotation.

    Args:
        scale: Gaussian scaling factors (standard deviations along principal axes).
               Shape (B, M, 3).
        rotation_quat: Gaussian rotations as quaternions (w, x, y, z).
                       Shape (B, M, 4). Assumed to be normalized.
        eps: Small epsilon added to scale to prevent division by zero.

    Returns:
        Inverse covariance matrices tensor with shape (B, M, 3, 3).

    Mathematical Principle:
        The covariance matrix Sigma for a 3D Gaussian is defined as:
            Sigma = R * S * S^T * R^T
        where R is the rotation matrix and S is the diagonal scaling matrix:
            S = diag(scale_x, scale_y, scale_z)
        We need the inverse covariance matrix Sigma^-1.
        Using matrix inverse properties ((AB)^-1 = B^-1 * A^-1, (A^T)^-1 = (A^-1)^T)
        and the fact that R is orthogonal (R^-1 = R^T), we get:
            Sigma^-1 = (R * S * S^T * R^T)^-1
                     = (R^T)^-1 * (S^T)^-1 * S^-1 * R^-1
                     = R * (S^T)^-1 * S^-1 * R^T
        Since S is diagonal, S^T = S, and (S^T)^-1 = S^-1. Thus:
            Sigma^-1 = R * S^-2 * R^T
                     = R * S^-2 * R^T
        where S^-2 = diag(1/scale_x^2, 1/scale_y^2, 1/scale_z^2).
        This function computes R from rotation_quat, computes S^-2 from scale,
        and performs the matrix multiplication R * S^-2 * R^T.
    """
    B, M, _ = scale.shape
    
    # Calculate the inverse scale squared diagonal elements: 1 / scale^2
    inv_scale = 1.0 / (scale + eps)
    inv_scale_sq_diag = inv_scale ** 2
    
    # Build the diagonal matrix S^-2 using torch.diag_embed
    S_inv_sq = torch.diag_embed(inv_scale_sq_diag) # Shape (B, M, 3, 3)

    # Convert quaternion to rotation matrix R
    R = quat_to_rot_matrix(rotation_quat) # Shape (B, M, 3, 3)

    # Calculate inverse covariance Sigma^-1 = R * S^-2 * R^T
    # R: (B, M, 3, 3), S_inv_sq: (B, M, 3, 3), R.transpose: (B, M, 3, 3)
    inv_cov = R @ S_inv_sq @ R.transpose(-1, -2) # Shape (B, M, 3, 3)
    return inv_cov

# Helper function to gather parameters for K nearest neighbors
def gather_gaussian_params(
    params: torch.Tensor, # Shape (B, M, D1, D2, ...)
    indices: torch.Tensor # Shape (B, N, K)
) -> torch.Tensor:      # Shape (B, N, K, D1, D2, ...)
    """
    Gathers parameters from a large tensor based on KNN indices using advanced indexing.
    This version avoids creating a massive intermediate tensor, unlike the previous gather-based approach.

    Args:
        params: Tensor containing parameters for all M items.
                Shape (B, M, D1, D2, ...).
        indices: Tensor containing the indices of the K nearest neighbors for each
                 of the N query points. Shape (B, N, K).

    Returns:
        Tensor containing the gathered parameters for the K neighbors of each query point.
        Shape (B, N, K, D1, D2, ...).

    Implementation Notes:
        - Assumes batch size B=1 for direct indexing. If B>1, this needs adjustment (e.g., looping).
        - Uses PyTorch's advanced indexing (params_squeezed[indices_squeezed]).
          This efficiently selects elements without materializing the huge intermediate
          tensor required by torch.gather in the previous implementation.
    """
    B, N, K = indices.shape
    _B, M = params.shape[:2]
    # feature_dims = params.shape[2:] # Not needed for indexing logic
    assert B == 1, "gather_gaussian_params_advanced_indexing currently only supports B=1"
    assert _B == 1, "gather_gaussian_params_advanced_indexing currently only supports B=1 for params"

    # Squeeze batch dimension (assuming B=1)
    params_squeezed = params.squeeze(0)   # Shape (M, D1, D2, ...)
    indices_squeezed = indices.squeeze(0) # Shape (N, K)

    # Use advanced indexing: M-dim tensor indexed by (N, K) indices
    # Result shape will be (N, K, D1, D2, ...)
    gathered_squeezed = params_squeezed[indices_squeezed]

    # Unsqueeze to add batch dimension back
    gathered = gathered_squeezed.unsqueeze(0) # Shape (1, N, K, D1, D2, ...)

    return gathered

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
        xyz_max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
        xyz_ratio: float = 10.
        
        top_K: int = 8 # Number of nearest neighbors to consider
        knn_backend: str = 'cuda-knn' # Changed default to cuda-knn

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
        self.xyz_max = lambda x: torch.tensor(self.cfg.xyz_max, device=x.device)

        # Configure KNN mode based on availability and config
        if self.cfg.knn_backend == 'cuda-knn':
            if HAS_CUDA_KNN:
                threestudio.info("Using CUDA KNN backend.")
                self.knn_mode = 'cuda-knn'
            else:
                threestudio.warning("WARNING: CUDA KNN extension not installed or cannot be loaded. Falling back to torch KNN implementation.")
                self.knn_mode = 'torch' # Fallback to torch
        elif self.cfg.knn_backend == 'torch':
            threestudio.info("Using PyTorch backend for KNN (might be slow for large M).")
            self.knn_mode = 'torch'
        else:
            raise ValueError(f"Unknown knn_backend: {self.cfg.knn_backend}")

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

    def _build_knn_index(self, points: Float[Tensor, "B M 3"]):
        """Builds a KNN index based on the active knn_mode."""
        B, M, _ = points.shape
        assert B == 1, "KNN index building currently only supports batch size 1"
        # Ensure points are on the correct device and contiguous
        points_flat = points.squeeze(0).to(self.device).contiguous() # Shape (M, 3)

        if self.knn_mode == 'cuda-knn':
            # Use CudaKNNIndex from the renderer's reference
            index = CudaKNNIndex()
            index.add(points_flat.unsqueeze(0)) # CudaKNNIndex might expect (1, M, 3)
            threestudio.debug(f"Built CUDA KNN index with {M} points.")
            return index
        elif self.knn_mode == 'torch':
            # Store points for PyTorch-based distance calculation
            class TorchKNNIndex:
                def __init__(self, data):
                    self.data = data # Shape (M, 3)
                    self.M = data.shape[0]

                def search(self, query: Float[Tensor, "Nq 3"], k: int):
                    # Ensure query is on the same device
                    query = query.to(self.data.device)
                    Nq = query.shape[0]
                    # Expand dims for broadcasting: (Nq, 1, 3) and (1, M, 3)
                    dist_sq = torch.sum((query.unsqueeze(1) - self.data.unsqueeze(0))**2, dim=-1) # Shape (Nq, M)
                    distances, indices = torch.topk(dist_sq, k=k, dim=-1, largest=False) # Shape (Nq, K)
                    # Return distances_sq (L2 squared) and indices
                    return distances, indices
            threestudio.de(f"Built Torch KNN placeholder with {M} points.")
            return TorchKNNIndex(points_flat)
        else:
            raise NotImplementedError(f"KNN backend {self.knn_mode} not implemented.")

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
                ) * self.cfg.xyz_ratio * self.xyz_max(triplane) + 
            self.xyz_center(triplane),
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

        # Build KNN index
        if pc_dict['position'].shape[0] == 1:
             pc_dict['index'] = self._build_knn_index(pc_dict['position'])
        else:
             print("[Warning] KNN index building for batch size > 1 not fully implemented in parse.")
             pc_dict['index'] = None

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
        space_cache: Dict[str, Any], # Expecting dict from parse(), now includes 'inv_cov', 'index', 'normal'
        output_normal: bool = False,
    ) -> Dict[str, Float[Tensor, "..."]]:
        """
        Computes weighted mixture of Gaussian colors, density, and an estimated SDF
        at query points, using only the top_K nearest neighbors.
        The SDF is estimated based on projection onto the normal defined by the smallest scale axis.
        """
        calculate_avg_normal = output_normal

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

        if index is None:
             raise ValueError("KNN index not found in space_cache. Ensure it's built during parsing.")
        if B > 1:
             print("[Warning] Forward pass with KNN might require adjustments for batch size > 1.")

        # Perform KNN search
        points_flat = points.view(B * N, 3)
        distances, indices = index.search(points_flat.unsqueeze(0) if self.knn_mode == 'cuda-knn' else points_flat, k=min(top_K, M))
        if self.knn_mode == 'cuda-knn':
            distances = distances.squeeze(0)
            indices = indices.squeeze(0)
        indices = indices.view(B, N, -1) # (B, N, K)
        K = indices.shape[-1]

        # Gather parameters of the K nearest neighbors
        gathered_pos = gather_gaussian_params(gauss_pos, indices)      # (B, N, K, 3)
        gathered_col = gather_gaussian_params(gauss_col, indices)      # (B, N, K, 3)
        gathered_opa = gather_gaussian_params(gauss_opa, indices)      # (B, N, K, 1)
        gathered_inv_cov = gather_gaussian_params(inv_cov, indices) # (B, N, K, 3, 3)
        gathered_normal = gather_gaussian_params(est_normals, indices)  # (B, N, K, 3) Gather precomputed normals

        # Calculate difference vector relative to K neighbors
        diff = points.unsqueeze(2) - gathered_pos # (B, N, K, 3)

        # Calculate Mahalanobis distance squared for K neighbors
        mahalanobis_sq = torch.einsum("bnki,bnkij,bnkj->bnk", diff, gathered_inv_cov, diff)

        # Calculate Gaussian density contribution (unnormalized) for K neighbors
        exponent = torch.clamp(-0.5 * mahalanobis_sq, max=20.0)
        gauss_density = torch.exp(exponent) # (B, N, K)

        # Calculate weights: density * opacity for K neighbors
        weights = gauss_density * gathered_opa.squeeze(-1) # (B, N, K)

        # Calculate sum of weights over K neighbors for normalization and density
        sum_weights = weights.sum(dim=-1, keepdim=True) + 1e-8 # (B, N, 1)

        # Normalize weights over K neighbors
        norm_weights = weights / sum_weights # (B, N, K)

        # Calculate interpolated color using K neighbors
        interpolated_color = torch.einsum("bnk,bnkc->bnc", norm_weights, gathered_col) # (B, N, 3)

        # Density is the sum of weights before normalization
        density = sum_weights # (B, N, 1)

        # --- Calculate SDF based on weighted projection onto normals ---
        signed_dist_k = torch.einsum("bnki,bnki->bnk", diff, gathered_normal) # (B, N, K)
        sdf = torch.sum(norm_weights * signed_dist_k, dim=-1, keepdim=True) # (B, N, 1)
        # -----------------------------------------------------------

        # Prepare output dictionary, ensuring outputs are flattened to (B*N, Dim)
        num_points_total = B * N
        out = {
            "features": interpolated_color.view(num_points_total, -1),
            "density": density.view(num_points_total, 1),
            "sdf": sdf.view(num_points_total, 1) # Add the estimated SDF
        }

        # Optional: Calculate the weighted average normal if requested
        if calculate_avg_normal:
            avg_normal = torch.einsum("bnk,bnkc->bnc", norm_weights, gathered_normal) # (B, N, 3)
            avg_normal = F.normalize(avg_normal, p=2, dim=-1)
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
