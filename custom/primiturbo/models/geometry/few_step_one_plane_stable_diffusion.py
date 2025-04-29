import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint # Keep import if other parts use it, otherwise remove

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.mesh import Mesh
from threestudio.utils.misc import broadcast, get_rank, C
from threestudio.utils.typing import *

from threestudio.utils.ops import get_activation
from threestudio.models.networks import get_encoding, get_mlp

from custom.triplaneturbo.models.geometry.utils import contract_to_unisphere_custom, sample_from_planes
from einops import rearrange

# --- Import KNN/KDN related stuff --- #
from ...extern.knn import HAS_CUDA_KNN, CudaKNNIndex # Original KNN
from ...extern.kdn import HAS_CUDA_KDN, CudaKDNIndex as CudaKDNIndexMahalanobis # New KDN, renamed to avoid conflict
# -----------------------------

# --- PyTorch KDN Reference Implementation (from test_kdn.py) ---
def compute_mahalanobis_sq_torch(
    query_points, # (B, N, D)
    reference_points, # (B, M, D)
    reference_inv_covariances, # (B, M, D, D)
    query_lengths, # (B,)
    reference_lengths, # (B,)
    k, # int
    debug_num_points: Optional[int] = None # Added parameter for limiting points
):
    """Pure PyTorch implementation for calculating Mahalanobis squared distances and finding top K."""
    B, N_orig, D = query_points.shape
    _B, M, _D1, _D2 = reference_inv_covariances.shape
    assert D == _D1 == _D2, "Dimension mismatch"
    assert B == _B, "Batch size mismatch"

    # Limit number of query points if specified for debugging
    N = N_orig
    if debug_num_points is not None:
        N = min(N_orig, debug_num_points)
        query_points = query_points[:, :N, :]
        # Adjust query_lengths accordingly, ensuring it doesn't exceed the new N
        query_lengths = torch.clamp(query_lengths, max=N)
        print(f"[DEBUG TORCH KDN] Limiting verification to first {N} query points.")

    # Allocate distance tensor based on potentially reduced N
    all_distances_sq = torch.full((B, N, M), float('inf'), device=query_points.device, dtype=query_points.dtype)

    for b in range(B):
        # Ensure lengths don't exceed tensor dimensions (original M and potentially reduced N)
        current_query_len = min(query_lengths[b].item(), N) # Use reduced N here
        current_ref_len = min(reference_lengths[b].item(), M)

        if current_query_len <= 0 or current_ref_len <= 0:
            continue
            
        b_query = query_points[b, :current_query_len]       # (N_valid, D)
        b_ref = reference_points[b, :current_ref_len]       # (M_valid, D)
        b_inv_cov = reference_inv_covariances[b, :current_ref_len] # (M_valid, D, D)
        
        N_valid = b_query.shape[0]
        M_valid = b_ref.shape[0]

        # Expand dimensions for broadcasting: (N_valid, 1, D) and (1, M_valid, D)
        diff = b_query.unsqueeze(1) - b_ref.unsqueeze(0) # (N_valid, M_valid, D)
        
        # Corrected einsum for d^T @ Sigma^-1 @ d
        try:
            dist_sq = torch.einsum('nmi,mij,nmj->nm', diff, b_inv_cov, diff)
        except RuntimeError as e:
            print(f"Error during einsum: {e}")
            print(f"Shapes: diff={diff.shape}, b_inv_cov={b_inv_cov.shape}")
            raise e
            
        # Fill the potentially smaller N dimension
        all_distances_sq[b, :N_valid, :M_valid] = dist_sq
        
    # Find top K smallest distances (operates on shape B, N, M)
    safe_k = min(k, M) 
    if safe_k <= 0: 
         top_k_distances_sq = torch.full((B, N, k), float('inf'), device=query_points.device, dtype=query_points.dtype)
         top_k_indices = torch.full((B, N, k), -1, device=query_points.device, dtype=torch.long)
         return top_k_distances_sq, top_k_indices
         
    top_k_distances_sq, top_k_indices = torch.topk(
        all_distances_sq, k=safe_k, dim=-1, largest=False, sorted=True
    )
    
    # Pad results if k > safe_k (i.e., if k > M)
    if k > safe_k:
        pad_size = k - safe_k
        pad_dist = torch.full((B, N, pad_size), float('inf'), device=query_points.device, dtype=top_k_distances_sq.dtype)
        pad_idx = torch.full((B, N, pad_size), -1, device=query_points.device, dtype=top_k_indices.dtype)
        top_k_distances_sq = torch.cat([top_k_distances_sq, pad_dist], dim=-1)
        top_k_indices = torch.cat([top_k_indices, pad_idx], dim=-1)

    inf_mask_in_topk = torch.isinf(top_k_distances_sq)
    top_k_indices[inf_mask_in_topk] = -1 

    # Return tensors shaped (B, N, k) where N might be smaller than original N
    return top_k_distances_sq, top_k_indices.long() 
# --------------------------------------------------------------

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
        xyz_scale: float = 1.0
        
        top_K: int = 8 # Number of nearest neighbors to consider
        knn_backend: str = 'cuda-knn' # Changed default to cuda-knn
        # forward_internal_chunk_size: Optional[int] = None # Removed chunking
        sdf_type: str = "normal_projection" # Options: "normal_projection", "mahalanobis", "none"

        # Neighbor search configuration
        neighbor_search_metric: str = 'l2' # 'l2' for KNN, 'mahalanobis' for KDN

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
        if self.cfg.neighbor_search_metric == 'l2':
            if self.cfg.knn_backend == 'cuda-knn':
                if HAS_CUDA_KNN:
                    threestudio.info("Using CUDA KNN backend (L2 distance).")
                    self.search_mode = 'knn-cuda'
                else:
                    threestudio.warning("CUDA KNN extension requested but not available/compiled. Falling back to torch KNN.")
                    self.search_mode = 'knn-torch' # Keep original torch fallback for KNN
            elif self.cfg.knn_backend == 'torch':
                 threestudio.info("Using PyTorch KNN backend (L2 distance, might be slow).")
                 self.search_mode = 'knn-torch'
            else:
                 raise ValueError(f"Unknown knn_backend: {self.cfg.knn_backend}")
        
        elif self.cfg.neighbor_search_metric == 'mahalanobis':
            if self.cfg.knn_backend == 'cuda-knn':
                 if HAS_CUDA_KDN:
                     threestudio.info("Using CUDA KDN backend (Mahalanobis distance).")
                     self.search_mode = 'kdn-cuda'
                 else:
                     threestudio.error("CUDA KDN extension requested (for Mahalanobis) but not available/compiled. Cannot proceed.")
                     raise ImportError("CUDA KDN extension failed to load.")
            elif self.cfg.knn_backend == 'torch':
                 threestudio.error("Torch backend is not supported for Mahalanobis (KDN) search.")
                 raise ValueError("Cannot use torch backend with Mahalanobis distance.")
            else:
                 raise ValueError(f"Unknown knn_backend: {self.cfg.knn_backend}")
        else:
            raise ValueError(f"Unknown neighbor_search_metric: {self.cfg.neighbor_search_metric}")
        
        if self.search_mode is None:
             raise RuntimeError("Neighbor search mode could not be determined.") # Should not happen

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
                              reference_lengths: Float[Tensor, "B"]):
        """Builds a KNN or KDN index based on the configured search_mode."""
        B, M, D = points.shape
        assert B == 1, "Index building currently only supports batch size 1"

        index = None
        if self.search_mode == 'knn-cuda':
            # Use CudaKNNIndex (original)
            index = CudaKNNIndex()
            # Assume original CudaKNNIndex add method takes (B, M, D) points
            # We might need to adjust this if knn.__init__.py is different
            index.add(points)
            # threestudio.debug(f"Built CUDA KNN index (L2) with {M} points.") # Commented out redundant log
        elif self.search_mode == 'kdn-cuda':
            # Use CudaKDNIndex (Mahalanobis)
            assert inv_covariances is not None, "Inverse covariances needed for KDN index."
            index = CudaKDNIndexMahalanobis()
            index.add(points, inv_covariances, reference_lengths)
            # threestudio.debug(f"Built CUDA KDN index (Mahalanobis) with {M} points.") # Commented out redundant log
        elif self.search_mode == 'knn-torch':
            # Use original TorchKNNIndex fallback
            # threestudio.debug(f"Building Torch KNN index (L2) with {M} points.") # Commented out redundant log
            points_flat = points.squeeze(0).contiguous() # Ensure it's on CPU/correct device later if needed
            
            # Define the fallback class locally or import if it exists elsewhere
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
                    distances, indices = torch.topk(dist_sq, k=min(k, self.M), dim=-1, largest=False) # Ensure k <= M
                    # Handle k > M case by padding
                    if k > self.M:
                        pad_size = k - self.M
                        pad_dist = torch.full((Nq, pad_size), float('inf'), device=query.device, dtype=distances.dtype)
                        pad_idx = torch.full((Nq, pad_size), -1, device=query.device, dtype=indices.dtype)
                        distances = torch.cat([distances, pad_dist], dim=-1)
                        indices = torch.cat([indices, pad_idx], dim=-1)
                    return distances, indices # Return L2 squared
            index = TorchKNNIndex(points_flat)
        else:
            raise NotImplementedError(f"Index building not implemented for search mode: {self.search_mode}")
            
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
             # Assuming B=1, all reference points are valid initially
             ref_lengths = torch.tensor([pc_dict['position'].shape[1]], 
                                        dtype=torch.int64, 
                                        device=pc_dict['position'].device)
             pc_dict['index'] = self._build_neighbor_index(
                 pc_dict['position'], 
                 pc_dict['inv_cov'], # Pass inv_cov, needed for KDN mode
                 ref_lengths
             )
        else:
             print("[Warning] Neighbor index building for batch size > 1 not fully implemented in parse.")
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

        if index is None:
             raise ValueError("KNN index not found in space_cache.")
        if B > 1:
            # Advanced indexing gather assumes B=1, needs adjustment if B>1
            raise NotImplementedError("Forward pass currently assumes B=1 due to gather implementation.")

        # Perform KNN/KDN search for all points at once
        points_flat = points.view(-1, 3) # Shape (B*N, 3)
        # Create query_lengths assuming all points are valid for B=1 case
        query_lengths_flat = torch.full((B,), N, dtype=torch.int64, device=points.device) 
        
        # Use the search method of the created index object
        # Assuming KDN search returns Mahalanobis^2, KNN search returns L2^2
        # We primarily need the indices for gathering
        # Need to adapt call based on index type if signatures differ significantly
        if self.search_mode == 'knn-torch':
            # TorchKNNIndex search expects (Nq, 3) query and k
             distances_sq, indices = index.search(points_flat, k=min(top_K, M))
        elif self.search_mode == 'knn-cuda':
            # Assuming original CudaKNNIndex search expects (B, Nq, 3) query and k?
            # Or maybe (Nq, 3)? Let's assume the latter for now based on old TorchKNNIndex
            # If it expects (B,N,D), need points, not points_flat
            # Let's try passing points_flat, similar to torch mode, assuming it handles it. Needs verification.
             distances_sq, indices = index.search(points_flat, k=min(top_K, M)) 
        elif self.search_mode == 'kdn-cuda':
             # CudaKDNIndex search expects (B, Nq, D) query, (B,) query_lengths, k
             # Need to reshape points_flat and query_lengths_flat to have batch dim
             distances_sq, indices = index.search(points, query_lengths_flat, k=min(top_K, M)) 
        else:
             raise RuntimeError(f"Search not implemented for mode {self.search_mode}")
            
        # Reshape indices and potentially distances to (B, N, K)
        # Indices shape might already be (B*N, K) or (B, N, K) depending on search impl.
        indices = indices.view(B, N, -1) # (B, N, K)
        # Distances are recalculated later using Mahalanobis, so initial distances_sq shape/value less critical
        # distances_sq = distances_sq.view(B, N, -1) # (B, N, K)
        K = indices.shape[-1]

        # --- KDN Debug Verification --- 
        if debug and self.search_mode == 'kdn-cuda':
            debug_points_to_check = 100 # Limit verification to first 100 points
            print(f"\n[DEBUG KDN] Running PyTorch KDN reference verification for first {debug_points_to_check} points...")
            # Need reference lengths for the PyTorch function
            ref_lengths_debug = torch.full((B,), M, dtype=torch.int64, device=points.device)
            # Slice query points for the reference calculation
            query_points_debug = points[:, :debug_points_to_check, :]
            query_lengths_debug = torch.clamp(query_lengths_flat, max=debug_points_to_check)
            
            torch_dist_sq, torch_idx = compute_mahalanobis_sq_torch(
                query_points=query_points_debug, # Use sliced points
                reference_points=gauss_pos, 
                reference_inv_covariances=inv_cov,
                query_lengths=query_lengths_debug, # Use potentially clamped lengths
                reference_lengths=ref_lengths_debug, 
                k=K, 
                debug_num_points=debug_points_to_check # Pass limit to function
            )
            print("[DEBUG KDN] PyTorch calculation done. Comparing results...")
            
            # Compare indices retrieved from CUDA KDN (slice CUDA results)
            cuda_indices = indices[:, :debug_points_to_check, :] # Slice CUDA indices
            indices_match = torch.all(torch_idx == cuda_indices)
            print(f"[DEBUG KDN] Indices Match (first {debug_points_to_check} points): {indices_match}")
            if not indices_match:
                mismatched = torch.where(torch_idx != cuda_indices)
                print(f"  - Found {len(mismatched[0])} mismatches in the first {debug_points_to_check} points.")
                for i in range(min(len(mismatched[0]), 5)):
                    b_idx, n_idx, k_idx = mismatched[0][i], mismatched[1][i], mismatched[2][i]
                    print(f"    Mismatch at B={b_idx.item()}, N={n_idx.item()}, K={k_idx.item()}: Torch={torch_idx[b_idx, n_idx, k_idx].item()}, CUDA={cuda_indices[b_idx, n_idx, k_idx].item()}")

            # Compare distances (slice CUDA results)
            cuda_dist_sq = distances_sq.view(B, N, K)[:, :debug_points_to_check, :] 
            dist_atol = 1e-4
            dist_rtol = 1e-3
            # Ensure shapes match before allclose
            if torch_dist_sq.shape == cuda_dist_sq.shape:
                distances_close = torch.allclose(torch_dist_sq, cuda_dist_sq, atol=dist_atol, rtol=dist_rtol)
                print(f"[DEBUG KDN] Distances Close (first {debug_points_to_check} points, atol={dist_atol}, rtol={dist_rtol}): {distances_close}")
                if not distances_close:
                    diff = torch.abs(torch_dist_sq - cuda_dist_sq)
                    max_diff = torch.max(diff)
                    print(f"  - Max distance difference: {max_diff.item()}")
            else:
                print(f"[DEBUG KDN] Shape mismatch for distance comparison: Torch={torch_dist_sq.shape}, CUDA={cuda_dist_sq.shape}")

            # Optional: Exit after debug check
            print("[DEBUG KDN] Exiting after verification.")
            import os; os._exit(0)
        # --- End KDN Debug Verification ---

        # Gather parameters of the K nearest neighbors (indices are relative to M)
        gathered_pos = gather_gaussian_params(gauss_pos, indices)      # (B, N, K, 3)
        gathered_col = gather_gaussian_params(gauss_col, indices)      # (B, N, K, 3)
        gathered_opa = gather_gaussian_params(gauss_opa, indices)      # (B, N, K, 1)
        gathered_inv_cov = gather_gaussian_params(inv_cov, indices) # (B, N, K, 3, 3)
        gathered_normal = gather_gaussian_params(est_normals, indices)  # (B, N, K, 3) Gather precomputed normals

        # Calculate difference vector relative to K neighbors
        diff = points.unsqueeze(2) - gathered_pos # (B, N, K, 3)

        # Calculate Mahalanobis distance squared for K neighbors
        mahalanobis_sq = torch.einsum("bnki,bnkij,bnkj->bnk", diff, gathered_inv_cov, diff)
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0.0) # Ensure non-negative

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

        # --- Calculate SDF based on cfg.sdf_type ---
        if "normal_projection" in self.cfg.sdf_type:
            signed_dist_k = torch.einsum("bnki,bnki->bnk", diff, gathered_normal) # (B, N, K)
            sdf = torch.sum(norm_weights * signed_dist_k, dim=-1, keepdim=True) # (B, N, 1)
        elif "mahalanobis" in self.cfg.sdf_type:
            mahalanobis_dist_k = torch.sqrt(mahalanobis_sq + 1e-8) # (B, N, K)
            sdf_k = mahalanobis_dist_k - 1.0
            sdf = torch.sum(norm_weights * sdf_k, dim=-1, keepdim=True) # (B, N, 1)
        elif self.cfg.sdf_type == "none":
            sdf = torch.zeros_like(density)
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
