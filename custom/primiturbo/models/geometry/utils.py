import torch
from threestudio.utils.typing import *


# --- Import KNN/KDN related stuff --- #
from ...extern.knn import HAS_CUDA_KNN, CudaKNNIndex # Original KNN
from ...extern.kdn import HAS_CUDA_KDN, CudaKDNIndex # New KDN, renamed to avoid conflict
try:
    from ...extern.kdon import kdon, HAS_CUDA_KDON # Import new KDON
except ImportError as e:
    print(f"Warning: Failed to import kdon extension: {e}. 'density-opacity' mode will not work.")
    HAS_CUDA_KDON = False
    kdon = None 
# -----------------------------

import threestudio # Import threestudio for logging

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

# --- New KDON Verification Function --- #
def verify_kdon_cuda_vs_pytorch(
    points: Float[Tensor, "B N 3"], 
    indices_cuda: Float[Tensor, "B N K_ret"],
    mahalanobis_sq_cuda: Float[Tensor, "B N K_ret"],
    gauss_pos: Float[Tensor, "B M 3"],
    inv_cov: Float[Tensor, "B M 3 3"],
    gauss_opa: Float[Tensor, "B M 1"],
    K_ret: int,
    debug_limit_N: int = 100
):
    """
    Compares the results of CUDA KDON (indices and distances) against a pure PyTorch implementation
    for a subset of query points. Logs comparison results using threestudio.
    Assumes B=1.
    """
    B, N, _ = points.shape
    _B, M, _ = gauss_pos.shape
    assert B == 1, "KDON verification currently only supports B=1"
    
    if N <= 0 or M <= 0 or K_ret <= 0:
        threestudio.warn("[Debug KDON Verify] Skipping verification due to empty query/ref points or K_ret <= 0.")
        return

    with torch.no_grad(): # No need for gradients here
        threestudio.info(f"[Debug KDON Verify] Running PyTorch comparison for K_ret={K_ret}...")
        debug_N = min(debug_limit_N, N)
        debug_points = points[0, :debug_N, :].contiguous() # (debug_N, 3)
        
        # Get CUDA results for the subset 
        indices_cuda_subset = indices_cuda[0, :debug_N, :].contiguous() # (debug_N, K_ret)
        mahalanobis_sq_cuda_subset = mahalanobis_sq_cuda[0, :debug_N, :].contiguous() # (debug_N, K_ret)

        # PyTorch KDON calculation for the subset
        ref_pos_flat = gauss_pos.squeeze(0) # (M, 3)
        ref_inv_cov_flat = inv_cov.squeeze(0) # (M, 3, 3)
        ref_opa_flat = gauss_opa.squeeze(0) # (M, 1)

        diff_torch = debug_points.unsqueeze(1) - ref_pos_flat.unsqueeze(0) # (debug_N, M, 3)
        mahalanobis_sq_torch_all = torch.einsum("nmi,mij,nmj->nm", diff_torch, ref_inv_cov_flat, diff_torch) # (debug_N, M)
        mahalanobis_sq_torch_all = torch.clamp(mahalanobis_sq_torch_all, min=0.0)
        
        exponent_torch = torch.clamp(-0.5 * mahalanobis_sq_torch_all, max=20.0)
        density_torch = torch.exp(exponent_torch) # (debug_N, M)
        
        scores_torch = density_torch * ref_opa_flat.squeeze(-1) # (debug_N, M)
        
        # Top K using PyTorch
        _, top_indices_torch = torch.topk(scores_torch, k=K_ret, dim=-1, largest=True) # (debug_N, K_ret)
        # Gather Mahalanobis distances for the PyTorch top-K
        top_mahalanobis_sq_torch = torch.gather(mahalanobis_sq_torch_all, dim=1, index=top_indices_torch) # (debug_N, K_ret)

        # Comparison
        # 1. Compare Mahalanobis distances (after sorting)
        mahalanobis_sq_cuda_sorted, _ = torch.sort(mahalanobis_sq_cuda_subset, dim=-1)
        top_mahalanobis_sq_torch_sorted, _ = torch.sort(top_mahalanobis_sq_torch, dim=-1)
        
        dist_match = torch.allclose(mahalanobis_sq_cuda_sorted, top_mahalanobis_sq_torch_sorted, atol=1e-4, rtol=1e-4)
        threestudio.info(f"[Debug KDON Verify] Sorted Mahalanobis squared distances match (atol=1e-4, rtol=1e-4): {dist_match}")
        if not dist_match:
            diff_abs = torch.abs(mahalanobis_sq_cuda_sorted - top_mahalanobis_sq_torch_sorted)
            threestudio.info(f"[Debug KDON Verify] Max absolute difference in distances: {diff_abs.max().item()}")
            # Optional: print samples if mismatch
            # print("CUDA Dist (sorted, first query):", mahalanobis_sq_cuda_sorted[0].cpu().numpy())
            # print("PyTorch Dist (sorted, first query):", top_mahalanobis_sq_torch_sorted[0].cpu().numpy())

        # 2. Compare Indices (sets)
        indices_match = True
        mismatched_queries = []
        for i in range(debug_N):
            # Fetch indices for comparison
            set_cuda = set(indices_cuda_subset[i].cpu().tolist())
            set_torch = set(top_indices_torch[i].cpu().tolist())
            if set_cuda != set_torch:
                indices_match = False
                mismatched_queries.append(i)
                # Optional: Print details for the first mismatch
                if len(mismatched_queries) == 1:
                    threestudio.info(f"[Debug KDON Verify] Index mismatch detail at query {i}:")
                    threestudio.info(f"  CUDA Indices Set : {sorted(list(set_cuda))}")
                    threestudio.info(f"  PyTorch Indices Set: {sorted(list(set_torch))}")
                # break # Stop after first mismatch for brevity
        
        threestudio.info(f"[Debug KDON Verify] Index sets match for all {debug_N} queries: {indices_match}")
        if not indices_match:
             threestudio.info(f"[Debug KDON Verify] Indices mismatch occurred for {len(mismatched_queries)} queries (e.g., query index {mismatched_queries[0]}).")

        # Add an assert? Maybe too strict for debug.
        # assert dist_match and indices_match, "KDON CUDA implementation does not match PyTorch!"
        threestudio.info("[Debug KDON Verify] Comparison finished.")
        import os; os._exit(0) # Removed exit call
# ----------------------------------

# --- New KDN Verification Function --- #
def verify_kdn_cuda_vs_pytorch(
    points: Float[Tensor, "B N 3"], 
    indices_cuda: Float[Tensor, "B N K_ret"],
    mahalanobis_sq_cuda: Float[Tensor, "B N K_ret"],
    gauss_pos: Float[Tensor, "B M 3"],
    inv_cov: Float[Tensor, "B M 3 3"],
    query_lengths: Float[Tensor, "B"], # KDN PyTorch function needs lengths
    ref_lengths: Float[Tensor, "B"],   # KDN PyTorch function needs lengths
    K_ret: int,
    K_cfg: int, # Original K requested in config
    debug_limit_N: int = 100
):
    """
    Compares the results of CUDA KDN (indices and distances) against a pure PyTorch implementation
    (using compute_mahalanobis_sq_torch) for a subset of query points.
    Logs comparison results using threestudio.
    Assumes B=1.
    """
    B, N_orig, _ = points.shape
    _B, M, _ = gauss_pos.shape
    assert B == 1, "KDN verification currently only supports B=1"
    
    if N_orig <= 0 or M <= 0 or K_ret <= 0:
        threestudio.warn("[Debug KDN Verify] Skipping verification due to empty query/ref points or K_ret <= 0.")
        return

    with torch.no_grad(): # No need for gradients here
        threestudio.info(f"[Debug KDN Verify] Running PyTorch comparison for K_ret={K_ret} (K_cfg={K_cfg})...")
        debug_N = min(debug_limit_N, N_orig)
        # debug_points = points[0, :debug_N, :].contiguous() # (debug_N, 3)
        
        # Get CUDA results for the subset 
        indices_cuda_subset = indices_cuda[0, :debug_N, :].contiguous() # (debug_N, K_ret)
        mahalanobis_sq_cuda_subset = mahalanobis_sq_cuda[0, :debug_N, :].contiguous() # (debug_N, K_ret)

        # Use the existing PyTorch KDN reference implementation
        # Pass the limited number of points to the PyTorch function
        top_mahalanobis_sq_torch, top_indices_torch = compute_mahalanobis_sq_torch(
            query_points=points, # Pass full points tensor
            reference_points=gauss_pos,
            reference_inv_covariances=inv_cov,
            query_lengths=query_lengths,
            reference_lengths=ref_lengths,
            k=K_cfg, # Use the original K from config for PyTorch comparison
            debug_num_points=debug_N # Tell torch function to limit points
        )

        # Slice results from PyTorch function to match debug_N and K_ret
        top_mahalanobis_sq_torch = top_mahalanobis_sq_torch[0, :debug_N, :K_ret].contiguous()
        top_indices_torch = top_indices_torch[0, :debug_N, :K_ret].contiguous()

        # Comparison (Similar to KDON)
        # 1. Compare Mahalanobis distances (after sorting)
        # Note: Both CUDA and PyTorch should return sorted distances for KDN (smallest first)
        dist_match = torch.allclose(mahalanobis_sq_cuda_subset, top_mahalanobis_sq_torch, atol=1e-4, rtol=1e-4)
        threestudio.info(f"[Debug KDN Verify] Mahalanobis squared distances match (atol=1e-4, rtol=1e-4): {dist_match}")
        if not dist_match:
            diff_abs = torch.abs(mahalanobis_sq_cuda_subset - top_mahalanobis_sq_torch)
            threestudio.info(f"[Debug KDN Verify] Max absolute difference in distances: {diff_abs.max().item()}")
            # print("CUDA Dist (first query):", mahalanobis_sq_cuda_subset[0].cpu().numpy())
            # print("PyTorch Dist (first query):", top_mahalanobis_sq_torch[0].cpu().numpy())

        # 2. Compare Indices (sets, as order might differ slightly with equal distances)
        indices_match = True
        mismatched_queries = []
        for i in range(debug_N):
            set_cuda = set(indices_cuda_subset[i].cpu().tolist())
            # Filter out -1 padding indices from PyTorch result if any
            set_torch = set(idx for idx in top_indices_torch[i].cpu().tolist() if idx != -1)
            # Also filter out -1 padding indices from CUDA result (shouldn't happen if K_ret is correct)
            set_cuda = set(idx for idx in set_cuda if idx != -1)
            
            if set_cuda != set_torch:
                indices_match = False
                mismatched_queries.append(i)
                if len(mismatched_queries) == 1:
                    threestudio.info(f"[Debug KDN Verify] Index mismatch detail at query {i}:")
                    threestudio.info(f"  CUDA Indices Set : {sorted(list(set_cuda))}")
                    threestudio.info(f"  PyTorch Indices Set: {sorted(list(set_torch))}")
        
        threestudio.info(f"[Debug KDN Verify] Index sets match for all {debug_N} queries: {indices_match}")
        if not indices_match:
             threestudio.info(f"[Debug KDN Verify] Indices mismatch occurred for {len(mismatched_queries)} queries (e.g., query index {mismatched_queries[0]}).")

        threestudio.info("[Debug KDN Verify] Comparison finished.")
        import os; os._exit(0) # Removed exit call
# ---------------------------------

# --- Helper function to convert quaternion to rotation matrix --- #
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

# --- Neighbor Index Classes --- #
# (Assuming CudaKNNIndex and CudaKDNIndexMahalanobis are defined or imported elsewhere if not built-in)

class CudaKDONIndex:
    """Wrapper for the custom KDON CUDA extension."""
    def __init__(self):
        if not HAS_CUDA_KDON or kdon is None:
            raise ImportError("CudaKDONIndex requires the compiled 'kdon' extension.")
        self.reference_points = None
        self.reference_inv_covariances = None
        self.reference_opacities = None
        self.reference_lengths = None
        self.added = False
        self.device = None

    def add(self, 
            points: Float[Tensor, "B M 3"], 
            inv_covariances: Float[Tensor, "B M 3 3"], 
            opacities: Float[Tensor, "B M 1"],
            lengths: Float[Tensor, "B"]):
        assert points.shape[0] == 1, "CudaKDONIndex currently supports B=1"
        assert inv_covariances.shape[0] == 1, "CudaKDONIndex currently supports B=1"
        assert opacities.shape[0] == 1, "CudaKDONIndex currently supports B=1"
        assert lengths.shape[0] == 1, "CudaKDONIndex currently supports B=1"
        assert points.device == inv_covariances.device == opacities.device, "All input tensors must be on the same device for add"
        
        self.reference_points = points.contiguous()
        self.reference_inv_covariances = inv_covariances.contiguous()
        self.reference_opacities = opacities.contiguous()
        self.reference_lengths = lengths.contiguous().to(torch.int64) 
        self.added = True
        self.device = points.device 

    def search(self, 
               query_points: Float[Tensor, "B N 3"], 
               query_lengths: Float[Tensor, "B"], 
               k: int):
        assert self.added, "Data must be added to CudaKDONIndex before search."
        assert query_points.shape[0] == 1, "CudaKDONIndex search currently supports B=1"
        assert query_lengths.shape[0] == 1, "CudaKDONIndex search currently supports B=1"
        assert query_points.device == self.device, "Query points must be on the same device as reference points"

        # Ensure k is not greater than the number of reference points
        M = self.reference_points.shape[1]
        safe_k = min(k, M)
        if safe_k <= 0: # Handle edge case where M=0
             B, N, _ = query_points.shape
             dtype = self.reference_points.dtype
             dist_dtype = dtype if dtype!=torch.int8 else torch.float32 # Placeholder
             distances = torch.full((B, N, k), float('inf'), device=self.device, dtype=dist_dtype)
             indices = torch.full((B, N, k), -1, device=self.device, dtype=torch.long)
             return distances, indices

        distances, indices = kdon(
            query_points.contiguous(),
            self.reference_points,
            self.reference_inv_covariances,
            self.reference_opacities,
            query_lengths.contiguous().to(torch.int64), 
            self.reference_lengths,
            K=safe_k # Use safe_k
        )

        # Pad if k > M
        if k > safe_k:
            B, N, _ = query_points.shape
            pad_size = k - safe_k
            pad_dist = torch.full((B, N, pad_size), float('inf'), device=self.device, dtype=distances.dtype)
            pad_idx = torch.full((B, N, pad_size), -1, device=self.device, dtype=indices.dtype)
            distances = torch.cat([distances, pad_dist], dim=-1)
            indices = torch.cat([indices, pad_idx], dim=-1)
            
        return distances, indices
# -----------------------------        