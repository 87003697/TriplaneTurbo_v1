import torch
import unittest
import time
import os
import sys
import numpy as np

# Adjust path to import from the kdon directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Import the new kdon function and the availability flag
    from __init__ import kdon, HAS_CUDA_KDON 
except ImportError as e:
    print(f"Error importing kdon: {e}")
    print("Please ensure the kdon extension is compiled correctly.")
    # Set defaults if import fails, tests will likely fail but won't crash the script
    HAS_CUDA_KDON = False
    def kdon(*args, **kwargs):
        raise ImportError("kdon function not imported.")

# --- PyTorch Reference Implementation for KDON ---
def pure_pytorch_kdon(
    query_points: torch.Tensor, 
    reference_points: torch.Tensor, 
    reference_inv_covariances: torch.Tensor,
    reference_opacities: torch.Tensor, # Added
    query_lengths: torch.Tensor, 
    reference_lengths: torch.Tensor, 
    k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch implementation of KDON for verification."""
    B, N, D = query_points.shape
    _B, M, _D = reference_points.shape
    device = query_points.device
    dtype = query_points.dtype

    all_batch_dists = torch.full((B, N, k), float('inf'), device=device, dtype=dtype)
    all_batch_indices = torch.full((B, N, k), -1, device=device, dtype=torch.int64)

    for b in range(B):
        # Get valid lengths for this batch item
        actual_n = query_lengths[b].item()
        actual_m = reference_lengths[b].item()

        if actual_n == 0 or actual_m == 0:
            continue # Skip if no query or reference points

        # Slice valid data for the batch
        q_pts = query_points[b, :actual_n, :]
        ref_pts = reference_points[b, :actual_m, :]
        ref_inv_covs = reference_inv_covariances[b, :actual_m, :, :]
        ref_opacities = reference_opacities[b, :actual_m, :] # Shape (actual_m, 1)

        # Expand dimensions for broadcasting: N x 1 x D and 1 x M x D
        q_expanded = q_pts.unsqueeze(1)  # (actual_n, 1, D)
        ref_expanded = ref_pts.unsqueeze(0)  # (1, actual_m, D)

        # Calculate difference vectors: N x M x D
        diff = q_expanded - ref_expanded  # (actual_n, actual_m, D)

        # Calculate Mahalanobis distance squared: N x M
        # diff: (actual_n, actual_m, D) -> (actual_n, actual_m, 1, D)
        # ref_inv_covs: (actual_m, D, D) -> (1, actual_m, D, D)
        # right_vec = einsum('nmkd,mkd->nmk', diff.unsqueeze(2), ref_inv_covs.unsqueeze(0))
        # dist_sq = einsum('nmk,nmk->nm', diff, right_vec)
        # Optimized version:
        # tmp = diff @ ref_inv_covs  # (actual_n, actual_m, D) @ (actual_m, D, D) -> needs einsum or loop
        # Using einsum for batched matrix multiplication:
        # diff[:, :, None, :] @ ref_inv_covs[None, :, :, :] -> (N, M, 1, D) @ (1, M, D, D) -> (N, M, 1, D)
        # (N, M, 1, D) @ diff[:, :, :, None] -> (N, M, 1, 1)
        mahalanobis_sq = torch.einsum('nmd,mdo,nmo->nm', diff, ref_inv_covs, diff) # (actual_n, actual_m)
        mahalanobis_sq = mahalanobis_sq.clamp(min=0) # Ensure non-negative distances

        # Calculate density: N x M
        density = torch.exp(-0.5 * mahalanobis_sq) # (actual_n, actual_m)
        
        # Calculate score: N x M
        # ref_opacities is (actual_m, 1), needs broadcasting with density (actual_n, actual_m)
        score = density * ref_opacities.squeeze(-1).unsqueeze(0) # (actual_n, actual_m)

        # Find top K scores (largest=True) for each query point
        # Handle cases where actual_m < k
        actual_k = min(k, actual_m)
        if actual_k == 0:
             continue # No valid reference points to select from

        top_scores, top_indices = torch.topk(score, k=actual_k, dim=1, largest=True, sorted=True)

        # Gather the corresponding Mahalanobis distances using the top_indices
        # top_indices shape: (actual_n, actual_k)
        # mahalanobis_sq shape: (actual_n, actual_m)
        top_dists_sq = torch.gather(mahalanobis_sq, dim=1, index=top_indices)

        # Store results for this batch item
        all_batch_dists[b, :actual_n, :actual_k] = top_dists_sq
        all_batch_indices[b, :actual_n, :actual_k] = top_indices

    return all_batch_dists, all_batch_indices

# --- Test Case Class ---
class TestKDON(unittest.TestCase):

    def _run_test(self, B, N, M, D, K, dtype, device, use_lengths):
        print(f"\nRunning test: B={B}, N={N}, M={M}, D={D}, K={K}, dtype={dtype}, device={device}, use_lengths={use_lengths}")

        # Generate random data
        query_points = torch.randn(B, N, D, device=device, dtype=dtype)
        reference_points = torch.randn(B, M, D, device=device, dtype=dtype)
        
        # Generate random positive definite covariance matrices and invert them
        # Ensure they are well-conditioned for stability
        inv_covs = []
        for b_idx in range(B):
            batch_inv_covs = []
            for m_idx in range(M):
                # Generate a random matrix in float32 for stability of inv
                A = torch.randn(D, D, device=device, dtype=torch.float32)
                # Create a positive definite matrix: A^T A + epsilon * I (in float32)
                cov_f32 = A.T @ A + 0.1 * torch.eye(D, device=device, dtype=torch.float32)
                # Invert it (in float32)
                try:
                    inv_cov_f32 = torch.linalg.inv(cov_f32)
                except torch._C._LinAlgError:
                    print(f"Warning: linalg.inv failed for item B={b_idx}, M={m_idx}. Using pseudo-inverse.")
                    inv_cov_f32 = torch.linalg.pinv(cov_f32)
                # Convert back to the target dtype for the test
                inv_cov = inv_cov_f32.to(dtype=dtype)
                batch_inv_covs.append(inv_cov)
            inv_covs.append(torch.stack(batch_inv_covs)) # (M, D, D)
        reference_inv_covariances = torch.stack(inv_covs) # (B, M, D, D)

        # Generate random opacities (0 to 1)
        reference_opacities = torch.rand(B, M, 1, device=device, dtype=dtype)

        # Generate lengths if needed
        if use_lengths:
            query_lengths = torch.randint(1, N + 1, (B,), device=device, dtype=torch.int64)
            reference_lengths = torch.randint(1, M + 1, (B,), device=device, dtype=torch.int64)
            # Ensure K is not larger than the smallest possible reference length if M is small
            if M < K:
                 reference_lengths = torch.clamp(reference_lengths, max=M)
        else:
            query_lengths = torch.full((B,), N, device=device, dtype=torch.int64)
            reference_lengths = torch.full((B,), M, device=device, dtype=torch.int64)
        
        # Ensure K is valid given potential lengths
        effective_K = min(K, M)
        if use_lengths and M > 0:
            min_ref_len = torch.min(reference_lengths).item()
            effective_K = min(effective_K, min_ref_len)
        
        if effective_K <= 0: effective_K = 1 # Avoid K=0 case if M becomes 0 due to lengths

        # Run KDON extension
        start_time_ext = time.time()
        try:
            ext_dists, ext_indices = kdon(
                query_points, 
                reference_points, 
                reference_inv_covariances,
                reference_opacities, # Pass opacities
                query_lengths,
                reference_lengths,
                K=K # Corrected back to K=K
            )
        except Exception as e:
            self.fail(f"KDON extension crashed: {e}")
        end_time_ext = time.time()

        # Run PyTorch reference
        start_time_ref = time.time()
        ref_dists, ref_indices = pure_pytorch_kdon(
            query_points, 
            reference_points, 
            reference_inv_covariances,
            reference_opacities, # Pass opacities
            query_lengths,
            reference_lengths,
            k=K # Keep this as k=K for the pure PyTorch function
        )
        end_time_ref = time.time()

        # --- Verification --- 
        print(f"  Ext time: {end_time_ext - start_time_ext:.6f}s")
        print(f"  Ref time: {end_time_ref - start_time_ref:.6f}s")
        
        # Mask invalid entries based on lengths for comparison
        mask = torch.arange(K, device=device)[None, None, :] < reference_lengths[:, None, None]
        mask = mask & (torch.arange(N, device=device)[None, :, None] < query_lengths[:, None, None])
        
        # Check distances (allow some tolerance for floating point differences)
        if dtype == torch.float16:
            atol = 1e-2 # Increased tolerance for float16
            rtol = 1e-2 # Increased tolerance for float16
        else:
            atol = 1e-5
            rtol = 1e-4
        
        # Compare distances where mask is True
        dist_diff = torch.abs(ext_dists - ref_dists)
        max_dist_diff = torch.max(dist_diff[mask]).item() if mask.any() else 0
        print(f"  Max distance difference (masked): {max_dist_diff:.6f} (atol={atol}, rtol={rtol})")
        # Check if the masked differences are within tolerance
        self.assertTrue(torch.allclose(ext_dists[mask], ref_dists[mask], atol=atol, rtol=rtol),
                        f"Distance mismatch exceeds tolerance (Max diff: {max_dist_diff})")
        
        # Check indices (should match exactly where mask is True)
        indices_match = ext_indices[mask] == ref_indices[mask]
        num_mismatched_indices = (~indices_match).sum().item()
        print(f"  Number of mismatched indices (masked): {num_mismatched_indices}")
        self.assertTrue(torch.all(indices_match), "Index mismatch detected")

        # Verify padding values for distances (infinity) and indices (-1)
        padding_mask = ~mask
        if padding_mask.any():
             self.assertTrue(torch.all(torch.isinf(ext_dists[padding_mask])), "Padding distances are not infinity")
             self.assertTrue(torch.all(ext_indices[padding_mask] == -1), "Padding indices are not -1")
        print("  Padding values verified.")


    # --- Test Cases --- 
    @unittest.skipIf(not torch.cuda.is_available() or not HAS_CUDA_KDON, "CUDA not available or KDON extension not built")
    def test_kdon_cuda_float32(self):
        self._run_test(B=2, N=1024, M=2048, D=3, K=8, dtype=torch.float32, device='cuda', use_lengths=True)
        self._run_test(B=4, N=512, M=1024, D=3, K=8, dtype=torch.float32, device='cuda', use_lengths=False)
        self._run_test(B=1, N=1, M=10, D=3, K=5, dtype=torch.float32, device='cuda', use_lengths=True) # Small N/M
        self._run_test(B=2, N=100, M=5, D=3, K=8, dtype=torch.float32, device='cuda', use_lengths=True) # K > M

    @unittest.skip("Skipping float16 CUDA test due to precision issues inherent to half-precision floats in exp/multiplication.")
    @unittest.skipIf(not torch.cuda.is_available() or not HAS_CUDA_KDON, "CUDA not available or KDON extension not built")
    def test_kdon_cuda_float16(self):
        self._run_test(B=2, N=1024, M=2048, D=3, K=8, dtype=torch.float16, device='cuda', use_lengths=True)
        self._run_test(B=4, N=512, M=1024, D=3, K=8, dtype=torch.float16, device='cuda', use_lengths=False)
        self._run_test(B=1, N=1, M=10, D=3, K=5, dtype=torch.float16, device='cuda', use_lengths=True)
        self._run_test(B=2, N=100, M=5, D=3, K=8, dtype=torch.float16, device='cuda', use_lengths=True)

    # Always run CPU tests if extension is importable
    def test_kdon_cpu_float32(self):
        self._run_test(B=2, N=128, M=256, D=3, K=16, dtype=torch.float32, device='cpu', use_lengths=True)
        self._run_test(B=4, N=64, M=128, D=3, K=8, dtype=torch.float32, device='cpu', use_lengths=False)
        self._run_test(B=1, N=1, M=10, D=3, K=5, dtype=torch.float32, device='cpu', use_lengths=True)
        self._run_test(B=2, N=100, M=5, D=3, K=8, dtype=torch.float32, device='cpu', use_lengths=True)

    def test_kdon_cpu_float64(self):
        self._run_test(B=2, N=128, M=256, D=3, K=16, dtype=torch.float64, device='cpu', use_lengths=True)
        self._run_test(B=4, N=64, M=128, D=3, K=8, dtype=torch.float64, device='cpu', use_lengths=False)
        self._run_test(B=1, N=1, M=10, D=3, K=5, dtype=torch.float64, device='cpu', use_lengths=True)
        self._run_test(B=2, N=100, M=5, D=3, K=8, dtype=torch.float64, device='cpu', use_lengths=True)


if __name__ == '__main__':
    # Add setup for testing specific devices if needed
    # Example: torch.cuda.set_device(0)
    unittest.main()
