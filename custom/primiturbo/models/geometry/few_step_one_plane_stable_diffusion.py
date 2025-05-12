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
                "require_intermediate_features": False,
            }
        )

        backbone: str = "few_step_one_plane_stable_diffusion"

        scaling_activation: str = "exp-0.1" # in ["exp-0.1", "sigmoid", "exp", "softplus"]
        opacity_activation: str = "sigmoid-0.1" # in ["sigmoid-0.1", "sigmoid", "sigmoid-mipnerf", "softplus"]
        rotation_activation: str = "normalize" # in ["normalize"]
        color_activation: str = "none" # in ["scale_-11_01", "sigmoid-mipnerf"]
        position_activation: str = "none" # in ["none"]
        
        xyz_center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
        xyz_scale: float = 1.0
        
        top_K: int = 8 # Number of nearest neighbors to consider
        knn_backend: str = 'cuda-knn' # Changed default to cuda-knn
        sdf_type: str = "none" # Options: "normal_projection", "mahalanobis", "mahalanobis_squared", "signed_mahalanobis", "signed_mahalanobis_squared", "none"
        gather_with_opacity: bool = True # whether to gather features with opacity in the knn aggregation
        neighbor_search_metric: str = 'l2'

        # --- New configs for hierarchical parsing ---
        hierarchical_parsing: bool = False
 
        scale_delta_residual_const: float = -0.1 # This is delta s, should be < 0
        render_intermediate_levels: bool = False # If True, parse() returns a list of pc_dict for each rendered level

    def configure(self) -> None:
        super().configure()

        print("The current device is: ", self.device)
        
        # Ensure space_generator_config reflects the need for intermediate features if hierarchical parsing is on
        if self.cfg.hierarchical_parsing and not self.cfg.space_generator_config.get("require_intermediate_features", False):
            threestudio.warn("hierarchical_parsing is True, but space_generator_config.require_intermediate_features is False. Forcing it to True.")
            self.cfg.space_generator_config["require_intermediate_features"] = True

        if self.cfg.backbone == "few_step_one_plane_stable_diffusion":
            from ...extern.few_step_one_plane_sd_modules import FewStepOnePlaneStableDiffusion as Generator
            self.space_generator = Generator(self.cfg.space_generator_config)
        else:
            raise ValueError(f"Unknown backbone {self.cfg.backbone}")
        
        self.scaling_activation_fn = get_activation(self.cfg.scaling_activation)
        self.opacity_activation_fn = get_activation(self.cfg.opacity_activation)
        self.rotation_activation_fn = get_activation(self.cfg.rotation_activation)
        self.color_activation_fn = get_activation(self.cfg.color_activation)
        self.position_activation_fn = get_activation(self.cfg.position_activation)

        self.xyz_center_t = lambda x: torch.tensor(self.cfg.xyz_center, device=x.device, dtype=x.dtype)

        # No hierarchical-specific activations to configure here anymore, as they are handled by the main ones or are identity.

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
        
        threestudio.debug(f"Determined search mode: {self.search_mode}")

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
                              reference_opacities: Optional[Float[Tensor, "B M 1"]],
                              reference_lengths: Float[Tensor, "B"]):
        """Builds a KNN or KDN index based on the configured search_mode."""
        B, M, D = points.shape
        assert B == 1, "Index building currently only supports batch size 1"

        index = None
        # Use the pre-determined search_mode from configure()
        search_mode = self.search_mode 

        threestudio.debug(f"Building index for search mode: {search_mode}")

        if search_mode == 'knn-cuda': 
            index = CudaKNNIndex() 
            index.add(points)
        elif search_mode == 'kdn-cuda': 
            assert inv_covariances is not None, "Inverse covariances needed for KDN index."
            index = CudaKDNIndex() 
            index.add(points, inv_covariances, reference_lengths)
        elif search_mode == 'kdon-cuda':
             assert inv_covariances is not None, "Inverse covariances needed for KDON index."
             assert reference_opacities is not None, "Reference opacities needed for KDON index."
             index = CudaKDONIndex()
             index.add(points, inv_covariances, reference_opacities, reference_lengths)
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
                    if safe_k <= 0:
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
            raise NotImplementedError(f"Index building not implemented for determined search mode: {search_mode}")
            
        return index

    def parse(
        self,
        decoded_outputs: Union[Float[Tensor, "B N Cout Hin Win"], List[Float[Tensor, "B N Ck Hk Wk"]]], # Adjusted N from 3 to generic N
        scale_factor: float, # Note: This parameter is NOT used by the rearrange logic below.
    ) -> Dict[str, Any]: # Ensure return type is List of pc_dicts

        ch_col_start, ch_col_end = 0, 3
        ch_pos_start, ch_pos_end = 3, 6
        ch_scl_start, ch_scl_end = 6, 9
        ch_rot_start, ch_rot_end = 9, 13
        ch_opa_start, ch_opa_end = 13, 14
        
        # --- Define interpolate_and_extract_params helper function at the beginning of parse method --- 
        def interpolate_and_extract_params(triplane_level_feat, level_idx, is_residual_level):
            # num_total_channels_level check (removed as level_output_dims is removed from config)
            # Assuming triplane_level_feat.shape[2] is always the expected total channels (e.g., 14)
            # If level-specific channel counts were needed, level_output_dims would be necessary.
            # For now, assume fixed total channels for all levels based on overall output_dim.
            
            # Simplified channel check based on space_generator_config.output_dim
            # This assumes all levels produce the same number of total raw channels.
            # If different levels have different output_dims, this logic needs level_output_dims config again.
            expected_total_channels = self.cfg.space_generator_config["output_dim"]
            if triplane_level_feat.shape[2] != expected_total_channels:
                    threestudio.warn(
                    f"Level {level_idx} triplane feature channel dim ({triplane_level_feat.shape[2]}) "
                    f"does not match expected total channels ({expected_total_channels}) from space_generator_config.output_dim. "
                    "Ensure channel splits and config are correct."
                )

            if ch_opa_end > triplane_level_feat.shape[2]: # Use actual channels from feature
                raise ValueError(
                    f"Level {level_idx}: Channel split for opacity (up to {ch_opa_end}) "
                    f"exceeds total available channels ({triplane_level_feat.shape[2]}) in the feature map."
                )

            # Intermediate activations within this function:
            # ALL parameters are passed through as raw network outputs from this function.
            # Final activations (e.g., exp for scale, sigmoid for opacity) will be
            # applied ONCE at the end of the main parse() method after all levels are accumulated.
            
            _color_int_act = lambda x: x
            _pos_int_act = lambda x: x
            _scl_int_act = lambda x: x 
            _rot_int_act = lambda x: x
            _opa_int_act = lambda x: x

            # No special handling for is_residual_level for _scl_int_act needed anymore,
            # as residual_scale_activation_fn has been removed. Raw output is always used.

            params = {
                "color": _color_int_act(rearrange(triplane_level_feat[:, :, ch_col_start:ch_col_end, :, :], "B N C H W -> B (N H W) C")),
                "position": _pos_int_act(rearrange(triplane_level_feat[:, :, ch_pos_start:ch_pos_end, :, :], "B N C H W -> B (N H W) C")),
                "scale": _scl_int_act(rearrange(triplane_level_feat[:, :, ch_scl_start:ch_scl_end, :, :], "B N C H W -> B (N H W) C")),
                "rotation": _rot_int_act(rearrange(triplane_level_feat[:, :, ch_rot_start:ch_rot_end, :, :], "B N C H W -> B (N H W) C")),
                "opacity": _opa_int_act(rearrange(triplane_level_feat[:, :, ch_opa_start:ch_opa_end, :, :], "B N C H W -> B (N H W) C")),
            }
            return params
        # --- End of interpolate_and_extract_params --- (now defined for the whole parse method)

        if not self.cfg.hierarchical_parsing:
            # --- Original Single Triplane Parsing Logic --- (Now uses the helper function)
            if isinstance(decoded_outputs, list):
                if len(decoded_outputs) > 1:
                    threestudio.warn("Received a list of triplanes in parse() but hierarchical_parsing is False. Using only the last triplane.")
                triplane = decoded_outputs[-1]
            else:
                triplane = decoded_outputs
            
            # triplane is expected to be (B, N_planes, C_total, H, W)
            # B = triplane.shape[0] # B is implicitly handled by rearrange and _activate_params_for_output

            # Use the helper function to get raw parameters
            raw_pc_dict = interpolate_and_extract_params(triplane_level_feat=triplane, level_idx=0, is_residual_level=False)
            
            # Now, apply activations using the unified helper function
            pc_dict = self._activate_params_for_output(raw_pc_dict)

        
        else:
            # --- Hierarchical Parsing Logic (GSGAN-like) ---
            if not isinstance(decoded_outputs, list) or not decoded_outputs:
                raise ValueError("Hierarchical parsing enabled, but decoded_outputs is not a non-empty list of triplanes.")
            
            num_levels = len(decoded_outputs)
            # Removed level_output_dims checks as the config was removed.
            # Add back if level-specific output dims become necessary.

            B = decoded_outputs[0].shape[0]
            collected_raw_params_list = [] # List to store raw pc_dicts for each rendered level if enabled
            current_params = {} # Stores params in raw/log space for accumulation

            for level_idx in range(num_levels):
                triplane_l = decoded_outputs[level_idx] 
                is_residual = level_idx > 0
                
                # parsed_l now contains raw network outputs (or raw residuals if is_residual_level for scale)
                parsed_l = interpolate_and_extract_params(triplane_l, level_idx, is_residual)

                if not is_residual: # Base level (l=0)
                    # Store raw outputs from the network
                    current_params["position"] = parsed_l["position"] 
                    current_params["scale"] = parsed_l["scale"]    # Expecting log-scale
                    current_params["rotation"] = parsed_l["rotation"] 
                    current_params["opacity"] = parsed_l["opacity"]  # Expecting logit-opacity
                    current_params["color"] = parsed_l["color"]
                else: # Residual levels (l > 0)
                    num_points_current_level = parsed_l["position"].shape[1]
                    num_points_prev_level = current_params["position"].shape[1]
                    
                    assert num_points_current_level > num_points_prev_level and num_points_current_level % num_points_prev_level == 0, "Current level must have more points and be a multiple of the previous level."
                    ratio = num_points_current_level // num_points_prev_level
                    for k_param_key in current_params: 
                        current_params[k_param_key] = current_params[k_param_key].repeat_interleave(ratio, dim=1)
                    
                    mu_prev_upsampled = current_params["position"]
                    log_scale_prev_upsampled = current_params["scale"] # This is s^{l-1} (log-scale)
                    quat_prev_upsampled = current_params["rotation"]
                    # color_prev_upsampled = current_params["color"] (raw)
                    # opacity_prev_upsampled = current_params["opacity"] (raw logits)
                    
                    # --- Position Update (Eqn. 3: mu^l = mu^{l-1} + R^{l-1}S^{l-1}hat_mu^l) ---
                    # hat_mu_l is raw residual from parsed_l["position"]
                    # For position update, R and S need to be in physical space.
                    # S^{l-1} from log_scale_prev_upsampled (physical scales for transformation)
                    # R^{l-1} from quat_prev_upsampled
                    R_prev_matrix = quat_to_rot_matrix(self.rotation_activation_fn(quat_prev_upsampled)) # Activate previous raw rotation for matrix
                    physical_scale_prev_diag = self.scaling_activation_fn(log_scale_prev_upsampled) # Activate previous raw log-scale for transformation
                    
                    hat_mu_l = parsed_l["position"] # raw local offset residual
                    
                    scaled_hat_mu_l = physical_scale_prev_diag * hat_mu_l 
                    world_offset = torch.einsum('bmik,bmk->bmi', R_prev_matrix, scaled_hat_mu_l)
                    current_params["position"] = mu_prev_upsampled + world_offset

                    # --- Scale Update (GSGAN-like, Option 2) ---
                    # parsed_l["scale"] is hat_s^l_pred (raw from net after residual_scale_activation_fn, which is identity by default)
                    raw_network_predicted_scale_delta = parsed_l["scale"] 
                    delta_s_const = self.cfg.scale_delta_residual_const
                    gsgan_effective_delta = -1 * torch.nn.functional.softplus(-(raw_network_predicted_scale_delta - delta_s_const)) + delta_s_const
                    current_params["scale"] = log_scale_prev_upsampled + gsgan_effective_delta # Accumulate in log-space
                    
                    # --- Rotation Update ---
                    # Additive residual for raw quaternion delta
                    current_params["rotation"] = quat_prev_upsampled + parsed_l["rotation"]
                    # Normalization will be part of the final activation for rotation
                    
                    # --- Color Update ---
                    # Additive residual for raw color
                    current_params["color"] = current_params["color"] + parsed_l["color"]
                    
                    # --- Opacity Update ---
                    # Additive residual for raw logit opacity
                    current_params["opacity"] = current_params["opacity"] + parsed_l["opacity"]

                # --- End of parameter updates for the current level_idx --- 

                if self.cfg.render_intermediate_levels:
                    collected_raw_params_list.append({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in current_params.items()})
                    
            # --- End of for level_idx in range(num_levels) loop --- 

            # Determine the set of raw parameters to process for final output
            if self.cfg.render_intermediate_levels:
                pc_dict = {}
                for name in current_params.keys():
                    pc_dict[name] = torch.cat([d[name] for d in collected_raw_params_list], dim=1)
                pc_dict = self._activate_params_for_output(pc_dict)
            else:
                pc_dict = self._activate_params_for_output(current_params)

        pc_dict['inv_cov'] = build_inverse_covariance(pc_dict['scale'], pc_dict['rotation'])

        scales_final = pc_dict['scale'] 
        quats_final = pc_dict['rotation']
        B_final, M_final, _ = scales_final.shape

        min_scale_indices = torch.argmin(scales_final, dim=-1)
        rot_mats = quat_to_rot_matrix(quats_final) 

        min_scale_indices_exp = min_scale_indices.view(B_final, M_final, 1, 1).expand(B_final, M_final, 3, 1)
        est_normals = torch.gather(rot_mats, 3, min_scale_indices_exp).squeeze(-1)
        pc_dict['normal'] = est_normals

        if pc_dict['position'].shape[0] == 1: 
            ref_lengths = torch.tensor([pc_dict['position'].shape[1]], dtype=torch.int64, device=pc_dict['position'].device)
            pc_dict['index'] = self._build_neighbor_index(
                pc_dict['position'], 
                pc_dict['inv_cov'], 
                pc_dict.get('opacity'), 
                ref_lengths
            )
        else:
            threestudio.debug("[Warning] Neighbor index building for batch size > 1 not typically done in parse. Setting index to None.")
            pc_dict['index'] = None 
        
        return pc_dict # Return the single, fully processed pc_dict


    def _activate_params_for_output(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """Applies final activations to a dictionary of raw (log/logit space) parameters."""
        # Ensure we are working with a copy to not modify the input dict if it's used elsewhere
        # (e.g. current_params that needs to remain raw for next iteration)
        activated_params = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in raw_params.items()}

        # Position: apply scaling, centering, and then configured activation
        # Ensure 'position' key exists before trying to access it for sum for centering
        if "position" in activated_params:
            position_sum_for_centering = activated_params["position"].sum(dim=(1,2), keepdim=True) if activated_params["position"].ndim >=3 else activated_params["position"].sum(dim=0, keepdim=True) # Handle B M C or M C
            activated_params["position"] = activated_params["position"] * self.cfg.xyz_scale + self.xyz_center_t(position_sum_for_centering)
            activated_params["position"] = self.position_activation_fn(activated_params["position"])
        
        if "scale" in activated_params:
            activated_params["scale"] = self.scaling_activation_fn(activated_params["scale"]) 
        if "opacity" in activated_params:
            activated_params["opacity"] = self.opacity_activation_fn(activated_params["opacity"]) 
        if "color" in activated_params:
            activated_params["color"] = self.color_activation_fn(activated_params["color"]) 
        if "rotation" in activated_params:
            activated_params["rotation"] = self.rotation_activation_fn(activated_params["rotation"])
        
        return activated_params
    
    def interpolate_encodings(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"], # Type hint might need update based on actual space_cache
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
        points: Float[Tensor, "*N Di"],
        space_cache: Dict[str, Any], # Changed type hint to Dict
        output_normal: bool = False,
        debug: bool = False,
    ) -> Dict[str, Float[Tensor, "..."]]:
        """
        Computes weighted mixture properties (color, density) and estimates SDF using
        the specified method (cfg.sdf_type). Uses KNN for efficiency.
        """
        calculate_avg_normal_output = output_normal
        B, N, _ = points.shape
        top_K = self.cfg.top_K

        gauss_pos = space_cache['position']
        gauss_col = space_cache['color']
        gauss_opa = space_cache['opacity']
        inv_cov = space_cache['inv_cov']
        index = space_cache['index']
        est_normals = space_cache['normal']
        M = gauss_pos.shape[1]
        # ref_lengths should be created based on gauss_pos's structure if B > 1 in space_cache
        # For B=1, this is fine.
        ref_lengths = torch.tensor([M], dtype=torch.int64, device=points.device)

        if index is None:
             raise ValueError("KNN index not found in space_cache.")
        if B > 1:
            raise NotImplementedError("Forward pass currently assumes B=1 due to gather implementation.")

        points_flat = points.view(-1, 3) 
        query_lengths_flat = torch.full((B,), N, dtype=torch.int64, device=points.device) 
        
        search_k = min(top_K, M) 
        
        if search_k <= 0: # No neighbors to search for if M is 0 or top_K is 0
            indices = torch.full((B, N, 0), -1, device=points.device, dtype=torch.long)
        
        elif self.search_mode == 'knn-torch':
             _, indices = index.search(points_flat, k=search_k) # index is TorchKNNIndex
        elif self.search_mode == 'knn-cuda':
             _, indices = index.search(points_flat, k=search_k) # index is CudaKNNIndex
        elif self.search_mode == 'kdn-cuda':
             # kdn-cuda search expects points (B,N,3), query_lengths (B,), k
             _, indices = index.search(points, query_lengths_flat, k=search_k) 
        elif self.search_mode == 'kdon-cuda': 
             _, indices = index.search(points, query_lengths_flat, k=search_k)
        else:
             raise RuntimeError(f"Search not implemented for mode {self.search_mode}")
            
        indices = indices.view(B, N, -1) # Ensure shape is B, N, K_ret
        K_ret = indices.shape[-1] 

        gathered_pos = gather_gaussian_params(gauss_pos, indices)
        gathered_col = gather_gaussian_params(gauss_col, indices)
        gathered_opa = gather_gaussian_params(gauss_opa, indices)
        gathered_inv_cov = gather_gaussian_params(inv_cov, indices)
        gathered_normal = gather_gaussian_params(est_normals, indices)

        diff = points.unsqueeze(2) - gathered_pos
        
        mahalanobis_sq = torch.einsum("bnki,bnkij,bnkj->bnk", diff, gathered_inv_cov, diff)
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0.0) 

        if debug and self.search_mode == 'kdn-cuda' and B == 1:
            verify_kdn_cuda_vs_pytorch(
                points=points,
                indices_cuda=indices,
                mahalanobis_sq_cuda=mahalanobis_sq,
                gauss_pos=gauss_pos,
                inv_cov=inv_cov,
                query_lengths=query_lengths_flat,
                ref_lengths=ref_lengths,
                K_ret=K_ret, 
                K_cfg=top_K,
                debug_limit_N=100
            )

        if debug and self.search_mode == 'kdon-cuda' and B == 1:
            verify_kdon_cuda_vs_pytorch(
                points=points,
                indices_cuda=indices, 
                mahalanobis_sq_cuda=mahalanobis_sq,
                gauss_pos=gauss_pos,
                inv_cov=inv_cov,
                gauss_opa=gauss_opa,
                K_ret=K_ret,
                debug_limit_N=100 
            )

        exponent = torch.clamp(-0.5 * mahalanobis_sq, max=20.0)
        gauss_density = torch.exp(exponent)

        weights = gauss_density  * (gathered_opa.squeeze(-1) if self.cfg.gather_with_opacity else 1)

        sum_weights = weights.sum(dim=-1, keepdim=True) + 1e-8

        norm_weights = weights / sum_weights

        interpolated_color = torch.einsum("bnk,bnkc->bnc", norm_weights, gathered_col)

        density = sum_weights if self.cfg.gather_with_opacity else torch.einsum("bnk,bnkc->bnc", norm_weights, gathered_opa)

        sdf_type = self.cfg.sdf_type.lower()
        if sdf_type == "normal_projection":
            signed_dist_k = torch.einsum("bnki,bnki->bnk", diff, gathered_normal)
            sdf = torch.sum(norm_weights * signed_dist_k * (1 if self.cfg.gather_with_opacity else gathered_opa.squeeze(-1)), dim=-1, keepdim=True)
        elif sdf_type == "mahalanobis":
            mahalanobis_dist_k = torch.sqrt(mahalanobis_sq + 1e-8)
            sdf_k = mahalanobis_dist_k - 1.0
            sdf = torch.sum(norm_weights * sdf_k * (1 if self.cfg.gather_with_opacity else gathered_opa.squeeze(-1)), dim=-1, keepdim=True)
        elif sdf_type == "mahalanobis_squared": 
            sdf = torch.sum(norm_weights * mahalanobis_sq * (1 if self.cfg.gather_with_opacity else gathered_opa.squeeze(-1)), dim=-1, keepdim=True)
        elif sdf_type == "signed_mahalanobis":
            mahalanobis_dist_k = torch.sqrt(mahalanobis_sq + 1e-8)
            signed_dist_k = torch.einsum("bnki,bnki->bnk", diff, gathered_normal)
            direction_sign = torch.sign(signed_dist_k + 1e-9)
            sdf_k = direction_sign * mahalanobis_dist_k 
            sdf = torch.sum(norm_weights * sdf_k * (1 if self.cfg.gather_with_opacity else gathered_opa.squeeze(-1)), dim=-1, keepdim=True)
        elif sdf_type == "signed_mahalanobis_squared":
            signed_dist_k = torch.einsum("bnki,bnki->bnk", diff, gathered_normal)
            direction_sign = torch.sign(signed_dist_k + 1e-9)
            sdf_k = direction_sign * mahalanobis_sq 
            sdf = torch.sum(norm_weights * sdf_k * (1 if self.cfg.gather_with_opacity else gathered_opa.squeeze(-1)), dim=-1, keepdim=True)
        elif sdf_type == "none":
            sdf = torch.zeros_like(density)
        else:
            raise ValueError(f"Unknown sdf_type: {self.cfg.sdf_type}")

        avg_normal = torch.einsum("bnk,bnkc->bnc", norm_weights, gathered_normal)
        avg_normal = F.normalize(avg_normal, p=2, dim=-1)

        num_points_total = B * N
        out = {
            "features": interpolated_color.view(num_points_total, -1),
            "density": density.view(num_points_total, 1),
            "sdf": sdf.view(num_points_total, 1) 
        }

        if self.cfg.sdf_type != "none":
            out["sdf_grad"] = avg_normal.view(num_points_total, 3)
        else:
            # Ensure sdf_grad has a consistent shape even if SDF is none
            out["sdf_grad"] = torch.zeros_like(points.view(num_points_total, 3))

        if calculate_avg_normal_output:
            out["normal"] = avg_normal.view(num_points_total, 3)

        return out

    def forward_sdf(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Dict[str, Any], # Changed type hint
    ) -> Float[Tensor, "*N 1"]:
        raise NotImplementedError("forward_sdf is not implemented yet.")

    def forward_field(
        self, 
        points: Float[Tensor, "*N Di"],
        space_cache: Dict[str, Any], # Changed type hint
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        raise NotImplementedError("forward_field is not implemented yet.")

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        raise NotImplementedError("forward_level is not implemented yet.")

    def export(
        self, 
        points: Float[Tensor, "*N Di"], # This argument might not be used if exporting pre-parsed Gaussians
        space_cache: Dict[str, Any],   # Expects space_cache to contain parsed Gaussians
    **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("export is not implemented yet.")
    

    def train(self, mode=True):
        super().train(mode)
        self.space_generator.train(mode)

    def eval(self):
        super().eval()
        self.space_generator.eval()

