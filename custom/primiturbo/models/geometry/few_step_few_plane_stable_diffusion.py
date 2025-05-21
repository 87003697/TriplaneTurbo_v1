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
    HAS_CUDA_KNN,
    CudaKNNIndex,
    gather_gaussian_params, build_inverse_covariance, quat_to_rot_matrix,
    verify_kdon_cuda_vs_pytorch,
    verify_kdn_cuda_vs_pytorch
)


@threestudio.register("few-step-few-plane-stable-diffusion")
class FewStepFewPlaneStableDiffusion(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_feature_dims: int = 3
        space_generator_config: dict = field(
            default_factory=lambda: {
                "pretrained_model_name_or_path": "stable-diffusion-2-1-base",
                "training_type": "lora_rank_4_self_lora_rank_4_cross_lora_rank_4_locon_rank_4",
                "output_dim": 14,
                "gradient_checkpoint": False,
                "require_intermediate_features": False,
            }
        )

        backbone: str = "few_step_few_plane_stable_diffusion"

        scaling_activation: str = "exp-0.1"
        opacity_activation: str = "sigmoid-0.1"
        rotation_activation: str = "normalize"
        color_activation: str = "none"
        position_activation: str = "none" # General activation for final Cartesian coords
        
        xyz_center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
        xyz_scale: float = 1.0
        
        top_K: int = 8
        knn_backend: str = 'cuda-knn'
        udf_type: str = "avg_l2" # Options: "none", "min_mahalanobis", "min_l2", "avg_mahalanobis", "avg_l2"

        gather_with_opacity: bool = True
        interpolation_mode: str = "mahalanobis-exp"  # Options: "mahalanobis-exp", "inverse_l2", "inverse_l1", "inverse_mahalanobis_sq", "inverse_mahalanobis"

        plane_attribute_mapping: List[Dict[str, Any]] = field(
            default_factory=lambda: [
                {"attribute_name": "color",    "plane_index": 0, "num_channels": 3},
                {"attribute_name": "position", "plane_index": 1, "num_channels": 3}, # Assumed to be Cartesian (x,y,z)
                {"attribute_name": "scale",    "plane_index": 2, "num_channels": 3},
                {"attribute_name": "rotation", "plane_index": 2, "num_channels": 4},
                {"attribute_name": "opacity",  "plane_index": 3, "num_channels": 1},
            ]
        )

        neighbor_search_metric: str = 'l2'

        # use_hierarchical_refinement: bool = False # 控制是否启用分层优化
        # num_refinement_levels: int = 3 # 例如，使用多少个特征层级 (包括最终层)
        # # 可以有更多参数来控制 HierarchicalGaussianRefiner 的具体行为
        # # 例如，每个层级的上采样率，MLP的维度等。
        # hierarchical_refiner_config: dict = field(default_factory=dict) # 传递给 Refiner 的配置

        # KNN Loss configuration
        knn_loss_K: int = 3 # Number of nearest neighbors for L_knn


    def _process_plane_attribute_mapping(self) -> None:
        """Processes and validates the plane_attribute_mapping configuration.
        It calculates 'channel_slice' for each attribute, which represents its slice in a 
        conceptual globally concatenated feature vector. The order is determined by 
        plane_index (primary sort key) and then original config order (secondary sort key).
        'plane_index' stored is the physical plane this attribute's data is associated with.
        The parse() method is expected to use this 'channel_slice' directly on features from 'plane_index'.
        This implies a specific understanding of how global slices map to local physical plane channels.
        A single attribute's channels CANNOT span multiple physical planes.
        Any 'channel_start_index' in the config is IGNORED.
        """
        self.processed_attribute_map: Dict[str, Dict[str, Any]] = {}
        required_attributes = {"color", "position", "scale", "rotation", "opacity"}
        defined_attributes = set()

        if not hasattr(self, 'space_generator') or not hasattr(self.space_generator, 'cfg'):
            raise RuntimeError("space_generator must be configured before processing attribute mapping.")

        module_num_physical_planes = self.space_generator.cfg.num_planes
        channels_per_physical_plane = self.space_generator.cfg.output_dim

        indexed_attribute_configs = []
        for i, item in enumerate(self.cfg.plane_attribute_mapping):
            if not all(k in item for k in ["attribute_name", "plane_index", "num_channels"]):
                raise ValueError(f"Invalid mapping item (missing keys): {item}. Must include 'attribute_name', 'plane_index', 'num_channels'.")
            indexed_attribute_configs.append((i, item))
        
        sorted_indexed_attribute_configs = sorted(indexed_attribute_configs, key=lambda x: (x[1]["plane_index"], x[0]))

        current_unified_channel_offset = 0
        for original_config_idx, mapping_item in sorted_indexed_attribute_configs:
            attr_name = mapping_item["attribute_name"]
            physical_plane_idx = mapping_item["plane_index"]
            num_ch = mapping_item["num_channels"]

            if not isinstance(physical_plane_idx, int) or not isinstance(num_ch, int):
                 raise ValueError(
                    f"Type error for '{attr_name}': 'plane_index' and 'num_channels' must be int. Got {mapping_item}"
                )
            if attr_name in defined_attributes:
                raise ValueError(f"Attribute '{attr_name}' defined multiple times.")
            defined_attributes.add(attr_name)

            if not (0 <= physical_plane_idx < module_num_physical_planes):
                raise ValueError(
                    f"Invalid plane_index {physical_plane_idx} for '{attr_name}'. Max is {module_num_physical_planes - 1}."
                )
            if not (num_ch > 0):
                 raise ValueError(f"num_channels for '{attr_name}' must be > 0, got {num_ch}.")

            # Calculate the unified channel slice based on global offset
            unified_channel_slice = slice(current_unified_channel_offset, current_unified_channel_offset + num_ch)
            
            self.processed_attribute_map[attr_name] = {
                "num_channels": num_ch,
                "plane_index": physical_plane_idx, # Physical plane this attribute is tied to
                "channel_slice": unified_channel_slice  # The globally unified slice
            }
            
            current_unified_channel_offset += num_ch

        missing_attributes = required_attributes - defined_attributes
        if missing_attributes:
            threestudio.warn(f"Potentially missing core attributes in plane_attribute_mapping: {missing_attributes}. Ensure this is intended.")


    def configure(self) -> None:
        super().configure()

        # print("The current device is: ", self.device) # Device is available after super().configure()

        # Step 1: Calculate required number of planes from plane_attribute_mapping
        if not self.cfg.plane_attribute_mapping:
            # If mapping is empty, maybe default to 1 plane or raise error,
            # For now, let's assume it implies 0 or 1 plane based on space_generator's default if any.
            # Or, more robustly, require at least one mapping if this feature is used.
            # Let's default to 1 if empty, assuming space_generator might have a default num_planes.
            # This might need refinement based on desired behavior for empty mapping.
            calculated_num_planes = 1 # Default or read from space_generator_config if it exists
            if "num_planes" in self.cfg.space_generator_config:
                 calculated_num_planes = self.cfg.space_generator_config.get("num_planes", 1)
            threestudio.warn("plane_attribute_mapping is empty or not defined. Defaulting or using space_generator_config.num_planes if present.")

        else:
            max_plane_index = -1
            for mapping_item in self.cfg.plane_attribute_mapping:
                plane_idx = mapping_item.get("plane_index")
                if isinstance(plane_idx, int) and plane_idx > max_plane_index:
                    max_plane_index = plane_idx
            
            if max_plane_index == -1: # No valid plane_index found in a non-empty mapping
                raise ValueError("plane_attribute_mapping is provided but contains no valid 'plane_index'.")
            calculated_num_planes = max_plane_index + 1
        
        threestudio.info(f"Calculated required number of planes: {calculated_num_planes} based on plane_attribute_mapping.")

        # Step 2: Prepare space_generator config
        # Create a mutable copy to avoid modifying the original Hydra config object
        current_space_generator_config = dict(self.cfg.space_generator_config)
        original_num_planes_in_config = current_space_generator_config.get("num_planes")

        current_space_generator_config["num_planes"] = calculated_num_planes
        
        if original_num_planes_in_config is not None and original_num_planes_in_config != calculated_num_planes:
            threestudio.warn(
                f"Overriding space_generator_config.num_planes (was {original_num_planes_in_config}) "
                f"with calculated value {calculated_num_planes} based on plane_attribute_mapping."
            )


        # Step 3: Instantiate space_generator
        if self.cfg.backbone == "few_step_few_plane_stable_diffusion":
            from ...extern.few_step_few_plane_sd_modules import FewStepFewPlaneStableDiffusion as Generator
            # Pass the modified config
            self.space_generator = Generator(current_space_generator_config)
        else:
            raise ValueError(f"Unknown backbone {self.cfg.backbone}")
        
        # Step 4: Process and validate attribute mapping using the now configured space_generator
        # Initialize this before calling _process_plane_attribute_mapping
        self.channels_used_per_plane: Dict[int, int] = {} 
        self._process_plane_attribute_mapping()
        
        self.scaling_activation_fn = get_activation(self.cfg.scaling_activation)
        self.opacity_activation_fn = get_activation(self.cfg.opacity_activation)
        self.rotation_activation_fn = get_activation(self.cfg.rotation_activation)
        self.color_activation_fn = get_activation(self.cfg.color_activation)
        self.position_activation_fn = get_activation(self.cfg.position_activation)

        self.xyz_center_t = lambda x: torch.tensor(self.cfg.xyz_center, device=x.device, dtype=x.dtype)

        self.search_mode = None
        metric = self.cfg.neighbor_search_metric.lower()
        backend = self.cfg.knn_backend.lower()

        if metric == 'l2':
            if backend == 'cuda-knn':
                assert HAS_CUDA_KNN, "CUDA KNN extension requested but not available/compiled. Cannot proceed."
                threestudio.info("Using CUDA KNN backend (L2 distance).")
                self.search_mode = 'knn-cuda'
            elif backend == 'torch':
                 threestudio.info("Using PyTorch KNN backend (L2 distance, might be slow).")
                 self.search_mode = 'knn-torch'
            else:
                 raise ValueError(f"Unknown knn_backend for L2: {self.cfg.knn_backend}")
        else:
            raise ValueError(f"Unknown neighbor_search_metric: {self.cfg.neighbor_search_metric}")
        
        # if self.cfg.use_hierarchical_refinement:
        #     self.hierarchical_refiner = HierarchicalGaussianRefiner(
        #         cfg=self.cfg.hierarchical_refiner_config,
        #         num_final_output_dim=sum(item['num_channels'] for item in self.cfg.plane_attribute_mapping), # 计算总输出维度
        #         num_planes=self.space_generator.cfg.num_planes
        #     ).to(self.device)


    def initialize_shape(self) -> None:
        # not used
        pass

    def denoise(
        self,
        noisy_input: Any,
        text_embed: Tensor,
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
                              points: Tensor, 
                              inv_covariances: Optional[Tensor], 
                              reference_opacities: Optional[Tensor],
                              reference_lengths: Tensor):
        """Builds a KNN or KDN index based on the configured search_mode."""
        B, M, D = points.shape
        assert B == 1, "Index building currently only supports batch size 1"

        index = None
        search_mode = self.search_mode 

        threestudio.debug(f"Building index for search mode: {search_mode}")

        if search_mode == 'knn-cuda': 
            index = CudaKNNIndex() 
            index.add(points)
        elif search_mode == 'knn-torch':
            threestudio.debug(f"Building Torch KNN index (L2) with {M} points.") 
            points_flat = points.squeeze(0).contiguous() 
            
            class TorchKNNIndex:
                def __init__(self, data):
                    self.data = data 
                    self.M = data.shape[0]
                def search(self, query: Tensor, k: int):
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
        decoded_outputs: Any, 
        scale_factor: float, # scale_factor is not used in the current logic but kept for signature consistency
    ) -> List[Dict[str, Any]]: # Return type changed to List of Dicts

        list_of_pc_dicts: List[Dict[str, Any]] = []
        feature_levels_to_process: List[Tensor] = []

        # Step 1: Determine the list of feature tensors to process
        if isinstance(decoded_outputs, list) and len(decoded_outputs) > 0 and \
           all(isinstance(item, torch.Tensor) and item.ndim == 5 for item in decoded_outputs):
            
            threestudio.debug(f"Processing {len(decoded_outputs)} levels for hierarchical combination.")

            # Reference B, P, C from the last feature map for consistency checks.
            # This assumes the list is not empty and all elements are 5D tensors as per the condition above.
            ref_feat_map_for_bpc = decoded_outputs[-1] 
            B_ref, NumPlanes_ref, C_ref, _, _ = ref_feat_map_for_bpc.shape

            current_hierarchical_sum_5D = None 
            
            for level_idx, current_raw_feat_5D in enumerate(decoded_outputs):
                # current_raw_feat_5D is already validated to be a 5D tensor by the outer if condition.
                B_curr, P_curr, C_curr, H_curr, W_curr = current_raw_feat_5D.shape
                
                # Validate B, P, C against the reference
                if B_curr != B_ref or P_curr != NumPlanes_ref:
                    threestudio.warn(f"Level {level_idx}: Skipping due to B/P mismatch with reference. Shape: {current_raw_feat_5D.shape}, Ref B,P: ({B_ref},{NumPlanes_ref})")
                    if current_hierarchical_sum_5D is not None:
                        feature_levels_to_process.append(current_hierarchical_sum_5D.clone())
                    continue 
                
                if C_curr != C_ref:
                    threestudio.warn(f"Level {level_idx} (raw shape {current_raw_feat_5D.shape}): Skipped contribution due to C mismatch (got {C_curr}, expected ref {C_ref}).")
                    if current_hierarchical_sum_5D is not None:
                        feature_levels_to_process.append(current_hierarchical_sum_5D.clone())
                    continue

                contribution_this_level_5D = current_raw_feat_5D

                if current_hierarchical_sum_5D is None:
                    current_hierarchical_sum_5D = contribution_this_level_5D.clone()
                    threestudio.debug(f"Level {level_idx} (raw shape {current_raw_feat_5D.shape}): Initialized sum. Sum shape: {current_hierarchical_sum_5D.shape}")
                else:
                    prev_sum_H, prev_sum_W = current_hierarchical_sum_5D.shape[-2:]
                    
                    upsampled_previous_sum_5D = current_hierarchical_sum_5D
                    if (prev_sum_H != H_curr) or (prev_sum_W != W_curr):
                        threestudio.debug(f"Level {level_idx}: Upsampling previous sum from ({prev_sum_H},{prev_sum_W}) to ({H_curr},{W_curr}).")
                        sum_reshaped_4D = current_hierarchical_sum_5D.reshape(B_ref * NumPlanes_ref, C_ref, prev_sum_H, prev_sum_W)
                        upsampled_sum_4D = F.interpolate(
                            sum_reshaped_4D, size=(H_curr, W_curr), mode='bilinear', align_corners=False
                        )
                        upsampled_previous_sum_5D = upsampled_sum_4D.view(B_ref, NumPlanes_ref, C_ref, H_curr, W_curr)
                    
                    current_hierarchical_sum_5D = upsampled_previous_sum_5D + contribution_this_level_5D
                    threestudio.debug(f"Level {level_idx} (raw shape {current_raw_feat_5D.shape}): Added contribution. Sum shape: {current_hierarchical_sum_5D.shape}")
                
                feature_levels_to_process.append(current_hierarchical_sum_5D.clone())

        elif isinstance(decoded_outputs, torch.Tensor) and decoded_outputs.ndim == 5:
            feature_levels_to_process = [decoded_outputs]
            threestudio.debug(f"Processing single 5D feature tensor as one level. Shape: {decoded_outputs.shape}")
        else:
            if isinstance(decoded_outputs, list):
                 threestudio.warn(f"decoded_outputs is a list, but not all elements are valid 5D Tensors. List content types: {[type(item) for item in decoded_outputs]}")
            else:
                 threestudio.warn(f"decoded_outputs has unexpected type or dimensions: {type(decoded_outputs)}, ndim: {decoded_outputs.ndim if isinstance(decoded_outputs, torch.Tensor) else 'N/A'}. Expected a list of 5D Tensors or a single 5D Tensor.")
            # feature_levels_to_process will remain empty

        # Step 2: Iterate through each processed feature level and parse attributes
        for level_idx, feature_planes_tensor in enumerate(feature_levels_to_process):
            threestudio.debug(f"Parsing attributes for feature level {level_idx} with shape {feature_planes_tensor.shape}")

            if not (isinstance(feature_planes_tensor, torch.Tensor) and feature_planes_tensor.ndim == 5):
                threestudio.warn(
                    f"Skipping attribute parsing for level {level_idx} due to invalid feature_planes_tensor (expected 5D Tensor, got {type(feature_planes_tensor)})."
                )
                continue

            B, NumPlanes_actual, ChannelsPerPlane_actual, H, W = feature_planes_tensor.shape
            
            if NumPlanes_actual != self.space_generator.cfg.num_planes:
                threestudio.warn(
                    f"Level {level_idx}: Mismatch in num_planes (tensor: {NumPlanes_actual}, config: {self.space_generator.cfg.num_planes}). Skipping."
                )
                continue
            if ChannelsPerPlane_actual != self.space_generator.cfg.output_dim:
                threestudio.warn(
                    f"Level {level_idx}: Mismatch in channels_per_plane (tensor: {ChannelsPerPlane_actual}, config: {self.space_generator.cfg.output_dim}). Skipping."
                )
                continue

            raw_params: Dict[str, Tensor] = {}
            num_points_per_batch = H * W

            valid_level = True
            for attr_name, mapping_info in self.processed_attribute_map.items():
                plane_idx: int = mapping_info["plane_index"]
                ch_slice: slice = mapping_info["channel_slice"]
                num_expected_channels = mapping_info['num_channels']
                
                selected_plane_features = feature_planes_tensor[:, plane_idx, :, :, :] 
                attribute_features_raw = selected_plane_features[:, ch_slice, :, :] 

                # # Normalize by the number of levels
                # attribute_features_raw = attribute_features_raw / (level_idx + 1)
                
                if attribute_features_raw.shape[1] != num_expected_channels:
                    threestudio.error(
                        f"Level {level_idx}, Attr '{attr_name}': Extracted {attribute_features_raw.shape[1]} channels from plane {plane_idx} using UNIFIED slice {ch_slice} (expected {num_expected_channels}). "
                        f"Physical plane channels: {selected_plane_features.shape[1]}. This level might be unusable."
                    )
                    valid_level = False
                    break 
                raw_params[attr_name] = rearrange(attribute_features_raw, "B C H W -> B (H W) C", H=H, W=W)
            
            if not valid_level:
                threestudio.warn(f"Skipping further processing for level {level_idx} due to attribute extraction errors.")
                continue

            pc_dict = self._activate_params_for_output(raw_params)

            if "scale" not in pc_dict or "rotation" not in pc_dict:
                threestudio.warn(f"Level {level_idx}: 'scale' or 'rotation' missing. Cannot build inv_cov. Skipping.")
                continue
            pc_dict['inv_cov'] = build_inverse_covariance(pc_dict['scale'], pc_dict['rotation'])

            if 'normal' not in pc_dict:
                if "scale" in pc_dict and "rotation" in pc_dict:
                    scales_final = pc_dict['scale']
                    quats_final = pc_dict['rotation']
                    B_final_norm, M_final_norm, _ = scales_final.shape
                    min_scale_indices = torch.argmin(scales_final, dim=-1)
                    rot_mats = quat_to_rot_matrix(quats_final)
                    min_scale_indices_exp = min_scale_indices.view(B_final_norm, M_final_norm, 1, 1).expand(B_final_norm, M_final_norm, 3, 1)
                    est_normals = torch.gather(rot_mats, 3, min_scale_indices_exp).squeeze(-1)
                    pc_dict['normal'] = est_normals
                else:
                    threestudio.debug(f"Level {level_idx}: Cannot estimate normals as 'scale' or 'rotation' is missing.")

            if "position" not in pc_dict:
                threestudio.warn(f"Level {level_idx}: 'position' missing. Cannot build KNN index. Skipping.")
                continue

            if pc_dict['position'].shape[0] == 1:
                ref_lengths = torch.tensor([pc_dict['position'].shape[1]], dtype=torch.int64, device=pc_dict['position'].device)
                pc_dict['index'] = self._build_neighbor_index(
                    pc_dict['position'], 
                    pc_dict.get('inv_cov'),
                    pc_dict.get('opacity'), 
                    ref_lengths
                )
            else:
                threestudio.debug(f"Level {level_idx}: KNN index building for batch size > 1 not performed. Setting index to None.")
                pc_dict['index'] = None
            
            list_of_pc_dicts.append(pc_dict)
            threestudio.debug(f"Successfully parsed attributes for level {level_idx}. pc_dict added.")
        
        # Determine what to return based on the number of dicts created
        # The original code had: return list_of_pc_dicts if len(list_of_pc_dicts) > 1 else list_of_pc_dicts[0]
        # This implies if only one dict, return the dict itself, otherwise the list.
        processed_output = list_of_pc_dicts[0] if len(list_of_pc_dicts) == 1 else list_of_pc_dicts

        # # Compute and assign KNN losses to the pc_dicts
        # self._compute_and_assign_knn_losses(list_of_pc_dicts)

        return processed_output

    def _compute_and_assign_knn_losses(self, list_of_pc_dicts: List[Dict[str, Any]]) -> None:
        if not list_of_pc_dicts: # Ensure there is at least one pc_dict to process
            return

        # First, ensure the base level (level 0) pc_dict has its knn_loss set and prepare its data for querying by higher levels.
        level0_pc_dict = list_of_pc_dicts[0]
        level0_pc_dict['knn_loss'] = torch.tensor(0.0, device=level0_pc_dict.get("position", torch.tensor([])).device)
        
        pos_level0_batched = level0_pc_dict.get('position') # Use .get for safety
        index_level0 = level0_pc_dict.get('index')

        if pos_level0_batched is None or index_level0 is None:
            # threestudio.warn("Level 0 position or index is None, skipping inter-level KNN loss for all higher levels.")
            print("WARN: Level 0 position or index is None, skipping inter-level KNN loss for all higher levels.")
            # Set knn_loss to 0 for all higher levels if level 0 data is missing
            for i, pc_dict_item in enumerate(list_of_pc_dicts):
                if i == 0:
                    continue
                # Ensure knn_loss key exists even if not calculated
                pc_dict_item['knn_loss'] = torch.tensor(0.0, device=pc_dict_item.get("position", torch.tensor([])).device if pc_dict_item.get("position") is not None else torch.device('cpu'))
            return # Exit after setting default losses

        batch_size_lvl0 = pos_level0_batched.shape[0]
        M0 = pos_level0_batched.shape[1]
        # threestudio.debug(f"KNN Loss: Level 0 has {M0} points. Batch size: {batch_size_lvl0}")
        print(f"DEBUG: KNN Loss: Level 0 has {M0} points. Batch size: {batch_size_lvl0}")
        assert batch_size_lvl0 == 1, f"Inter-level KNN loss currently only supports batch size 1 for level 0, got B={batch_size_lvl0}."

        # Iterate through all pc_dicts to set their knn_loss
        for i, pc_dict_item in enumerate(list_of_pc_dicts):
            if i == 0: # Skip level 0 itself, its loss is already set to 0
                continue

            if self.training: # Calculate KNN loss only during training for higher levels
                if "position" in pc_dict_item and pc_dict_item["position"] is not None and pc_dict_item["position"].numel() > 0:
                    positions_higher_level_batched = pc_dict_item["position"]  # Shape (1, M_high, 3)
                    
                    if positions_higher_level_batched.shape[0] != 1:
                        threestudio.warn(f"Inter-level KNN loss for item {i} currently only supports B=1, got B={positions_higher_level_batched.shape[0]}. Setting knn_loss to 0.")
                        pc_dict_item['knn_loss'] = torch.tensor(0.0, device=positions_higher_level_batched.device)
                        continue
                    
                    M_high = positions_higher_level_batched.shape[1]
                    M0 = pos_level0_batched.shape[1] # Redundant but harmless
                    K_inter_level = min(self.cfg.knn_loss_K, M0 if M0 > 0 else 0)

                    # threestudio.debug(f"KNN Loss: Item {i}, M_high={M_high}, M0={M0}, K_cfg={self.cfg.knn_loss_K}, K_inter_level={K_inter_level}")
                    print(f"DEBUG: KNN Loss: Item {i}, M_high={M_high}, M0={M0}, K_cfg={self.cfg.knn_loss_K}, K_inter_level={K_inter_level}")

                    if M_high > 0 and M0 > 0 and K_inter_level > 0:
                        l_knn_val = self._calculate_inter_level_knn_loss(
                            query_positions_batched=positions_higher_level_batched,
                            ref_positions_batched=pos_level0_batched,
                            ref_index=index_level0,
                            k_neighbors=K_inter_level
                        )
                        # threestudio.debug(f"Calculated inter-level L_knn ({K_inter_level}-NN for item {i} querying level 0): {l_knn_val.item()}")
                        print(f"DEBUG: Calculated inter-level L_knn ({K_inter_level}-NN for item {i} querying level 0): {l_knn_val.item()}")
                    else:
                        l_knn_val = torch.tensor(0.0, device=positions_higher_level_batched.device)
                    
                    pc_dict_item['knn_loss'] = l_knn_val
                else:  # No position data or empty position tensor for higher level item i
                    # threestudio.debug(f"KNN Loss: Item {i}, no position data or empty tensor. Setting knn_loss to 0.")
                    print(f"DEBUG: KNN Loss: Item {i}, no position data or empty tensor. Setting knn_loss to 0.")
                    pc_dict_item['knn_loss'] = torch.tensor(0.0, device=pc_dict_item.get("position", torch.tensor([])).device if pc_dict_item.get("position") is not None else torch.device('cpu'))
            else: # Not training
                # threestudio.debug(f"KNN Loss: Item {i}, not training. Setting knn_loss to 0.")
                print(f"DEBUG: KNN Loss: Item {i}, not training. Setting knn_loss to 0.")
                pc_dict_item['knn_loss'] = torch.tensor(0.0, device=pc_dict_item.get("position", torch.tensor([])).device if pc_dict_item.get("position") is not None else torch.device('cpu'))

    def _calculate_inter_level_knn_loss(
        self,
        query_positions_batched: Tensor, # Shape (1, M_high, 3)
        ref_positions_batched: Tensor,   # Shape (1, M0, 3)
        ref_index: Any,                  # KNN index built on ref_positions_batched
        k_neighbors: int
    ) -> Tensor:
        """
        Calculates the KNN loss for query_positions_batched by finding k_neighbors 
        in ref_positions_batched using ref_index.
        Assumes batch size is 1 for both inputs.
        """
        # threestudio.debug(f"_calc_knn: query_shape={query_positions_batched.shape}, ref_shape={ref_positions_batched.shape}, k={k_neighbors}")
        print(f"DEBUG: _calc_knn: query_shape={query_positions_batched.shape}, ref_shape={ref_positions_batched.shape}, k={k_neighbors}")

        # Squeeze batch dimension for consistent processing, as B=1 is asserted before calling
        query_points_flat = query_positions_batched.squeeze(0) # Shape: (M_high, 3)
        ref_points_flat = ref_positions_batched.squeeze(0)     # Shape: (M0, 3)

        knn_indices_flat: Tensor # Shape: (M_high, k_neighbors)
        if self.search_mode == 'knn-cuda':
            # CudaKNNIndex.search expects (B,N,D) query and returns (B,N,K) indices
            # query_positions_batched is (1, M_high, 3)
            # ref_index is CudaKNNIndex, its .search method needs query_lengths
            actual_query_lengths = torch.tensor([query_positions_batched.shape[1]], dtype=torch.int64, device=query_positions_batched.device)
            _, knn_indices_batched = ref_index.search(
                query_positions_batched.contiguous(), 
                k=k_neighbors, 
                query_lengths=actual_query_lengths
            )
            knn_indices_flat = knn_indices_batched.squeeze(0)
        elif self.search_mode == 'knn-torch':
            # TorchKNNIndex.search expects (N,D) query and returns (N,K) indices
            # It does not take query_lengths as an argument.
            _, knn_indices_flat = ref_index.search(query_points_flat.contiguous(), k=k_neighbors) 
        else:
            raise NotImplementedError(f"Inter-level KNN search not implemented for search_mode: {self.search_mode}")
        
        # knn_indices_flat contains indices into ref_points_flat (which has M0 points)
        
        # Gather K nearest neighbor positions from ref_points_flat
        # Shape: (M_high, k_neighbors, 3)
        gathered_neighbors_flat = ref_points_flat[knn_indices_flat]
        
        # Expand query points to match shape of gathered neighbors for distance calculation
        # query_points_flat is (M_high, 3) -> unsqueeze to (M_high, 1, 3)
        # Shape: (M_high, k_neighbors, 3)
        expanded_query_flat = query_points_flat.unsqueeze(1).expand_as(gathered_neighbors_flat)
        
        # Calculate squared L2 distances
        # Shape: (M_high, k_neighbors)
        dist_sq_to_ref_neighbors_flat = (expanded_query_flat - gathered_neighbors_flat).pow(2).sum(dim=2)
        
        l_knn_val = torch.tensor(0.0, device=query_positions_batched.device)
        if dist_sq_to_ref_neighbors_flat.numel() > 0:
            l_knn_val = torch.mean(dist_sq_to_ref_neighbors_flat)
        
        return l_knn_val

    def _activate_params_for_output(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        activated_params = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in raw_params.items()}

        if "position" in activated_params and isinstance(activated_params["position"], torch.Tensor):
            pos_tensor_raw = activated_params["position"] # Shape (B, N, 3)
            
            if pos_tensor_raw.shape[-1] != 3:
                raise ValueError(
                    f"Position tensor must have 3 channels in the last dimension (for Cartesian coordinates), "
                    f"got shape {pos_tensor_raw.shape}"
                )

            # Directly use pos_tensor_raw as Cartesian coordinates (v1 logic)
            threestudio.debug("Activating position with format v1 (Cartesian)")
            cartesian_coords_intermediate = pos_tensor_raw 
            
            # Apply common scaling and centering to the Cartesian coordinates
            xyz_center_tensor = self.xyz_center_t(cartesian_coords_intermediate)
            cartesian_coords_scaled_centered = cartesian_coords_intermediate * self.cfg.xyz_scale + xyz_center_tensor.unsqueeze(0).unsqueeze(0)
            
            # Apply final generic position activation (usually 'none')
            activated_params["position"] = self.position_activation_fn(cartesian_coords_scaled_centered)

        if "scale" in activated_params and isinstance(activated_params["scale"], torch.Tensor):
            raw_scale_tensor = activated_params["scale"]

            # Apply main activation (for X, Y, and Z)
            main_scale_activation_fn = get_activation(self.cfg.scaling_activation)
            activated_scale_tensor = main_scale_activation_fn(raw_scale_tensor)
            activated_params["scale"] = activated_scale_tensor
        
        if "opacity" in activated_params and isinstance(activated_params["opacity"], torch.Tensor):
            activated_params["opacity"] = self.opacity_activation_fn(activated_params["opacity"]) 
        
        if "color" in activated_params and isinstance(activated_params["color"], torch.Tensor):
            activated_params["color"] = self.color_activation_fn(activated_params["color"]) 
        
        if "rotation" in activated_params and isinstance(activated_params["rotation"], torch.Tensor):
            activated_params["rotation"] = self.rotation_activation_fn(activated_params["rotation"])
        
        return activated_params
    
    def interpolate_encodings(
        self,
        points: Tensor,
        space_cache: Tensor,
        only_geo: bool = False,
    ):
        raise NotImplementedError("interpolate_encodings is not implemented yet.")


    def rescale_points(
        self,
        points: Tensor,
    ):
        raise NotImplementedError("rescale_points is not implemented yet.")

    def forward(
        self,
        points: Tensor,
        space_cache: Dict[str, Any],
        output_normal: bool = False,
        debug: bool = False,
    ) -> Dict[str, Float[Tensor, "..."]]:
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
        # ref_lengths = torch.tensor([M], dtype=torch.int64, device=points.device) # Not used in current logic

        if index is None:
             raise ValueError("KNN index not found in space_cache.")
        if B > 1:
            raise NotImplementedError("Forward pass currently assumes B=1 due to gather implementation.")

        points_flat = points.view(-1, 3) 
        # query_lengths_flat = torch.full((B,), N, dtype=torch.int64, device=points.device) # Not used
        
        search_k = min(top_K, M) 
        
        if search_k <= 0:
            indices = torch.full((B, N, 0), -1, device=points.device, dtype=torch.long)
        
        elif self.search_mode == 'knn-torch':
             _, indices = index.search(points_flat, k=search_k)
        elif self.search_mode == 'knn-cuda':
             _, indices = index.search(points_flat, k=search_k)
        else:
             raise RuntimeError(f"Search not implemented for mode {self.search_mode}")
            
        indices = indices.view(B, N, -1)
        K_ret = indices.shape[-1]
        assert K_ret > 0, "K_ret must be greater than 0"

        gathered_pos = gather_gaussian_params(gauss_pos, indices)
        gathered_col = gather_gaussian_params(gauss_col, indices)
        gathered_opa = gather_gaussian_params(gauss_opa, indices)
        gathered_inv_cov = gather_gaussian_params(inv_cov, indices)
        gathered_normal = gather_gaussian_params(est_normals, indices)

        diff = points.unsqueeze(2) - gathered_pos # Shape (B, N, K, 3)
        
        # --- Distance Calculations ---
        mahalanobis_sq = torch.einsum("bnki,bnkij,bnkj->bnk", diff, gathered_inv_cov, diff)
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0.0)

        dist_sq_l2 = torch.sum(diff**2, dim=-1)
        dist_sq_l2 = torch.clamp(dist_sq_l2, min=0.0)

        dist_l1 = torch.sum(torch.abs(diff), dim=-1) # Shape (B, N, K)
        dist_l1 = torch.clamp(dist_l1, min=0.0)

        # --- Gaussian Density Calculation (based on interpolation_mode) ---
        gauss_density = torch.zeros(B, N, K_ret, device=points.device, dtype=points.dtype)
        inverse_distance_epsilon = 1e-8
        if self.cfg.interpolation_mode == "mahalanobis-exp":
            exponent = torch.clamp(-0.5 * mahalanobis_sq, max=20.0)
            gauss_density = torch.exp(exponent)
        elif self.cfg.interpolation_mode == "inverse_l2":
            dist_l2_sqrt = torch.sqrt(dist_sq_l2) # Use precomputed dist_sq_l2
            gauss_density = 1.0 / (dist_l2_sqrt + inverse_distance_epsilon)
        elif self.cfg.interpolation_mode == "inverse_l1":
            gauss_density = 1.0 / (dist_l1 + inverse_distance_epsilon)
        elif self.cfg.interpolation_mode == "inverse_mahalanobis_sq":
            gauss_density = 1.0 / (mahalanobis_sq + inverse_distance_epsilon)
        elif self.cfg.interpolation_mode == "inverse_mahalanobis":
            mahalanobis_dist_sqrt = torch.sqrt(mahalanobis_sq) # Use precomputed mahalanobis_sq
            gauss_density = 1.0 / (mahalanobis_dist_sqrt + inverse_distance_epsilon)
        else:
            raise ValueError(f"Unknown interpolation_mode: {self.cfg.interpolation_mode}")

        # --- Weights and Interpolation ---
        weights = gauss_density  * (gathered_opa.squeeze(-1) if self.cfg.gather_with_opacity else 1.0)
        sum_weights = weights.sum(dim=-1, keepdim=True) + 1e-8
        norm_weights = weights / sum_weights

        interpolated_color = torch.einsum("bnk,bnkc->bnc", norm_weights, gathered_col)

        
        # Density can be sum of weights, or weighted sum of opacities if not gathered with opacity initially
        interpolated_density_val = sum_weights 
        if not self.cfg.gather_with_opacity : # if opacity was not part of weights for density
            interpolated_density_val = torch.einsum("bnk,bnkc->bnc", norm_weights, gathered_opa)

        # --- UDF Calculation ---
        udf_type_lower = self.cfg.udf_type.lower()
        epsilon_val = 1e-8 # Epsilon for sqrt and clamping to prevent NaN

        if udf_type_lower == "min_mahalanobis":
            min_mahalanobis_sq_val, _ = torch.min(mahalanobis_sq, dim=-1, keepdim=True)
            udf = torch.sqrt(torch.clamp(min_mahalanobis_sq_val, min=epsilon_val))
        elif udf_type_lower == "min_l2":
            min_l2_sq_val, _ = torch.min(dist_sq_l2, dim=-1, keepdim=True)
            udf = torch.sqrt(torch.clamp(min_l2_sq_val, min=epsilon_val))
        elif udf_type_lower == "avg_mahalanobis":
            mahalanobis_dist = torch.sqrt(torch.clamp(mahalanobis_sq, min=epsilon_val))
            udf = torch.mean(mahalanobis_dist, dim=-1, keepdim=True)
        elif udf_type_lower == "avg_l2":
            l2_dist = torch.sqrt(torch.clamp(dist_sq_l2, min=epsilon_val))
            udf = torch.mean(l2_dist, dim=-1, keepdim=True)
        elif udf_type_lower == "none":
            udf = torch.zeros_like(points[..., :1])
        else:
            raise ValueError(f"Unknown udf_type: {self.cfg.udf_type}")

        # --- Averaged Normal (if neighbors exist) ---
        avg_normal_calculated = torch.einsum("bnk,bnkc->bnc", norm_weights, gathered_normal)
        avg_normal = F.normalize(avg_normal_calculated, p=2, dim=-1)

        num_points_total = B * N
        out = {
            "features": interpolated_color.view(num_points_total, -1),
            "density": interpolated_density_val.view(num_points_total, 1),
            "udf": udf.view(num_points_total, 1) 
        }

        if calculate_avg_normal_output:
            out["normal"] = avg_normal.view(num_points_total, 3)

        return out

    def forward_sdf(
        self,
        points: Tensor,
        space_cache: Dict[str, Any],
    ) -> Tensor:
        raise NotImplementedError("forward_sdf is not implemented yet.")

    def forward_field(
        self, 
        points: Tensor,
        space_cache: Dict[str, Any],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        raise NotImplementedError("forward_field is not implemented yet.")

    def forward_level(
        self, field: Tensor, threshold: float
    ) -> Tensor:
        raise NotImplementedError("forward_level is not implemented yet.")

    def export(
        self, 
        points: Tensor,
        space_cache: Dict[str, Any],
    **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("export is not implemented yet.")
    

    def train(self, mode=True):
        super().train(mode)
        self.space_generator.train(mode)

    def eval(self):
        super().eval()
        self.space_generator.eval()

    def export_gaussian_attributes_as_images(
        self,
        decoded_outputs: Union[Tensor, List[Tensor]],
        prefix: str = "",
    ) -> Dict[str, Tensor]:
        images = {}

        feature_maps_to_process = []
        level_prefixes = []

        if isinstance(decoded_outputs, list):
            if len(decoded_outputs) > 1:
                threestudio.debug("export_gaussian_attributes_as_images: Received a list of feature maps. Visualizing based on the last one.")
            if not decoded_outputs:
                threestudio.warn("export_gaussian_attributes_as_images: decoded_outputs list is empty.")
                return images
            feature_maps_to_process.append(decoded_outputs[-1])
            level_prefixes.append("_final_level_")
        elif isinstance(decoded_outputs, torch.Tensor):
            feature_maps_to_process.append(decoded_outputs)
            level_prefixes.append("")
        else:
            threestudio.warn(f"export_gaussian_attributes_as_images: decoded_outputs type {type(decoded_outputs)} not supported.")
            return images

        for level_idx, current_level_feature_planes_tensor in enumerate(feature_maps_to_process):
            if not (isinstance(current_level_feature_planes_tensor, torch.Tensor) and current_level_feature_planes_tensor.ndim == 5):
                threestudio.warn(f"Skipping visualization for level {level_idx} due to unexpected feature map format.")
                continue

            _B, _NumPlanes, _CperPlane, H_feat, W_feat = current_level_feature_planes_tensor.shape
            current_level_prefix = prefix + level_prefixes[level_idx]

            for attr_name, mapping_info in self.processed_attribute_map.items():
                plane_idx = mapping_info["plane_index"]
                ch_slice = mapping_info["channel_slice"]
                num_attr_channels = mapping_info["num_channels"]

                attribute_features_for_vis = current_level_feature_planes_tensor[:, plane_idx, ch_slice, :, :]
                
                # Defaults
                activation_fn = lambda x: x 
                should_normalize = True

                if attr_name == "color": 
                    activation_fn = lambda x: torch.sigmoid(self.color_activation_fn(x)) # User's preference
                    should_normalize = False 
                elif attr_name == "position": 
                    activation_fn = self.position_activation_fn # Will be wrapped for visualization
                    # should_normalize will be handled specifically below for position_rgb_feat
                elif attr_name == "scale": 
                    activation_fn = self.scaling_activation_fn # Used to get activated_scale before norm
                    # should_normalize will be handled specifically below for scale_magnitude_feat
                elif attr_name == "opacity": 
                    activation_fn = self.opacity_activation_fn
                    should_normalize = False 
                elif attr_name == "rotation":
                    current_activation_fn = self.rotation_activation_fn 
                    if num_attr_channels == 4: 
                        vis_rot_features = attribute_features_for_vis[:, :3, :, :]
                        images[f"{current_level_prefix}{attr_name}_quat_xyz_feat"] = self.prepare_attribute_image_for_export(
                            vis_rot_features, current_activation_fn, num_output_channels=3, target_H=H_feat, target_W=W_feat,
                            apply_min_max_normalization=True 
                        )
                    elif num_attr_channels == 3:
                         images[f"{current_level_prefix}{attr_name}_feat"] = self.prepare_attribute_image_for_export(
                            attribute_features_for_vis, current_activation_fn, num_output_channels=3, target_H=H_feat, target_W=W_feat,
                            apply_min_max_normalization=True 
                        )
                    else:
                        threestudio.warn(f"Cannot visualize rotation attribute '{attr_name}' with {num_attr_channels} channels easily.")
                    continue 
                
                # Special handling for position visualization to use position_rgb_feat key and (val+1)/2 mapping
                if attr_name == "position":
                    key_name = f"{current_level_prefix}position_rgb_feat"
                    position_vis_activation_fn = lambda x: (self.position_activation_fn(x).clamp(min=0.0, max=1.0) + 1.0) / 2.0
                    images[key_name] = self.prepare_attribute_image_for_export(
                        attribute_features_for_vis, 
                        position_vis_activation_fn, 
                        num_output_channels=3, 
                        target_H=H_feat, 
                        target_W=W_feat,
                        apply_min_max_normalization=False # Handled by vis_activation_fn
                    )
                # Special handling for scale visualization to use scale_magnitude_feat key and calculate magnitude
                elif attr_name == "scale":
                    key_name = f"{current_level_prefix}scale_magnitude_feat"
                    activated_scale = self.scaling_activation_fn(attribute_features_for_vis) 
                    scale_magnitude = torch.norm(activated_scale, p=2, dim=1, keepdim=True) 
                    images[key_name] = self.prepare_attribute_image_for_export(
                        scale_magnitude, 
                        lambda x: x, # Magnitude is already the value
                        num_output_channels=1, 
                        target_H=H_feat, 
                        target_W=W_feat,
                        apply_min_max_normalization=True # Normalize the magnitude
                    )
                # Generic visualization for other 1-channel (grayscale) or 3-channel (RGB) attributes (e.g. color, opacity)
                elif (num_attr_channels == 1 or num_attr_channels == 3) and attr_name not in ["position", "scale", "rotation"]:
                    key_name = f"{current_level_prefix}{attr_name}_feat"
                    images[key_name] = self.prepare_attribute_image_for_export(
                        attribute_features_for_vis, 
                        activation_fn, # Use the one determined above (e.g. for color, opacity)
                        num_output_channels=num_attr_channels,
                        target_H=H_feat, 
                        target_W=W_feat,
                        apply_min_max_normalization=should_normalize # Use the one determined above
                    )
                elif attr_name not in ["position", "scale", "rotation"]: # Avoid double warning
                    threestudio.warn(
                        f"Attribute '{attr_name}' with {num_attr_channels} channels cannot be directly visualized as RGB/Grayscale. Skipping."
                    )
        return images

    def prepare_attribute_image_for_export(self, 
                                           raw_features_ch_slice: torch.Tensor, 
                                           activation_fn: Callable, 
                                           num_output_channels: int, 
                                           target_H: int, 
                                           target_W: int,
                                           apply_min_max_normalization: bool = True):
        current_batch_size = raw_features_ch_slice.shape[0]
        activated_attr = activation_fn(raw_features_ch_slice)
        activated_attr_hwc = activated_attr.permute(0, 2, 3, 1)
        C_attr_activated = activated_attr_hwc.shape[-1]

        if C_attr_activated == num_output_channels:
            img = activated_attr_hwc
        elif C_attr_activated == 1 and num_output_channels == 3:
            img = activated_attr_hwc.repeat(1, 1, 1, 3)
        elif C_attr_activated == 3 and num_output_channels == 1:
            threestudio.debug("prepare_attribute_image: Converting 3 channels to 1 by averaging. Consider pre-calculating magnitude.")
            img = torch.mean(activated_attr_hwc, dim=-1, keepdim=True)
        else:
            threestudio.warn(
                f"Channel mismatch in prepare_attribute_image: activated channels {C_attr_activated}, "
                f"target output channels {num_output_channels}. Taking first {num_output_channels} or padding."
            )
            if C_attr_activated > num_output_channels:
                img = activated_attr_hwc[..., :num_output_channels]
            else:
                padding = torch.zeros(current_batch_size, target_H, target_W, num_output_channels - C_attr_activated, device=activated_attr_hwc.device, dtype=activated_attr_hwc.dtype)
                img = torch.cat([activated_attr_hwc, padding], dim=-1)
                
        if img.shape[1] != target_H or img.shape[2] != target_W:
            img = F.interpolate(img.permute(0,3,1,2), size=(target_H, target_W), mode='bilinear', align_corners=False).permute(0,2,3,1)

        if apply_min_max_normalization:
            min_val = torch.amin(img, dim=(1,2,3), keepdim=True)
            max_val = torch.amax(img, dim=(1,2,3), keepdim=True)
            img_normalized = (img - min_val) / (max_val - min_val + 1e-8)
            img_final = torch.clamp(img_normalized, 0.0, 1.0)
        else:
            img_final = torch.clamp(img, 0.0, 1.0)

        return img_final
