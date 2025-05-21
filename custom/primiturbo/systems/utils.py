import torch
from threestudio.systems.base import BaseLift3DSystem # For type hinting system_object
from threestudio.utils.typing import * # For Dict, Any etc.
import threestudio # For threestudio.info
import os

# Helper function for center point depth visualization
# Can now optionally compute disparity if near and far are provided
def visualize_center_depth(depth_map, near=None, far=None):
    if not isinstance(depth_map, torch.Tensor):
            raise TypeError(f"Expected depth_map to be a torch.Tensor, got {type(depth_map)}")
    if depth_map.numel() == 0: # Handle empty tensor case
            return torch.zeros_like(depth_map)
            
    mask_inf = torch.isinf(depth_map)

    # --- Determine effective near/far for processing ---
    finite_vals = depth_map[~mask_inf & ~torch.isnan(depth_map)] # Exclude NaNs as well
    min_depth_in_map = finite_vals.min().item() if finite_vals.numel() > 0 else 0.0
    max_depth_in_map = finite_vals.max().item() if finite_vals.numel() > 0 else 1.0
    
    # Use provided near/far if valid, otherwise estimate from map's content
    use_provided_near_far = (
        near is not None and far is not None and
        isinstance(near, (int, float)) and isinstance(far, (int, float)) and
        near < far
    )

    effective_near = float(near) if use_provided_near_far else float(min_depth_in_map)
    effective_far = float(far) if use_provided_near_far else float(max_depth_in_map)

    # Ensure near < far for calculations, handle potential zero range from map content
    epsilon = 1e-7
    if effective_near >= effective_far - epsilon:
            # If near >= far (or too close), try using map range directly if different
            if min_depth_in_map < max_depth_in_map - epsilon:
                effective_near = float(min_depth_in_map)
                effective_far = float(max_depth_in_map)
            else: # If map range is also zero/invalid, create a minimal valid range
                effective_far = effective_near + 1.0 

    # --- Process depth map ---
    # Clamp depth to the effective range first
    # Need to handle potential NaNs before clamping
    depth_no_nan = torch.nan_to_num(depth_map, nan=effective_far) # Replace NaN with far
    depth_clamped = torch.clamp(depth_no_nan, effective_near, effective_far)
    
    # Replace original inf values with the far plane value for background
    depth_proc = depth_clamped.clone() # Use clone to avoid modifying clamped tensor inplace
    depth_proc[mask_inf] = effective_far # Set background to far

    # --- Calculate visualization map ---
    vis_map = torch.zeros_like(depth_proc) # Initialize output map
    valid_range = (effective_far - effective_near) > epsilon

    if use_provided_near_far and valid_range:
        # --- Option 1: Calculate disparity directly ---
        # Formula: (far - depth) / (far - near) -> 1=Near, 0=Far (White=Near)
        disparity = (effective_far - depth_proc) / (effective_far - effective_near)
        vis_map = torch.clamp(disparity, 0.0, 1.0)
        # Ensure background (originally inf, now far) is strictly 0 (black)
        vis_map[mask_inf] = 0.0
    else:
        # --- Option 2: Fallback to normalization + INVERSION ---
        # Input depth is positive distance: [Nearest Dist, Farthest Dist]
        # Normalize based on the effective range: [0=Near, 1=Far]
        # (effective_near is the nearest distance, effective_far is the farthest)
        if valid_range:
            depth_norm = (depth_proc - effective_near) / (effective_far - effective_near)
        else:
            depth_norm = torch.zeros_like(depth_proc) # Avoid division by zero
        
        depth_norm = torch.clamp(depth_norm, 0.0, 1.0) # Clamp normalization result
        # RE-INTRODUCE Inversion to get [1=Near, 0=Far] -> White=Near, Black=Far
        # vis_map = depth_norm # Previous version for negative Z input
        vis_map = 1.0 - depth_norm
        # Set background (originally inf, now represented by far distance) to black (0).
        vis_map[mask_inf] = 0.0 

    # Ensure final map is within [0, 1]
    vis_map = torch.clamp(vis_map, 0.0, 1.0)
    
    return vis_map


# +++++ HELPER FUNCTION FOR ATTRIBUTE VISUALIZATION GRID +++++
def save_attribute_visualization_grid(
    system_object: BaseLift3DSystem,
    batch: Dict[str, Any],
    item_idx_in_batch: int, # Index of the item within the current dataloader batch
    attribute_images_dict: Dict[str, torch.Tensor],
    phase: str, # "val" or "test"
    debug: bool = False, # Add a debug flag
    save_path: str = "" # Add a save_path parameter
):
    """
    Collects and saves a 2x2 grid of attribute images (color, position, scale, opacity)
    for a given item in a batch, handling hierarchical levels if present.
    """
    # print(f"[DEBUG save_attribute_visualization_grid] Called. item_idx_in_batch: {item_idx_in_batch}, phase: '{phase}', debug: {debug}")
    if not attribute_images_dict:
        if debug:
            threestudio.info(f"[Debug Vis Grid] Attribute images dictionary is empty for item {item_idx_in_batch}. Skipping.")
        # print("[DEBUG save_attribute_visualization_grid] attribute_images_dict is empty. Returning.")
        return

    # Determine batch size of the attribute images themselves (usually 1)
    # Take the first key to get an image tensor and its batch dim
    first_key = next(iter(attribute_images_dict))
    attr_img_batch_size = attribute_images_dict[first_key].shape[0]

    # Determine the actual index into the attribute image tensors.
    # This handles if attr_img_batch_size is 1, but we are iterating through a larger dataloader batch.
    current_attr_tensor_idx = item_idx_in_batch % attr_img_batch_size
    # print(f"[DEBUG save_attribute_visualization_grid] attr_img_batch_size: {attr_img_batch_size}, current_attr_tensor_idx: {current_attr_tensor_idx}")


    # Check for hierarchical levels
    has_levels = any(key.startswith("level_") or key.startswith("_final_level_") for key in attribute_images_dict.keys()) # Adjusted to include _final_level_
    num_levels = 0
    if has_levels:
        level_indices = set()
        for key in attribute_images_dict.keys():
            if key.startswith("level_"):
                try:
                    level_indices.add(int(key.split("_")[1]))
                except (ValueError, IndexError):
                    if debug:
                        threestudio.info(f"[Debug Vis Grid] Could not parse level index from key: {key}")
            elif key.startswith("_final_level_"): # Handle the case where only final_level is present
                level_indices.add(0) # Treat final_level as level 0 if it's the only one
        
        if level_indices:
            num_levels = max(level_indices) + 1 if any(k.startswith("level_") for k in attribute_images_dict.keys()) else 1 # If only final_level, num_levels is 1
        else:
            if debug and has_levels:
                threestudio.info(f"[Debug Vis Grid] Detected level prefix but could not parse valid level indices. Treating as single level.")
            has_levels = False # Fallback to no levels if parsing failed
    
    if not has_levels and any(key.startswith("_final_level_") for key in attribute_images_dict.keys()):
        # This handles the case where export_gaussian_attributes_as_images produced only a _final_level_ due to single tensor input
        has_levels = True # We will treat this as one "level"
        num_levels = 1
    elif not has_levels: # No level prefixes at all
        num_levels = 1

    if debug:
        threestudio.info(f"[Debug Vis Grid] Processing item {item_idx_in_batch}, found {num_levels} level(s). Attribute dict keys: {list(attribute_images_dict.keys())}")
    # else:
        # Add a non-debug print for key information if debug is false, to still get some output
        # print(f"[save_attribute_visualization_grid] Processing item {item_idx_in_batch}, num_levels: {num_levels}. Keys: {list(attribute_images_dict.keys())}")

    # --- Determine prompt_name_part ONCE for the current item_idx_in_batch --- 
    prompt_name_part = None
    # Try to get from batch['name']
    if "name" in batch and isinstance(batch.get("name"), (list, tuple)) and item_idx_in_batch < len(batch["name"]):
        candidate_name = batch['name'][item_idx_in_batch]
        if isinstance(candidate_name, str) and candidate_name.strip():
            prompt_name_part = candidate_name.replace(',', '').replace('.', '').replace(' ', '_').strip()
            if not prompt_name_part: prompt_name_part = None # Ensure empty string after replace isn't kept
    
    if prompt_name_part is None: # If not found in batch['name'], try batch['prompt']
        if "prompt" in batch and isinstance(batch.get("prompt"), (list, tuple)) and item_idx_in_batch < len(batch["prompt"]):
            candidate_prompt = batch['prompt'][item_idx_in_batch]
            if isinstance(candidate_prompt, str) and candidate_prompt.strip():
                prompt_name_part = candidate_prompt.replace(',', '').replace('.', '').replace(' ', '_').strip()
                if not prompt_name_part: prompt_name_part = None # Ensure empty string after replace isn't kept

    if prompt_name_part is None: 
        if debug:
            log_index_str = f"item_idx_in_batch {item_idx_in_batch}"
            try:
                actual_item_global_index = batch['index'][item_idx_in_batch].item()
                log_index_str = f"item_idx_in_batch {item_idx_in_batch} (global index {actual_item_global_index})"
            except Exception: 
                pass 
            threestudio.info(f"[Debug Vis Grid] Skipping attribute image for {log_index_str} as no valid 'name' or 'prompt' found.")
        # print(f"[DEBUG save_attribute_visualization_grid] No valid name/prompt for item {item_idx_in_batch}. Returning.")
        return 
    # --- End prompt_name_part determination and early exit ---
    # print(f"[DEBUG save_attribute_visualization_grid] Determined prompt_name_part: '{prompt_name_part}'")

    for level_i in range(num_levels):
        if has_levels and num_levels > 1 and any(k.startswith("level_") for k in attribute_images_dict.keys()):
            level_prefix = f"level_{level_i}_"
        elif has_levels and num_levels == 1 and any(key.startswith("_final_level_") for k in attribute_images_dict.keys()):
            level_prefix = "_final_level_" 
        else:
            level_prefix = "" 
        
        current_level_images_to_grid = []
        # print(f"  [DEBUG save_attribute_visualization_grid] Level {level_i}, using prefix: '{level_prefix}'")
        
        key_color = level_prefix + "color_feat"
        key_pos = level_prefix + "position_rgb_feat"
        key_scale = level_prefix + "scale_magnitude_feat"
        key_opa = level_prefix + "opacity_feat"
        
        expected_keys = [key_color, key_pos, key_scale, key_opa]
        if debug:
            threestudio.info(f"[Debug Vis Grid] Item {item_idx_in_batch}, Level {level_i}: Expecting keys: {expected_keys}")
        # else:
            # print(f"  [save_attribute_visualization_grid] Item {item_idx_in_batch}, Level {level_i}: Expecting keys: {expected_keys}")

        if key_color in attribute_images_dict:
            img_tensor = attribute_images_dict[key_color][current_attr_tensor_idx] 
            current_level_images_to_grid.append({"type": "rgb", "img": img_tensor, "kwargs": {"data_range": (0,1), "data_format": "HWC"}})
            if debug: threestudio.info(f"[Debug Vis Grid] Found {key_color}, shape: {img_tensor.shape}")
            # else: print(f"    Found {key_color}, shape: {img_tensor.shape}")
        elif debug:
            threestudio.info(f"[Debug Vis Grid] Missing {key_color}")
        # else:
            # print(f"    Missing {key_color}")
        
        if key_pos in attribute_images_dict:
            img_tensor = attribute_images_dict[key_pos][current_attr_tensor_idx]
            current_level_images_to_grid.append({"type": "rgb", "img": img_tensor, "kwargs": {"data_range": (0,1), "data_format": "HWC"}})
            if debug: threestudio.info(f"[Debug Vis Grid] Found {key_pos}, shape: {img_tensor.shape}")
            # else: print(f"    Found {key_pos}, shape: {img_tensor.shape}")
        elif debug:
            threestudio.info(f"[Debug Vis Grid] Missing {key_pos}")
        # else:
            # print(f"    Missing {key_pos}")
            
        if key_scale in attribute_images_dict:
            img_tensor = attribute_images_dict[key_scale][current_attr_tensor_idx]
            current_level_images_to_grid.append({"type": "grayscale", "img": img_tensor, "kwargs": {"data_range": (0,1)}})
            if debug: threestudio.info(f"[Debug Vis Grid] Found {key_scale}, shape: {img_tensor.shape}")
            # else: print(f"    Found {key_scale}, shape: {img_tensor.shape}")
        elif debug:
            threestudio.info(f"[Debug Vis Grid] Missing {key_scale}")
        # else:
            # print(f"    Missing {key_scale}")
            
        if key_opa in attribute_images_dict:
            img_tensor = attribute_images_dict[key_opa][current_attr_tensor_idx]
            current_level_images_to_grid.append({"type": "grayscale", "img": img_tensor, "kwargs": {"data_range": (0,1)}})
            if debug: threestudio.info(f"[Debug Vis Grid] Found {key_opa}, shape: {img_tensor.shape}")
            # else: print(f"    Found {key_opa}, shape: {img_tensor.shape}")
        elif debug:
            threestudio.info(f"[Debug Vis Grid] Missing {key_opa}")
        # else:
            # print(f"    Missing {key_opa}")
        
        if len(current_level_images_to_grid) == 4:
            # print(f"    [DEBUG save_attribute_visualization_grid] Found all 4 images for level {level_i}. Proceeding to save grid.")
            top_level_attrs_dir_relative = f"it{system_object.true_global_step}-{phase}-attr"
            image_name_relative_to_save_path = f"{top_level_attrs_dir_relative}/{prompt_name_part}.png"
            
            if debug:
                # print(f"[DEBUG save_attribute_visualization_grid] prompt_name_part: {prompt_name_part}")
                # print(f"[DEBUG save_attribute_visualization_grid] top_level_attrs_dir_relative (for mkdir): {os.path.join(save_path, top_level_attrs_dir_relative)}")
                # print(f"[DEBUG save_attribute_visualization_grid] image_name_relative_to_save_path: {image_name_relative_to_save_path}")
                pass # Keep debug prints inside this block if needed for very verbose output
            
            full_image_path = os.path.join(save_path, image_name_relative_to_save_path)
            # print(f"    [DEBUG save_attribute_visualization_grid] Saving grid to: {full_image_path}")
            system_object.save_image_grid(
                full_image_path,
                current_level_images_to_grid, 
                name=f"{phase}_attrs_{level_prefix}combined_step", 
                step=system_object.true_global_step
            )
            # print(f"    [DEBUG save_attribute_visualization_grid] Grid saved for level {level_i}.")
        elif debug:
            threestudio.info(f"[Debug Vis Grid] Item {item_idx_in_batch}, Level {level_i}: Did not find all 4 attribute images. Found {len(current_level_images_to_grid)}. Skipping grid save for this level.")
        # else:
            # print(f"  [save_attribute_visualization_grid] Item {item_idx_in_batch}, Level {level_i}: Did not find all 4 attribute images. Found {len(current_level_images_to_grid)}. Skipping grid save for this level.")

# ----- END HELPER FUNCTION -----
