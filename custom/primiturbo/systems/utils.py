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
    if not attribute_images_dict:
        if debug:
            threestudio.info(f"[Debug Vis Grid] Attribute images dictionary is empty for item {item_idx_in_batch}. Skipping.")
        return

    # Determine batch size of the attribute images themselves (usually 1)
    # Take the first key to get an image tensor and its batch dim
    first_key = next(iter(attribute_images_dict))
    attr_img_batch_size = attribute_images_dict[first_key].shape[0]

    # Determine the actual index into the attribute image tensors.
    # This handles if attr_img_batch_size is 1, but we are iterating through a larger dataloader batch.
    current_attr_tensor_idx = item_idx_in_batch % attr_img_batch_size


    # Check for hierarchical levels
    has_levels = any(key.startswith("level_") for key in attribute_images_dict.keys())
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
        if level_indices:
            num_levels = max(level_indices) + 1
        else:
            if debug and has_levels: # has_levels was true but couldn't find valid level indices
                threestudio.info(f"[Debug Vis Grid] Detected 'level_' prefix but could not parse valid level indices. Treating as single level.")
            has_levels = False # Fallback to no levels if parsing failed

    if not has_levels:
        num_levels = 1 # Treat as a single level for the loop

    if debug:
        threestudio.info(f"[Debug Vis Grid] Processing item {item_idx_in_batch}, found {num_levels} level(s). Attribute dict keys: {list(attribute_images_dict.keys())}")

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

    if prompt_name_part is None: # If still None, this item should be skipped (would have defaulted to "item_X")
        if debug:
            log_index_str = f"item_idx_in_batch {item_idx_in_batch}"
            try:
                # Try to get a global index for richer logging, but don't fail if not present/accessible
                actual_item_global_index = batch['index'][item_idx_in_batch].item()
                log_index_str = f"item_idx_in_batch {item_idx_in_batch} (global index {actual_item_global_index})"
            except Exception: 
                pass # Keep log_index_str as is if global index retrieval fails
            threestudio.info(f"[Debug Vis Grid] Skipping attribute image for {log_index_str} as no valid 'name' or 'prompt' found.")
        return # Exit the function, do not save for this item_idx_in_batch
    # --- End prompt_name_part determination and early exit ---

    for level_i in range(num_levels):
        level_prefix = f"level_{level_i}_" if has_levels and num_levels > 1 else "" # only add prefix if genuinely multilevel
        
        current_level_images_to_grid = []
        
        key_color = level_prefix + "color_feat"
        key_pos = level_prefix + "position_rgb_feat"
        key_scale = level_prefix + "scale_magnitude_feat"
        key_opa = level_prefix + "opacity_feat"
        
        expected_keys = [key_color, key_pos, key_scale, key_opa]
        if debug:
            threestudio.info(f"[Debug Vis Grid] Item {item_idx_in_batch}, Level {level_i}: Expecting keys: {expected_keys}")

        # Color (RGB)
        if key_color in attribute_images_dict:
            img_tensor = attribute_images_dict[key_color][current_attr_tensor_idx] # HWC
            current_level_images_to_grid.append({"type": "rgb", "img": img_tensor, "kwargs": {"data_range": (0,1), "data_format": "HWC"}})
            if debug: threestudio.info(f"[Debug Vis Grid] Found {key_color}, shape: {img_tensor.shape}")
        elif debug:
            threestudio.info(f"[Debug Vis Grid] Missing {key_color}")
        
        # Position (RGB)
        if key_pos in attribute_images_dict:
            img_tensor = attribute_images_dict[key_pos][current_attr_tensor_idx]
            current_level_images_to_grid.append({"type": "rgb", "img": img_tensor, "kwargs": {"data_range": (0,1), "data_format": "HWC"}})
            if debug: threestudio.info(f"[Debug Vis Grid] Found {key_pos}, shape: {img_tensor.shape}")
        elif debug:
            threestudio.info(f"[Debug Vis Grid] Missing {key_pos}")
            
        # Scale Magnitude (Grayscale)
        if key_scale in attribute_images_dict:
            img_tensor = attribute_images_dict[key_scale][current_attr_tensor_idx]
            current_level_images_to_grid.append({"type": "grayscale", "img": img_tensor, "kwargs": {"data_range": (0,1)}})
            if debug: threestudio.info(f"[Debug Vis Grid] Found {key_scale}, shape: {img_tensor.shape}")
        elif debug:
            threestudio.info(f"[Debug Vis Grid] Missing {key_scale}")
            
        # Opacity (Grayscale)
        if key_opa in attribute_images_dict:
            img_tensor = attribute_images_dict[key_opa][current_attr_tensor_idx]
            current_level_images_to_grid.append({"type": "grayscale", "img": img_tensor, "kwargs": {"data_range": (0,1)}})
            if debug: threestudio.info(f"[Debug Vis Grid] Found {key_opa}, shape: {img_tensor.shape}")
        elif debug:
            threestudio.info(f"[Debug Vis Grid] Missing {key_opa}")
        
        if len(current_level_images_to_grid) == 4:
            # Determine filename parts
            # item_idx_in_batch is the index within the current dataloader batch.
            # batch['index'] usually contains global indices if dataset provides them.
            # We should use item_idx_in_batch to correctly get name/prompt for the current item.
            # --- PROMPT_NAME_PART is now determined and validated above this loop ---
            # No longer need to determine actual_item_global_index or prompt_name_part here.
            # The old block for this (approx lines 192-201) is replaced by the logic above the loop.
            
            # Define the top-level directory for all attribute images of this step and phase
            # Example: it0-val-attr
            top_level_attrs_dir_relative = f"it{system_object.true_global_step}-{phase}-attr"
            
            # # Ensure the top-level attribute directory exists (relative to save_path)
            # os.makedirs(os.path.join(save_path, top_level_attrs_dir_relative), exist_ok=True)

            # Define the image filename. It will be directly under top_level_attrs_dir_relative.
            # Example: it0-val-attr/a_20-sided_die_made_out_of_glass.png
            # Example: it0-val-attr/item_1.png
            image_name_relative_to_save_path = f"{top_level_attrs_dir_relative}/{prompt_name_part}.png"
            
            if debug:
                print(f"[DEBUG save_attribute_visualization_grid] prompt_name_part: {prompt_name_part}")
                print(f"[DEBUG save_attribute_visualization_grid] top_level_attrs_dir_relative (for mkdir): {os.path.join(save_path, top_level_attrs_dir_relative)}")
                print(f"[DEBUG save_attribute_visualization_grid] image_name_relative_to_save_path: {image_name_relative_to_save_path}")
            
            # Full path for the combined image
            full_image_path = os.path.join(save_path, image_name_relative_to_save_path)
            
            # Create and save the PIL image (assuming combined_image_grid_pil is defined correctly before this)
            system_object.save_image_grid(
                full_image_path,
                current_level_images_to_grid, # List of 4 image dicts
                name=f"{phase}_attrs_{level_prefix}combined_step", 
                step=system_object.true_global_step
            )
        elif debug:
            threestudio.info(f"[Debug Vis Grid] Item {item_idx_in_batch}, Level {level_i}: Did not find all 4 attribute images. Found {len(current_level_images_to_grid)}. Skipping grid save for this level.")

# ----- END HELPER FUNCTION -----
