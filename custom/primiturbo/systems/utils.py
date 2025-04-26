import torch

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