# Import necessary libraries
import threestudio
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from threestudio.models.renderers.base import Renderer
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.background.base import BaseBackground
from threestudio.utils.typing import *
from typing import Union, Tuple, Dict, Any # Add Union, Tuple, Dict, Any

# Register the new renderer
@threestudio.register("dual-renderer")
class DualRenderers(Renderer):
    """
    A wrapper renderer that uses two base renderers:
    1. high_res_renderer: Performs full-resolution rendering.
    2. low_res_renderer: Processes either a sampled subset of rays or a downsampled version of the full view,
                       based on the configuration (low_res_mode).
    Output format also depends on the configuration.
    """

    # --- Configuration Dataclass ---
    @dataclass
    class Config(Renderer.Config):
        # High-resolution renderer configuration
        high_res_renderer_type: str = ""
        high_res_renderer: Optional[Renderer.Config] = None # Assuming the base renderer is of VolumeRenderer type

        # Low-resolution renderer configuration
        low_res_renderer_type: str = ""
        low_res_renderer: Optional[Renderer.Config] = None # Assuming the base renderer is of VolumeRenderer type

        # Mode selector for low-resolution processing
        low_res_mode: str = 'sample' # Options: 'sample', 'downsample'

        # --- Config for 'sample' mode ---
        num_low_res_samples_per_view: int = 1024 # Number of rays (N) to sample per view
        # Source for calculating sampling probability
        sample_source: str = "opacity" # Which key from high_res_output to use for sampling probability
        guidance_processing: str = 'softmax' # How to process sample source ('softmax', 'normalize', 'raw')
        temperature: float = 0.1 # Temperature for softmax guidance processing

        # --- Config for passing guidance data to low-res renderer (both modes) ---
        guidance_source: str = "none" # Which key from high_res_output to pass as 'gs_depth' kwarg to low-res renderer (after downsampling)

        # --- Unified Output configuration ---
        allow_dual_output: bool = False # If True, return (high_res, low_res). If False, return combined high_res dict.
                                       # In 'sample' mode, low_res is sparse. In 'downsample' mode, this is ignored (always returns tuple).

    cfg: Config

    # --- Initialization and Configuration ---
    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        """Configure the two base renderers"""
        # Configure the high-resolution renderer
        threestudio.debug(f"Configuring high-resolution renderer: {self.cfg.high_res_renderer_type}")
        self.high_res_renderer: Renderer = threestudio.find(self.cfg.high_res_renderer_type)(
            self.cfg.high_res_renderer,
            geometry=geometry,
            material=material,
            background=background,
        )

        # Configure the low-resolution (sampled or downsampled) renderer
        threestudio.debug(f"Configuring low-resolution ('{self.cfg.low_res_mode}' mode) renderer: {self.cfg.low_res_renderer_type}")
        self.low_res_renderer: Renderer = threestudio.find(self.cfg.low_res_renderer_type)(
            self.cfg.low_res_renderer,
            geometry=geometry, # Assumes shared components, modify if needed
            material=material,
            background=background,
        )

    # --- Main Forward Method --- 
    def forward(
        self,
        rays_o: Float[Tensor, "B H_low W_low 3"],      # Low-resolution rays origin
        rays_d: Float[Tensor, "B H_low W_low 3"],      # Low-resolution rays direction
        rays_o_rasterize: Float[Tensor, "B H_high W_high 3"], # High-resolution rays origin
        rays_d_rasterize: Float[Tensor, "B H_high W_high 3"], # High-resolution rays direction
        light_positions: Float[Tensor, "B 3"],
        **kwargs
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Dispatches to training or evaluation forward method based on self.training."""
        if self.training:
            # Call training-specific forward logic
            return self._forward_train(
                rays_o_low=rays_o,
                rays_d_low=rays_d,
                rays_o_high=rays_o_rasterize,
                rays_d_high=rays_d_rasterize,
                light_positions=light_positions,
                **kwargs
            )
        else:
            # Call evaluation-specific forward logic
            return self._forward_eval( # self._forward_train( 
                rays_o_low=rays_o,
                rays_d_low=rays_d,
                rays_o_high=rays_o_rasterize,
                rays_d_high=rays_d_rasterize,
                light_positions=light_positions,
                **kwargs
            )

    # --- Training Forward ---
    def _forward_train(
        self,
        rays_o_low: Float[Tensor, "B H_low W_low 3"],
        rays_d_low: Float[Tensor, "B H_low W_low 3"],
        rays_o_high: Float[Tensor, "B H_high W_high 3"],
        rays_d_high: Float[Tensor, "B H_high W_high 3"],
        light_positions: Float[Tensor, "B 3"],
        **kwargs
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]: # Return type depends on mode
        """
        Training forward logic.
        Runs high-res, then runs low-res based on mode ('sample' or 'downsample').
        'sample' mode combines results into a single dict by scattering.
        'downsample' mode returns a tuple (high_res, low_res).
        """
        B, H_high, W_high, _ = rays_o_high.shape
        _, H_low, W_low, _ = rays_o_low.shape # Get low-res dims from input
        device = rays_o_high.device

        # 1. High-resolution rendering
        # Uses the high-resolution rays (rays_o_high, rays_d_high)
        out_high_res = self.high_res_renderer(
            rays_o_rasterize=rays_o_high, # Assuming high-res renderer expects this name
            rays_d_rasterize=rays_d_high,
            light_positions=light_positions,
            **kwargs
        )

        # --- Branch based on low_res_mode for training ---
        if self.cfg.low_res_mode == 'sample':
            # ========== Sample Mode (Training) ========== 
            N = self.cfg.num_low_res_samples_per_view
            num_pixels_per_view = H_high * W_high

            if N > num_pixels_per_view:
                threestudio.info(f"num_low_res_samples_per_view ({N}) > H*W ({num_pixels_per_view}). Clamping N to H*W.")
                N = num_pixels_per_view

            # 2a. Calculate guidance probabilities for sampling
            # Uses cfg.sample_source and cfg.guidance_processing
            if self.cfg.sample_source == "none":
                 guidance_probs = torch.ones(B, num_pixels_per_view, device=device) / num_pixels_per_view
            elif self.cfg.sample_source not in out_high_res:
                 raise ValueError(f"Sample source '{self.cfg.sample_source}' not found in high_res_renderer output keys: {list(out_high_res.keys())}")
            else:
                raw_sample_guidance = out_high_res[self.cfg.sample_source].detach()
                # Ensure shape is (B, H_high, W_high) or (B, H_high, W_high, C)
                if raw_sample_guidance.ndim == 4 and raw_sample_guidance.shape[1:3] == (H_high, W_high):
                    raw_sample_guidance = raw_sample_guidance[..., 0] 
                elif raw_sample_guidance.ndim == 3 and raw_sample_guidance.shape == (B, H_high, W_high):
                    pass # Already correct shape
                else:
                    raise ValueError(f"Unexpected shape for sample_source '{self.cfg.sample_source}': {raw_sample_guidance.shape}. Expected (B, H, W) or (B, H, W, C).")
                
                guidance_flat = raw_sample_guidance.view(B, -1)
                # Process into probabilities
                if self.cfg.guidance_processing == 'softmax':
                    guidance_probs = F.softmax(guidance_flat / self.cfg.temperature, dim=-1)
                elif self.cfg.guidance_processing == 'normalize':
                    guidance_flat = F.relu(guidance_flat) 
                    norm = guidance_flat.sum(dim=-1, keepdim=True)
                    guidance_probs = guidance_flat / (norm + 1e-6)
                elif self.cfg.guidance_processing == 'raw':
                    guidance_probs = F.relu(guidance_flat)
                else:
                    raise ValueError(f"Unknown guidance_processing mode: {self.cfg.guidance_processing}")
                
                if torch.isnan(guidance_probs).any() or (guidance_probs.sum(dim=-1) < 1e-6).any():
                     raise RuntimeError(f"Guidance probabilities became NaN or all zero after processing '{self.cfg.guidance_processing}' mode. Check guidance values and temperature.")

            # 3a. Sample N rays per view without replacement based on probabilities
            sampled_pixel_indices = torch.multinomial(guidance_probs, N, replacement=False) # (B, N)
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N) # (B, N)
            flat_indices = batch_indices * num_pixels_per_view + sampled_pixel_indices # (B, N)
            flat_indices_global = flat_indices.view(-1) # (B*N), global indices for flattened tensor

            # 4a. Prepare inputs for low-res (sampled) renderer
            # Gather rays and light positions corresponding to the sampled indices
            rays_o_flat = rays_o_high.view(B * num_pixels_per_view, 3)
            rays_d_flat = rays_d_high.view(B * num_pixels_per_view, 3)
            rays_o_sampled_flat = rays_o_flat[flat_indices_global] # (B*N, 3)
            rays_d_sampled_flat = rays_d_flat[flat_indices_global] # (B*N, 3)
            light_positions_sampled_flat = light_positions.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 3) # (B*N, 3)

            # Reshape inputs to (B, N, ...)
            rays_o_sampled = rays_o_sampled_flat.view(B, N, 3)
            rays_d_sampled = rays_d_sampled_flat.view(B, N, 3)
            light_positions_sampled = light_positions_sampled_flat.view(B, N, 3)

            # Prepare kwargs, sampling spatial ones, expanding per-view ones
            low_res_kwargs = {}
            guidance_data_sampled = None # For storing sampled guidance data

            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == B:
                    if v.ndim >= 3 and v.shape[1:3] == (H_high, W_high):
                        # Sample spatial tensor based on indices
                        v_flat = v.view(B * num_pixels_per_view, *v.shape[3:])
                        v_sampled_flat = v_flat[flat_indices_global]
                        low_res_kwargs[k] = v_sampled_flat.view(B, N, *v.shape[3:])
                    elif v.ndim >= 2: 
                        # Pass OTHER per-view tensors directly without expansion
                        # If low_res_renderer needs expanded versions, it must handle it internally
                        low_res_kwargs[k] = v 
                else: 
                     low_res_kwargs[k] = v # Pass non-tensors or non-batched tensors as is

            # Extract and sample guidance data (from cfg.guidance_source) if configured
            if self.cfg.guidance_source != "none":
                 if self.cfg.guidance_source not in out_high_res:
                     raise ValueError(f"Guidance source '{self.cfg.guidance_source}' for guidance_data not found in high_res_renderer output keys: {list(out_high_res.keys())}")
                 
                 raw_guidance_content = out_high_res[self.cfg.guidance_source].detach()
                 original_guidance_shape = raw_guidance_content.shape

                 # Ensure guidance has spatial dimensions for sampling
                 if raw_guidance_content.ndim >= 3 and original_guidance_shape[1:3] == (H_high, W_high):
                     guidance_channels = original_guidance_shape[3:]
                     guidance_content_flat = raw_guidance_content.view(B * num_pixels_per_view, *guidance_channels)
                     # Gather features at sampled locations
                     guidance_data_sampled_flat = guidance_content_flat[flat_indices_global] # Shape (B*N, C)
                     # Reshape to (B, N, C) for the low-res renderer
                     guidance_data_sampled = guidance_data_sampled_flat.view(B, N, *guidance_channels)
                     low_res_kwargs['gs_depth'] = guidance_data_sampled
                 else:
                     raise ValueError(f"Guidance source '{self.cfg.guidance_source}' has unsuitable shape for sampling: {original_guidance_shape}. Expected spatial dims (B, H, W, C).")

            # 5a. Low-resolution rendering on sampled rays
            out_low_res = self.low_res_renderer(
                rays_o=rays_o_sampled, 
                rays_d=rays_d_sampled,
                light_positions=light_positions_sampled,
                **low_res_kwargs
            )

            # 6a. Combine results by scattering sparse low-res results into high-res dict
            combined_output = out_high_res.copy()
            for key, low_res_value in out_low_res.items():
                # Ensure corresponding key exists and types match
                if key in combined_output and isinstance(low_res_value, torch.Tensor) and isinstance(combined_output[key], torch.Tensor):
                    high_res_value = combined_output[key]
                    original_shape = high_res_value.shape
                    # Check if low_res_value shape matches (B, N, ...)
                    if low_res_value.shape[0] == B and low_res_value.shape[1] == N and high_res_value.numel() >= B * N:
                        value_shape_per_ray = low_res_value.shape[2:] 

                        # Flatten target high-res tensor using reshape for safety
                        high_res_flat = high_res_value.reshape(B * num_pixels_per_view, *value_shape_per_ray)
                        # Flatten source low-res tensor using reshape for safety
                        low_res_value_flat = low_res_value.reshape(B * N, *value_shape_per_ray)
                        # Scatter low-res values into high-res tensor at sampled indices
                        high_res_flat[flat_indices_global] = low_res_value_flat
                        # Reshape back and update the output dictionary using reshape for safety
                        combined_output[key] = high_res_flat.reshape(original_shape)

            # Return the combined dictionary
            return combined_output

        elif self.cfg.low_res_mode == 'downsample':
            # ========== Downsample Mode (Training) ========== 
            # This mode uses the pre-defined low-resolution rays (rays_o_low, rays_d_low)
            
            # 3b. Optional: Extract and downsample guidance data (from cfg.guidance_source)
            guidance_data_downsampled = None
            if self.cfg.guidance_source != "none":
                if self.cfg.guidance_source not in out_high_res:
                     raise ValueError(f"Guidance source '{self.cfg.guidance_source}' specified but not found in high_res_renderer output for downsampling.")

                raw_guidance_content = out_high_res[self.cfg.guidance_source].detach()
                
                # Ensure guidance has spatial dimensions H_high, W_high
                if raw_guidance_content.ndim >= 3 and raw_guidance_content.shape[1:3] == (H_high, W_high):
                    # Downsample the guidance content to H_low, W_low
                    permute_dims = list(range(raw_guidance_content.ndim))
                    channel_dim_index = -1 
                    h_dim_index, w_dim_index = -3, -2
                    permute_dims_to = permute_dims[:h_dim_index] + [permute_dims[channel_dim_index]] + permute_dims[h_dim_index:channel_dim_index]
                    guidance_permuted = raw_guidance_content.permute(*permute_dims_to)
                    
                    guidance_downsampled_permuted = F.interpolate(
                        guidance_permuted, 
                        (H_low, W_low), # Target low resolution
                        mode="bilinear", 
                        align_corners=False
                    )
                    
                    # Calculate the inverse permutation to restore original dimension order
                    permute_dims_to_tensor = torch.tensor(permute_dims_to, device=raw_guidance_content.device)
                    permute_dims_back = torch.argsort(permute_dims_to_tensor).tolist()
                    guidance_data_downsampled = guidance_downsampled_permuted.permute(*permute_dims_back)

                else:
                     raise ValueError(f"Cannot handle guidance source '{self.cfg.guidance_source}' shape for downsampling guidance data: {raw_guidance_content.shape}. Expected at least 3 dims with shape (B, H, W, C).")

            # 4b. Prepare inputs for low-res (downsampled) renderer
            low_res_kwargs = kwargs.copy()
            # Add the downsampled guidance data if created
            if guidance_data_downsampled is not None:
                low_res_kwargs['gs_depth'] = guidance_data_downsampled

            # 5b. Low-resolution rendering using the provided low-res rays
            out_low_res = self.low_res_renderer(
                rays_o=rays_o_low, # Use low-res rays directly
                rays_d=rays_d_low,
                light_positions=light_positions,
                **low_res_kwargs
            )
            # Always return tuple in downsample mode during training
            return out_high_res, out_low_res

        else:
            # Should not happen if cfg validation is done
            raise ValueError(f"Invalid low_res_mode: {self.cfg.low_res_mode}. Must be 'sample' or 'downsample'.")

    # --- Evaluation Forward ---
    def _forward_eval(
        self,
        rays_o_low: Float[Tensor, "B H_low W_low 3"],
        rays_d_low: Float[Tensor, "B H_low W_low 3"],
        rays_o_high: Float[Tensor, "B H_high W_high 3"],
        rays_d_high: Float[Tensor, "B H_high W_high 3"],
        light_positions: Float[Tensor, "B 3"],
        **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]: # Always returns tuple
        """
        Evaluation forward logic.
        Runs high-res and low-res renderers independently using their respective ray inputs.
        Passes full high-resolution guidance data (if configured via guidance_source) to low-res renderer.
        Always returns a tuple (high_res_output, low_res_output).
        """
        # 1. High-resolution rendering using high-res rays
        out_high_res = self.high_res_renderer(
            rays_o_rasterize=rays_o_high, # Use high-res rays
            rays_d_rasterize=rays_d_high,
            light_positions=light_positions,
            **kwargs # Pass original kwargs
        )

        # 2. Prepare inputs for low-resolution renderer
        low_res_kwargs = kwargs.copy() 
        # Extract full high-resolution guidance data if configured
        if self.cfg.guidance_source != "none":
            if self.cfg.guidance_source not in out_high_res:
                 raise ValueError(f"Guidance source '{self.cfg.guidance_source}' specified for eval but not found in high_res_renderer output.")
            
            raw_guidance_content = out_high_res[self.cfg.guidance_source].detach()

            # === Simplified Downsampling logic (mirroring _forward_train) ===
            H_high, W_high = rays_o_high.shape[1], rays_o_high.shape[2] # Get high-res dims
            H_low, W_low = rays_o_low.shape[1], rays_o_low.shape[2]     # Get low-res dims

            # Ensure guidance has spatial dimensions H_high, W_high
            if raw_guidance_content.ndim >= 3 and raw_guidance_content.shape[1:3] == (H_high, W_high):
                # Downsample the guidance content to H_low, W_low
                permute_dims = list(range(raw_guidance_content.ndim))
                # Assume channel dim is last, H is -3, W is -2 for permutation
                # Modify if your data format is different (e.g., B, C, H, W)
                channel_dim_index = -1 
                h_dim_index, w_dim_index = -3, -2
                # Bring channel dim to the front for interpolate: B,C,H,W or C,H,W
                permute_dims_to = permute_dims[:h_dim_index] + [permute_dims[channel_dim_index]] + permute_dims[h_dim_index:channel_dim_index]
                guidance_permuted = raw_guidance_content.permute(*permute_dims_to).contiguous() # Ensure contiguous

                guidance_downsampled_permuted = F.interpolate(
                    guidance_permuted, 
                    (H_low, W_low), # Target low resolution
                    mode="bilinear", 
                    align_corners=False
                )

                # Calculate the inverse permutation to restore original dimension order
                permute_dims_to_tensor = torch.tensor(permute_dims_to, device=raw_guidance_content.device)
                permute_dims_back = torch.argsort(permute_dims_to_tensor).tolist()
                guidance_data_downsampled = guidance_downsampled_permuted.permute(*permute_dims_back)

                low_res_kwargs['gs_depth'] = guidance_data_downsampled
                guidance_data_high_res = guidance_data_downsampled # Update variable for clarity if needed later
            else:
                 # Raise error if shape is unsuitable, consistent with train logic
                 raise ValueError(f"Cannot handle guidance source '{self.cfg.guidance_source}' shape for downsampling guidance data in eval: {raw_guidance_content.shape}. Expected at least 3 dims with shape (..., H_high, W_high, C). Adjust permutation logic if needed.")
            # === END Simplified Downsampling logic ===
        
        # 3. Low-resolution rendering using low-res rays
        out_low_res = self.low_res_renderer(
            rays_o=rays_o_low, # Use low-res rays directly
            rays_d=rays_d_low,
            light_positions=light_positions,
            **low_res_kwargs # Pass kwargs + potentially high-res guidance
        )

        # 4. Always return tuple in eval mode
        return out_high_res, out_low_res

    # --- Helper Methods ---
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False) -> None:
        """Update the internal states of both base renderers (e.g., learning rate schedulers)"""
        self.high_res_renderer.update_step(epoch, global_step, on_load_weights)
        self.low_res_renderer.update_step(epoch, global_step, on_load_weights)

    def train(self, mode=True):
        """Set the training mode for both base renderers"""
        self.high_res_renderer.train(mode)
        self.low_res_renderer.train(mode)
        return super().train(mode=mode)

    def eval(self):
        """Set the evaluation mode for both base renderers"""
        self.high_res_renderer.eval()
        self.low_res_renderer.eval()
        return super().eval()
