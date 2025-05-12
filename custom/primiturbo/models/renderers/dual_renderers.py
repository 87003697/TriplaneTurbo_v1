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
        sample_source: str = "opacity" # Which key from high_res_output to use for sampling probability
        sample_source_processing: str = 'softmax' # How to process sample_source to get probabilities (e.g., 'softmax-0.1', 'softmax', 'normalize', 'raw')

        # --- Config for passing guidance data to low-res renderer (both modes) ---
        guidance_source: str = "none" # Which key from high_res_output to pass as 'gs_depth' kwarg to low-res renderer (after downsampling)

        # --- Config for evaluation mode ---
        eval_training: bool = False # rendering what is in training mode

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

    # --- Helper method for sampling probabilities ---
    def _get_sampling_probabilities(
        self,
        out_high_res: Dict[str, Any],
        H_high: int,
        W_high: int,
        B: int,
        device: torch.device
    ) -> torch.Tensor:
        """Calculate guidance probabilities for sampling based on cfg.sample_source."""
        num_pixels_per_view = H_high * W_high

        if self.cfg.sample_source == "none":
            guidance_probs = torch.ones(B, num_pixels_per_view, device=device) / num_pixels_per_view
        else:
            if self.cfg.sample_source == "depth_gradient":
                depth_key = "depth" # Default key for depth map
                if depth_key not in out_high_res:
                    raise ValueError(
                        f"Depth key '{depth_key}' for 'depth_gradient' sample source not found "
                        f"in high_res_renderer output keys: {list(out_high_res.keys())}"
                    )
                depth_map = out_high_res[depth_key].detach()
                if depth_map.ndim == 4:
                    if depth_map.shape[1] == H_high and depth_map.shape[2] == W_high and depth_map.shape[3] == 1:
                        depth_map_conv = depth_map.permute(0, 3, 1, 2)
                    elif depth_map.shape[1] == 1 and depth_map.shape[2] == H_high and depth_map.shape[3] == W_high:
                        depth_map_conv = depth_map
                    else:
                        raise ValueError(f"Unexpected depth_map shape for 'depth_gradient': {depth_map.shape}.")
                elif depth_map.ndim == 3 and depth_map.shape[0] == B and depth_map.shape[1] == H_high and depth_map.shape[2] == W_high:
                    depth_map_conv = depth_map.unsqueeze(1)
                else:
                    raise ValueError(f"Unexpected depth_map shape for 'depth_gradient': {depth_map.shape}.")
                depth_map_conv = torch.nan_to_num(depth_map_conv, nan=100.0, posinf=100.0, neginf=0.0)
                sobel_x_kernel = torch.tensor(
                    [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], 
                    device=device, dtype=depth_map_conv.dtype
                ).reshape(1, 1, 3, 3)
                sobel_y_kernel = torch.tensor(
                    [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], 
                    device=device, dtype=depth_map_conv.dtype
                ).reshape(1, 1, 3, 3)
                Gx_depth = F.conv2d(depth_map_conv, sobel_x_kernel, padding=1)
                Gy_depth = F.conv2d(depth_map_conv, sobel_y_kernel, padding=1)
                gradient_magnitude = torch.sqrt(Gx_depth**2 + Gy_depth**2 + 1e-6)
                raw_sample_guidance = gradient_magnitude.squeeze(1)
            elif self.cfg.sample_source in out_high_res:
                raw_sample_guidance = out_high_res[self.cfg.sample_source].detach()
                if raw_sample_guidance.ndim == 4 and raw_sample_guidance.shape[1:3] == (H_high, W_high):
                    raw_sample_guidance = raw_sample_guidance[..., 0] 
                elif raw_sample_guidance.ndim == 3 and raw_sample_guidance.shape == (B, H_high, W_high):
                    pass # Already correct shape
                else:
                    raise ValueError(
                        f"Unexpected shape for sample_source '{self.cfg.sample_source}': {raw_sample_guidance.shape}. "
                        f"Expected (B, H, W) or (B, H, W, C)."
                    )
            else:
                raise ValueError(
                    f"Sample source '{self.cfg.sample_source}' must be 'none', 'depth_gradient', "
                    f"or a key in high_res_renderer output. Got: '{self.cfg.sample_source}', "
                    f"Available keys in high_res_output: {list(out_high_res.keys())}"
                )
            
            guidance_flat = raw_sample_guidance.view(B, -1)
            
            current_processing_mode = self.cfg.sample_source_processing
            temperature = 0.1 # Default temperature

            if current_processing_mode.startswith('softmax'):
                parts = current_processing_mode.split('-')
                if len(parts) == 2:
                    try:
                        temperature = float(parts[1])
                        if temperature <= 0:
                            raise ValueError("Temperature for softmax must be positive.")
                    except ValueError:
                        raise ValueError(
                            f"Invalid temperature format in sample_source_processing: '{current_processing_mode}'. "
                            f"Expected 'softmax-value' where value is a positive float (e.g., 'softmax-0.1')."
                        )
                elif len(parts) > 2:
                     raise ValueError(
                        f"Invalid format for sample_source_processing: '{current_processing_mode}'. Too many hyphens."
                    )
                # If len(parts) == 1 (i.e., just 'softmax'), use default temperature 0.1
                
                guidance_probs = F.softmax(guidance_flat / temperature, dim=-1)

            elif current_processing_mode == 'normalize':
                guidance_flat = F.relu(guidance_flat)
                norm = guidance_flat.sum(dim=-1, keepdim=True)
                is_all_zero = norm < 1e-9
                uniform_prob = 1.0 / guidance_flat.shape[-1]
                guidance_probs = torch.where(is_all_zero, torch.full_like(guidance_flat, uniform_prob), guidance_flat / (norm + 1e-6))
            elif current_processing_mode == 'raw':
                guidance_probs = F.relu(guidance_flat)
            else:
                raise ValueError(f"Unknown sample_source_processing mode: {current_processing_mode}")
            
            if torch.isnan(guidance_probs).any() or (guidance_probs.sum(dim=-1) < 1e-6).any():
                raise RuntimeError(
                    f"Guidance probabilities became NaN or all zero after processing "
                    f"'{current_processing_mode}' mode with sample_source '{self.cfg.sample_source}'. "
                    f"Check guidance values and temperature (if applicable)."
                )
        return guidance_probs

    # --- Helper method for preparing guidance data ---
    def _prepare_guidance_for_low_res(
        self,
        out_high_res: Dict[str, Any],
        H_high: int,
        W_high: int,
        B: int,
        device: torch.device,
        processing_mode: str, # 'sample' or 'downsample'
        flat_indices_global: Optional[torch.Tensor] = None,
        N_samples: Optional[int] = None,
        H_low: Optional[int] = None,
        W_low: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Prepares guidance data (from cfg.guidance_source) for the low-resolution renderer."""
        if self.cfg.guidance_source == "none":
            return None

        if self.cfg.guidance_source not in out_high_res:
            raise ValueError(
                f"Guidance source '{self.cfg.guidance_source}' not found in high_res_renderer output keys: "
                f"{list(out_high_res.keys())}"
            )
        
        raw_guidance_content = out_high_res[self.cfg.guidance_source].detach()
        original_guidance_shape = raw_guidance_content.shape

        if processing_mode == 'sample':
            if not (raw_guidance_content.ndim >= 3 and original_guidance_shape[1:3] == (H_high, W_high)):
                raise ValueError(
                    f"Guidance source '{self.cfg.guidance_source}' has unsuitable shape for sampling: "
                    f"{original_guidance_shape}. Expected spatial dims (B, H, W, C)."
                )
            if flat_indices_global is None or N_samples is None:
                raise ValueError("flat_indices_global and N_samples must be provided for 'sample' mode.")

            guidance_channels = original_guidance_shape[3:]
            guidance_content_flat = raw_guidance_content.view(B * H_high * W_high, *guidance_channels)
            guidance_data_sampled_flat = guidance_content_flat[flat_indices_global]
            return guidance_data_sampled_flat.view(B, N_samples, *guidance_channels)
        
        elif processing_mode == 'downsample':
            if not (raw_guidance_content.ndim >= 3 and original_guidance_shape[1:3] == (H_high, W_high)):
                raise ValueError(
                    f"Guidance source '{self.cfg.guidance_source}' has unsuitable shape for downsampling: "
                    f"{original_guidance_shape}. Expected at least 3 dims with shape (B, H, W, C)."
                )
            if H_low is None or W_low is None:
                raise ValueError("H_low and W_low must be provided for 'downsample' mode.")

            permute_dims = list(range(raw_guidance_content.ndim))
            # Assuming channel dim is last, H is -3, W is -2. Modify if data format is B, C, H, W
            # For B, H, W, C -> B, C, H, W for interpolate
            channel_dim_index = -1 
            h_dim_index, w_dim_index = -3, -2 
            if raw_guidance_content.ndim == 3: # B, H, W -> B, 1, H, W
                raw_guidance_content = raw_guidance_content.unsqueeze(channel_dim_index) # B,H,W,1 or B,H,1,W if channel_dim_index wrong
                # if we unsqueezed, channel is now last. For interpolate, need it at dim 1 (idx after batch)
                # This needs careful handling if original is B,H,W. Standardizing on B,H,W,C input.
                # For B,H,W, a common approach is to make it B,1,H,W
                # For now, let's assume input is B,H,W,C or B,C,H,W that becomes B,H,W,C
                # If input was B,H,W and we want B,1,H,W for interpolate, permute will be different
                # This section of permutation logic assumes input has a channel dimension already (e.g. B,H,W,C)
                # If not, the permute logic might need to adapt or input needs unsqueezing to B,1,H,W first
                # For B,H,W,C: target B,C,H,W
                if channel_dim_index == -1: # If channels are last
                    permute_dims_to = permute_dims[:h_dim_index] + [permute_dims[channel_dim_index]] + permute_dims[h_dim_index:channel_dim_index]
                else: # If channels are e.g. at index 1 (B,C,H,W)
                    # This branch might not be hit if we always expect B,H,W,C as input to this function
                    permute_dims_to = permute_dims # Already B,C,H,W
            elif raw_guidance_content.ndim == 4: # B,H,W,C or B,C,H,W
                if original_guidance_shape[1] == H_high: # B,H,W,C
                    permute_dims_to = [0, 3, 1, 2] # B, C, H, W
                elif original_guidance_shape[1] == raw_guidance_content.shape[-1] and original_guidance_shape[2] == H_high: # B,C,H,W
                    permute_dims_to = list(range(4)) # Already B,C,H,W
                else:
                    raise ValueError(f"Unsupported 4D shape for downsampling: {original_guidance_shape}")
            else: # ndim < 3 or > 4 not handled well by current permute
                raise ValueError(f"Guidance source for downsampling has {raw_guidance_content.ndim} dims. Expected 3 (B,H,W assumed to be B,H,W,1 effectively) or 4 (B,H,W,C or B,C,H,W).")

            guidance_permuted = raw_guidance_content.permute(*permute_dims_to).contiguous()
            guidance_downsampled_permuted = F.interpolate(
                guidance_permuted, (H_low, W_low), mode="bilinear", align_corners=False
            )
            permute_dims_to_tensor = torch.tensor(permute_dims_to, device=device)
            permute_dims_back = torch.argsort(permute_dims_to_tensor).tolist()
            return guidance_downsampled_permuted.permute(*permute_dims_back)
        
        else:
            raise ValueError(f"Unknown processing_mode: {processing_mode}")

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
            func = self._forward_train if self.cfg.eval_training else self._forward_eval

            return func( # self._forward_eval( # 
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
            num_pixels_per_view_high = H_high * W_high

            if N > num_pixels_per_view_high:
                threestudio.info(f"num_low_res_samples_per_view ({N}) > H*W ({num_pixels_per_view_high}). Clamping N to H*W.")
                N = num_pixels_per_view_high

            # 2a. Calculate guidance probabilities for sampling
            guidance_probs = self._get_sampling_probabilities(out_high_res, H_high, W_high, B, device)
            
            # 3a. Sample N rays per view without replacement based on probabilities
            sampled_pixel_indices = torch.multinomial(guidance_probs, N, replacement=False) # (B, N)
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N) # (B, N)
            flat_indices = batch_indices * num_pixels_per_view_high + sampled_pixel_indices # (B, N)
            flat_indices_global = flat_indices.view(-1) # (B*N), global indices for flattened tensor

            # 4a. Prepare inputs for low-res (sampled) renderer
            rays_o_flat = rays_o_high.view(B * num_pixels_per_view_high, 3)
            rays_d_flat = rays_d_high.view(B * num_pixels_per_view_high, 3)
            rays_o_sampled_flat = rays_o_flat[flat_indices_global] # (B*N, 3)
            rays_d_sampled_flat = rays_d_flat[flat_indices_global] # (B*N, 3)
            light_positions_sampled_flat = light_positions.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 3) # (B*N, 3)

            rays_o_sampled = rays_o_sampled_flat.view(B, N, 3)
            rays_d_sampled = rays_d_sampled_flat.view(B, N, 3)
            light_positions_sampled = light_positions_sampled_flat.view(B, N, 3)

            low_res_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == B:
                    if v.ndim >= 3 and v.shape[1:3] == (H_high, W_high):
                        v_flat = v.view(B * num_pixels_per_view_high, *v.shape[3:])
                        v_sampled_flat = v_flat[flat_indices_global]
                        low_res_kwargs[k] = v_sampled_flat.view(B, N, *v.shape[3:])
                    elif v.ndim >= 2: 
                        low_res_kwargs[k] = v 
                else: 
                     low_res_kwargs[k] = v

            # Prepare guidance data for low-res renderer
            guidance_data_for_low_res = self._prepare_guidance_for_low_res(
                out_high_res, H_high, W_high, B, device,
                processing_mode='sample',
                flat_indices_global=flat_indices_global,
                N_samples=N
            )
            if guidance_data_for_low_res is not None:
                low_res_kwargs['gs_depth'] = guidance_data_for_low_res

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
                if key in combined_output and isinstance(low_res_value, torch.Tensor) and isinstance(combined_output[key], torch.Tensor):
                    high_res_value = combined_output[key]
                    original_shape = high_res_value.shape
                    if low_res_value.shape[0] == B and low_res_value.shape[1] == N and high_res_value.numel() >= B * N:
                        value_shape_per_ray = low_res_value.shape[2:] 
                        high_res_flat = high_res_value.reshape(B * num_pixels_per_view_high, *value_shape_per_ray)
                        low_res_value_flat = low_res_value.reshape(B * N, *value_shape_per_ray)
                        high_res_flat[flat_indices_global] = low_res_value_flat
                        combined_output[key] = high_res_flat.reshape(original_shape)

            # Return the combined dictionary
            return combined_output

        elif self.cfg.low_res_mode == 'downsample':
            # ========== Downsample Mode (Training) ========== 
            # This mode uses the pre-defined low-resolution rays (rays_o_low, rays_d_low)
            
            # 3b. Optional: Extract and downsample guidance data (from cfg.guidance_source)
            guidance_data_for_low_res = self._prepare_guidance_for_low_res(
                out_high_res, H_high, W_high, B, device,
                processing_mode='downsample',
                H_low=H_low, W_low=W_low
            )
            if guidance_data_for_low_res is not None:
                low_res_kwargs = kwargs.copy()
                low_res_kwargs['gs_depth'] = guidance_data_for_low_res

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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]: 
        """
        Evaluation forward logic.
        Runs high-res and low-res renderers independently using their respective ray inputs.
        Passes downsampled high-resolution guidance data (if configured) to low-res renderer.
        Always returns a tuple (high_res_output, low_res_output).
        """
        B, H_high, W_high, _ = rays_o_high.shape
        _, H_low_eval, W_low_eval, _ = rays_o_low.shape # Get low-res dims from input
        device = rays_o_high.device

        # 1. High-resolution rendering using high-res rays
        out_high_res = self.high_res_renderer(
            rays_o_rasterize=rays_o_high, # Use high-res rays
            rays_d_rasterize=rays_d_high,
            light_positions=light_positions,
            **kwargs # Pass original kwargs
        )

        # 2. Prepare inputs for low-resolution renderer
        low_res_kwargs = kwargs.copy() 
        
        # Use helper function to get downsampled guidance data
        guidance_data_for_low_res = self._prepare_guidance_for_low_res(
            out_high_res, 
            H_high, 
            W_high, 
            B, 
            device,
            processing_mode='downsample', 
            H_low=H_low_eval, 
            W_low=W_low_eval
        )
        
        if guidance_data_for_low_res is not None:
            low_res_kwargs['gs_depth'] = guidance_data_for_low_res
        
        # 3. Low-resolution rendering using low-res rays
        out_low_res = self.low_res_renderer(
            rays_o=rays_o_low, # Use low-res rays directly
            rays_d=rays_d_low,
            light_positions=light_positions,
            **low_res_kwargs # Pass kwargs + potentially processed guidance
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
