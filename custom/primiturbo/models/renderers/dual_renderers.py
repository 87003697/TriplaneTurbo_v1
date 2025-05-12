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
        guidance_source_processing: str = "none" # Options: "none", "cut_min_max"

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
                raise ValueError("depth_gradient sample source is not supported yet.")
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
    def _apply_guidance_source_processing(self, data: torch.Tensor, processing_type: str) -> torch.Tensor:
        """Applies post-processing to the guidance data."""
        if processing_type == "none":
            return data
        elif processing_type == "cut_min_max":
            original_shape = data.shape
            B = original_shape[0]

            if data.ndim == 4: # Shape (B, H, W, C_actual)
                N_elements = data.shape[1] * data.shape[2]
                C_eff = data.shape[3]
                flat_data = data.contiguous().view(B, N_elements, C_eff)
            else:
                threestudio.warn(f"cut_min_max for data with unhandled ndim={data.ndim} (shape {original_shape}) is not applied.")
                return data

            processed_flat_data = flat_data.clone()
            for b_idx in range(B):
                for c_idx in range(C_eff):
                    current_slice = flat_data[b_idx, :, c_idx]
                    u_vals = torch.unique(current_slice, sorted=True)
                    num_unique = u_vals.numel()

                    if num_unique >= 2:
                        true_min_val = u_vals[0]
                        true_max_val = u_vals[-1] # last unique value
                        
                        # Determine replacements
                        replacement_for_min = u_vals[1] # Next distinct value after true_min_val
                        replacement_for_max = u_vals[-2] # Previous distinct value before true_max_val

                        # Create masks based on the original values in current_slice
                        is_true_min = (current_slice == true_min_val)
                        is_true_max = (current_slice == true_max_val)
                        
                        # Apply replacements to the corresponding slice in processed_flat_data
                        if num_unique == 2:
                            # Min becomes max, max becomes min
                            processed_flat_data[b_idx, :, c_idx][is_true_min] = true_max_val
                            processed_flat_data[b_idx, :, c_idx][is_true_max] = true_min_val
                        else: # num_unique > 2
                            processed_flat_data[b_idx, :, c_idx][is_true_min] = replacement_for_min
                            processed_flat_data[b_idx, :, c_idx][is_true_max] = replacement_for_max
            
            return processed_flat_data.view(original_shape)
        else:
            raise ValueError(f"Unknown guidance_source_processing type: '{processing_type}'")

    def _prepare_guidance_for_low_res(
        self,
        out_high_res: Dict[str, Any],
        H_high: int,
        W_high: int,
        B: int,
        device: torch.device,
        processing_mode: str, 
        flat_indices_global: Optional[torch.Tensor] = None,
        N_samples: Optional[int] = None,
        H_low: Optional[int] = None,
        W_low: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        if self.cfg.guidance_source == "none":
            return None

        if self.cfg.guidance_source not in out_high_res:
            raise ValueError(
                f"Guidance source '{self.cfg.guidance_source}' not found in high_res_renderer output keys: "
                f"{list(out_high_res.keys())}"
            )
        
        raw_guidance_content = out_high_res[self.cfg.guidance_source].detach()

        # Apply post-processing to the raw_guidance_content first, if specified
        if self.cfg.guidance_source_processing != "none":
            raw_guidance_content = self._apply_guidance_source_processing(
                raw_guidance_content, self.cfg.guidance_source_processing
            )
        
        original_guidance_shape = raw_guidance_content.shape # Get shape after potential processing
        processed_guidance_data: Optional[torch.Tensor] = None

        if processing_mode == 'sample':
            # Ensure shape is suitable for sampling after potential processing
            # The primary check is that it still has the expected spatial dimensions H_high, W_high
            shape_ok_for_sampling = (
                raw_guidance_content.ndim >= 2 and 
                raw_guidance_content.shape[0] == B and
                (
                    (raw_guidance_content.ndim >= 3 and raw_guidance_content.shape[1:3] == (H_high, W_high)) or
                    (raw_guidance_content.ndim == 2 and raw_guidance_content.shape[1] == H_high * W_high) # Allow B, H*W format
                )
            )
            if not shape_ok_for_sampling:
                raise ValueError(
                    f"Guidance source '{self.cfg.guidance_source}' (after processing '{self.cfg.guidance_source_processing}') "
                    f"has unsuitable shape for sampling: {original_guidance_shape}. Expected B,H,W[,C] or B,H*W."
                )
            if flat_indices_global is None or N_samples is None:
                 raise ValueError("flat_indices_global and N_samples must be provided for 'sample' mode.")

            # Determine channels after processing. If processing reduced to B,H,W, channels is empty.
            guidance_channels = original_guidance_shape[3:] if raw_guidance_content.ndim > 3 and original_guidance_shape[1:3] == (H_high, W_high) else ()
            
            # Flatten for sampling. Handles B,H,W or B,H,W,C (original_guidance_shape might be B,X if processing flattened it)
            # We rely on H_high, W_high for the total number of pixels for safety if shape changed.
            num_pixels_high_res = H_high * W_high
            guidance_content_flat = raw_guidance_content.reshape(B * num_pixels_high_res, *guidance_channels)            
            guidance_data_sampled_flat = guidance_content_flat[flat_indices_global]
            
            if guidance_channels: 
                processed_guidance_data = guidance_data_sampled_flat.view(B, N_samples, *guidance_channels)
            else: 
                processed_guidance_data = guidance_data_sampled_flat.view(B, N_samples)
        
        elif processing_mode == 'downsample':
            # Ensure shape is suitable for downsampling after potential processing
            shape_ok_for_downsampling = (
                raw_guidance_content.ndim >= 3 and 
                raw_guidance_content.shape[0] == B and
                (
                    (raw_guidance_content.ndim == 3 and raw_guidance_content.shape[1:3] == (H_high, W_high)) or
                    (raw_guidance_content.ndim == 4 and original_guidance_shape[1:3] == (H_high, W_high)) or # B,H,W,C
                    (raw_guidance_content.ndim == 4 and original_guidance_shape[2:4] == (H_high, W_high)) # B,C,H,W
                )
            )
            if not shape_ok_for_downsampling:
                raise ValueError(
                    f"Guidance source '{self.cfg.guidance_source}' (after processing '{self.cfg.guidance_source_processing}') "
                    f"has unsuitable shape for downsampling: {original_guidance_shape}. Expected B,H,W[,C] or B,C,H,W format compatible with H_high, W_high."
                )
            if H_low is None or W_low is None:
                raise ValueError("H_low and W_low must be provided for 'downsample' mode.")

            current_guidance_content_permute_in = raw_guidance_content
            if raw_guidance_content.ndim == 3: # B, H, W
                current_guidance_content_permute_in = raw_guidance_content.unsqueeze(1) # -> B, 1, H, W
                permute_dims_to = list(range(4)) 
            elif raw_guidance_content.ndim == 4: # B, H, W, C or B, C, H, W
                if current_guidance_content_permute_in.shape[1] == H_high and current_guidance_content_permute_in.shape[2] == W_high : # B,H,W,C
                    permute_dims_to = [0, 3, 1, 2] # B,C,H,W
                elif current_guidance_content_permute_in.shape[2] == H_high and current_guidance_content_permute_in.shape[3] == W_high: # B,C,H,W
                     permute_dims_to = list(range(4)) 
                else:
                    raise ValueError(f"Unsupported 4D shape for downsampling after processing: {current_guidance_content_permute_in.shape}, expecting compatibility with H_high={H_high}, W_high={W_high}")
            else: 
                raise ValueError(f"Guidance source for downsampling has {raw_guidance_content.ndim} dims after processing. Expected 3 or 4.")

            guidance_permuted = current_guidance_content_permute_in.permute(*permute_dims_to).contiguous()
            
            interpolate_kwargs = {"mode": "bilinear"} 
            if interpolate_kwargs["mode"] != "area":
                 interpolate_kwargs["align_corners"] = False

            guidance_downsampled_permuted = F.interpolate(
                guidance_permuted, (H_low, W_low), **interpolate_kwargs
            )
            permute_dims_to_tensor = torch.tensor(permute_dims_to, device=device)
            permute_dims_back = torch.argsort(permute_dims_to_tensor).tolist()
            processed_guidance_data = guidance_downsampled_permuted.permute(*permute_dims_back)
        
        else:
            raise ValueError(f"Unknown processing_mode: {processing_mode}")

        
        return processed_guidance_data

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
            
            # Initialize low_res_kwargs first
            low_res_kwargs = kwargs.copy() 

            # 3b. Optional: Extract and downsample guidance data (from cfg.guidance_source)
            guidance_data_for_low_res = self._prepare_guidance_for_low_res(
                out_high_res, H_high, W_high, B, device,
                processing_mode='downsample',
                H_low=H_low, W_low=W_low
            )
            if guidance_data_for_low_res is not None:
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
