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
from typing import Union, Tuple, Dict, Any

# import inspect # REMOVED: No longer used

# Register the new renderer
@threestudio.register("defered-dual-renderer")
class DeferedDualRenderers(Renderer):
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

        # --- Config for 'sample' mode --- (Now the primary mode)
        num_low_res_samples_per_view: int = 1024 # Number of rays (N) to sample per view

        # --- Config for sampling source in 2nd phase ---
        sample_source: str = "grad_norm" # Which source to use for sampling: "grad_norm", "opacity", "depth", etc.
        sample_source_processing: str = 'softmax-0.1' # How to process sample_source to get probabilities

        # --- Config for passing guidance data to low-res renderer (both modes) ---
        guidance_source: str = "none" # Key from high_res_output for aux guidance to low_res_renderer

    cfg: Config

    # +++ Added for phase-based rendering and caching +++
    _cached_high_res_output: Optional[Dict[str, Any]]
    _comp_rgb_for_grad_extraction: Optional[Tensor] = None # To store the comp_rgb from phase 1
    # +++ End Added +++

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
        threestudio.debug(f"Configuring low-resolution (aux guidance always uses sample mode) renderer: {self.cfg.low_res_renderer_type}")
        self.low_res_renderer: Renderer = threestudio.find(self.cfg.low_res_renderer_type)(
            self.cfg.low_res_renderer,
            geometry=geometry, # Assumes shared components, modify if needed
            material=material,
            background=background,
        )
        # +++ Added for phase-based rendering and caching +++
        self._cached_high_res_output = None
        self._comp_rgb_for_grad_extraction = None
        # +++ End Added +++

    # +++ Helper method for processing raw guidance to probabilities +++
    def _process_raw_guidance_to_probabilities(
        self,
        guidance_flat: torch.Tensor, # Shape: B, num_pixels
        processing_mode: str,
        B: int,
        num_pixels: int, 
        device: torch.device,
        # Optional: specific temperature for softmax, if not taken from processing_mode string
        softmax_temperature_override: Optional[float] = None 
    ) -> torch.Tensor:
        """Processes a flat guidance tensor into sampling probabilities."""
        
        temperature = softmax_temperature_override if softmax_temperature_override is not None else 0.1 # Default temperature

        if processing_mode.startswith('softmax'):
            if softmax_temperature_override is None: # Parse from string if not overridden
                parts = processing_mode.split('-')
                if len(parts) == 2:
                    try:
                        temperature = float(parts[1])
                        if temperature <= 0:
                            raise ValueError("Temperature for softmax must be positive.")
                    except ValueError:
                        raise ValueError(
                            f"Invalid temperature format in sample_source_processing: '{processing_mode}'. "
                            f"Expected 'softmax-value' where value is a positive float (e.g., 'softmax-0.1')."
                        )
                elif len(parts) > 2:
                     raise ValueError(
                        f"Invalid format for sample_source_processing: '{processing_mode}'. Too many hyphens."
                    )
            # If len(parts) == 1 (i.e., just 'softmax') and no override, default temperature 0.1 is used.
            
            sampling_weights_flat = F.softmax(guidance_flat / temperature, dim=-1)

        elif processing_mode == 'normalize':
            guidance_flat_processed = F.relu(guidance_flat) # Ensure non-negative before sum
            norm = guidance_flat_processed.sum(dim=-1, keepdim=True)
            is_all_zero = norm < 1e-9
            uniform_prob = 1.0 / num_pixels
            sampling_weights_flat = torch.where(
                is_all_zero, 
                torch.full_like(guidance_flat_processed, uniform_prob), 
                guidance_flat_processed / (norm + 1e-6) # Add epsilon for stability
            )
        elif processing_mode == 'raw':
            sampling_weights_flat = F.relu(guidance_flat) # Values will be used as weights, ensure non-negative
        else:
            raise ValueError(f"Unknown sample_source_processing mode: {processing_mode}")

        # Robustness check
        sum_is_problematic = (sampling_weights_flat.sum(dim=-1) < 1e-6).any() if processing_mode != 'raw' else False
        if torch.isnan(sampling_weights_flat).any() or sum_is_problematic:
            threestudio.warn(
                f"Sampling_weights_flat became NaN or sum is too small after processing "
                f"'{processing_mode}'. Falling back to uniform."
            )
            sampling_weights_flat = torch.ones(B, num_pixels, device=device) / num_pixels
        
        return sampling_weights_flat
    # +++ End Helper method +++

    # +++ Helper method for sampling probabilities (similar to dual_renderers.py) +++
    def _get_sampling_probabilities(
        self,
        out_high_res: Dict[str, Any],
        H_high: int,
        W_high: int,
        B: int,
        device: torch.device,
        guidance_for_sampling_flat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate guidance probabilities for sampling based on cfg.sample_source."""
        num_pixels_per_view = H_high * W_high

        if self.cfg.sample_source == "grad_norm":
            # Use gradient norm from comp_rgb.grad (our primary deferred rendering feature)
            if guidance_for_sampling_flat is not None:
                guidance_flat = guidance_for_sampling_flat
            else:
                threestudio.warn("grad_norm sample source requested but guidance_for_sampling_flat is None. Using uniform.")
                return torch.ones(B, num_pixels_per_view, device=device) / num_pixels_per_view
        elif self.cfg.sample_source == "none":
            return torch.ones(B, num_pixels_per_view, device=device) / num_pixels_per_view
        else:
            # Use other sources from high_res_renderer output (like dual_renderers.py)
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
                guidance_flat = raw_sample_guidance.view(B, -1)
            else:
                raise ValueError(
                    f"Sample source '{self.cfg.sample_source}' must be 'grad_norm', 'none', 'depth_gradient', "
                    f"or a key in high_res_renderer output. Got: '{self.cfg.sample_source}', "
                    f"Available keys in high_res_output: {list(out_high_res.keys())}"
                )
        
        # Process the guidance using sample_source_processing
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
            guidance_flat_processed = F.relu(guidance_flat)
            norm = guidance_flat_processed.sum(dim=-1, keepdim=True)
            is_all_zero = norm < 1e-9
            uniform_prob = 1.0 / guidance_flat.shape[-1]
            guidance_probs = torch.where(is_all_zero, torch.full_like(guidance_flat_processed, uniform_prob), guidance_flat_processed / (norm + 1e-6))
        elif current_processing_mode == 'raw':
            guidance_probs = F.relu(guidance_flat)
        else:
            raise ValueError(f"Unknown sample_source_processing mode: {current_processing_mode}")
        
        if torch.isnan(guidance_probs).any() or (guidance_probs.sum(dim=-1) < 1e-6).any():
            threestudio.warn(
                f"Guidance probabilities became NaN or all zero after processing "
                f"'{current_processing_mode}' mode with sample_source '{self.cfg.sample_source}'. "
                f"Falling back to uniform sampling."
            )
            guidance_probs = torch.ones(B, num_pixels_per_view, device=device) / num_pixels_per_view
        
        return guidance_probs
    # +++ End Helper method +++

    # --- Helper method for preparing guidance data ---
    def _prepare_guidance_for_low_res(
        self,
        out_high_res: Dict[str, Any],
        H_high: int,
        W_high: int,
        B: int,
        device: torch.device,
        flat_indices_global: Optional[torch.Tensor] = None,
        N_samples: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        if self.cfg.guidance_source == "none":
            return None

        if self.cfg.guidance_source not in out_high_res:
            threestudio.warn(f"Guidance '{self.cfg.guidance_source}' not found. No guidance passed.")
            return None
        
        raw_guidance_content = out_high_res[self.cfg.guidance_source].detach()
        original_guidance_shape = raw_guidance_content.shape 
        
        # Only 'sample' logic remains
        shape_ok_for_sampling = (raw_guidance_content.ndim >= 2 and 
                raw_guidance_content.shape[0] == B and
            ((raw_guidance_content.ndim >= 3 and raw_guidance_content.shape[1:3] == (H_high, W_high)) or
             (raw_guidance_content.ndim == 2 and raw_guidance_content.shape[1] == H_high * W_high)))
        if not shape_ok_for_sampling:
            raise ValueError(f"Guidance source '{self.cfg.guidance_source}' has unsuitable shape for sampling: {original_guidance_shape}.")
        if flat_indices_global is None or N_samples is None:
            raise ValueError("flat_indices_global and N_samples must be provided for preparing sampled guidance data.")

        guidance_channels = original_guidance_shape[3:] if raw_guidance_content.ndim > 3 and original_guidance_shape[1:3] == (H_high, W_high) else ()
        guidance_content_flat = raw_guidance_content.reshape(B * H_high * W_high, *guidance_channels)            
        guidance_data_sampled_flat = guidance_content_flat[flat_indices_global]
        
        processed_guidance_data = guidance_data_sampled_flat.view(B, N_samples, *guidance_channels) if guidance_channels else guidance_data_sampled_flat.view(B, N_samples)
        return processed_guidance_data

    # --- Main Forward Method --- 
    def forward(
        self,
        rays_o: Tensor, 
        rays_d: Tensor, 
        rays_o_rasterize: Tensor, 
        rays_d_rasterize: Tensor, 
        light_positions: Tensor, 
        phase: str,
        **kwargs
    ) -> Dict[str, Any]:
        
        # Determine if we are in training or evaluation mode
        if self.training:
            return self._forward_train(
                rays_o_low=rays_o,
                rays_d_low=rays_d,
                rays_o_high=rays_o_rasterize,
                rays_d_high=rays_d_rasterize,
                light_positions=light_positions,
                phase=phase,
                **kwargs
            )
        else:
            return self._forward_eval(
                rays_o_low=rays_o,
                rays_d_low=rays_d,
                rays_o_high=rays_o_rasterize,
                rays_d_high=rays_d_rasterize,
                light_positions=light_positions,
                phase=phase,
                **kwargs
            )

    # +++ Helper method for preparing sampling in 2nd phase +++
    def _prepare_sampling_for_2nd_phase(
        self,
        guidance_for_sampling_flat: Optional[torch.Tensor], 
        B_cameras: int, 
        num_pixels_per_view_high: int,
        actual_N_samples: int,
        device: torch.device,
        out_high_res: Dict[str, Any],
        H_high: int,
        W_high: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        # Use the new _get_sampling_probabilities method
        sampling_weights_flat = self._get_sampling_probabilities(
            out_high_res=out_high_res,
            H_high=H_high,
            W_high=W_high,
            B=B_cameras,
            device=device,
            guidance_for_sampling_flat=guidance_for_sampling_flat
        )
        
        try:
            sampled_indices_flat_batch = torch.multinomial(
                sampling_weights_flat, 
                actual_N_samples,
                replacement=False  # 改为无放回采样，确保采样的像素都是独特的
            ) 
        except RuntimeError as e:
            threestudio.warn(f"Multinomial failed: {e}. Falling back to uniform random.")
            sampled_indices_flat_batch = torch.randint(
                0, num_pixels_per_view_high, (B_cameras, actual_N_samples), device=device
            )

        if 'rays_o_high_res_flat' not in self._cached_high_res_output or \
           'rays_d_high_res_flat' not in self._cached_high_res_output:
            raise RuntimeError("Flattened high-resolution rays not found in cache.")

        rays_o_high_flat = self._cached_high_res_output['rays_o_high_res_flat'] 
        rays_d_high_flat = self._cached_high_res_output['rays_d_high_res_flat'] 
        
        indices_for_gather = sampled_indices_flat_batch.unsqueeze(-1).expand(-1, -1, 3)
        
        sampled_rays_o_flat = torch.gather(rays_o_high_flat, 1, indices_for_gather) 
        sampled_rays_d_flat = torch.gather(rays_d_high_flat, 1, indices_for_gather) 

        offset = torch.arange(B_cameras, device=device) * num_pixels_per_view_high
        sampled_indices_flat_global = (sampled_indices_flat_batch + offset.unsqueeze(1)).view(-1)
        
        return sampled_rays_o_flat, sampled_rays_d_flat, sampled_indices_flat_global
    # +++ End Helper method +++

    # --- Training Forward --- (Primary focus of refactoring)
    def _forward_train(
        self,
        rays_o_low: Tensor, 
        rays_d_low: Tensor, 
        rays_o_high: Tensor, 
        rays_d_high: Tensor, 
        light_positions: Tensor, 
        phase: str,
        **kwargs
    ) -> Dict[str, Any]: 
        B_cameras, H_high, W_high, _ = rays_o_high.shape 
        device = rays_o_high.device
        outputs = {}

        if phase == "1st":
            self._comp_rgb_for_grad_extraction = None 

            out_high_res = self.high_res_renderer(
                rays_o_rasterize=rays_o_high,
                rays_d_rasterize=rays_d_high,
                light_positions=light_positions,
                **kwargs  # Pass all other kwargs from the system
            )
            
            if 'comp_rgb' not in out_high_res:
                raise RuntimeError("Phase 1: 'comp_rgb' not found in high_res_renderer output.")
            
            # Store comp_rgb and ensure gradients are retained for non-leaf tensors
            self._comp_rgb_for_grad_extraction = out_high_res['comp_rgb']
            if not self._comp_rgb_for_grad_extraction.is_leaf:
                self._comp_rgb_for_grad_extraction.retain_grad()
            
            self._cached_high_res_output = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in out_high_res.items()}
            self._cached_high_res_output['rays_o_high_res_flat'] = rays_o_high.reshape(B_cameras, H_high * W_high, 3).detach()
            self._cached_high_res_output['rays_d_high_res_flat'] = rays_d_high.reshape(B_cameras, H_high * W_high, 3).detach()
            
            outputs.update(out_high_res) 
            outputs['phase1_complete'] = True 
            return outputs

        elif phase == "2nd":
            if self._cached_high_res_output is None or self._comp_rgb_for_grad_extraction is None:
                raise RuntimeError("Phase '2nd' called but cached info from phase '1st' is missing.")

            guidance_for_sampling_flat: Optional[torch.Tensor] = None
            if self._comp_rgb_for_grad_extraction.grad is not None:
                comp_rgb_grad = self._comp_rgb_for_grad_extraction.grad 
                comp_rgb_grad_norm = torch.norm(comp_rgb_grad, p=2, dim=-1) 
                num_pixels_high = H_high * W_high
                guidance_for_sampling_flat = comp_rgb_grad_norm.view(B_cameras, num_pixels_high).detach().clone() 
            else:
                threestudio.warn("Phase 2: Gradient for cached 'comp_rgb' is None. Sampling will fall back.")
            
            actual_N_samples = self.cfg.num_low_res_samples_per_view
            num_pixels_per_view_high = H_high * W_high

            sampling_results = self._prepare_sampling_for_2nd_phase(
                guidance_for_sampling_flat=guidance_for_sampling_flat, B_cameras=B_cameras,
                num_pixels_per_view_high=num_pixels_per_view_high, actual_N_samples=actual_N_samples, device=device,
                out_high_res=self._cached_high_res_output, H_high=H_high, W_high=W_high)

            if sampling_results is None: 
                outputs.update(self._cached_high_res_output) 
                outputs["comp_rgb"] = self._cached_high_res_output["comp_rgb"]
                if "comp_rgb_ref" in self._cached_high_res_output: outputs["comp_rgb_ref"] = self._cached_high_res_output["comp_rgb_ref"]
                return outputs

            sampled_rays_o_flat, sampled_rays_d_flat, sampled_indices_flat_global = sampling_results
            
            # Prepare kwargs for low_res_renderer (volume renderer)
            low_res_render_kwargs = kwargs.copy() # Start with a copy

            # Pass sampled rays as 'rays_o' and 'rays_d' for volume renderer
            low_res_render_kwargs["rays_o"] = sampled_rays_o_flat.view(B_cameras, actual_N_samples, 1, 3)
            low_res_render_kwargs["rays_d"] = sampled_rays_d_flat.view(B_cameras, actual_N_samples, 1, 3)
            
            # Add light_positions parameter
            low_res_render_kwargs["light_positions"] = light_positions
            
            # Update height and width for the low_res_renderer
            low_res_render_kwargs["height"] = actual_N_samples
            low_res_render_kwargs["width"] = 1
            
            # Remove original rasterize rays if they were in kwargs, to avoid conflict
            low_res_render_kwargs.pop('rays_o_rasterize', None)
            low_res_render_kwargs.pop('rays_d_rasterize', None)
            
            # space_cache argument - ensure it uses the correct one if specified
            low_res_render_kwargs["space_cache"] = kwargs.get('space_cache_low_res', kwargs.get('space_cache'))
            
            # CRITICAL: Ensure text_embed is passed to low_res_renderer
            if 'text_embed' in kwargs:
                low_res_render_kwargs['text_embed'] = kwargs['text_embed']

            # Only use 'sample' logic for preparing auxiliary guidance from guidance_source
            prepared_guidance = self._prepare_guidance_for_low_res(
                self._cached_high_res_output, H_high, W_high, B_cameras, device,
                flat_indices_global=sampled_indices_flat_global, 
                N_samples=actual_N_samples)
            if prepared_guidance is not None and self.cfg.guidance_source != "none":
                guidance_key = getattr(self.cfg, 'guidance_source_as_key', self.cfg.guidance_source)
                low_res_render_kwargs[guidance_key] = prepared_guidance 
                # Also pass as gs_depth for depth estimator
                low_res_render_kwargs['gs_depth'] = prepared_guidance

            out_low_res = self.low_res_renderer(**low_res_render_kwargs)
            
            # Simplified scattering: last sample wins for a given pixel if duplicates exist.
            combined_output = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self._cached_high_res_output.items()}

            # 参考dual_renderers.py的融合逻辑，对所有低分辨率输出进行融合
            for key, low_res_value in out_low_res.items():
                if key in combined_output and isinstance(low_res_value, torch.Tensor) and isinstance(combined_output[key], torch.Tensor):
                    high_res_value = combined_output[key]
                    original_shape = high_res_value.shape
                    
                    # 检查低分辨率输出是否符合预期格式 (B, N, ...)
                    if low_res_value.shape[0] == B_cameras and low_res_value.shape[1] == actual_N_samples and high_res_value.numel() >= B_cameras * actual_N_samples:
                        value_shape_per_ray = low_res_value.shape[2:] 
                        num_pixels_per_view_high = H_high * W_high
                        
                        # 展平高分辨率值
                        high_res_flat = high_res_value.reshape(B_cameras * num_pixels_per_view_high, *value_shape_per_ray)
                        # 展平低分辨率值
                        low_res_value_flat = low_res_value.reshape(B_cameras * actual_N_samples, *value_shape_per_ray)
                        
                        # 执行融合：将低分辨率结果散布到对应的高分辨率位置
                        high_res_flat[sampled_indices_flat_global] = low_res_value_flat
                        combined_output[key] = high_res_flat.reshape(original_shape)
                        
            # 添加低分辨率渲染器的额外输出用于调试
            for k, v in out_low_res.items():
                if k not in combined_output:
                    combined_output[f"low_res_{k}"] = v
            
            outputs.update(combined_output)
            outputs['phase2_complete'] = True
            return outputs
        else:
            raise ValueError(f"Unknown phase: {phase}")

    # --- Evaluation Forward ---
    def _forward_eval(
        self,
        rays_o_low: Tensor, 
        rays_d_low: Tensor, 
        rays_o_high: Tensor, 
        rays_d_high: Tensor, 
        light_positions: Tensor, 
        phase: str,
        **kwargs
    ) -> Dict[str, Any]:
        B_cameras, H_high, W_high, _ = rays_o_high.shape
        device = rays_o_high.device
        outputs = {}

        self._comp_rgb_for_grad_extraction = None 
        out_high_res = self.high_res_renderer(
            rays_o_rasterize=rays_o_high, rays_d_rasterize=rays_d_high, 
            light_positions=light_positions, **kwargs)
        
        self._cached_high_res_output = None 
        outputs.update(out_high_res)
        outputs['phase1_complete'] = True
        return outputs

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

