import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import math # For Kaiming init

from diffusers.models.attention_processor import Attention, AttnProcessor, LoRAAttnProcessor
from threestudio.utils.typing import *
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from diffusers.loaders import AttnProcsLayers
from threestudio.utils.base import BaseModule
from dataclasses import dataclass

from diffusers.models.lora import LoRAConv2dLayer, LoRACompatibleConv
from threestudio.utils.misc import cleanup


# --- LoRALinearLayerwBias (used for vanilla LoRA) ---
class LoRALinearLayerwBias(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None, 
        dtype: Optional[torch.dtype] = None,
        with_bias: bool = False
    ):
        super().__init__()
        self.rank = rank 
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.with_bias = with_bias
        if with_bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        
        self.network_alpha = network_alpha
        self.in_features = in_features 
        self.out_features = out_features 

        if self.rank > 0:
            nn.init.normal_(self.down.weight, std=1 / self.rank)
        else: 
            nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        if self.with_bias:
            nn.init.zeros_(self.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return torch.zeros_like(self.up(self.down(hidden_states)))
        
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None and self.rank > 0:
            up_hidden_states = up_hidden_states * (self.network_alpha / self.rank)
        
        output = up_hidden_states.to(orig_dtype)
        if self.with_bias:
            output = output + self.bias
        return output

# --- FewLoRAConv2dLayer (formerly QuadLoRAConv2dLayer, optimized for few_v1) ---
class FewLoRAConv2dLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, Tuple[int, int], str] = 0,
        network_alpha: Optional[float] = None,
        with_bias: bool = False,
        locon_type: str = "few_v1", # Changed from quad_v1
        num_planes: int = 4,
    ):
        super().__init__()
        assert locon_type in ["few_v1", "vanilla_v1", "vanilla_v2", "none"], "The LoCON type is not supported." # Changed quad_v1 to few_v1
        self.locon_type = locon_type
        self.rank = rank
        self.network_alpha = network_alpha
        self.num_planes = num_planes
        self.out_features = out_features
        self.in_features = in_features # Store for init
        self.with_bias = with_bias # Store for init

        if isinstance(kernel_size, int): self.kernel_size_tuple = (kernel_size, kernel_size)
        else: self.kernel_size_tuple = kernel_size
        if isinstance(stride, int): self.stride_tuple = (stride, stride)
        else: self.stride_tuple = stride
        if isinstance(padding, int): self.padding_tuple = (padding, padding)
        elif padding == 'same': # Handle 'same' padding if necessary, F.conv2d needs int/tuple
             # For 'same' padding with stride 1, padding = (kernel_size - 1) // 2
             self.padding_tuple = tuple((k - 1) // 2 for k in self.kernel_size_tuple)
        else: self.padding_tuple = padding


        if self.locon_type != "none" and self.rank > 0:
            if locon_type == "few_v1":
                # Stacked weights for grouped convolution
                self.few_down_weight = nn.Parameter(torch.empty(self.num_planes * self.rank, self.in_features, *self.kernel_size_tuple))
                self.few_up_weight = nn.Parameter(torch.empty(self.num_planes * self.out_features, self.rank, 1, 1))
                if self.with_bias: # Bias for the up-projection
                    self.few_up_bias = nn.Parameter(torch.zeros(self.num_planes * self.out_features))
                else:
                    self.register_parameter('few_up_bias', None)

            elif locon_type == "vanilla_v1":
                self.down = nn.Conv2d(self.in_features, self.rank, kernel_size=self.kernel_size_tuple, stride=self.stride_tuple, padding=self.padding_tuple, bias=False)
                self.up = nn.Conv2d(self.rank, self.out_features, kernel_size=(1, 1), stride=(1, 1), bias=self.with_bias)
            elif locon_type == "vanilla_v2":
                self.down = nn.Conv2d(self.in_features, self.rank, kernel_size=(1,1), stride=(1,1), padding=self.padding_tuple, bias=False)
                self.up = nn.Conv2d(self.rank, self.out_features, kernel_size=self.kernel_size_tuple, stride=self.stride_tuple, bias=self.with_bias)
            self._init_weights()

    def _init_weights(self):
        if self.locon_type == "few_v1" and self.rank > 0:
            for i in range(self.num_planes):
                # Initialize slices of the stacked weights
                # Down weights: (rank, in_features, K, K)
                # Kaiming uniform for Conv2d down weight like diffusers LoRAConv2DLayer
                # std for normal init was 1/rank. For Kaiming, it's derived from fan_in.
                # nn.init.normal_(self.few_down_weight.data[i * self.rank : (i + 1) * self.rank], std=1 / self.rank)
                # Let's use Kaiming Uniform as it's common for conv layers
                kaiming_uniform_params = nn.Linear(self.in_features * self.kernel_size_tuple[0] * self.kernel_size_tuple[1], self.rank)
                nn.init.kaiming_uniform_(self.few_down_weight.data[i * self.rank : (i + 1) * self.rank], a=math.sqrt(5))

                # Up weights: (out_features, rank, 1, 1) -> init to zeros
                nn.init.zeros_(self.few_up_weight.data[i * self.out_features : (i + 1) * self.out_features])
            if self.with_bias and self.few_up_bias is not None: # Bias already init to zeros
                pass 

        elif "vanilla" in self.locon_type and self.rank > 0:
            # Kaiming uniform for Conv2d down weight like diffusers LoRAConv2DLayer
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.weight)
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.locon_type == "none" or self.rank == 0:
            batch_eff, _, h_in, w_in = hidden_states.shape
            # Calculate output height and width based on conv parameters if needed
            # For now, assume H_out, W_out are same as H_in, W_in (common for LoRA-like additions)
            # More robust: calculate H_out, W_out using self.stride_tuple, self.padding_tuple, self.kernel_size_tuple
            # This is tricky if self.kernel_size_tuple is not defined (e.g. rank=0 path taken early)
            # For simplicity, zero tensor with input H,W for now.
            return torch.zeros(batch_eff, self.out_features, h_in, w_in, device=hidden_states.device, dtype=hidden_states.dtype)

        orig_dtype = hidden_states.dtype

        if self.locon_type == "few_v1":
            if self.num_planes <= 0: raise ValueError("Number of planes must be positive.")
            B_eff, C_in, H_in, W_in = hidden_states.shape
            B_orig = B_eff // self.num_planes

            # Reshape for grouped convolution: (B_orig, num_planes * C_in, H_in, W_in)
            # This doesn't seem right for how weights are (P*R, C_in, K,K)
            # The input for F.conv2d with groups=P should be (B_orig, C_in_total, H, W) where C_in_total = P * C_in_per_group
            # And weight (R_total, C_in_per_group, K,K) where R_total = P * R_per_group
            # Our weights: self.few_down_weight (P*R, C_in, K, K)
            # So, input to F.conv2d should be (B_orig, P*C_in, H, W)

            # Original hidden_states: (B_orig * num_planes, C_in, H_in, W_in)
            # We need to interleave C_in from different planes for grouped conv, if C_in is C_in_per_group
            # Or, treat as B_orig batches of (num_planes * C_in, H, W) data? 
            # Let's try to process B_orig batches, each containing all plane data interleaved in channels. 
            # hidden_states: (B_orig * num_planes, C_in, H, W)
            # -> permute to (B_orig, num_planes, C_in, H, W)
            # -> reshape to (B_orig, num_planes * C_in, H, W)
            x_reshaped = hidden_states.view(B_orig, self.num_planes, C_in, H_in, W_in).permute(0, 2, 1, 3, 4).contiguous()
            x_reshaped = x_reshaped.view(B_orig, C_in * self.num_planes, H_in, W_in)
            
            # Down conv
            # Weight: (num_planes * rank, in_features, K, K)
            # Input:  (B_orig, num_planes * in_features, H_in, W_in)
            # groups = num_planes. Each group takes in_features channels.
            down_res = F.conv2d(x_reshaped.to(self.few_down_weight.dtype), 
                                self.few_down_weight, 
                                bias=None, 
                                stride=self.stride_tuple, 
                                padding=self.padding_tuple, 
                                groups=self.num_planes)
            # Output: (B_orig, num_planes * rank, H_down, W_down)

            # Up conv
            # Weight: (num_planes * out_features, rank, 1, 1)
            # Input:  (B_orig, num_planes * rank, H_down, W_down)
            # groups = num_planes. Each group takes rank channels.
            up_res = F.conv2d(down_res, 
                              self.few_up_weight, 
                              bias=self.few_up_bias, 
                              stride=(1,1), # Up conv stride is 1
                              padding=(0,0),  # Up conv padding is 0
                              groups=self.num_planes)
            # Output: (B_orig, num_planes * out_features, H_up, W_up)

            # Reshape back to (B_orig * num_planes, out_features, H_up, W_up)
            lora_hidden_states = up_res.view(B_orig * self.num_planes, self.out_features, up_res.size(-2), up_res.size(-1))
            
        elif "vanilla" in self.locon_type:
            # Determine dtype from the first available conv layer's weight for vanilla
            main_dtype_layer = self.down
            dtype = main_dtype_layer.weight.dtype
            lora_hidden_states = self.up(self.down(hidden_states.to(dtype)))

        if self.network_alpha is not None and self.rank > 0:
            lora_hidden_states = lora_hidden_states * (self.network_alpha / self.rank)
        
        return lora_hidden_states.to(orig_dtype)

# --- FewSelfAttentionLoRAAttnProcessor (formerly QuadSelfAttentionLoRAAttnProcessor) ---
class FewSelfAttentionLoRAAttnProcessor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        with_bias: bool = False,
        lora_type: str = "few_v1", # Changed from quad_v1
        num_planes: int = 4,
    ):
        super().__init__()
        assert lora_type in ["few_v1", "vanilla", "none"], "The LoRA type is not supported." # Changed quad_v1 to few_v1
        self.lora_type = lora_type
        self.rank = rank
        self.hidden_size = hidden_size
        self.network_alpha = network_alpha
        self.with_bias = with_bias
        self.num_planes = num_planes

        if lora_type == "few_v1" and rank > 0:
            for proj_name in ["q", "k", "v", "out"]:
                setattr(self, f"{proj_name}_lora_down_w_stacked", nn.Parameter(torch.empty(self.num_planes, hidden_size, rank)))
                setattr(self, f"{proj_name}_lora_up_w_stacked", nn.Parameter(torch.empty(self.num_planes, rank, hidden_size)))
                if with_bias:
                    setattr(self, f"{proj_name}_lora_bias_stacked", nn.Parameter(torch.empty(self.num_planes, hidden_size)))
            self._init_stacked_weights()

        elif lora_type == "vanilla": 
            self.to_q_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
            self.to_k_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
            self.to_v_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
            self.to_out_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)

    def _init_stacked_weights(self):
        for proj_name in ["q", "k", "v", "out"]:
            nn.init.normal_(getattr(self, f"{proj_name}_lora_down_w_stacked"), std=1 / self.rank)
            nn.init.zeros_(getattr(self, f"{proj_name}_lora_up_w_stacked"))
            if self.with_bias:
                nn.init.zeros_(getattr(self, f"{proj_name}_lora_bias_stacked"))

    def _apply_lora_stacked(self, x_plane_batched, proj_name):
        down_w = getattr(self, f"{proj_name}_lora_down_w_stacked")
        up_w = getattr(self, f"{proj_name}_lora_up_w_stacked")
        
        lora_down = torch.einsum('bpsi,pir->bpsr', x_plane_batched, down_w)
        lora_up = torch.einsum('bpsr,pro->bpso', lora_down, up_w)

        if self.network_alpha is not None and self.rank > 0:
            lora_up = lora_up * (self.network_alpha / self.rank)
        
        if self.with_bias:
            bias = getattr(self, f"{proj_name}_lora_bias_stacked")
            lora_up = lora_up + bias.unsqueeze(0).unsqueeze(2)
        return lora_up

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if attn.spatial_norm is not None: hidden_states = attn.spatial_norm(hidden_states, temb)
        if input_ndim == 4:
            batch_size_eff, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size_eff, channel, height * width).transpose(1, 2)
        
        batch_size_eff, sequence_length, _ = hidden_states.shape
        
        if self.num_planes <= 0: raise ValueError("Number of planes must be positive.")
        original_batch_size = batch_size_eff // self.num_planes

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size_eff)
        if attn.group_norm is not None: hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query_orig = attn.to_q(hidden_states)
        if self.lora_type == "few_v1":
            if self.rank > 0:
                h_for_lora = hidden_states.view(original_batch_size, self.num_planes, sequence_length, self.hidden_size)
                lora_q = self._apply_lora_stacked(h_for_lora, "q")
                lora_q_contrib = lora_q.reshape(batch_size_eff, sequence_length, self.hidden_size)
                query = query_orig + scale * lora_q_contrib
            else: # few_v1 with rank == 0
                query = query_orig
        elif self.lora_type == "vanilla": 
            lora_q_contrib = self.to_q_lora(hidden_states)
            query = query_orig + scale * lora_q_contrib
        else:
            lora_q_contrib = torch.zeros_like(query_orig)
            query = query_orig + scale * lora_q_contrib
        
        encoder_hidden_states_eff = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        if encoder_hidden_states_eff.shape[0] != batch_size_eff and encoder_hidden_states is None: 
            encoder_hidden_states_eff = hidden_states

        if attn.norm_cross: encoder_hidden_states_eff = attn.norm_encoder_hidden_states(encoder_hidden_states_eff)

        key_orig = attn.to_k(encoder_hidden_states_eff)
        value_orig = attn.to_v(encoder_hidden_states_eff)

        if self.lora_type == "few_v1":
            if self.rank > 0:
                ehs_for_lora = encoder_hidden_states_eff.view(original_batch_size, self.num_planes, -1, self.hidden_size)
                lora_k = self._apply_lora_stacked(ehs_for_lora, "k")
                lora_v = self._apply_lora_stacked(ehs_for_lora, "v")
                lora_k_contrib = lora_k.reshape(batch_size_eff, -1, self.hidden_size)
                lora_v_contrib = lora_v.reshape(batch_size_eff, -1, self.hidden_size)
                key = key_orig + scale * lora_k_contrib
                value = value_orig + scale * lora_v_contrib
            else: # few_v1 with rank == 0
                key = key_orig
                value = value_orig
        elif self.lora_type == "vanilla":
            lora_k_contrib = self.to_k_lora(encoder_hidden_states_eff)
            lora_v_contrib = self.to_v_lora(encoder_hidden_states_eff)
            key = key_orig + scale * lora_k_contrib
            value = value_orig + scale * lora_v_contrib
        else:
            lora_k_contrib = torch.zeros_like(key_orig)
            lora_v_contrib = torch.zeros_like(value_orig)
            key = key_orig + scale * lora_k_contrib
            value = value_orig + scale * lora_v_contrib

        
        query_reshaped = query.view(original_batch_size, sequence_length * self.num_planes, self.hidden_size)
        key_reshaped = key.reshape(original_batch_size, -1, self.hidden_size)
        value_reshaped = value.reshape(original_batch_size, -1, self.hidden_size)

        query_heads = attn.head_to_batch_dim(query_reshaped) 
        key_heads = attn.head_to_batch_dim(key_reshaped) 
        value_heads = attn.head_to_batch_dim(value_reshaped) 

        attention_probs = attn.get_attention_scores(query_heads, key_heads, None) 
        hidden_states_attn = torch.bmm(attention_probs, value_heads)
        
        hidden_states_attn = attn.batch_to_head_dim(hidden_states_attn)
        
        hidden_states_attn = hidden_states_attn.view(batch_size_eff, sequence_length, self.hidden_size)

        hidden_states_out_orig = attn.to_out[0](hidden_states_attn)
        hidden_states_out_final = hidden_states_out_orig # Initialize final output

        if self.lora_type == "few_v1":
            if self.rank > 0:
                h_attn_for_lora = hidden_states_attn.view(original_batch_size, self.num_planes, sequence_length, self.hidden_size)
                lora_out = self._apply_lora_stacked(h_attn_for_lora, "out")
                lora_out_contrib = lora_out.reshape(batch_size_eff, sequence_length, self.hidden_size)
                hidden_states_out_final = hidden_states_out_orig + scale * lora_out_contrib
            # else: few_v1 with rank == 0, hidden_states_out_final remains hidden_states_out_orig
        elif self.lora_type == "vanilla":
            lora_out_contrib = self.to_out_lora(hidden_states_attn)
            hidden_states_out_final = hidden_states_out_orig + scale * lora_out_contrib
        else:
            lora_out_contrib = torch.zeros_like(hidden_states_out_orig)
            hidden_states_out_final = hidden_states_out_orig + scale * lora_out_contrib
        
        hidden_states = attn.to_out[1](hidden_states_out_final)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size_eff, channel, height, width)
        if attn.residual_connection: hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

# --- FewCrossAttentionLoRAAttnProcessor (formerly QuadCrossAttentionLoRAAttnProcessor) ---
class FewCrossAttentionLoRAAttnProcessor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        with_bias: bool = False,
        lora_type: str = "few_v1", # Changed from quad_v1
        num_planes: int = 4,
    ):
        super().__init__()
        assert lora_type in ["few_v1", "vanilla", "none"], "The LoRA type is not supported." # Changed quad_v1 to few_v1
        self.lora_type = lora_type
        self.rank = rank
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.network_alpha = network_alpha
        self.with_bias = with_bias
        self.num_planes = num_planes

        if lora_type == "few_v1" and rank > 0:
            setattr(self, "q_lora_down_w_stacked", nn.Parameter(torch.empty(self.num_planes, hidden_size, rank)))
            setattr(self, "q_lora_up_w_stacked", nn.Parameter(torch.empty(self.num_planes, rank, hidden_size)))
            setattr(self, "k_lora_down_w_stacked", nn.Parameter(torch.empty(self.num_planes, cross_attention_dim, rank)))
            setattr(self, "k_lora_up_w_stacked", nn.Parameter(torch.empty(self.num_planes, rank, hidden_size)))
            setattr(self, "v_lora_down_w_stacked", nn.Parameter(torch.empty(self.num_planes, cross_attention_dim, rank)))
            setattr(self, "v_lora_up_w_stacked", nn.Parameter(torch.empty(self.num_planes, rank, hidden_size)))
            setattr(self, "out_lora_down_w_stacked", nn.Parameter(torch.empty(self.num_planes, hidden_size, rank)))
            setattr(self, "out_lora_up_w_stacked", nn.Parameter(torch.empty(self.num_planes, rank, hidden_size)))
            if with_bias:
                for proj_name in ["q", "k", "v", "out"]:
                    setattr(self, f"{proj_name}_lora_bias_stacked", nn.Parameter(torch.empty(self.num_planes, hidden_size)))
            self._init_stacked_weights()

        elif lora_type == "vanilla":
            self.to_q_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)
            self.to_k_lora = LoRALinearLayerwBias(cross_attention_dim, hidden_size, rank, network_alpha, with_bias=with_bias)
            self.to_v_lora = LoRALinearLayerwBias(cross_attention_dim, hidden_size, rank, network_alpha, with_bias=with_bias)
            self.to_out_lora = LoRALinearLayerwBias(hidden_size, hidden_size, rank, network_alpha, with_bias=with_bias)

    def _init_stacked_weights(self):
        for proj_name in ["q", "k", "v", "out"]:
            down_w = getattr(self, f"{proj_name}_lora_down_w_stacked")
            up_w = getattr(self, f"{proj_name}_lora_up_w_stacked")
            nn.init.normal_(down_w, std=1 / self.rank)
            nn.init.zeros_(up_w)
            if self.with_bias:
                nn.init.zeros_(getattr(self, f"{proj_name}_lora_bias_stacked"))

    def _apply_lora_stacked(self, x_plane_batched, proj_name):
        down_w = getattr(self, f"{proj_name}_lora_down_w_stacked")
        up_w = getattr(self, f"{proj_name}_lora_up_w_stacked")
        
        lora_down = torch.einsum('bpsi,pir->bpsr', x_plane_batched, down_w)
        lora_up = torch.einsum('bpsr,pro->bpso', lora_down, up_w)

        if self.network_alpha is not None and self.rank > 0:
            lora_up = lora_up * (self.network_alpha / self.rank)
        
        if self.with_bias:
            bias = getattr(self, f"{proj_name}_lora_bias_stacked")
            lora_up = lora_up + bias.unsqueeze(0).unsqueeze(2)
        return lora_up

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None):
        
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if attn.spatial_norm is not None: hidden_states = attn.spatial_norm(hidden_states, temb)
        if input_ndim == 4:
            batch_size_eff, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size_eff, channel, height * width).transpose(1, 2)
        
        batch_size_eff, sequence_length_q, _ = hidden_states.shape
        
        if self.num_planes <= 0: raise ValueError("Number of planes must be positive.")
        original_batch_size = batch_size_eff // self.num_planes

        if encoder_hidden_states is None: encoder_hidden_states = hidden_states
        if attn.norm_cross: encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        batch_size_ehs = encoder_hidden_states.shape[0]
        seq_len_ehs = encoder_hidden_states.shape[1]

        query_orig = attn.to_q(hidden_states)
        if self.lora_type == "few_v1":
            if self.rank > 0:
                h_for_lora_q = hidden_states.view(original_batch_size, self.num_planes, sequence_length_q, self.hidden_size)
                lora_q = self._apply_lora_stacked(h_for_lora_q, "q")
                lora_q_contrib = lora_q.reshape(batch_size_eff, sequence_length_q, self.hidden_size)
                query = query_orig + scale * lora_q_contrib
            else: # few_v1 with rank == 0
                query = query_orig
        elif self.lora_type == "vanilla":
            lora_q_contrib = self.to_q_lora(hidden_states)
            query = query_orig + scale * lora_q_contrib
        else:
            lora_q_contrib = torch.zeros_like(query_orig)
            query = query_orig + scale * lora_q_contrib
        
        key_orig = attn.to_k(encoder_hidden_states)
        value_orig = attn.to_v(encoder_hidden_states)

        if self.lora_type == "few_v1":
            if self.rank > 0:
                if batch_size_ehs == original_batch_size:
                    ehs_for_lora_kv = encoder_hidden_states.unsqueeze(1).repeat(1, self.num_planes, 1, 1)
                elif batch_size_ehs == batch_size_eff:
                    ehs_for_lora_kv = encoder_hidden_states.view(original_batch_size, self.num_planes, seq_len_ehs, self.cross_attention_dim)
                else: raise ValueError("EHS batch size incompatible for few_v1 LoRA.")

                lora_k = self._apply_lora_stacked(ehs_for_lora_kv, "k")
                lora_v = self._apply_lora_stacked(ehs_for_lora_kv, "v")
            
                # lora_k/v are (original_batch_size, self.num_planes, seq_len_ehs, self.hidden_size)
                # We want contributions to be (batch_size_eff, seq_len_ehs, self.hidden_size)
                lora_k_contrib = lora_k.reshape(batch_size_eff, seq_len_ehs, self.hidden_size)
                lora_v_contrib = lora_v.reshape(batch_size_eff, seq_len_ehs, self.hidden_size)
                
                # Ensure key_orig and value_orig match batch_size_eff before adding
                if key_orig.shape[0] == original_batch_size:
                    key_orig_eff = key_orig.repeat_interleave(self.num_planes, dim=0)
                    value_orig_eff = value_orig.repeat_interleave(self.num_planes, dim=0)
                elif key_orig.shape[0] == batch_size_eff:
                    key_orig_eff = key_orig
                    value_orig_eff = value_orig
                else:
                    raise ValueError(
                        f"key_orig/value_orig batch size ({key_orig.shape[0]}) is unexpected. "
                        f"Expected original_batch_size ({original_batch_size}) or batch_size_eff ({batch_size_eff})."
                    )
                
                # Check sequence length consistency; key_orig_eff.shape[1] should be seq_len_ehs
                if key_orig_eff.shape[1] != seq_len_ehs:
                    raise ValueError(
                        f"Sequence length mismatch for key. Expected {seq_len_ehs}, got {key_orig_eff.shape[1]}"
                    )

                key = key_orig_eff + scale * lora_k_contrib
                value = value_orig_eff + scale * lora_v_contrib
            else: # few_v1 with rank == 0
                key = key_orig
                value = value_orig
        elif self.lora_type == "vanilla":
            lora_k_contrib = self.to_k_lora(encoder_hidden_states)
            lora_v_contrib = self.to_v_lora(encoder_hidden_states)
            key = key_orig + scale * lora_k_contrib
            value = value_orig + scale * lora_v_contrib
        else:
            lora_k_contrib = torch.zeros_like(key_orig)
            lora_v_contrib = torch.zeros_like(value_orig)
            key = key_orig + scale * lora_k_contrib
            value = value_orig + scale * lora_v_contrib
        
        query_reshaped = query.view(original_batch_size, sequence_length_q * self.num_planes, self.hidden_size)
        key_reshaped = key.reshape(original_batch_size, -1, self.hidden_size)
        value_reshaped = value.reshape(original_batch_size, -1, self.hidden_size)

        query_heads = attn.head_to_batch_dim(query_reshaped) 
        key_heads = attn.head_to_batch_dim(key_reshaped) 
        value_heads = attn.head_to_batch_dim(value_reshaped) 

        attention_probs = attn.get_attention_scores(query_heads, key_heads, None) 
        hidden_states_attn = torch.bmm(attention_probs, value_heads)
        
        hidden_states_attn = attn.batch_to_head_dim(hidden_states_attn)
        
        hidden_states_attn = hidden_states_attn.view(batch_size_eff, sequence_length_q, self.hidden_size)

        hidden_states_out_orig = attn.to_out[0](hidden_states_attn)
        hidden_states_out_final = hidden_states_out_orig # Initialize final output

        if self.lora_type == "few_v1":
            if self.rank > 0:
                h_attn_for_lora = hidden_states_attn.view(original_batch_size, self.num_planes, sequence_length_q, self.hidden_size)
                lora_out = self._apply_lora_stacked(h_attn_for_lora, "out")
                lora_out_contrib = lora_out.reshape(batch_size_eff, sequence_length_q, self.hidden_size)
                hidden_states_out_final = hidden_states_out_orig + scale * lora_out_contrib
            # else: few_v1 with rank == 0, hidden_states_out_final remains hidden_states_out_orig
        elif self.lora_type == "vanilla":
            lora_out_contrib = self.to_out_lora(hidden_states_attn)
            hidden_states_out_final = hidden_states_out_orig + scale * lora_out_contrib
        else:
            lora_out_contrib = torch.zeros_like(hidden_states_out_orig)
            hidden_states_out_final = hidden_states_out_orig + scale * lora_out_contrib
        
        hidden_states = attn.to_out[1](hidden_states_out_final)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size_eff, channel, height, width)
        if attn.residual_connection: hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class CustomVAEDecoder_FewPlane(nn.Module):
    def __init__(self, original_decoder, vae_config, num_planes, output_dim_final, intermediate_output_dims=None):
        super().__init__()
        self.original_decoder_config = vae_config # Store original VAE config
        self.conv_in = original_decoder.conv_in
        self.up_blocks = original_decoder.up_blocks
        self.mid_block = getattr(original_decoder, 'mid_block', None)
        self.conv_norm_out = getattr(original_decoder, 'conv_norm_out', None)
        self.conv_act = getattr(original_decoder, 'conv_act', F.silu)
        self.conv_out = original_decoder.conv_out # This is the final output layer

        self.num_planes = num_planes
        self.output_dim_final = output_dim_final 
        
        self.intermediate_mapping_layers = nn.ModuleList()
        self.collected_feature_channels = [] # To store IN channels for mapping layers

        # Determine input channels for mapping layers based on collected features
        if self.mid_block is not None:
            # Output channels of mid_block is usually the last channel size in block_out_channels
            self.collected_feature_channels.append(self.original_decoder_config.block_out_channels[-1])

        # Output channels for up_blocks are typically reversed block_out_channels
        rev_block_out_channels = list(reversed(self.original_decoder_config.block_out_channels))
        
        for i in range(len(self.up_blocks)):
            if i < len(rev_block_out_channels):
                self.collected_feature_channels.append(rev_block_out_channels[i])
            else:
                # Fallback: try to get out_channels from the block itself if rev_block_out_channels is too short
                # This path is less likely for standard diffusers VAEs.
                current_block_module = self.up_blocks[i]
                current_block_out_channels = None
                # Attempt to find an 'out_channels' attribute, common in diffusers blocks
                if hasattr(current_block_module, 'out_channels'):
                    current_block_out_channels = current_block_module.out_channels
                elif hasattr(current_block_module, 'resnets') and len(current_block_module.resnets) > 0 and \
                     hasattr(current_block_module.resnets[-1], 'conv2') and \
                     hasattr(current_block_module.resnets[-1].conv2, 'out_channels'):
                     current_block_out_channels = current_block_module.resnets[-1].conv2.out_channels
                
                if current_block_out_channels is not None:
                     self.collected_feature_channels.append(current_block_out_channels)
                else:
                    print(f"[WARN] CustomVAEDecoder_FewPlane: Could not reliably determine output channels for up_block {i}. Using previous or default.")
                    if self.collected_feature_channels: # Use last known channel
                        self.collected_feature_channels.append(self.collected_feature_channels[-1])
                    else: # Very unlikely (e.g. no mid_block and first up_block fails)
                        self.collected_feature_channels.append(self.output_dim_final) # Last resort fallback

        actual_num_intermediate_features = len(self.collected_feature_channels)
        target_output_dims_for_map_layers = []

        if intermediate_output_dims and len(intermediate_output_dims) == actual_num_intermediate_features:
            target_output_dims_for_map_layers = intermediate_output_dims
        else:
            if intermediate_output_dims: # Was provided but wrong length
                print(
                    f"[WARN] CustomVAEDecoder_FewPlane: intermediate_output_dims (len {len(intermediate_output_dims)}) "
                    f"does not match actual_num_intermediate_features ({actual_num_intermediate_features}). "
                    f"Defaulting all intermediate mapping layers to output_dim_final ({self.output_dim_final})."
                )
            target_output_dims_for_map_layers = [self.output_dim_final] * actual_num_intermediate_features
        
        for i in range(actual_num_intermediate_features):
            in_ch = self.collected_feature_channels[i]
            out_ch = target_output_dims_for_map_layers[i]
            conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0) # Match reference style
            if conv_layer.bias is not None:
                nn.init.zeros_(conv_layer.bias)
            if conv_layer.weight is not None and i > 0: # Only initialize weights for the first layer
                nn.init.zeros_(conv_layer.weight)
            self.intermediate_mapping_layers.append(conv_layer)

    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        intermediate_features_raw = []
        # x 初始形状: (B_eff, C_latent, H_latent, W_latent)
        # B_eff = original_batch_size * num_planes

        x = self.conv_in(x)
        # intermediate_features_raw.append(x) # 可选：添加 conv_in 后的特征

        if self.mid_block is not None:
            x = self.mid_block(x)
            intermediate_features_raw.append(x) # mid_block 输出

        for up_block_idx, up_block in enumerate(self.up_blocks):
            x = up_block(x)
            intermediate_features_raw.append(x) # 每个 up_block 输出

        if self.conv_norm_out is not None:
            x = self.conv_norm_out(x)
        
        x = self.conv_act(x)
        final_output = self.conv_out(x) # 形状: (B_eff, output_dim_final, H_out, W_out)

        processed_intermediate_features = []
        # Ensure the number of collected raw features matches the number of mapping layers.
        # This check helps catch discrepancies between __init__ logic for channels and forward collection.
        if len(intermediate_features_raw) != len(self.intermediate_mapping_layers):
             print(
                f"[WARN] CustomVAEDecoder_FewPlane: Mismatch between collected raw features ({len(intermediate_features_raw)}) "
                f"and mapping layers ({len(self.intermediate_mapping_layers)}). Mapping may be incorrect or incomplete."
            )

        for i, feat_raw in enumerate(intermediate_features_raw):
            if i < len(self.intermediate_mapping_layers): # Check if a mapping layer exists for this feature
                mapped_feat = self.intermediate_mapping_layers[i](feat_raw)
            else:
                # This case implies more raw features were collected in forward() than mapping layers created in __init__().
                print(f"[WARN] CustomVAEDecoder_FewPlane: No mapping layer for raw feature index {i}. Using raw feature as fallback.")
                mapped_feat = feat_raw
            
            # Reshape the (potentially mapped) feature
            orig_B = mapped_feat.shape[0] // self.num_planes
            C_feat, H_feat, W_feat = mapped_feat.shape[1], mapped_feat.shape[2], mapped_feat.shape[3]
            processed_intermediate_features.append(
                mapped_feat.view(orig_B, self.num_planes, C_feat, H_feat, W_feat)
            )
        
        orig_B_final = final_output.shape[0] // self.num_planes
        final_output_reshaped = final_output.view(orig_B_final, self.num_planes, self.output_dim_final, *final_output.shape[-2:])
        processed_intermediate_features.append(final_output_reshaped)

        return processed_intermediate_features


class FewStepFewPlaneStableDiffusion(BaseModule):
    """
    Few-step Few Plane Stable Diffusion module.
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        training_type: str = "lora_rank_4_self_lora_rank_4_cross_lora_rank_4_locon_rank_4" # Example, may need update if "few_v1" is default
        output_dim: int = 14
        num_planes: int = 4  
        gradient_checkpoint: bool = False
        inherit_conv_out: bool = False
        self_lora_type: str = "few_v1" # few_v1
        cross_lora_type: str = "few_v1" # few_v1
        locon_type: str = "few_v1" # few_v1
        vae_self_lora_type: str = "vanilla" 
        vae_cross_lora_type: str = "vanilla" 
        vae_locon_type: str = "vanilla_v1" 
        w_lora_bias: bool = False
        w_locon_bias: bool = False
        network_alpha: Optional[float] = None

        require_intermediate_features: bool = False


    cfg: Config

    def configure(self) -> None:
        self.output_dim = self.cfg.output_dim
        self.num_planes = self.cfg.num_planes 
        model_path = self.cfg.pretrained_model_name_or_path
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
        del vae.encoder
        del vae.quant_conv
        cleanup()
        training_type = self.cfg.training_type
        self.self_lora_rank = 0
        if "self_lora_rank_" in training_type:
            match = re.search(r"self_lora_rank_(\d+)", training_type)
            if match: self.self_lora_rank = int(match.group(1))
        self.cross_lora_rank = 0
        if "cross_lora_rank_" in training_type:
            match = re.search(r"cross_lora_rank_(\d+)", training_type)
            if match: self.cross_lora_rank = int(match.group(1))
        self.locon_rank = 0
        if "locon_rank_" in training_type:
            match = re.search(r"locon_rank_(\d+)", training_type)
            if match: self.locon_rank = int(match.group(1))
        self.w_lora_bias = "with_lora_bias" in training_type or self.cfg.w_lora_bias
        self.w_locon_bias = "with_locon_bias" in training_type or self.cfg.w_locon_bias
        self.network_alpha = self.cfg.network_alpha

        assert "lora" in training_type or "locon" in training_type or "full" in training_type, "Training type not supported."

        if not "full" in training_type:
            trainable_params = {}
            @dataclass
            class SubModules:
                unet: UNet2DConditionModel
                vae: AutoencoderKL

            self.submodules = SubModules(unet=unet.to(self.device), vae=vae.to(self.device))
            for param in self.unet.parameters(): param.requires_grad_(False)
            for param in self.vae.parameters(): param.requires_grad_(False)

            if "lora" in training_type:
                unet_lora_attn_procs = self._set_attn_processor(
                    self.unet, self_lora_rank_val=self.self_lora_rank,
                    cross_lora_rank_val=self.cross_lora_rank, self_lora_type_val=self.cfg.self_lora_type,
                    cross_lora_type_val=self.cfg.cross_lora_type, with_bias_val=self.w_lora_bias,
                    num_planes_val=self.num_planes 
                )
                self.unet.set_attn_processor(unet_lora_attn_procs)
                trainable_params.update(self.unet.attn_processors)
                vae_lora_attn_procs = self._set_attn_processor(
                    self.vae, self_lora_rank_val=self.self_lora_rank,
                    cross_lora_rank_val=self.cross_lora_rank, self_lora_type_val=self.cfg.vae_self_lora_type,
                    cross_lora_type_val=self.cfg.vae_cross_lora_type, with_bias_val=self.w_lora_bias, is_vae=True,
                    num_planes_val=self.num_planes 
                )
                self.vae.set_attn_processor(vae_lora_attn_procs)
                trainable_params.update(self.vae.attn_processors)

            if "locon" in training_type:
                unet_locon_layers = self._set_conv_processor(
                    self.unet, rank_val=self.locon_rank, locon_type_val=self.cfg.locon_type,
                    with_bias_val=self.w_locon_bias, num_planes_val=self.num_planes 
                )
                trainable_params.update(unet_locon_layers)
                vae_locon_layers = self._set_conv_processor(
                    self.vae, rank_val=self.locon_rank, locon_type_val=self.cfg.vae_locon_type,
                    with_bias_val=self.w_locon_bias, num_planes_val=self.num_planes 
                )
                trainable_params.update(vae_locon_layers)

            conv_out_orig = self.vae.decoder.conv_out
            conv_out_new = nn.Conv2d(
                in_channels=conv_out_orig.in_channels,
                out_channels=self.cfg.output_dim, kernel_size=3, padding=1
            )
            if conv_out_new.bias is not None:
                nn.init.zeros_(conv_out_new.bias)
            if conv_out_new.weight is not None and self.cfg.require_intermediate_features:
                nn.init.zeros_(conv_out_new.weight)
            if self.cfg.inherit_conv_out and self.cfg.output_dim >= conv_out_orig.out_channels:
                conv_out_new.weight.data[:conv_out_orig.out_channels, :, :, :] = conv_out_orig.weight.data
                conv_out_new.bias.data[:conv_out_orig.out_channels] = conv_out_orig.bias.data
            self.vae.decoder.conv_out = conv_out_new
            trainable_params["vae.decoder.conv_out"] = conv_out_new

            # === 新增：替换为 Custom VAE Decoder ===
            if self.cfg.require_intermediate_features: # 从配置读取
                original_vae_decoder = self.vae.decoder
                # (可选) 定义你希望中间特征映射到的维度
                # intermediate_output_dims_list = [self.cfg.output_dim] * num_intermediate_layers 
                custom_decoder = CustomVAEDecoder_FewPlane(
                    original_decoder=original_vae_decoder,
                    vae_config=self.vae.config, # Pass the VAE's config
                    num_planes=self.num_planes,
                    output_dim_final=self.cfg.output_dim,
                    # intermediate_output_dims=intermediate_output_dims_list # 可选
                )
                self.vae.decoder = custom_decoder
                # 如果 intermediate_mapping_layers 是可训练的，也需要加入 trainable_params
                if hasattr(custom_decoder, 'intermediate_mapping_layers') and len(custom_decoder.intermediate_mapping_layers) > 0:
                    trainable_params["vae.decoder.intermediate_mapping_layers"] = custom_decoder.intermediate_mapping_layers
            # =====================================

            self.peft_layers = AttnProcsLayers(trainable_params).to(self.device)
            self.peft_layers._load_state_dict_pre_hooks.clear()
            self.peft_layers._state_dict_hooks.clear()

        elif training_type == "full":
            self.unet = unet.to(self.device)
            self.vae = vae.to(self.device)
            unet_lora_attn_procs = self._set_attn_processor(
                self.unet, self_lora_rank_val=0, cross_lora_rank_val=0,
                self_lora_type_val="none", cross_lora_type_val="none", with_bias_val=False,
                num_planes_val=self.num_planes 
            )
            self.unet.set_attn_processor(unet_lora_attn_procs)
            vae_lora_attn_procs = self._set_attn_processor(
                self.vae, self_lora_rank_val=0, cross_lora_rank_val=0,
                self_lora_type_val="none", cross_lora_type_val="none", with_bias_val=False, is_vae=True,
                num_planes_val=self.num_planes 
            )
            self.vae.set_attn_processor(vae_lora_attn_procs)

            conv_out_orig = self.vae.decoder.conv_out
            conv_out_new = nn.Conv2d(
                in_channels=conv_out_orig.in_channels,
                out_channels=self.cfg.output_dim, kernel_size=3, padding=1
            ).to(self.device)
            if self.cfg.inherit_conv_out and self.cfg.output_dim >= conv_out_orig.out_channels:
                conv_out_new.weight.data[:conv_out_orig.out_channels, :, :, :] = conv_out_orig.weight.data.to(self.device)
                conv_out_new.bias.data[:conv_out_orig.out_channels] = conv_out_orig.bias.data.to(self.device)
            self.vae.decoder.conv_out = conv_out_new
        else:
            raise NotImplementedError("The training type is not supported.")

        if self.cfg.gradient_checkpoint:
            self.unet.enable_gradient_checkpointing()

    @property
    def unet(self):
        return self.submodules.unet if hasattr(self, "submodules") else self._unet_full
    @property
    def vae(self):
        return self.submodules.vae if hasattr(self, "submodules") else self._vae_full
    def __setattr__(self, name, value):
        if name == "unet" and not hasattr(self, "submodules"): self._unet_full = value
        elif name == "vae" and not hasattr(self, "submodules"): self._vae_full = value
        else: super().__setattr__(name, value)

    def _set_conv_processor(
        self,
        module,
        rank_val: int,
        locon_type_val: str,
        with_bias_val: bool,
        num_planes_val: int, 
        conv_compatible_name: str = "LoRACompatibleConv",
    ):
        locon_layers = {}
        current_locon_type = "none" if rank_val == 0 else locon_type_val
        current_rank = 0 if current_locon_type == "none" else rank_val

        for _name, _module in module.named_modules():
            if _module.__class__.__name__ == conv_compatible_name:
                # Ensure kernel_size, stride, padding are correctly obtained from _module
                # as they might be int or tuple.
                kernel_size = _module.kernel_size
                stride = _module.stride
                padding = _module.padding

                locon_layer = FewLoRAConv2dLayer( # Changed from QuadLoRAConv2dLayer
                    in_features=_module.in_channels,
                    out_features=_module.out_channels,
                    rank=current_rank, 
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    with_bias=with_bias_val,
                    locon_type=current_locon_type, 
                    network_alpha=self.network_alpha,
                    num_planes=num_planes_val 
                )
                _module.lora_layer = locon_layer
                key_name = f"{_name}.lora_layer"
                locon_layers[key_name] = locon_layer
        return locon_layers

    def _set_attn_processor(
            self,
            module,
            self_lora_rank_val: int,
            cross_lora_rank_val: int,
            self_lora_type_val: str,
            cross_lora_type_val: str,
            with_bias_val: bool,
            num_planes_val: int, 
            is_vae: bool = False,
        ):
        attn_procs = {}
        self_attn_suffix = "processor" if is_vae else "attn1.processor"
        
        for name in module.attn_processors.keys():
            cross_attention_dim_val = module.config.cross_attention_dim if hasattr(module.config, "cross_attention_dim") else None
            
            up_blocks_prefix = "up_blocks."
            down_blocks_prefix = "down_blocks."

            if name.startswith("mid_block"): hidden_size_val = module.config.block_out_channels[-1]
            elif name.startswith(up_blocks_prefix):
                block_id_str = name[len(up_blocks_prefix):].split('.')[0]
                block_id = int(block_id_str)
                hidden_size_val = list(reversed(module.config.block_out_channels))[block_id]
            elif name.startswith(down_blocks_prefix):
                block_id_str = name[len(down_blocks_prefix):].split('.')[0]
                block_id = int(block_id_str)
                hidden_size_val = module.config.block_out_channels[block_id]
            elif name.startswith("decoder") and is_vae:
                hidden_size_val = module.config.block_out_channels[-1]
            else:
                hidden_size_val = module.config.block_out_channels[0]

            if name.endswith(self_attn_suffix):
                current_rank = self_lora_rank_val
                current_lora_type = "none" if current_rank == 0 else self_lora_type_val
                
                if current_lora_type == "none":
                    attn_procs[name] = FewSelfAttentionLoRAAttnProcessor( # Changed from Quad...
                        hidden_size_val, rank=0, lora_type="none", with_bias=False,
                        network_alpha=self.network_alpha, num_planes=num_planes_val 
                    )
                elif current_lora_type == "vanilla":
                    attn_procs[name] = FewSelfAttentionLoRAAttnProcessor( # Changed from Quad...
                        hidden_size_val, rank=current_rank, network_alpha=self.network_alpha,
                        with_bias=with_bias_val, lora_type="vanilla", num_planes=num_planes_val 
                    )
                elif current_lora_type == "few_v1": # Changed from quad_v1
                    attn_procs[name] = FewSelfAttentionLoRAAttnProcessor( # Changed from Quad...
                        hidden_size_val, rank=current_rank, network_alpha=self.network_alpha,
                        with_bias=with_bias_val, lora_type="few_v1", num_planes=num_planes_val 
                    )
                else: attn_procs[name] = AttnProcessor()
            else: 
                current_rank = cross_lora_rank_val
                current_lora_type = "none" if current_rank == 0 else cross_lora_type_val
                if cross_attention_dim_val is None and not name.endswith("attn2.processor"):
                    attn_procs[name] = AttnProcessor()
                    continue

                if current_lora_type == "none":
                    attn_procs[name] = FewCrossAttentionLoRAAttnProcessor( # Changed from Quad...
                        hidden_size_val, cross_attention_dim_val, rank=0, lora_type="none", with_bias=False,
                        network_alpha=self.network_alpha, num_planes=num_planes_val 
                    )
                elif current_lora_type == "vanilla":
                    attn_procs[name] = FewCrossAttentionLoRAAttnProcessor( # Changed from Quad...
                        hidden_size_val, cross_attention_dim_val, rank=current_rank,
                        network_alpha=self.network_alpha,
                        with_bias=with_bias_val, lora_type="vanilla", num_planes=num_planes_val 
                    )
                elif current_lora_type == "few_v1": # Changed from quad_v1
                    attn_procs[name] = FewCrossAttentionLoRAAttnProcessor( # Changed from Quad...
                        hidden_size_val, cross_attention_dim_val, rank=current_rank,
                        network_alpha=self.network_alpha,
                        with_bias=with_bias_val, lora_type="few_v1", num_planes=num_planes_val 
                    )
                else: attn_procs[name] = AttnProcessor()
        return attn_procs

    def forward(
        self,
        text_embed: Tensor, # Float[Tensor, "B L D"],
        noisy_latents: Tensor, # Float[Tensor, "B_eff C H W"],
    ):
        raise NotImplementedError("The forward function is not implemented.")

    def forward_denoise(
        self,
        text_embed,
        noisy_input,
        t,
    ):
        original_batch_size = text_embed.shape[0] 
        if self.num_planes <= 0: raise ValueError("Number of planes must be positive for forward_denoise.")

        effective_unet_batch_size = original_batch_size * self.num_planes

        if text_embed.ndim == 3:
            text_embed_unet = text_embed.repeat_interleave(self.num_planes, dim=0)
        elif text_embed.ndim == 4: 
            text_embed_unet = text_embed.reshape(effective_unet_batch_size, *text_embed.shape[2:])
        else:
            raise ValueError(f"text_embed shape {text_embed.shape} not supported.")

        if noisy_input.ndim == 5 and noisy_input.shape[0] == original_batch_size and noisy_input.shape[1] == self.num_planes:
            noisy_input_unet = noisy_input.reshape(effective_unet_batch_size, noisy_input.shape[2], noisy_input.shape[3], noisy_input.shape[4])
        elif noisy_input.ndim == 4 and noisy_input.shape[0] == original_batch_size:
            noisy_input_unet = noisy_input.repeat_interleave(self.num_planes, dim=0)
        elif noisy_input.ndim == 4 and noisy_input.shape[0] == effective_unet_batch_size:
            noisy_input_unet = noisy_input
        else:
            raise ValueError(
                f"noisy_input shape {noisy_input.shape} not supported. "
                f"Expected 5D (orig_bs={original_batch_size}, num_planes={self.num_planes}, C, H, W) "
                f"or 4D (orig_bs={original_batch_size}, C, H, W) "
                f"or 4D (eff_bs={effective_unet_batch_size}, C, H, W)."
            )
        
        if t.ndim == 0: 
            timesteps_unet = t.unsqueeze(0).repeat(effective_unet_batch_size)
        elif t.ndim == 1 and t.shape[0] == original_batch_size:
            timesteps_unet = t.repeat_interleave(self.num_planes, dim=0)
        elif t.ndim == 1 and t.shape[0] == effective_unet_batch_size:
            timesteps_unet = t
        else:
            raise ValueError(
                f"timesteps (t) shape {t.shape} not supported. Expected scalar, "
                f"or 1D with size original_batch_size ({original_batch_size}) "
                f"or 1D with size effective_unet_batch_size ({effective_unet_batch_size})."
            )

        if noisy_input_unet.shape[0] != text_embed_unet.shape[0]:
            raise ValueError(
                f"Batch size mismatch between noisy_input_unet ({noisy_input_unet.shape[0]}) "
                f"and text_embed_unet ({text_embed_unet.shape[0]}). Should be {effective_unet_batch_size}"
            )
        
        if noisy_input_unet.shape[0] != timesteps_unet.shape[0]:
            raise ValueError(
                f"Batch size mismatch between noisy_input_unet ({noisy_input_unet.shape[0]}) "
                f"and timesteps_unet ({timesteps_unet.shape[0]}). Should be {effective_unet_batch_size}"
            )

        noise_pred = self.unet(
            noisy_input_unet,
            timesteps_unet, 
            encoder_hidden_states=text_embed_unet
        ).sample

        return noise_pred

    def forward_decode(
        self,
        latents: torch.FloatTensor, # 预期形状 (B_eff, C_latent, H_latent, W_latent)
    ):
        B_eff = latents.shape[0]
        # original_batch_size = B_eff // self.num_planes # 如果需要原始批次大小

        if self.cfg.require_intermediate_features and \
           isinstance(self.vae.decoder, CustomVAEDecoder_FewPlane):
            
            # CustomVAEDecoder_FewPlane.forward 内部接收 (B_eff, C, H, W)
            # 并返回 final_output_reshaped (orig_B, num_planes, C_out, H_out, W_out)
            # 和 processed_intermediate_features (列表，每个元素是 (orig_B, num_planes, C_feat, H_feat, W_feat))
            return self.vae.decode(latents).sample
        else:
            # 原始逻辑
            final_output_flat = self.vae.decode(latents).sample # (B_eff, output_dim, H_out, W_out)
            
            # 计算 original_batch_size 用于 view
            if self.num_planes == 0: # 避免除以零
                raise ValueError("num_planes cannot be zero for reshaping.")
            original_batch_size = B_eff // self.num_planes
            if B_eff % self.num_planes != 0:
                raise ValueError(f"Effective batch size {B_eff} is not divisible by num_planes {self.num_planes}")

            return final_output_flat.view(original_batch_size, 
                                          self.num_planes, 
                                          self.cfg.output_dim, 
                                          *final_output_flat.shape[-2:])