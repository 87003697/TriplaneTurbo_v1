import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re

from diffusers.models.attention_processor import Attention, AttnProcessor, LoRAAttnProcessor, LoRALinearLayer
from threestudio.utils.typing import *
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from diffusers.loaders import AttnProcsLayers
from threestudio.utils.base import BaseModule
from dataclasses import dataclass

from diffusers.models.lora import LoRAConv2dLayer
from threestudio.utils.misc import cleanup


class CustomVAEDecoder(nn.Module):
    def __init__(self, original_decoder):
        super().__init__()
        self.conv_in = original_decoder.conv_in
        self.up_blocks = original_decoder.up_blocks # This is typically a ModuleList
        
        # getattr is used for optional attributes like mid_block
        self.mid_block = getattr(original_decoder, 'mid_block', None)
        
        self.conv_norm_out = getattr(original_decoder, 'conv_norm_out', None)
        self.conv_act = getattr(original_decoder, 'conv_act', F.silu) # Use original act or default to silu
        self.conv_out = original_decoder.conv_out # This will be the already modified one

        # record the number of intermediate features
        self.num_intermediate_features = len(self.up_blocks) - 3 # the last two up_blocks are as the same resolution as the output resolution
        # record the channel size of the intermediate features
        self.intermediate_channels = [
            self.mid_block.resnets[-1].conv2.out_channels
        ]
        for idx, up_block in enumerate(self.up_blocks):
            if idx <= self.num_intermediate_features:
                self.intermediate_channels.append(up_block.upsamplers[0].out_channels)


    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        
        intermediate_features = []
        x = self.conv_in(x)

        # save the intermediate features and its resolution
        intermediate_features.append(x)


        if self.mid_block is not None:
            x = self.mid_block(x)
            # Optionally, one could consider mid_block's output as an intermediate feature here
            # For now, focusing on up_blocks as per "each resolution" implies upsampling stages

        for idx, up_block in enumerate(self.up_blocks):
            x = up_block(x)
            
            # save the intermediate features and its resolution
            if idx <= self.num_intermediate_features: # the last two up_blocks are as the same resolution as the output resolution
                intermediate_features.append(x)

        if self.conv_norm_out is not None:
            x = self.conv_norm_out(x)
        
        x = self.conv_act(x)
        output = self.conv_out(x)

        return output, intermediate_features


class FewStepOnePlaneStableDiffusion(BaseModule):
    """
    Few-step One Plane Stable Diffusion module.
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        training_type: str = "lora_rank_4",
        output_dim: int = 14
        gradient_checkpoint: bool = False
        inherit_conv_out: bool = False
        require_intermediate_features: bool = False

    cfg: Config

    def configure(self) -> None:

        self.output_dim = self.cfg.output_dim
        self.num_planes = 1

        # we only use the unet and vae model here
        model_path = self.cfg.pretrained_model_name_or_path
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
        # the encoder is not needed
        del vae.encoder
        del vae.quant_conv 
        cleanup()

        # set the training type
        training_type = self.cfg.training_type



        assert "lora" in training_type or "locon" in training_type or "full" in training_type, "The training type is not supported."
 
        if not "full" in training_type: # then paramter-efficient training

            # save trainable parameters
            trainable_params_gen = {}
            trainable_params_dec = {}
            assert "lora" in training_type or "locon" in training_type, "The training type is not supported."
            @dataclass
            class SubModules:
                unet: UNet2DConditionModel
                vae: AutoencoderKL

            self.submodules = SubModules(
                unet=unet.to(self.device),
                vae=vae.to(self.device),
            )

            # free all the parameters
            for param in self.unet.parameters():
                param.requires_grad_(False)
            for param in self.vae.parameters():
                param.requires_grad_(False)

            ############################################################
            # overwrite the unet and vae with the customized processors

            if "lora" in training_type:

                # parse the rank from the training type, with the template "lora_rank_{}"
                assert "self_lora_rank" in training_type, "The self_lora_rank is not specified."
                rank = re.search(r"self_lora_rank_(\d+)", training_type).group(1)
                self.self_lora_rank = int(rank)

                assert "cross_lora_rank" in training_type, "The cross_lora_rank is not specified."
                rank = re.search(r"cross_lora_rank_(\d+)", training_type).group(1)
                self.cross_lora_rank = int(rank)


                # specify the attn_processor for unet
                lora_attn_procs = self._set_attn_processor(
                    self.unet, 
                    self_attn_name="attn1.processor",
                )
                self.unet.set_attn_processor(lora_attn_procs)
                # update the trainable parameters
                trainable_params_gen.update(self.unet.attn_processors)

                # specify the attn_processor for vae
                lora_attn_procs = self._set_attn_processor(
                    self.vae, 
                    self_attn_name="processor",
                )
                self.vae.set_attn_processor(lora_attn_procs)
                # update the trainable parameters
                trainable_params_dec.update(self.vae.attn_processors)

            if "locon" in training_type:
                # parse the rank from the training type, with the template "locon_rank_{}"
                rank = re.search(r"locon_rank_(\d+)", training_type).group(1)
                self.locon_rank = int(rank)


                # specify the conv_processor for unet
                locon_procs = self._set_conv_processor(
                    self.unet,
                )

                # update the trainable parameters
                trainable_params_gen.update(locon_procs)

                # specify the conv_processor for vae
                locon_procs = self._set_conv_processor(
                    self.vae,
                )
                # update the trainable parameters
                trainable_params_dec.update(locon_procs)

            # overwrite the outconv
            conv_out_orig = self.vae.decoder.conv_out
            conv_out_new = nn.Conv2d(
                in_channels=128, # conv_out_orig.in_channels, hard-coded
                out_channels=self.cfg.output_dim, kernel_size=3, padding=1
            )

            if self.cfg.inherit_conv_out:
                # copy the weights from the original conv_out
                conv_out_new.weight.data[:3, :, :, :] = conv_out_orig.weight.data
                conv_out_new.bias.data[:3] = conv_out_orig.bias.data

            # update the trainable parameters
            self.vae.decoder.conv_out = conv_out_new
            trainable_params_dec["vae.decoder.conv_out"] = conv_out_new

            # save the trainable parameters
            self.gen_layers = AttnProcsLayers(trainable_params_gen).to(self.device)
            self.gen_layers._load_state_dict_pre_hooks.clear()
            self.gen_layers._state_dict_hooks.clear()      

            self.dec_layers = AttnProcsLayers(trainable_params_dec).to(self.device)
            self.dec_layers._load_state_dict_pre_hooks.clear()
            self.dec_layers._state_dict_hooks.clear()      

        elif training_type == "full": # full parameter training

            # just nullify the parameters
            self.self_lora_rank = 0
            self.cross_lora_rank = 0
            self.w_lora_bias = False

            self.unet = unet.to(self.device)
            self.vae = vae.to(self.device)

            # overwrite the outconv
            conv_out_orig = self.vae.decoder.conv_out
            conv_out_new = nn.Conv2d(
                in_channels=128, # conv_out_orig.in_channels, hard-coded
                out_channels=self.cfg.output_dim, kernel_size=3, padding=1
            )
            if self.cfg.inherit_conv_out:
                # copy the weights from the original conv_out
                conv_out_new.weight.data[:3, :, :, :] = conv_out_orig.weight.data
                conv_out_new.bias.data[:3] = conv_out_orig.bias.data
                
            # update the trainable parameters
            self.vae.decoder.conv_out = conv_out_new.to(self.device)

            ############################################################
            # overwrite the unet and vae with the customized processors

            # specify the attn_processor for unet
            lora_attn_procs = self._set_attn_processor(
                self.unet, 
                self_attn_name="attn1.processor",
                self_lora_type="none",
                cross_lora_type="none",
            )
            self.unet.set_attn_processor(lora_attn_procs)
        else:
            raise NotImplementedError("The training type is not supported.")

        # Replace VAE decoder with custom decoder to extract intermediate features
        # This is done after all modifications to self.vae.decoder (like conv_out replacement, LoRA/LoCon)
        # are complete, so CustomVAEDecoder inherits them.
        if self.cfg.require_intermediate_features:
            original_vae_decoder = self.vae.decoder
            custom_decoder = CustomVAEDecoder(original_vae_decoder)
            self.vae.decoder = custom_decoder

            # record the number of intermediate features
            self.num_intermediate_features = custom_decoder.num_intermediate_features
            # record the channel size of the intermediate features
            self.intermediate_channels = custom_decoder.intermediate_channels

            # init new layers to map the intermediate features to the output dimensions
            self.intermediate_to_output_layers = nn.ModuleList()
            for in_channel in self.intermediate_channels:
                self.intermediate_to_output_layers.append(
                    nn.Conv2d(in_channel, self.cfg.output_dim, kernel_size=3, padding=1)
                )
            trainable_params_dec["intermediate_to_output_layers"] = self.intermediate_to_output_layers

        if self.cfg.gradient_checkpoint:
            self.unet.enable_gradient_checkpointing()
            self.vae.enable_gradient_checkpointing()

    @property
    def unet(self):
        return self.submodules.unet

    @property
    def vae(self):
        return self.submodules.vae

    def _set_conv_processor(
        self,
        module,
        conv_name: str = "LoRACompatibleConv",
    ):
        locon_procs = {}
        for _name, _module in module.named_modules():
            if _module.__class__.__name__ == conv_name:
                # append the locon processor to the module
                locon_proc = LoRAConv2dLayer(
                    in_features=_module.in_channels,
                    out_features=_module.out_channels,
                    rank=self.locon_rank,
                    kernel_size=_module.kernel_size,
                    stride=_module.stride,
                    padding=_module.padding,
                )
                # add the locon processor to the module
                _module.lora_layer = locon_proc
                # update the trainable parameters
                key_name = f"{_name}.lora_layer"
                locon_procs[key_name] = locon_proc
        return locon_procs



    def _set_attn_processor(
            self, 
            module,
            self_attn_name: str = "attn1.processor",
        ):
        lora_attn_procs = {}
        for name in module.attn_processors.keys():

            if name.startswith("mid_block"):
                hidden_size = module.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(module.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = module.config.block_out_channels[block_id]
            elif name.startswith("decoder"):
                # special case for decoder in SD
                hidden_size = 512

            if name.endswith(self_attn_name):
                # it is self-attention
                cross_attention_dim = None
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size = hidden_size, 
                    lora_rank = self.self_lora_rank
                )
            else:
                # it is cross-attention
                cross_attention_dim = module.config.cross_attention_dim
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size = hidden_size, 
                    cross_attention_dim = cross_attention_dim, 
                    lora_rank = self.cross_lora_rank
                )
        return lora_attn_procs

    def forward(
        self,
        text_embed,
        styles,
    ):

        raise NotImplementedError("The forward function is not implemented.")

        
    def forward_denoise(
        self, 
        text_embed,
        noisy_input,
        t,
    ):

        batch_size = text_embed.size(0)
        noise_shape = noisy_input.size(-2)

        if text_embed.ndim == 3:
            # same text_embed for all planes
            # text_embed = text_embed.repeat(self.num_planes, 1, 1) # wrong!!!
            text_embed = text_embed.repeat_interleave(self.num_planes, dim=0)
        elif text_embed.ndim == 4:
            # different text_embed for each plane
            text_embed = text_embed.view(batch_size * self.num_planes, *text_embed.shape[-2:])
        else:
            raise ValueError("The text_embed should be either 3D or 4D.")

        noisy_input = noisy_input.view(-1, 4, noise_shape, noise_shape)
        noise_pred = self.unet(
            noisy_input,
            t,
            encoder_hidden_states=text_embed
        ).sample


        return noise_pred

    def forward_decode(
        self,
        latents,
    ):
        latents = latents.view(-1, 4, *latents.shape[-2:])
        if self.cfg.require_intermediate_features:

            # AutoencoderKL.decode typically returns a DecoderOutput object, and .sample gives the decoder's output.
            # If CustomVAEDecoder returns (output, features), then .sample will be (output, features).
            triplane_output, intermediate_features_raw = self.vae.decode(latents).sample
            triplane_output = triplane_output.view(-1, self.num_planes, self.cfg.output_dim, *triplane_output.shape[-2:])

            # Reshape intermediate features to match (batch_size, num_planes, C, H, W)
            features_reshaped = []
            for idx, feat in enumerate(intermediate_features_raw):
                # feat shape is (batch_size * self.num_planes, channels, height, width)
                feat_reshaped = self.intermediate_to_output_layers[idx](feat)
                features_reshaped.append(feat_reshaped.view(-1, self.num_planes, *feat_reshaped.shape[1:]))
            
            features_reshaped.append(triplane_output.view(-1, self.num_planes, self.cfg.output_dim, *triplane_output.shape[-2:]))
            return features_reshaped
        
        else:
            triplane_output = self.vae.decode(latents).sample
            return triplane_output.view(-1, self.num_planes, self.cfg.output_dim, *triplane_output.shape[-2:])