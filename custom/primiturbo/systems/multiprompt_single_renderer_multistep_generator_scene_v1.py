import os
import shutil
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_rank, get_device, barrier
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from functools import partial

from tqdm import tqdm
from threestudio.utils.misc import barrier
from threestudio.models.mesh import Mesh

from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)

from torch.autograd import Variable, grad as torch_grad
from threestudio.utils.ops import SpecifyGradient
from threestudio.systems.utils import parse_optimizer, parse_scheduler, get_parameters

from .utils import visualize_center_depth, save_attribute_visualization_grid

from threestudio.utils.misc import C

from custom.primiturbo.models.geometry.utils import CudaKNNIndex, position_pull_loss, scale_smooth_loss

def sample_timesteps(
    all_timesteps: List,
    num_parts: int,
    batch_size: int = 1,
):
    # separate the timestep into num_parts_training parts
    timesteps = []

    for i in range(num_parts):
        length_timestep = len(all_timesteps) // num_parts
        timestep = all_timesteps[
            i * length_timestep : (i + 1) * length_timestep
        ]
        # sample only one from the timestep
        idx = torch.randint(0, len(timestep), (batch_size,))
        timesteps.append(timestep[idx])

    return timesteps

@threestudio.register("multiprompt-single-renderer-multistep-generator-scene-system-v1")
class MultipromptSingleRendererMultiStepGeneratorSceneSystemV1(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):

        # validation related
        visualize_samples: bool = False

        # renderering related
        rgb_as_latents: bool = False

        # initialization related
        initialize_shape: bool = True

        # if the guidance requires training
        train_guidance: bool = False

        # scheduler path
        scheduler_dir: str = "pretrained/stable-diffusion-2-1-base"

        # the followings are related to the multi-step diffusion
        num_parts_training: int = 4

        num_steps_training: int = 50
        num_steps_sampling: int = 50

        
        sample_scheduler: str = "ddpm" #any of "ddpm", "ddim"
        noise_scheduler: str = "ddim"

        gradient_accumulation_steps: int = 1

        training_type: str = "rollout-rendering-distillation-last-step" # "progressive-rendering-distillation" or "rollout-rendering-distillation" or "rollout-rendering-distillation-last-step"
        multi_step_module_name: Optional[str] = "space_generator.gen_layers"

        min_scale_factor: float = 1.0

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        if self.cfg.train_guidance: # if the guidance requires training, then it is initialized here
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # Sampler for training
        self.noise_scheduler = self._configure_scheduler(self.cfg.noise_scheduler)
        self.is_training_sde = True if self.cfg.noise_scheduler == "ddpm" else False

        # Sampler for inference
        self.sample_scheduler = self._configure_scheduler(self.cfg.sample_scheduler)

        # This property activates manual optimization.
        self.automatic_optimization = False 


    def _configure_scheduler(self, scheduler: str):
        assert scheduler in ["ddpm", "ddim", "dpm"]
        if scheduler == "ddpm":
            return DDPMScheduler.from_pretrained(
                self.cfg.scheduler_dir,
                subfolder="scheduler",
            )
        elif scheduler == "ddim":
            return DDIMScheduler.from_pretrained(
                self.cfg.scheduler_dir,
                subfolder="scheduler",
            )
        elif scheduler == "dpm":
            return DPMSolverMultistepScheduler.from_pretrained(
                self.cfg.scheduler_dir,
                subfolder="scheduler",
            )


    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training

        if not self.cfg.train_guidance: # if the guidance does not require training, then it is initialized here
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # initialize SDF
        if self.cfg.initialize_shape:
            # info
            if get_device() == "cuda_0": # only report from one process
                threestudio.info("Initializing shape...")
            
            # check if attribute exists
            if not hasattr(self.geometry, "initialize_shape"):
                threestudio.info("Geometry does not have initialize_shape method. skip.")
            else:
                self.geometry.initialize_shape()



    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        # for gradient accumulation
        opt = self.optimizers()
        opt.zero_grad()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        # for gradient accumulation
        # update the weights with the remaining gradients
        opt = self.optimizers()
        try:
            opt.step()
            opt.zero_grad()
        except:
            pass


    def forward_rendering(
        self,
        batch: Dict[str, Any],
    ):

        assert not self.cfg.rgb_as_latents, "rgb_as_latents is not supported for single renderer"

        render_out = self.renderer(**batch)

        if isinstance(render_out, tuple):
            render_out, render_out_2nd = render_out
        else:
            render_out_2nd = None

        return render_out, render_out_2nd
    
    def compute_guidance_n_loss(
        self,
        out: Dict[str, Any],
        out_2nd: Dict[str, Any],
        idx: int,
        **batch,
    ) -> Dict[str, Any]:
        # guidance for the first renderer
        guidance_rgb = out["comp_rgb"]

        # specify the timestep range for the guidance
        timestep_range = None

        # collect the guidance
        if "prompt_utils" not in batch:
            batch["prompt_utils"] = batch["guidance_utils"]

        # the guidance is computed in two steps
        guidance_out = self.guidance(
            guidance_rgb, 
            normal=out["comp_normal_cam_vis"] if "comp_normal_cam_vis" in out else None,
            depth=out["disparity"] if "disparity" in out else None,
            **batch, 
            rgb_as_latents=self.cfg.rgb_as_latents,
            timestep_range=timestep_range,
        )

        loss_dict = self._compute_loss(guidance_out, out, renderer="1st", step = idx, has_grad = guidance_rgb.requires_grad, **batch)

        return {
            "fidelity_loss": loss_dict["fidelity_loss"],
            "regularization_loss": loss_dict["regularization_loss"],            
        }

    def _set_timesteps(
        self,
        scheduler,
        num_steps: int,
    ):
        scheduler.set_timesteps(num_steps)
        timesteps_orig = scheduler.timesteps
        timesteps_delta = scheduler.config.num_train_timesteps - 1 - timesteps_orig.max() 
        timesteps = timesteps_orig + timesteps_delta
        return timesteps


    def diffusion_reverse(
        self,
        batch: Dict[str, Any],
    ):
        prompt_utils = batch["condition_utils"] if "condition_utils" in batch else batch["prompt_utils"]
        if "prompt_target" in batch:
           raise NotImplementedError
        else:
            # more general case
            text_embed_cond = prompt_utils.get_global_text_embeddings()
            text_embed_uncond = prompt_utils.get_uncond_text_embeddings()
        
        if None: # This condition seems to be always False, keeping original logic
            text_embed = torch.cat(
                [
                    text_embed_cond,
                    text_embed_uncond,
                ],
                dim=0,
            )
        else:
            text_embed = text_embed_cond

        timesteps = self._set_timesteps(
            self.sample_scheduler,
            self.cfg.num_steps_sampling,
        )

        latents = batch.pop("noise")

        for i, t in enumerate(timesteps):

            # prepare inputs
            noisy_latent_input = self.sample_scheduler.scale_model_input(
                latents, 
                t
            )
            # predict the noise added
            pred = self.geometry.denoise(
                noisy_input = noisy_latent_input,
                text_embed = text_embed, # TODO: text_embed might be null
                timestep = t.to(self.device),
            )
            results = self.sample_scheduler.step(pred, t, latents)
            latents = results.prev_sample
            latents_denoised = results.pred_original_sample

        # decode the latent to 3D representation
        space_cache_decoded_raw = self.geometry.decode(
            latents = latents_denoised,
        )
        space_cache_parsed = self.geometry.parse(space_cache_decoded_raw, scale_factor = 1.0) # Assuming scale_factor=1.0 for val/test
        return space_cache_parsed, space_cache_decoded_raw

    def training_step(
        self,
        batch_list: List[Dict[str, Any]],
        batch_idx
    ):
        if self.cfg.training_type == "progressive-rendering-distillation":
            return self._training_step_progressive_rendering_distillation(batch_list, batch_idx)
        elif self.cfg.training_type == "rollout-rendering-distillation":
            return self._training_step_rollout_rendering_distillation(batch_list, batch_idx)
        elif self.cfg.training_type == "rollout-rendering-distillation-last-step":
            return self._training_step_rollout_rendering_distillation(batch_list, batch_idx, only_last_step = True)
        else:
            raise ValueError(f"Training type {self.cfg.training_type} not supported")

    def _fake_gradient(self, module):
        loss = 0
        for param in module.parameters():
            if param.requires_grad:
                loss += 0.0 * param.sum()
        return loss

    def _training_step_rollout_rendering_distillation(
        self,
        batch_list: List[Dict[str, Any]],
        batch_idx,
        only_last_step = False,
    ):
        """
            Diffusion Forward Process
            but supervised by the 2D guidance
        """

        all_timesteps = self._set_timesteps(
            self.noise_scheduler,
            self.cfg.num_steps_training,
        )

        timesteps = sample_timesteps(
            all_timesteps,
            num_parts = self.cfg.num_parts_training, 
            batch_size=1, #batch_size,
        )

        # zero the gradients
        opt = self.optimizers()

        # load the coefficients to the GPU
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
    
        cond_trajectory = []
        _noisy_latents_input_trajectory = []
        gradient_trajectory = []
        # _denoised_latents_trajectory = [] # for DEBUG
        # _noise_pred_trajectory = [] # for DEBUG


        # the starting latent
        if self.is_training_sde:
            _denoised_latent = batch_list[0]["noise"]
        else:
            _latent = batch_list[0]["noise"]

        # rollout the denoising process
        for i, (t, batch) in enumerate(zip(timesteps, batch_list)):
            # prepare the text embeddings as input
            prompt_utils = batch["condition_utils"] if "condition_utils" in batch else batch["prompt_utils"]
            if "prompt_target" in batch:
                raise NotImplementedError
            else:
                # more general case
                cond = prompt_utils.get_global_text_embeddings()
                uncond = prompt_utils.get_uncond_text_embeddings()
                batch["text_embed_bg"] = prompt_utils.get_global_text_embeddings(use_local_text_embeddings = False)
                batch["text_embed"] = cond
            
            text_embed = cond
            cond_trajectory.append(text_embed)
            
            # record the latent
            with torch.no_grad():

                # prepare the input for the denoiser
                if self.is_training_sde:
                    _noisy_latent_input = self.noise_scheduler.add_noise(
                        _denoised_latent,
                        batch_list[i]["noise"],
                        t
                    ) if i > 0 else batch_list[i]["noise"]
                else:
                    _noisy_latent_input = self.noise_scheduler.scale_model_input(
                        _latent, 
                        t
                    )
                _noisy_latents_input_trajectory.append(_noisy_latent_input)
                
                # predict the noise added
                _noise_pred = self.geometry.denoise(
                    noisy_input = _noisy_latent_input,
                    text_embed = text_embed, # TODO: text_embed might be null
                    timestep = t.to(self.device),
                )
                # _noise_pred_trajectory.append(_noise_pred) # for DEBUG
                results = self.noise_scheduler.step(
                    _noise_pred, 
                    t.to(self.device), 
                     _noisy_latent_input
                )
                _denoised_latent = results.pred_original_sample
                _latent = results.prev_sample

                # _denoised_latents_trajectory.append(_denoised_latent) # for DEBUG

            if only_last_step and i < len(timesteps) - 1:
                continue
            else:
                latent_var = Variable(_denoised_latent.detach(), requires_grad=True)
                # decode the latent to 3D representation
                space_cache = self.geometry.decode(
                    latents = latent_var,
                )

                space_cache_var = Variable(space_cache, requires_grad=True)
                
                # during the rollout, we can compute the gradient of the space cache and store it
                space_cache_var_parsed = self.geometry.parse(space_cache_var, scale_factor = self.scale_factor_list[i].item())
                batch["space_cache"] = space_cache_var_parsed
                    
                # render the image and compute the gradients
                out, out_2nd = self.forward_rendering(batch)
                loss_dict = self.compute_guidance_n_loss(
                    out, out_2nd, idx = i, **batch
                )
                fidelity_loss = loss_dict["fidelity_loss"]
                regularization_loss = loss_dict["regularization_loss"]

                # # check the gradients for DEBUG
                # self._check_trainable_params(opt_other)
                # self._check_trainable_params(opt_multi_step)

                # store gradients
                loss_render = (
                    fidelity_loss + regularization_loss
                )  / self.cfg.gradient_accumulation_steps / (1 if only_last_step else self.cfg.num_steps_training)

                self.manual_backward(loss_render)
                
                # 计算位置牵引损失和缩放平滑损失
                loss_reg = self._compute_loss_post_grad(
                    space_cache_parsed = self.geometry.parse(space_cache, scale_factor = self.scale_factor_list[i].item()),
                    space_cache_parsed_grad = self.geometry.parse(space_cache_var.grad, scale_factor = self.scale_factor_list[i].item()),
                    step = i
                )
                # 继续原来的前向传播和梯度传递
                loss_dec = SpecifyGradient.apply(
                    space_cache,
                    space_cache_var.grad  # 使用原始梯度或修改后的梯度
                )
                loss_fake = self._fake_gradient(self.geometry.space_generator)
                self.manual_backward(loss_dec + 0 * loss_fake + loss_reg)
                
                gradient_trajectory.append(latent_var.grad)
                
                # # check the gradient
                # for name, param in self.geometry.space_generator.vae.named_parameters():
                #     # if the param requires grad but not in the gradient trajectory, then print it
                #     if param.requires_grad and param.grad is None:
                #         print(f"Parameter {name} requires grad but not in the gradient trajectory")
                
        B = space_cache[0].shape[0]  if isinstance(space_cache, list) else space_cache.shape[0]
        # the rollout is done, now we can compute the gradient of the denoised latents
        noise_pred_batch = self.geometry.denoise(
            noisy_input =  torch.cat(_noisy_latents_input_trajectory, dim=0),
            text_embed = torch.cat(cond_trajectory, dim=0),
            timestep = torch.cat(timesteps, dim=0).repeat_interleave(B).to(self.device)
        )

        # iterative over the denoised latents
        if self.is_training_sde:
            denoised_latent = batch_list[0]["noise"]
        else:
            latent = batch_list[0]["noise"]
        
        denoised_latent_batch = []

        for i, (
            noise_pred, 
            t,
            # _noise_pred, # for DEBUG
            # _denoised_latent, # for DEBUG
        ) in enumerate(
            zip(
                noise_pred_batch.chunk(self.cfg.num_parts_training), 
                timesteps,
                # _noise_pred_trajectory, # for DEBUG
                # _denoised_latents_trajectory, # for DEBUG
            )
        ):

            # print(
            #     "\nStep:", i
            # )
            # print(
            #     "noise_pred_gap:",
            #     (noise_pred - _noise_pred).norm().item()
            # )
            # predict the noise added

            if self.is_training_sde:
                noisy_latent_input = self.noise_scheduler.add_noise(
                    denoised_latent,
                    batch_list[i]["noise"],
                    t
                ) if i > 0 else batch_list[i]["noise"]
            else:
                noisy_latent_input = self.noise_scheduler.scale_model_input(
                    latent,
                    t
                )
            results = self.noise_scheduler.step(
                noise_pred, 
                t.to(self.device), 
                noisy_latent_input
            )
            latent = results.prev_sample # do not detach here, we want to keep the gradient
            denoised_latent = results.pred_original_sample
            # print(
            #     "denoised_latent_gap:",
            #     (denoised_latent - _denoised_latent).norm().item()
            # )

            # record the denoised latent
            if only_last_step and i < len(timesteps) - 1:
                continue
            else:
                denoised_latent_batch.append(denoised_latent)

        # Removed gradient filtering logic

        loss_gen = SpecifyGradient.apply(
            torch.cat(denoised_latent_batch, dim=0),
            torch.cat(gradient_trajectory, dim=0)
        )

        # because self.manual_backward() will not work if the decoder, renderer, or background has no grad
        loss_fake = self._fake_gradient(self.geometry) + self._fake_gradient(self.background)
        # Restored original backward call
        self.manual_backward(loss_gen / self.cfg.gradient_accumulation_steps / (1 if only_last_step else self.cfg.num_steps_training) + 0 * loss_fake)

        # update the weights
        if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
            opt.step()
            opt.zero_grad()



    def _training_step_progressive_rendering_distillation(
        self,
        batch_list: List[Dict[str, Any]],
        batch_idx
    ):
        """
            Diffusion Forward Process
            but supervised by the 2D guidance
        """
        latent = batch_list[0]["noise"]

        all_timesteps = self._set_timesteps(
            self.noise_scheduler,
            self.cfg.num_steps_training,
        )

        timesteps = sample_timesteps(
            all_timesteps,
            num_parts = self.cfg.num_parts_training, 
            batch_size=1, #batch_size,
        )

        # zero the gradients
        opt = self.optimizers()

        # load the coefficients to the GPU
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
    
        for i, (t, batch) in enumerate(zip(timesteps, batch_list)):

            # prepare the text embeddings as input
            prompt_utils = batch["condition_utils"] if "condition_utils" in batch else batch["prompt_utils"]
            if "prompt_target" in batch:
                raise NotImplementedError
            else:
                # more general case
                cond = prompt_utils.get_global_text_embeddings()
                uncond = prompt_utils.get_uncond_text_embeddings()
                batch["text_embed_bg"] = prompt_utils.get_global_text_embeddings(use_local_text_embeddings = False)
                batch["text_embed"] = cond

            # choose the noise to be added
            noise = torch.randn_like(latent)

            # add noise to the latent
            noisy_latent = self.noise_scheduler.add_noise(
                latent,
                noise,
                t,
            )
 
            # prepare the text embeddings as input
            text_embed = cond
            # if torch.rand(1) < 0: 
            #     text_embed = uncond

            noise_pred = self.geometry.denoise(
                noisy_input = noisy_latent,
                text_embed = text_embed, # TODO: text_embed might be null
                timestep = t.to(self.device),
            )

            denoised_latents = self.noise_scheduler.step(
                noise_pred, 
                t.to(self.device), 
                noisy_latent
            ).pred_original_sample


            # decode the latent to 3D representation
            space_cache = self.geometry.decode(
                latents = denoised_latents,
            )

            batch["space_cache"] = self.geometry.parse(space_cache, scale_factor = self.scale_factor_list[i].item())

            # render the image and compute the gradients
            out, out_2nd = self.forward_rendering(batch)
            loss_dict = self.compute_guidance_n_loss(
                out, out_2nd, idx = i, **batch
            )
            fidelity_loss = loss_dict["fidelity_loss"]
            regularization_loss = loss_dict["regularization_loss"]


            weight_fide = 1.0 / self.cfg.num_parts_training
            weight_regu = 1.0 / self.cfg.num_parts_training

            loss = weight_fide * fidelity_loss + weight_regu * regularization_loss
            self.manual_backward(loss / self.cfg.gradient_accumulation_steps)
            
            # prepare for the next iteration
            latent = denoised_latents.detach()
            
        # update the weights
        if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
            opt.step()
            opt.zero_grad()

    def validation_step(self, batch, batch_idx):

        # prepare the text embeddings as input
        prompt_utils = batch["condition_utils"] if "condition_utils" in batch else batch["prompt_utils"]
        if "prompt_target" in batch:
            raise NotImplementedError
        else:
            # more general case
            batch["text_embed"] = prompt_utils.get_global_text_embeddings()
            batch["text_embed_bg"] = prompt_utils.get_global_text_embeddings(use_local_text_embeddings = False)
    
        # Call diffusion_reverse and get both parsed and raw decoded outputs
        space_cache_parsed_val, raw_decoded_features_val = self.diffusion_reverse(batch)
        batch["space_cache"] = space_cache_parsed_val
        
        out, out_2nd = self.forward_rendering(batch)

        batch_size = out['comp_rgb'].shape[0]
        for batch_idx in tqdm(range(batch_size), desc="Saving val images"):
            self._save_image_grid(batch, batch_idx, out, phase="val", render="1st")
            if out_2nd:
                self._save_image_grid(batch, batch_idx, out_2nd, phase="val", render="2nd")
        
        # --- Save Gaussian attribute images for the first item in the batch ---
        if hasattr(self.geometry, "export_gaussian_attributes_as_images"):
            # Assuming raw_decoded_features_val might contain features for the whole batch,
            # export_gaussian_attributes_as_images should ideally handle this or be designed
            # to extract for a single item if needed, or return a dict appropriate for item_idx_in_batch=0.
            attribute_images_dict_val = self.geometry.export_gaussian_attributes_as_images(raw_decoded_features_val)
            if attribute_images_dict_val: # Ensure dict is not empty
                save_attribute_visualization_grid(
                    system_object=self,
                    batch=batch, # Pass the whole batch
                    item_idx_in_batch=0, # Focus on the first item
                    attribute_images_dict=attribute_images_dict_val,
                    phase="val",
                    debug=False, # Or self.cfg.debug or a dedicated debug flag
                )

        # --- End save Gaussian attribute images ---
        
        if self.cfg.visualize_samples:
            raise NotImplementedError

    def test_step(self, batch, batch_idx, return_space_cache = False, render_images = True):

        # prepare the text embeddings as input
        prompt_utils = batch["condition_utils"] if "condition_utils" in batch else batch["prompt_utils"]
        if "prompt_target" in batch:
            raise NotImplementedError
        else:
            # more general case
            batch["text_embed"] = prompt_utils.get_global_text_embeddings()
            batch["text_embed_bg"] = prompt_utils.get_global_text_embeddings(use_local_text_embeddings = False)
    
        # Call diffusion_reverse and get both parsed and raw decoded outputs
        space_cache_parsed_test, raw_decoded_features_test = self.diffusion_reverse(batch)
        batch["space_cache"] = space_cache_parsed_test

        if render_images:
            out, out_2nd = self.forward_rendering(batch)
            batch_size = out['comp_rgb'].shape[0]

            for batch_idx in tqdm(range(batch_size), desc="Saving test images"):
                self._save_image_grid(batch, batch_idx, out, phase="test", render="1st")
                if out_2nd:
                    self._save_image_grid(batch, batch_idx, out_2nd, phase="test", render="2nd")

        # --- Save Gaussian attribute images for the first item in the batch ---
        if render_images and hasattr(self.geometry, "export_gaussian_attributes_as_images"):
            attribute_images_dict_test = self.geometry.export_gaussian_attributes_as_images(raw_decoded_features_test)
            if attribute_images_dict_test:
                save_attribute_visualization_grid(
                    system_object=self,
                    batch=batch,
                    item_idx_in_batch=0,
                    attribute_images_dict=attribute_images_dict_test,
                    phase="test",
                    debug=False, # Or self.cfg.debug
                )

        # --- End save Gaussian attribute images ---
        
        if return_space_cache:
            return batch["space_cache"]


    def _compute_loss_post_grad(
        self,
        space_cache_parsed,
        space_cache_parsed_grad,
        step: int = 0,
    ):
        loss_sum = 0
        # 位置牵引损失
        loss_pos_pull = 0
        if hasattr(self.cfg.loss, "lambda_position_pull") and self.C(self.cfg.loss.lambda_position_pull) > 0:

            loss_pos_pull = position_pull_loss(
                position = space_cache_parsed["position"], # 当前点位置
                position_grad = space_cache_parsed_grad["position"],  # 位置梯度
                opacity = space_cache_parsed["opacity"]  # 当前点不透明度
            ) 
            self.log(f"train/loss_position_pull_{step}", loss_pos_pull)
            loss_pos_pull = loss_pos_pull * self.C(self.cfg.loss.lambda_position_pull)

        loss_scale_smooth = 0
        if hasattr(self.cfg.loss, "lambda_scale_smooth") and self.C(self.cfg.loss.lambda_scale_smooth) > 0:
            loss_scale_smooth = scale_smooth_loss(
                position = space_cache_parsed["position"], # 当前点位置
                position_grad = space_cache_parsed_grad["position"],  # 位置梯度
                scale = space_cache_parsed["scale"]  # 当前点缩放
            )
            self.log(f"train/loss_scale_smooth_{step}", loss_scale_smooth)
            loss_scale_smooth = loss_scale_smooth * self.C(self.cfg.loss.lambda_scale_smooth)

        loss_sum += loss_pos_pull + loss_scale_smooth
        return loss_sum


    def _compute_loss(
        self,
        guidance_out: Dict[str, Any],
        out: Dict[str, Any],
        renderer: str = "1st",
        step: int = 0,
        has_grad: bool = True,
        **batch,
    ):
        
        assert renderer in ["1st", "2nd"]

        fide_loss = 0.0
        regu_loss = 0.0
        for name, value in guidance_out.items():
            if renderer == "1st":
                self.log(f"train/{name}_{step}", value)
                if name.startswith("loss_"):
                    fide_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
            else:
                self.log(f"train/{name}_2nd_{step}", value)
                if name.startswith("loss_"):
                    fide_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_") + "_2nd"])

        # position loss #########################################################
        if renderer == "1st" and hasattr(self.cfg.loss, "lambda_position") and self.C(self.cfg.loss.lambda_position) != 0:
            xyz_mean = torch.stack([batch_item["gs_xyz"] for batch_item in batch["space_cache"]]).mean().abs()
            self.log(f"train/loss_position_{step}", xyz_mean)
            regu_loss += self.C(self.cfg.loss.lambda_position) * xyz_mean
        if renderer == "2nd" and hasattr(self.cfg.loss, "lambda_position_2nd") and self.C(self.cfg.loss.lambda_position_2nd) != 0:
            xyz_mean = torch.stack([batch_item["gs_xyz"] for batch_item in batch["space_cache"]]).mean().abs()
            self.log(f"train/loss_position_2nd_{step}", xyz_mean)
            regu_loss += self.C(self.cfg.loss.lambda_position_2nd) * xyz_mean

        # scale loss #########################################################
        if renderer == "1st" and hasattr(self.cfg.loss, "lambda_scales") and self.C(self.cfg.loss.lambda_scales) != 0:
            scale_sum = torch.stack([batch_item["gs_scale"] for batch_item in batch["space_cache"]]).mean()
            self.log(f"train/loss_scales_{step}", scale_sum)
            regu_loss += self.C(self.cfg.loss.lambda_scales) * scale_sum
        if renderer == "2nd" and hasattr(self.cfg.loss, "lambda_scales_2nd") and self.C(self.cfg.loss.lambda_scales_2nd) != 0:
            scale_sum = torch.stack([batch_item["gs_scale"] for batch_item in batch["space_cache"]]).mean()
            self.log(f"train/loss_scales_2nd_{step}", scale_sum)
            regu_loss += self.C(self.cfg.loss.lambda_scales_2nd) * scale_sum
        
        # sdf points loss #########################################################
        if renderer == "1st" and hasattr(self.cfg.loss, "lambda_sdf_points") and self.C(self.cfg.loss.lambda_sdf_points) != 0:
            mean_sdf = torch.norm(out['sdf_points'], p=2, dim=-1).mean()
            self.log(f"train/loss_sdf_points_{step}", mean_sdf)
            regu_loss += mean_sdf * self.C(self.cfg.loss.lambda_sdf_points)
        if renderer == "2nd" and hasattr(self.cfg.loss, "lambda_sdf_points_2nd") and self.C(self.cfg.loss.lambda_sdf_points_2nd) != 0:
            mean_sdf = torch.norm(out['sdf_points'], p=2, dim=-1).mean()
            self.log(f"train/loss_sdf_points_2nd_{step}", mean_sdf)
            regu_loss += mean_sdf * self.C(self.cfg.loss.lambda_sdf_points_2nd)

        # sdf eikonal loss #########################################################
        if (renderer == "1st" and self.C(self.cfg.loss.lambda_eikonal) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_eikonal_2nd) > 0):
            if 'sdf_grad' not in out:
                raise ValueError(
                    "sdf is required for eikonal loss, no sdf is found in the output."
                )
            
            if isinstance(out["sdf_grad"], torch.Tensor):
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
            else:
                loss_eikonal = 0
                for sdf_grad in out["sdf_grad"]:
                    loss_eikonal += (
                        (torch.linalg.norm(sdf_grad, ord=2, dim=-1) - 1.0) ** 2
                    ).mean()
                loss_eikonal /= len(out["sdf_grad"])

            if renderer == "1st":
                self.log(f"train/loss_eikonal_{step}", loss_eikonal)
                regu_loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)
            else:
                self.log(f"train/loss_eikonal_2nd_{step}", loss_eikonal)
                regu_loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal_2nd)

        # sparsity loss #########################################################
        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        if renderer == "1st":
            self.log(f"train/loss_sparsity_{step}", loss_sparsity, prog_bar=False if step % self.cfg.num_steps_training != self.cfg.num_steps_training - 1 else True)
            if hasattr(self.cfg.loss, "lambda_sparsity") and self.C(self.cfg.loss.lambda_sparsity) != 0:
                regu_loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
        else:
            self.log(f"train/loss_sparsity_2nd_{step}", loss_sparsity, prog_bar=False if step % self.cfg.num_steps_training != self.cfg.num_steps_training - 1 else True)
            if hasattr(self.cfg.loss, "lambda_sparsity_2nd") and self.C(self.cfg.loss.lambda_sparsity_2nd) != 0:
                regu_loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity_2nd)

        # opacity loss #########################################################
        if renderer == "1st" and hasattr(self.cfg.loss, "lambda_opaque") and self.C(self.cfg.loss.lambda_opaque) != 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log(f"train/loss_opaque_{step}", loss_opaque)
            regu_loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        if renderer == "2nd" and hasattr(self.cfg.loss, "lambda_opaque_2nd") and self.C(self.cfg.loss.lambda_opaque_2nd) != 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log(f"train/loss_opaque_2nd_{step}", loss_opaque)
            regu_loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque_2nd)

        # detach the loss if necessary
        if not has_grad:
            if hasattr(fide_loss, "requires_grad") and fide_loss.requires_grad:
                fide_loss = fide_loss.detach()
                
        if not has_grad:
            if hasattr(regu_loss, "requires_grad") and regu_loss.requires_grad:
                regu_loss = regu_loss.detach()

        return {
            "fidelity_loss": fide_loss,
            "regularization_loss": regu_loss,
        }


    def _save_image_grid(
        self, 
        batch,
        batch_idx,
        out,
        phase="val",
        render="1st",
    ):
        
        assert phase in ["val", "test"]

        # save the image with the same name as the prompt
        if "name" in batch:
            name = batch['name'][0].replace(',', '').replace('.', '').replace(' ', '_')
        else:
            name = batch['prompt'][0].replace(',', '').replace('.', '').replace(' ', '_')
        # specify the image name
        image_name  = f"it{self.true_global_step}-{phase}-{render}/{name}/{str(batch['index'][batch_idx].item())}.png"
        # specify the verbose name
        verbose_name = f"{phase}_{render}_step"

        self.save_image_grid(
            image_name,
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][batch_idx] if not self.cfg.rgb_as_latents else out["decoded_rgb"][batch_idx],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_cam_vis_white"][batch_idx] if "comp_normal_cam_vis_white" in out else out["comp_normal"][batch_idx],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out and out["comp_normal"] is not None
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        # Pass adaptively calculated near/far to visualize_center_depth
                        "img": visualize_center_depth(
                            out["comp_center_point_depth"][batch_idx, :, :, 0],
                            near=None, #calculated_near, # Pass calculated or None
                            far=None, # far=calculated_far   # Pass calculated or None
                        ),
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "comp_center_point_depth" in out and out["comp_center_point_depth"] is not None
                else [ # Fallback: show opacity if alpha blend depth not available
                    {
                        "type": "grayscale",
                        "img": out["opacity"][batch_idx, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
            )
            + (
                [
                    {
                        "type": "grayscale",
                        # Visualize positive depth using the same function as center_point_depth
                        "img": visualize_center_depth(
                            out["depth"][batch_idx, :, :, 0], # Pass positive depth directly
                            near=None,
                            far=None
                        ),
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "depth" in out
                else []
            ),
            name=verbose_name,
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        barrier() # wait until all GPUs finish rendering images
        filestems = [
            f"it{self.true_global_step}-val-{render}"
            for render in ["1st", "2nd"] # Include "2nd"
        ]
        if get_rank() == 0: # only remove from one process
            for filestem in filestems:
                dir_name = os.path.join(self.get_save_dir(), filestem)
                if os.path.exists(dir_name):
                    files = os.listdir(dir_name)
                    files = [f for f in files if os.path.isdir(os.path.join(dir_name, f))]
                    for prompt in tqdm(
                        files,
                            desc="Generating validation videos",
                        ):
                            try:
                                self.save_img_sequence(
                                    os.path.join(filestem, prompt),
                                    os.path.join(filestem, prompt),
                                    "(\d+)\.png",
                                    save_format="mp4",
                                    fps=10,
                                    name="validation_epoch_end",
                                    step=self.true_global_step,
                                    multithreaded=True,
                                )
                            except Exception as e:
                                self.save_img_sequence(
                                    os.path.join(filestem, prompt),
                                    os.path.join(filestem, prompt),
                                    "(\d+)\.png",
                                    save_format="mp4",
                                    fps=10,
                                    name="validation_epoch_end",
                                    step=self.true_global_step,
                                    # multithreaded=True,
                                )

    def on_test_epoch_end(self):
        barrier() # wait until all GPUs finish rendering images
        filestems = [
            f"it{self.true_global_step}-test-{render}"
            for render in ["1st", "2nd"]
        ]
        if get_rank() == 0: # only remove from one process
            for filestem in filestems:
                dir_name = os.path.join(self.get_save_dir(), filestem)
                if os.path.exists(dir_name):
                    files = os.listdir(dir_name)
                    files = [f for f in files if os.path.isdir(os.path.join(dir_name, f))]
                    for prompt in tqdm(
                        files,
                        desc="Generating validation videos",
                    ):
                        try:
                            self.save_img_sequence(
                                os.path.join(filestem, prompt),
                                os.path.join(filestem, prompt),
                                "(\d+)\.png",
                                save_format="mp4",
                                fps=30,
                                name="test",
                                step=self.true_global_step,
                                multithreaded=True,
                            )
                        except Exception as e:
                            self.save_img_sequence(
                                os.path.join(filestem, prompt),
                                os.path.join(filestem, prompt),
                                "(\d+)\.png",
                                save_format="mp4",
                                fps=10,
                                name="validation_epoch_end",
                                step=self.true_global_step,
                                # multithreaded=True,
                            )


    def on_predict_start(self) -> None:
        self.exporter: Exporter = threestudio.find(self.cfg.exporter_type)(
            self.cfg.exporter,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )

    def predict_step(self, batch, batch_idx):
        space_cache = self.test_step(batch, batch_idx, render_images=self.exporter.cfg.save_video, return_space_cache=True)
        # update the space_cache into the exporter
        exporter_output: List[ExporterOutput] = self.exporter(space_cache)

        # specify the name
        if "name" in batch:
            name = batch['name'][0].replace(',', '').replace('.', '').replace(' ', '_')
        else:
            name = batch['prompt'][0].replace(',', '').replace('.', '').replace(' ', '_')

        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            if not hasattr(self, save_func_name):
                raise ValueError(f"{save_func_name} not supported by the SaverMixin")
            save_func = getattr(self, save_func_name)
            save_func(f"it{self.true_global_step}-export/{name}/{out.save_name}", **out.params)

    def on_predict_epoch_end(self) -> None:
        if self.exporter.cfg.save_video:
            self.on_test_epoch_end()


    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        self.min_scale_factor = C(
            self.cfg.min_scale_factor, epoch, global_step
        )

        # given the self.cfg.num_steps_training, linearly interpolate the scale_factor from self.min_scale_factor to 1.0 for each step
        self.scale_factor_list = torch.linspace(self.min_scale_factor, 1.0, self.cfg.num_steps_training)

    # def on_test_epoch_start(self) -> None:
    #     # save state_dict
    #     state_dict = self.geometry.state_dict() # <class 'collections.OrderedDict'>
    #     save_state_dict = {}
    #     # save a compact state_dict to save space
    #     for key, value in state_dict.items():
    #         # add "geometry." to the key
    #         new_key = "geometry." + key
    #         if "peft_layers" in key:
    #             save_state_dict[new_key] = value
    #             continue

    #         if key == "bbox":
    #             save_state_dict[new_key] = value
    #             continue

    #         if "feature_network" in key:
    #             save_state_dict[new_key] = value
    #             continue

    #         if "sdf_network" in key:
    #             save_state_dict[new_key] = value
    #             continue

    #         if "deformation_network" in key:
    #             save_state_dict[new_key] = value
    #             continue

    #     save_dict = {}
    #     # add other parameters
    #     save_dict["epoch"] = 0
    #     save_dict["global_step"] = 0
    #     save_dict["pytorch-lightning_version"] = '2.0.0'
    #     save_dict["state_dict"] = save_state_dict
    #     torch.save(save_dict, "pretrained/triplane_turbo_sd_v1.pth")
    #     super().on_test_epoch_start()

