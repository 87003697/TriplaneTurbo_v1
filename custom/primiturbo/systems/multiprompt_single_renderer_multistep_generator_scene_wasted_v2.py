import os
import shutil
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_rank, get_device
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

def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]

def tv_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

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

@threestudio.register("multiprompt-single-renderer-multistep-generator-scene-system")
class MultipromptSingleRendererMultiStepGeneratorSceneSystem(BaseLift3DSystem):
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

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        if self.cfg.train_guidance: # if the guidance requires training, then it is initialized here
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # Sampler for training
        self.noise_scheduler = self._configure_scheduler(self.cfg.noise_scheduler)
        self.is_training_odd = True if self.cfg.noise_scheduler == "ddpm" else False

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

        render_out = self.renderer(batch)

        # decode the rgb as latents only in testing and validation
        if self.cfg.rgb_as_latents and not self.training: 
            # get the rgb
            if "comp_rgb" not in render_out:
                raise ValueError(
                    "comp_rgb is required for rgb_as_latents, no comp_rgb is found in the output."
                )
            else:
                out_image = render_out["comp_rgb"]
                out_image = self.guidance.decode_latents(
                    out_image.permute(0, 3, 1, 2)
                ).permute(0, 2, 3, 1) 
                render_out['decoded_rgb'] = out_image


        return render_out, {}
    
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
        
        if None:
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

            if None:
                noisy_latent_input = torch.cat([noisy_latent_input] * 2, dim=0)


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
        space_cache = self.geometry.decode(
            latents = latents_denoised,
        )
        space_cache_parsed = self.geometry.parse(space_cache)

        return space_cache_parsed
    
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
    
        noisy_latents_input_trajectory = []
        noise_trajectory = []
        gradient_trajectory = []
        _latent_trajectory = []
        _denoised_latents_trajectory = []
        _noise_pred_trajectory = []
        


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

            # choose the noise to be added
            noise = torch.randn_like(latent)
            noise_trajectory.append(noise)

            with torch.no_grad():
                _latent_trajectory.append(latent)
                # add noise to the latent
                noisy_latent = self.noise_scheduler.add_noise(
                    latent,
                    noise,
                    t,
                )
                noisy_latents_input_trajectory.append(noisy_latent)

                # prepare the input for the denoiser
                noisy_latent_input = noisy_latent #torch.cat([noisy_latent] * 2, dim=0)
                text_embed = cond

                # predict the noise added
                noise_pred = self.geometry.denoise(
                    noisy_input = noisy_latent_input,
                    text_embed = text_embed, # TODO: text_embed might be null
                    timestep = t.to(self.device),
                )
                _noise_pred_trajectory.append(noise_pred)
                denoised_latents = self.noise_scheduler.step(
                    noise_pred, 
                    t.to(self.device), 
                    noisy_latent
                ).pred_original_sample
                _denoised_latents_trajectory.append(denoised_latents)
                latent = denoised_latents # important!!!!!!

            if only_last_step and i < len(timesteps) - 1:
                # print(f"skipping the {i}-th step of denoised latents")
                continue
            else:
                latent_var = Variable(denoised_latents, requires_grad=True)
                # decode the latent to 3D representation
                space_cache = self.geometry.decode(
                    latents = latent_var,
                )
                # during the rollout, we can compute the gradient of the space cache and store it
                space_cache_parsed = self.geometry.parse(space_cache)
                batch["space_cache"] = space_cache_parsed
                    
                # render the image and compute the gradients
                out, out_2nd = self.forward_rendering(batch)
                loss_dict = self.compute_guidance_n_loss(
                    out, out_2nd, idx = i, **batch
                )
                fidelity_loss = loss_dict["fidelity_loss"]
                regularization_loss = loss_dict["regularization_loss"]

                weight_fide = 1.0# / self.cfg.num_parts_training
                weight_regu = 1.0# / self.cfg.num_parts_training

                # loss = weight_fide * fidelity_loss + weight_regu * regularization_loss
                # store gradients
                loss_var = (
                    weight_fide * fidelity_loss + weight_regu * regularization_loss
                )  / self.cfg.gradient_accumulation_steps
                loss_var.backward()
                gradient_trajectory.append(latent_var.grad)
                
        # the rollout is done, now we can compute the gradient of the denoised latents
        noise_pred_batch = self.geometry.denoise(
            noisy_input =  torch.cat(noisy_latents_input_trajectory, dim=0),
            text_embed = text_embed.repeat(self.cfg.num_parts_training, 1, 1),
            timestep = torch.cat(timesteps, dim=0).repeat_interleave(space_cache.shape[0]).to(self.device)
        )

        # iterative over the denoised latents
        latent_with_grad = batch_list[0]["noise"]
        latent_batch = []
        for i, (
                noise_pred_with_grad, t, noise, 
                _latent, _noisy_latent, _noise_pred, _denoised_latent, 
            ) in enumerate(
                zip(
                    noise_pred_batch.chunk(self.cfg.num_parts_training), 
                    timesteps, 
                    noise_trajectory, 
                    _latent_trajectory,
                    noisy_latents_input_trajectory,
                    _noise_pred_trajectory,
                    _denoised_latents_trajectory,
                )
            ):
            # compute the gradient of the denoised latent
            noisy_latent_with_grad = self.noise_scheduler.add_noise(
                latent_with_grad,
                noise,
                t,
            )
            denoised_latent_with_grad = self.noise_scheduler.step(
                noise_pred_with_grad, 
                t.to(self.device), 
                noisy_latent_with_grad
            ).pred_original_sample
            # print(
            #     "\nStep:", i
            # )
            # print(
            #     "latent_gap:", 
            #     (latent_with_grad - _latent).norm().item()
            # )
            # print(
            #     "noisy_latent_gap:",
            #     (noisy_latent_with_grad - _noisy_latent).norm().item()
            # )
            # print(
            #     "noise_pred_gap:",
            #     (noise_pred_with_grad - _noise_pred).norm().item()
            # )
            # print(
            #     "denoised_latent_gap:",
            #     (denoised_latent_with_grad - _denoised_latent).norm().item()
            # )

            latent_with_grad = denoised_latent_with_grad # do not detach here, we want to keep the gradient
            if only_last_step and i < len(timesteps) - 1:
                # print(f"skipping the {i}-th step of denoised latents")
                continue
            else:
                latent_batch.append(denoised_latent_with_grad)

        loss = SpecifyGradient.apply(
            torch.cat(latent_batch, dim=0),
            torch.cat(gradient_trajectory, dim=0)
        )
        loss.backward()

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

            # batch["space_cache"] = self.geometry.parse(space_cache)
            space_cache_var = Variable(space_cache, requires_grad=True)
            space_cache_parsed = self.geometry.parse(space_cache_var)
            batch["space_cache"] = space_cache_parsed

            # render the image and compute the gradients
            out, out_2nd = self.forward_rendering(batch)
            loss_dict = self.compute_guidance_n_loss(
                out, out_2nd, idx = i, **batch
            )
            fidelity_loss = loss_dict["fidelity_loss"]
            regularization_loss = loss_dict["regularization_loss"]


            weight_fide = 1.0 / self.cfg.num_parts_training
            weight_regu = 1.0 / self.cfg.num_parts_training

            # loss = weight_fide * fidelity_loss + weight_regu * regularization_loss
            # store gradients
            loss_var = weight_fide * fidelity_loss + weight_regu * regularization_loss
            loss_var.backward()
            loss = SpecifyGradient.apply(
                space_cache,
                space_cache_var.grad
            )
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
    
        batch["space_cache"]  = self.diffusion_reverse(batch)
        out, out_2nd = self.forward_rendering(batch)

        batch_size = out['comp_rgb'].shape[0]

        for batch_idx in tqdm(range(batch_size), desc="Saving val images"):
            self._save_image_grid(batch, batch_idx, out, phase="val", render="1st")
            # self._save_image_grid(batch, batch_idx, out_2nd, phase="val", render="2nd")
                
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
    
            batch["space_cache"] = self.diffusion_reverse(batch)

        if render_images:
            out, out_2nd = self.forward_rendering(batch)
            batch_size = out['comp_rgb'].shape[0]

            for batch_idx in tqdm(range(batch_size), desc="Saving test images"):
                self._save_image_grid(batch, batch_idx, out, phase="test", render="1st")
                # self._save_image_grid(batch, batch_idx, out_2nd, phase="test", render="2nd")

        if return_space_cache:
            return batch["space_cache"]


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

        if (renderer == "1st" and self.C(self.cfg.loss.lambda_position) != 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_position_2nd) != 0):
            xyz_mean = torch.stack(
                [
                    batch_item["gs_xyz"] for batch_item in batch["space_cache"]
                ]
            ).mean().abs()
            if renderer == "1st":
                self.log(f"train/loss_position_{step}", xyz_mean)
                regu_loss += self.C(self.cfg.loss.lambda_position) * xyz_mean
            else:
                self.log(f"train/loss_position_2nd_{step}", xyz_mean)
                regu_loss += self.C(self.cfg.loss.lambda_position_2nd) * xyz_mean

        if (renderer == "1st" and self.C(self.cfg.loss.lambda_scales) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_scales_2nd) > 0):
            scale_sum = torch.stack(
                [
                    batch_item["gs_scale"] for batch_item in batch["space_cache"]
                ]
            ).mean()
            if renderer == "1st":
                self.log(f"train/scales_{step}", scale_sum)
                regu_loss += self.C(self.cfg.loss.lambda_scales) * scale_sum
            else:
                self.log(f"train/scales_2nd_{step}", scale_sum)
                regu_loss += self.C(self.cfg.loss.lambda_scales_2nd) * scale_sum


        if (renderer == "1st" and self.C(self.cfg.loss.lambda_depth_tv_loss) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_depth_tv_loss_2nd) > 0):
            loss_depth_tv = tv_loss(out["depth"].permute(0, 3, 1, 2))
            if renderer == "1st":
                self.log(f"train/loss_depth_tv_{step}", loss_depth_tv)
                regu_loss += self.C(self.cfg.loss.lambda_depth_tv_loss) * loss_depth_tv
            else:
                self.log(f"train/loss_depth_tv_2nd_{step}", loss_depth_tv)
                regu_loss += self.C(self.cfg.loss.lambda_depth_tv_loss_2nd) * loss_depth_tv
            

        if (renderer == "1st" and self.C(self.cfg.loss.lambda_sparsity) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_sparsity_2nd) > 0):
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            if renderer == "1st":
                self.log(f"train/loss_sparsity_{step}", loss_sparsity, prog_bar=False if step % 4 != 3 else True)
                regu_loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
            else:
                self.log(f"train/loss_sparsity_2nd_{step}", loss_sparsity, prog_bar=False if step % 4 != 3 else True)
                regu_loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity_2nd)


        if (renderer == "1st" and self.C(self.cfg.loss.lambda_opaque) > 0) or (renderer == "2nd" and self.C(self.cfg.loss.lambda_opaque_2nd) > 0):
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            if renderer == "1st":
                self.log(f"train/loss_opaque_{step}", loss_opaque)
                regu_loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
            else:
                self.log(f"train/loss_opaque_2nd_{step}", loss_opaque)
                regu_loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque_2nd)

        if "inv_std" in out:
            self.log("train/inv_std", out["inv_std"], prog_bar=True)

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

        # normalize the depth
        normalize = lambda x: (x - x.min()) / (x.max() - x.min())

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
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][batch_idx, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out['disparity'][batch_idx, :, :, 0] if 'disparity' in out else normalize(out["depth"][batch_idx, :, :, 0]),
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
            for render in ["1st"] # ["1st", "2nd"]
        ]
        if get_rank() == 0: # only remove from one process
            for filestem in filestems:
                files = os.listdir(os.path.join(self.get_save_dir(), filestem))
                files = [f for f in files if os.path.isdir(os.path.join(self.get_save_dir(), filestem, f))]
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
                    except:
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
            for render in ["1st"] # ["1st", "2nd"]
        ]
        if get_rank() == 0: # only remove from one process
            for filestem in filestems:
                files = os.listdir(os.path.join(self.get_save_dir(), filestem))
                files = [f for f in files if os.path.isdir(os.path.join(self.get_save_dir(), filestem, f))]
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
                    except:
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
