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

        training_type: str = "rollout-rendering-distillation" # "progressive-rendering-distillation" or "rollout-rendering-distillation" or "rollout-rendering-distillation-last-step"

        multi_step_module_name: Optional[str] = "space_generator.gen_layers"

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


        self.automatic_optimization = False # we manually update the weights
    
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
        opt_multi_step, opt_other = self.optimizers()
        opt_multi_step.zero_grad()
        opt_other.zero_grad()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        # for gradient accumulation
        # update the weights with the remaining gradients
        opt_multi_step, opt_other = self.optimizers()
        try:
            opt_multi_step.step()
            opt_other.step()
            opt_multi_step.zero_grad()
            opt_other.zero_grad()
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
        self.geometry.eval()
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
        # get the optimizers
        opt_other, opt_multi_step = self.optimizers()

        # set the timesteps
        all_timesteps = self._set_timesteps(
            self.noise_scheduler,
            self.cfg.num_steps_training,
        )

        timesteps = sample_timesteps(
            all_timesteps,
            num_parts = self.cfg.num_parts_training, 
            batch_size=1, #batch_size,
        )

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
        self.toggle_optimizer(opt_other)
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
                latent_var = Variable(_denoised_latent, requires_grad=True)

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

                # # check the gradients for DEBUG
                # self._check_trainable_params(opt_other)
                # self._check_trainable_params(opt_multi_step)

                # store gradients
                loss_dec = (
                    fidelity_loss + regularization_loss
                )  / self.cfg.gradient_accumulation_steps
                loss_fake = self._fake_gradient(opt_multi_step)

                self._check_trainable_params(opt_other)
                # self._check_trainable_params(opt_multi_step)
                self.manual_backward(loss_dec + 0 * loss_fake)
                # self.trainer.strategy.backward(loss_dec, opt_other)
                self._check_trainable_params(opt_other)
                # self._check_trainable_params(opt_multi_step)

                gradient_trajectory.append(latent_var.grad.clone())
                print("now the gradient of the decoder is", latent_var.grad.norm().item())

        # update the weights of the decoder
        if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
            opt_other.step()
            opt_other.zero_grad()
        self.untoggle_optimizer(opt_other) # during the rollout, only the decoder is updated

        # the rollout is done, now we can compute the gradient of the denoised latents
        self.toggle_optimizer(opt_multi_step)  # during this process, only the multi-step module is updated
        noise_pred_batch = self.geometry.denoise(
            noisy_input =  torch.cat(_noisy_latents_input_trajectory, dim=0),
            text_embed = torch.cat(cond_trajectory, dim=0),
            timestep = torch.cat(timesteps, dim=0).repeat_interleave(
                    space_cache.shape[0]
                ).to(self.device)
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

        loss_gen = SpecifyGradient.apply(
            torch.cat(denoised_latent_batch, dim=0),
            torch.cat(gradient_trajectory, dim=0)
        )

        # # check the gradients for DEBUG
        # self._check_trainable_params(opt_other)
        # self._check_trainable_params(opt_multi_step)
        # self._check_model_gradients()
        self.manual_backward(loss_gen / self.cfg.gradient_accumulation_steps)
        
        
        # update the weights of the multi-step module
        if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
            opt_multi_step.step()
            opt_multi_step.zero_grad()
        self.untoggle_optimizer(opt_multi_step)



    # def on_before_backward(self, loss):
    #     # for debugging
    #     name_list = []
    #     for name, param in self.named_parameters():
    #         if param.requires_grad and param.grad is None:
    #             name_list.append(name)

    # def on_after_backward(self):
    #     # for debugging
    #     for name, param in self.named_parameters():
    #         if param.requires_grad and param.grad is None:
    #             print(name)

    def _check_trainable_params(self, opt):
        # list all the parameters contained in the multi-step optimizer
        for param_group in opt.param_groups:

            params_has_grad = []
            params_no_grad = []
            for idx, param in enumerate(param_group['params']):
                if param.requires_grad:
                    if param.grad is None:
                        params_no_grad.append(idx)
                    else:
                        params_has_grad.append(idx)
            if len(params_has_grad) > 0 and len(params_no_grad) == 0:
                print("All the parameters in {} have gradients".format(param_group['name']))
            elif len(params_has_grad) == 0 and len(params_no_grad) > 0:
                print("All the parameters in {} do not have gradients".format(param_group['name']))
            elif len(params_has_grad) > 0 and len(params_no_grad) > 0:
                print("Some parameters in {} have gradients, and some do not".format(param_group['name']))
            else:
                print("No parameters in {}".format(param_group['name']))

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

    def _fake_gradient(self, optimizer):
        loss = 0
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.requires_grad:
                    loss += 0.0 * param.sum()
        return loss
        

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



    def configure_optimizers(self):
        if self.cfg.multi_step_module_name is None:
            return super().configure_optimizers()
        else:
            # all parameters except the space generator's specific parameters by self.cfg.module_name_seperate_training
            # first make sure the module_name_seperate_training is a valid module name
            module_name = self.cfg.multi_step_module_name

            # then make sure the self.cfg.optimizer.params has separate parameters for the two parts
            params_multi_step = {}
            params_other = {}
            for param_name, param in self.cfg.optimizer.params.items():
                if module_name in param_name:
                    params_multi_step[param_name] = param
                else:
                    params_other[param_name] = param
            assert len(params_other) > 0, "No parameters are not in the multi-step module."
            assert len(params_multi_step) > 0, "No parameters are in the multi-step module."
            
            # then create the optimizers and schedulers for the multi-step module
            config_multi_step = self.cfg.optimizer.copy()
            config_multi_step.params = params_multi_step
            ret_multi_step = {
                "optimizer": parse_optimizer(config_multi_step, self),
            }
            if self.cfg.scheduler is not None:
                ret_multi_step.update(
                    {
                        "lr_scheduler": parse_scheduler(self.cfg.scheduler, ret_multi_step["optimizer"]),
                    }
                )


            # then create the optimizers and schedulers for the other module
            config_other = self.cfg.optimizer.copy()
            config_other.params = params_other
            ret_other = {
                "optimizer": parse_optimizer(config_other, self),
            }
            if self.cfg.scheduler is not None:
                ret_other.update(
                    {
                        "lr_scheduler": parse_scheduler(self.cfg.scheduler, ret_other["optimizer"]),
                    }
                )

            # then return the optimizers and schedulers
            return ret_other, ret_multi_step # the multi-step module is the first one in the returned tuple
            

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

    def _check_model_gradients(self):
        """Prints parameters without gradients after backward pass"""
        no_grad_params = []
        with_grad_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    no_grad_params.append(name)
                else:
                    with_grad_params.append(name)
        
        print(f"\nParameters WITH gradients: {len(with_grad_params)}")
        print(f"Parameters WITHOUT gradients: {len(no_grad_params)}")
        
        if len(no_grad_params) > 0:
            print("\nParameters without gradients:")
            for name in no_grad_params:
                print(f"  - {name}")

