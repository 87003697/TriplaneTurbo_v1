import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.mesh import Mesh
from threestudio.utils.misc import broadcast, get_rank, C
from threestudio.utils.typing import *

from threestudio.utils.ops import get_activation
from threestudio.models.networks import get_encoding, get_mlp

from custom.triplaneturbo.models.geometry.utils import contract_to_unisphere_custom, sample_from_planes
from einops import rearrange

@threestudio.register("few-step-one-plane-stable-diffusion")
class FewStepOnePlaneStableDiffusion(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_feature_dims: int = 3
        space_generator_config: dict = field(
            default_factory=lambda: {
                "pretrained_model_name_or_path": "stable-diffusion-2-1-base",
                "training_type": "lora",
                "output_dim": 15,
                "gradient_checkpoint": False,
            }
        )

        backbone: str = "few_step_one_plane_stable_diffusion" #TODO: change to few_step_few_plane_stable_diffusion

        scaling_activation: str = "exp-0.1" # in ["exp-0.1", "sigmoid", "exp", "softplus"]
        opacity_activation: str = "sigmoid-0.1" # in ["sigmoid-0.1", "sigmoid", "sigmoid-mipnerf", "softplus"]
        rotation_activation: str = "normalize" # in ["normalize"]
        color_activation: str = "sigmoid-mipnerf" # in ["scale_-11_01", "sigmoid-mipnerf"]

        xyz_center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
        xyz_max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
        xyz_radius_max: float = 1.0

        radius_init: float = 1.0

    def configure(self) -> None:
        super().configure()

        print("The current device is: ", self.device)
        
        # set up the space generator
        if self.cfg.backbone == "few_step_one_plane_stable_diffusion":
            from ...extern.few_step_one_plane_sd_modules import FewStepOnePlaneStableDiffusion as Generator
            self.space_generator = Generator(self.cfg.space_generator_config)
        else:
            raise ValueError(f"Unknown backbone {self.cfg.backbone}")
        
        self.scaling_activation = get_activation(self.cfg.scaling_activation)
        self.opacity_activation = get_activation(self.cfg.opacity_activation)
        self.rotation_activation = get_activation(self.cfg.rotation_activation)
        self.color_activation = get_activation(self.cfg.color_activation)

        self.xyz_center = lambda x: torch.tensor(self.cfg.xyz_center, device=x.device)
        self.xyz_max = lambda x: torch.tensor(self.cfg.xyz_max, device=x.device)
        self.xyz_radius_max = self.cfg.xyz_radius_max

    def radius_shift(
        self,
        points: Float[Tensor, "*N 3"],
        radius_init: float = 0.8,
    ) -> Float[Tensor, "*N 1"]:
        """
        Shift the points to the radius_init
        """
        direction = points / torch.norm(points, dim=1, p=2, keepdim=True)
        distance = torch.norm(points, dim=1, p=2) + radius_init
        return direction * distance

    def initialize_shape(self) -> None:
        # not used
        pass

    def denoise(
        self,
        noisy_input: Any,
        text_embed: Float[Tensor, "B C"],
        timestep
    ) -> Any:
        output = self.space_generator.forward_denoise(
            text_embed = text_embed,
            noisy_input = noisy_input,
            t = timestep
        )
        return output
    
    def decode(
        self,
        latents: Any,
    ) -> Any:
        triplane = self.space_generator.forward_decode(
            latents = latents
        )
        return triplane

    def parse(
        self,
        triplane: Float[Tensor, "B 3 C//3 H W"],
    ) -> List[Dict[str, Float[Tensor, "..."]]]:
        B, _, C, H, W = triplane.shape
        pc_list = []
        for i in range(B):
            pc_list.append(
                {
                    "gs_rgb": self.color_activation(
                        rearrange(
                            triplane[i, :, 0:3, :, :], 
                            "N C H W -> (N H W) C"
                        )
                    ),
                    "gs_xyz": rearrange(
                        self.radius_shift(
                            triplane[i, :, 3:6, :, :],
                            radius_init=self.cfg.radius_init
                        ),
                        "N C H W -> (N H W) C"
                        ) * self.xyz_max(triplane) + 
                        self.xyz_center(triplane), # plus center
                    "gs_scale": self.scaling_activation(
                        rearrange(
                            triplane[i, :, 6:9, :, :],
                            "N C H W -> (N H W) C"
                        )
                    ),
                    "gs_rotation": self.rotation_activation(
                        rearrange(
                            triplane[i, :, 9:13, :, :], 
                            "N C H W -> (N H W) C"
                        )
                    ),
                    "gs_opacity": self.opacity_activation(
                        rearrange(
                            triplane[i, :, 13:14, :, :], 
                            "N C H W -> (N H W) C"
                        )
                    )
                }
            )
        return pc_list

    def interpolate_encodings(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
        only_geo: bool = False,
    ):
        raise NotImplementedError("interpolate_encodings is not implemented yet.")


    def rescale_points(
        self,
        points: Float[Tensor, "*N Di"],
    ):
        raise NotImplementedError("rescale_points is not implemented yet.")

    def forward(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Any,
        output_normal: bool = False,
    ) -> Dict[str, Float[Tensor, "..."]]:
        raise NotImplementedError("forward is not implemented yet.")

    def forward_sdf(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    ) -> Float[Tensor, "*N 1"]:
        raise NotImplementedError("forward_sdf is not implemented yet.")

    def forward_field(
        self, 
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        raise NotImplementedError("forward_field is not implemented yet.")

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        # TODO: is this function correct?
        raise NotImplementedError("forward_level is not implemented yet.")

    def export(
        self, 
        points: Float[Tensor, "*N Di"], 
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("export is not implemented yet.")
    

    def train(self, mode=True):
        super().train(mode)
        self.space_generator.train(mode)

    def eval(self):
        super().eval()
        self.space_generator.eval()
