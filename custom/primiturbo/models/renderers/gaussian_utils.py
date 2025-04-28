
import os
import torch
from torch.nn import functional as F

import numpy as np
from plyfile import PlyData, PlyElement

from threestudio.utils.typing import *
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from torch.utils.checkpoint import _get_autocast_kwargs


class Camera:
    def __init__(self,
        C2W: Tensor, fxfycxcy: Tensor, h: int, w: int,
        znear: float = 0.01, zfar: float = 100.,
    ):
        self.fxfycxcy = fxfycxcy.clone().float()
        self.C2W = C2W.clone().float()
        self.W2C = self.C2W.inverse()

        self.znear = znear
        self.zfar = zfar
        self.h = h
        self.w = w

        fx, fy, cx, cy = self.fxfycxcy[0], self.fxfycxcy[1], self.fxfycxcy[2], self.fxfycxcy[3]
        self.tanfovX = 1 / (2 * fx)  # `tanHalfFovX` actually
        self.tanfovY = 1 / (2 * fy)  # `tanHalfFovY` actually
        self.fovX = 2 * torch.atan(self.tanfovX)
        self.fovY = 2 * torch.atan(self.tanfovY)
        self.shiftX = 2 * cx - 1
        self.shiftY = 2 * cy - 1

        def getProjectionMatrix(znear, zfar, fovX, fovY, shiftX, shiftY):
            tanHalfFovY = torch.tan((fovY / 2))
            tanHalfFovX = torch.tan((fovX / 2))

            top = tanHalfFovY * znear
            bottom = -top
            right = tanHalfFovX * znear
            left = -right

            P = torch.zeros(4, 4, device=fovX.device)

            z_sign = 1

            P[0, 0] = 2 * znear / (right - left)
            P[1, 1] = 2 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left) + shiftX
            P[1, 2] = (top + bottom) / (top - bottom) + shiftY
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P

        self.world_view_transform = self.W2C.transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.fovX, self.fovY, self.shiftX, self.shiftY).transpose(0, 1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.C2W[:3, 3]
        
class GaussianModel:
    def __init__(self):
        self.xyz = None
        self.rgb = None
        self.scale = None
        self.rotation = None
        self.opacity = None

        self.sh_degree = 0

    def set_data(self, xyz: Tensor, rgb: Tensor, scale: Tensor, rotation: Tensor, opacity: Tensor):
        self.xyz = xyz
        self.rgb = rgb
        self.scale = scale
        self.rotation = rotation
        self.opacity = opacity
        return self

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "GaussianModel":
        self.xyz = self.xyz.to(device, dtype)
        self.rgb = self.rgb.to(device, dtype)
        self.scale = self.scale.to(device, dtype)
        self.rotation = self.rotation.to(device, dtype)
        self.opacity = self.opacity.to(device, dtype)
        return self

    def save_ply(self, path: str, opacity_threshold: float = 0.):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.xyz.detach().cpu().numpy()
        f_dc = self.rgb.detach().cpu().numpy()
        rgb = (f_dc * 255.).clip(0., 255.).astype(np.uint8)
        opacity = self.opacity.detach().cpu().numpy()
        scale = self.scale.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        # Filter out points with low opacity
        mask = (opacity > opacity_threshold).squeeze()
        xyz = xyz[mask]
        f_dc = f_dc[mask]
        opacity = opacity[mask]
        scale = scale[mask]
        rotation = rotation[mask]
        rgb = rgb[mask]

        dtype_full = [(attribute, "f4") for attribute in self._construct_list_of_attributes()]
        dtype_full.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, opacity, scale, rotation, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path: str):
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ), axis=1)
        f_dc = np.stack((
            np.asarray(plydata.elements[0]["f_dc_0"]),
            np.asarray(plydata.elements[0]["f_dc_1"]),
            np.asarray(plydata.elements[0]["f_dc_2"]),
        ), axis=1)
        opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        scale = np.stack((
            np.asarray(plydata.elements[0]["scale_0"]),
            np.asarray(plydata.elements[0]["scale_1"]),
            np.asarray(plydata.elements[0]["scale_2"]),
        ), axis=1)
        rotation = np.stack((
            np.asarray(plydata.elements[0]["rot_0"]),
            np.asarray(plydata.elements[0]["rot_1"]),
            np.asarray(plydata.elements[0]["rot_2"]),
            np.asarray(plydata.elements[0]["rot_3"]),
        ), axis=1)

        self.xyz = torch.from_numpy(xyz).float()
        self.rgb = torch.from_numpy(f_dc).float()
        self.opacity = torch.from_numpy(opacity).float()
        self.scale = torch.from_numpy(scale).float()
        self.rotation = torch.from_numpy(rotation).float()

    def _construct_list_of_attributes(self):
        l = ["x", "y", "z"]
        for i in range(self.rgb.shape[1]):
            l.append(f"f_dc_{i}")
        l.append("opacity")
        for i in range(self.scale.shape[1]):
            l.append(f"scale_{i}")
        for i in range(self.rotation.shape[1]):
            l.append(f"rot_{i}")
        return l


class Depth2Normal(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delzdelxkernel = torch.tensor(
            [
                [0.00000, 0.00000, 0.00000],
                [-1.00000, 0.00000, 1.00000],
                [0.00000, 0.00000, 0.00000],
            ]
        )
        self.delzdelykernel = torch.tensor(
            [
                [0.00000, -1.00000, 0.00000],
                [0.00000, 0.00000, 0.00000],
                [0.0000, 1.00000, 0.00000],
            ]
        )

    def forward(self, x):
        B, C, H, W = x.shape
        delzdelxkernel = self.delzdelxkernel.view(1, 1, 3, 3).to(x.device)
        delzdelx = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelxkernel, padding=1
        ).reshape(B, C, H, W)
        delzdelykernel = self.delzdelykernel.view(1, 1, 3, 3).to(x.device)
        delzdely = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelykernel, padding=1
        ).reshape(B, C, H, W)
        # 使用负号以匹配常见约定（法线指向外）
        normal = -torch.cross(delzdelx, delzdely, dim=1)
        return normal
