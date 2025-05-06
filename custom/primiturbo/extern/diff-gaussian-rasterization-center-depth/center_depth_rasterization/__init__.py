#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
import sys
import os
import importlib

# 避免循环导入
_C = None

def _load_c_extension():
    """尝试通过绝对路径加载 C++ 扩展模块"""
    global _C
    if _C is None:
        try:
            # 获取当前模块的绝对路径
            module_path = os.path.dirname(os.path.abspath(__file__))
            # 构建 _C.so 文件的绝对路径
            extension_path = os.path.join(module_path, '_C.cpython-311-x86_64-linux-gnu.so')

            if not os.path.exists(extension_path):
                # 尝试在conda环境的site-packages中查找
                import site
                for site_path in site.getsitepackages():
                    possible_path = os.path.join(site_path, 'center_depth_rasterization-0.0.0-py3.11-linux-x86_64.egg', 
                                                'center_depth_rasterization', '_C.cpython-311-x86_64-linux-gnu.so')
                    if os.path.exists(possible_path):
                        extension_path = possible_path
                        break
            
            if not os.path.exists(extension_path):
                raise ImportError(f"Cannot find _C extension at {extension_path}")
            
            print(f"Found extension at: {extension_path}")
            
            # 使用importlib.machinery加载动态库
            spec = importlib.machinery.ModuleSpec(
                name="_C",
                loader=importlib.machinery.ExtensionFileLoader("_C", extension_path),
                origin=extension_path
            )
            _C = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_C)
            
            print("成功加载 C++ 扩展")
        except Exception as e:
            print(f"加载 C++ 扩展时出错: {e}", file=sys.stderr)
            raise
    return _C

def rasterize_gaussians_center_depth(*args, **kwargs):
    """rasterize_gaussians_center_depth 函数的包装器"""
    c_module = _load_c_extension()
    return c_module.rasterize_gaussians_center_depth(*args, **kwargs)

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    c_module = _load_c_extension()
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        c_module = _load_c_extension()

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.kernel_size,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.require_coord,
            raster_settings.require_depth,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, coord, mcoord, alpha, normal, depth, mdepth, radii, geomBuffer, binningBuffer, imgBuffer = c_module.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, coord, mcoord, alpha, normal, depth, mdepth, radii, geomBuffer, binningBuffer, imgBuffer = c_module.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, normal, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha)
        return color, radii, coord, mcoord, depth, mdepth, alpha, normal

    @staticmethod
    def backward(ctx, grad_color, grad_radii, grad_coord, grad_mcoord, grad_depth, grad_mdepth, grad_alpha,grad_normal):
        c_module = _load_c_extension()

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, normal, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                raster_settings.kernel_size,
                grad_color,
                grad_coord,
                grad_mcoord,
                grad_depth,
                grad_mdepth,
                grad_alpha,
                grad_normal,
                normal,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                raster_settings.require_coord,
                raster_settings.require_depth,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = c_module.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = c_module.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    kernel_size : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    require_depth : bool
    require_coord : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            c_module = _load_c_extension()
            raster_settings = self.raster_settings
            visible = c_module.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )
    
    def integrate(self, points3D, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, view2gaussian_precomp = None):
        c_module = _load_c_extension()
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if view2gaussian_precomp is None:
            view2gaussian_precomp = torch.Tensor([])

        # Invoke C++/CUDA integration routine
        return c_module.integrate_gaussians(
            points3D,
            means3D, 
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            view2gaussian_precomp,
            raster_settings.viewmatrix, 
            raster_settings.projmatrix, 
            raster_settings.campos, 
            raster_settings.sh_degree,
            raster_settings.scale_modifier
        )

