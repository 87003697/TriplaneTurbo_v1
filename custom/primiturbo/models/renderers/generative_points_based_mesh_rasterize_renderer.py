from dataclasses import dataclass
from functools import partial
from tqdm import tqdm
import os
import sys

import nerfacc
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.estimators import ImportanceEstimator
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial

from threestudio.models.renderers.nvdiff_rasterizer import NVDiffRasterizer

from threestudio.utils.misc import get_device, C
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
from threestudio.models.mesh import Mesh

from threestudio.utils.ops import scale_tensor as scale_tensor


@threestudio.register("generative-point-based-mesh-rasterize-renderer")
class GenerativePointBasedMeshRasterizeRenderer(NVDiffRasterizer):
    @dataclass
    class Config(NVDiffRasterizer.Config):
        # the following are from NeuS/tmp.py #########
        isosurface_resolution: int = 128

        isosurface_remove_outliers: bool = False
        isosurface_outlier_n_faces_threshold: Union[int, float] = 0.01

        context_type: str = "cuda"
        isosurface_method: str = "mt" # "mt" or "mc-cpu" or "diffmc"

        enable_bg_rays: bool = False
        normal_direction: str = "camera" # "camera" or "world" or "front"

        # sdf forcing strategy for generative space
        sdf_grad_shrink: float = 1.

        def_grad_shrink: float = 1. # Note: Deformation might not be present in point-based cache

        allow_empty_flag: bool = True

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())
        # overwrite the geometry
        # self.geometry.isosurface = self.isosurface # Let the geometry handle its own isosurface logic if needed

        assert self.cfg.isosurface_method in ["mt", "mc-cpu", "diffmc"], "Invalid isosurface method"
        if self.cfg.isosurface_method == "mt":
            from threestudio.models.isosurface import MarchingTetrahedraHelper
            # Ensure the tets file path is correct relative to the threestudio installation or project structure
            tets_path = f"load/tets/{self.cfg.isosurface_resolution}_tets.npz"
            if not os.path.exists(tets_path):
                 # Try finding relative to threestudio install path if not found directly
                 try:
                     import threestudio
                     base_dir = os.path.dirname(threestudio.__file__)
                     tets_path = os.path.join(base_dir, '..', tets_path) # Go up one level from threestudio module
                     if not os.path.exists(tets_path):
                         threestudio.warn(f"Tets file not found at {tets_path}, Marching Tetrahedra might fail.")
                 except ImportError:
                     threestudio.warn(f"Tets file not found at {tets_path} and threestudio path couldn't be determined.")

            self.isosurface_helper = MarchingTetrahedraHelper(
                self.cfg.isosurface_resolution,
                tets_path,
            )
        elif self.cfg.isosurface_method == "mc-cpu":
            from threestudio.models.isosurface import  MarchingCubeCPUHelper
            self.isosurface_helper = MarchingCubeCPUHelper(
                self.cfg.isosurface_resolution,
            )
        elif self.cfg.isosurface_method == "diffmc":
            from threestudio.models.isosurface import  DiffMarchingCubeHelper
            self.isosurface_helper = DiffMarchingCubeHelper(
                self.cfg.isosurface_resolution,
            )
        else:
             raise NotImplementedError(f"Isosurface method {self.cfg.isosurface_method} not supported.")


        # detect if the sdf is empty
        self.empty_flag = False

        # follow InstantMesh
        grid_res = self.cfg.isosurface_resolution
        if grid_res > 0:
            v = torch.zeros([grid_res] * 3, dtype=torch.bool,)
            center_start = max(0, grid_res // 2 - 1)
            center_end = min(grid_res, grid_res // 2 + 1)
            v[center_start:center_end, center_start:center_end, center_start:center_end] = True
            self.center_indices = torch.nonzero(v.reshape(-1)).to(self.device)

            v = torch.zeros([grid_res] * 3, dtype=torch.bool,)
            v[:2, :, :] = True; v[-2:, :, :] = True
            v[:, :2, :] = True; v[:, -2:, :] = True
            v[:, :, :2] = True; v[:, :, -2:] = True
            self.border_indices = torch.nonzero(v.reshape(-1)).to(self.device)
        else:
            self.center_indices = None
            self.border_indices = None


    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        space_cache: Dict[str, Union[Float[Tensor, "B ..."], List[Float[Tensor, "B ..."]]]], # Expecting point-based cache
        text_embed: Optional[Float[Tensor, "B C"]] = None, # Usually tied to space_cache batch size
        noise: Optional[Float[Tensor, "B C"]] = None, # May not be directly used if geometry uses space_cache
        render_rgb: bool = True,
        rays_d_rasterize: Optional[Float[Tensor, "B H W 3"]] = None, # For background if needed
        camera_distances: Optional[Float[Tensor, "B"]] = None,
        c2w: Optional[Float[Tensor, "B 4 4"]] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:

        # Determine batch size from space_cache (assuming 'position' key exists and holds relevant tensor/list)
        if "position" not in space_cache:
            raise ValueError("GenerativePointBasedMeshRasterizeRenderer requires 'position' key in space_cache")

        if isinstance(space_cache["position"], torch.Tensor):
            batch_size_space_cache = space_cache["position"].shape[0]
        elif isinstance(space_cache["position"], list):
            batch_size_space_cache = space_cache["position"][0].shape[0] # Assuming list elements have consistent batch size
        else:
            raise TypeError("space_cache['position'] must be a Tensor or list of Tensors")

        if mvp_mtx.shape[0] % batch_size_space_cache != 0:
             raise ValueError(f"mvp_mtx batch size ({mvp_mtx.shape[0]}) must be a multiple of space_cache batch size ({batch_size_space_cache})")
        num_views_per_batch = mvp_mtx.shape[0] // batch_size_space_cache


        # Extract mesh using the point-based space_cache
        mesh_list = self.isosurface(space_cache)

        # detect if the sdf is empty (flag set within isosurface)
        if self.empty_flag:
            is_emtpy = True
            self.empty_flag = False # Reset flag
        else:
            is_emtpy = False

        out_list = []
        # Process each mesh extracted from the batch of space_caches
        for batch_idx, mesh in enumerate(mesh_list):
            _mvp_mtx: Float[Tensor, "NV 4 4"]  = mvp_mtx[batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch]
            _camera_positions: Float[Tensor, "NV 3"] = camera_positions[batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch]
            _light_positions: Float[Tensor, "NV 3"] = light_positions[batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch]
            _c2w: Optional[Float[Tensor, "NV 4 4"]] = c2w[batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch] if c2w is not None else None
            _camera_distances: Optional[Float[Tensor, "NV"]] = camera_distances[batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch] if camera_distances is not None else None
            _rays_d_rasterize: Optional[Float[Tensor, "NV H W 3"]] = rays_d_rasterize[batch_idx * num_views_per_batch : (batch_idx + 1) * num_views_per_batch] if rays_d_rasterize is not None else None
            _text_embed: Optional[Float[Tensor, "1 C"]] = text_embed[batch_idx:batch_idx+1] if text_embed is not None else None # Pass single text embed for this batch item


            v_pos_clip: Float[Tensor, "NV Nv 4"] = self.ctx.vertex_transform(
                mesh.v_pos, _mvp_mtx
            )

            # do rasterization
            if self.training: # optimize for fewer views during training
                rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
                gb_feat, _ = self.ctx.interpolate(v_pos_clip, rast, mesh.t_pos_idx)
                depth = gb_feat[..., -2:-1]
            else: # evaluation might require many views, rasterize in chunks
                rast_list = []
                depth_list = []
                n_views_per_rasterize = 4 # Adjust chunk size based on memory
                for i in range(0, v_pos_clip.shape[0], n_views_per_rasterize):
                    rast_chunk, _ = self.ctx.rasterize(v_pos_clip[i:i+n_views_per_rasterize], mesh.t_pos_idx, (height, width))
                    rast_list.append(rast_chunk)
                    gb_feat_chunk, _ = self.ctx.interpolate(v_pos_clip[i:i+n_views_per_rasterize], rast_chunk, mesh.t_pos_idx)
                    depth_list.append(gb_feat_chunk[..., -2:-1])
                rast = torch.cat(rast_list, dim=0)
                depth = torch.cat(depth_list, dim=0)

            mask = rast[..., 3:] > 0

            # special case when no points are visible
            effective_mask = mask.clone()
            if effective_mask.sum() == 0: # no visible points
                # set the mask to be the first point (or handle differently)
                effective_mask[:1, :1, :1] = True # Modify a copy for selection, keep original mask for lerping

            mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

            # disparity calculation (if camera_distances provided)
            disparity_norm = torch.zeros_like(depth) # Default
            if _camera_distances is not None:
                far = (_camera_distances + torch.sqrt(3.0)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                near = (_camera_distances - torch.sqrt(3.0)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # Clamp depth to avoid issues when depth > far
                # Use mask to avoid division by zero or negative values if near == far
                valid_range = (far > near) & (far > 0) & (near >= 0)
                disparity_tmp = torch.where(valid_range, depth.clamp(min=near, max=far), far)
                disparity_norm = torch.where(
                    valid_range,
                    (far - disparity_tmp) / (far - near).clamp(min=1e-6),
                    torch.zeros_like(depth) # Assign 0 disparity if range is invalid
                )
                disparity_norm = disparity_norm.clamp(0, 1)
                disparity_norm = torch.lerp(torch.zeros_like(depth), disparity_norm, mask.float())
                disparity_norm = self.ctx.antialias(disparity_norm, rast, v_pos_clip, mesh.t_pos_idx)


            out = {
                "opacity": mask_aa if not is_emtpy else torch.zeros_like(mask_aa), # Return 0 opacity if empty
                "mesh": mesh,
                "depth": depth if not is_emtpy else torch.zeros_like(depth),
                "disparity": disparity_norm if not is_emtpy else torch.zeros_like(disparity_norm),
            }

            # Normals
            gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
            gb_normal = F.normalize(gb_normal, dim=-1)
            # Lerp with background normal (e.g., (0,0,0) or (0.5, 0.5, 0.5)) before AA for smoother edges
            # Using (0,0,0) which becomes (0.5, 0.5, 0.5) after scaling to [0, 1] seems common
            gb_normal_world_aa = torch.lerp(
                 #torch.zeros_like(gb_normal) + 0.5,
                 torch.tensor([0.0, 0.0, 0.0], device=gb_normal.device).view(1, 1, 1, 3),
                 (gb_normal + 1.0) / 2.0, # Scale to [0, 1]
                 mask.float()
            )
            gb_normal_world_aa = self.ctx.antialias(
                gb_normal_world_aa, rast, v_pos_clip, mesh.t_pos_idx
            )
            # Output world normal scaled to [0, 1]
            out.update({"comp_normal": gb_normal_world_aa if not is_emtpy else torch.zeros_like(gb_normal_world_aa)})


            # Normal transformations based on cfg.normal_direction
            if _c2w is not None and self.cfg.normal_direction in ["camera", "front"]:
                 # Common background normals
                 bg_normal_cam = torch.tensor([0.5, 0.5, 1.0], device=gb_normal.device).view(1, 1, 1, 3) # Blueish background
                 bg_normal_white = torch.ones_like(gb_normal)

                 # Select C2W matrix for transformation
                 if self.cfg.normal_direction == "camera":
                     c2w_transform = _c2w
                 elif self.cfg.normal_direction == "front":
                     # Use the first view's C2W for all views in this sub-batch
                     c2w_transform = _c2w[0:1].repeat(num_views_per_batch, 1, 1)
                 else: # Should not happen based on outer check
                     c2w_transform = _c2w

                 w2c: Float[Tensor, "NV 4 4"] = torch.inverse(c2w_transform)
                 rotate: Float[Tensor, "NV 3 3"] = w2c[:, :3, :3]

                 # Transform world normals gb_normal (shape NV, H, W, 3) to camera space
                 # Reshape gb_normal: NV, H, W, 3 -> NV*H*W, 1, 3
                 # Reshape rotate: NV, 3, 3 -> NV, 1, 1, 3, 3 -> NV*H*W, 3, 3 (match points)
                 num_views, H, W, _ = gb_normal.shape
                 gb_normal_flat = gb_normal.reshape(num_views * H * W, 1, 3)
                 rotate_expanded = rotate.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1, -1).reshape(num_views * H * W, 3, 3)

                 # Perform rotation: (NV*H*W, 1, 3) @ (NV*H*W, 3, 3) -> (NV*H*W, 1, 3)
                 gb_normal_cam_flat = gb_normal_flat @ rotate_expanded
                 gb_normal_cam = gb_normal_cam_flat.reshape(num_views, H, W, 3) # Reshape back

                 # Optional flip (often needed for camera space normals depending on convention)
                 # This flip aligns with OpenGL/Blender camera space (+Y up, -Z forward) typically used
                 # flip_yz = torch.eye(3, device=w2c.device)
                 # flip_yz[1, 1] = -1
                 # flip_yz[2, 2] = -1
                 # gb_normal_cam = gb_normal_cam @ flip_yz

                 # Pixel space flip (Common in some pipelines like RichDreamer)
                 flip_xy = torch.eye(3, device=w2c.device)
                 flip_xy[0, 0] = -1.0
                 # flip_xy[1, 1] = -1.0 # Sometimes Y needs flipping too
                 gb_normal_cam = gb_normal_cam @ flip_xy

                 gb_normal_cam = F.normalize(gb_normal_cam, dim=-1)
                 gb_normal_cam_01 = (gb_normal_cam + 1.0) / 2.0 # Scale to [0, 1]

                 # Lerp with background and antialias
                 camera_gb_normal_bg = torch.lerp(bg_normal_cam, gb_normal_cam_01, mask.float())
                 camera_gb_normal_bg = self.ctx.antialias(camera_gb_normal_bg, rast, v_pos_clip, mesh.t_pos_idx)

                 camera_gb_normal_bg_white = torch.lerp(bg_normal_white, gb_normal_cam_01, mask.float())
                 camera_gb_normal_bg_white = self.ctx.antialias(camera_gb_normal_bg_white, rast, v_pos_clip, mesh.t_pos_idx)

                 out.update({
                     "comp_normal_cam_vis": camera_gb_normal_bg if not is_emtpy else torch.zeros_like(camera_gb_normal_bg),
                     "comp_normal_cam_vis_white": camera_gb_normal_bg_white if not is_emtpy else torch.zeros_like(camera_gb_normal_bg_white),
                 })


            # RGB Rendering
            if render_rgb:
                # Prepare the space_cache slice for the current batch item
                # Assumes space_cache contains tensors or lists of tensors with batch dim
                space_cache_slice = {}
                try:
                    for key, value in space_cache.items():
                        if isinstance(value, torch.Tensor):
                            space_cache_slice[key] = value[batch_idx: batch_idx+1]
                        elif isinstance(value, list):
                            # Assuming list elements are tensors with batch dim
                            space_cache_slice[key] = [v[batch_idx: batch_idx+1] for v in value]
                        # else: ignore non-tensor/list items if any
                except Exception as e:
                    print(f"Error slicing space_cache: {e}. Key: {key}, Value type: {type(value)}")
                    # Handle error, maybe skip RGB rendering for this item
                    continue # Skip to next mesh item

                selector = effective_mask[..., 0] # Use the mask that guarantees some points if original mask sum is 0

                gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
                gb_viewdirs = F.normalize(
                    gb_pos - _camera_positions[:, None, None, :], # Use per-view camera positions
                    dim=-1
                )
                gb_light_positions = _light_positions[:, None, None, :].expand(
                    -1, height, width, -1
                ) # Use per-view light positions

                positions = gb_pos[selector] # Select visible points

                if positions.shape[0] == 0 and mask.sum() > 0:
                     # This case should ideally not happen if effective_mask logic is correct
                     # but as a fallback, maybe skip material evaluation
                     threestudio.warn("No positions selected for material evaluation despite non-empty mask.")
                     rgb_fg = torch.zeros_like(gb_pos) # Fallback
                elif positions.shape[0] == 0 and mask.sum() == 0:
                     # If mask was truly empty, output zeros
                     rgb_fg = torch.zeros_like(gb_pos)
                else:
                    # Query geometry and material using the point-based space_cache
                    geo_out = self.geometry(
                        positions.unsqueeze(0), # Geometry might expect batch dim (B=1, N, 3)
                        space_cache_slice,      # Pass the sliced cache for this item
                        output_normal=self.material.requires_normal, # Check if material needs normals from geometry
                        output_sdf=False # We likely don't need SDF here, only surface attrs
                    )

                    extra_geo_info = {}
                    # Use interpolated normals/tangents from mesh rasterization
                    # unless geometry explicitly provides different shading normals/tangents
                    if self.material.requires_normal:
                        if "shading_normal" in geo_out:
                             # Need to carefully manage shapes if geometry returns normals
                             # For now, use interpolated normals
                             extra_geo_info["shading_normal"] = gb_normal[selector] if not is_emtpy else torch.zeros_like(gb_normal[selector])
                        else:
                             extra_geo_info["shading_normal"] = gb_normal[selector] if not is_emtpy else torch.zeros_like(gb_normal[selector])


                    if self.material.requires_tangent:
                        # Check if mesh has tangents, interpolate if so
                        if hasattr(mesh, 'v_tng') and mesh.v_tng is not None:
                            gb_tangent, _ = self.ctx.interpolate_one(
                                mesh.v_tng, rast, mesh.t_pos_idx
                            )
                            gb_tangent = F.normalize(gb_tangent, dim=-1)
                            extra_geo_info["tangent"] = gb_tangent[selector] if not is_emtpy else torch.zeros_like(gb_tangent[selector])
                        # elif "tangent" in geo_out: pass geometry tangent through
                        # else: tangent might not be available

                    # Remove fields geometry might return but are handled by rasterizer/mesh
                    # geo_out.pop("shading_normal", None)
                    geo_out.pop("sdf", None)
                    geo_out.pop("sdf_grad", None)
                    geo_out.pop("normal", None) # Use interpolated normal unless specifically overridden

                    # Call material
                    rgb_fg_values = self.material(
                        viewdirs=gb_viewdirs[selector],
                        positions=positions,
                        light_positions=gb_light_positions[selector],
                        **extra_geo_info,
                        **geo_out # Pass remaining features from geometry
                    )

                    # Scatter calculated RGB values back to the image grid
                    gb_rgb_fg = torch.zeros_like(gb_pos)
                    gb_rgb_fg[selector] = rgb_fg_values


                # Background calculation
                if self.cfg.enable_bg_rays and _rays_d_rasterize is not None:
                    view_dirs_bg = _rays_d_rasterize # Use provided background rays
                else:
                    view_dirs_bg = gb_viewdirs # Use foreground view directions as approximation

                # Pass the correct text embedding for the background if it uses hypernet
                bg_text_embed = _text_embed if "text_embed_bg" not in kwargs else kwargs["text_embed_bg"][batch_idx:batch_idx+1]

                if hasattr(self.background, "enabling_hypernet") and self.background.enabling_hypernet:
                    gb_rgb_bg = self.background(
                        dirs=view_dirs_bg,
                        text_embed=bg_text_embed
                    )
                else:
                    gb_rgb_bg = self.background(dirs=view_dirs_bg)

                # Composite foreground and background
                gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
                gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

                out.update(
                    {
                        "comp_rgb": gb_rgb_aa if not is_emtpy else torch.zeros_like(gb_rgb_aa),
                        "comp_rgb_fg": gb_rgb_fg if not is_emtpy else torch.zeros_like(gb_rgb_fg), # Non-AA foreground
                        "comp_rgb_bg": gb_rgb_bg if not is_emtpy else torch.zeros_like(gb_rgb_bg), # Background
                    }
                )

            out_list.append(out)

        # Combine results from all meshes in the batch
        combined_out = {}
        if not out_list: # Handle case where mesh_list was empty
             # Return dummy tensors with expected batch size if possible
             print("Warning: No meshes processed, returning empty output dict.")
             # TODO: Define expected output shapes and return zeros
             return combined_out

        # Determine keys to combine (exclude per-mesh objects like 'mesh')
        keys_to_combine = [k for k in out_list[0].keys() if k != 'mesh']

        for key in keys_to_combine:
            if isinstance(out_list[0][key], torch.Tensor):
                 # Concatenate tensors along the batch dimension (dim=0)
                 try:
                    combined_out[key] = torch.cat([o[key] for o in out_list], dim=0)
                 except Exception as e:
                     print(f"Error concatenating key '{key}': {e}")
                     # Handle error, maybe skip this key or assign None
                     combined_out[key] = None # Or some default
            # else: # Handle non-tensor values if needed (e.g., lists of scalars?)
                 # For now, assume all combined keys are tensors

        # Add the list of meshes separately
        combined_out['mesh'] = [o['mesh'] for o in out_list]

        # Include losses/metrics if geometry calculated any (e.g., sdf loss)
        # These might be lists if calculated per-mesh/point
        if "sdf" in out_list[0]: combined_out["sdf"] = [o["sdf"] for o in out_list if "sdf" in o]
        if "sdf_grad" in out_list[0]: combined_out["sdf_grad"] = [o["sdf_grad"] for o in out_list if "sdf_grad" in o]


        return combined_out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        # Update shrinkage factors based on schedule C(...)
        self.sdf_grad_shrink = C(self.cfg.sdf_grad_shrink, epoch, global_step)
        self.def_grad_shrink = C(self.cfg.def_grad_shrink, epoch, global_step)

    def isosurface(self, space_cache: Dict[str, Union[Float[Tensor, "B ..."], List[Float[Tensor, "B ..."]]]]) -> List[Mesh]:
        """
        Extracts isosurfaces (meshes) from the SDF field defined by the geometry
        using the provided point-based space_cache.
        """
        if self.cfg.isosurface_resolution <= 0:
            threestudio.warn("isosurface_resolution is 0, skipping mesh extraction.")
            return []

        # Determine batch size from space_cache
        if "position" not in space_cache:
            raise ValueError("isosurface requires 'position' key in space_cache")

        key_for_batch_size = "position" # Or another key guaranteed to exist and have batch dim
        if isinstance(space_cache[key_for_batch_size], torch.Tensor):
            batch_size = space_cache[key_for_batch_size].shape[0]
        elif isinstance(space_cache[key_for_batch_size], list):
            batch_size = space_cache[key_for_batch_size][0].shape[0] # Assuming list elements have consistent batch size
        else:
            raise TypeError(f"space_cache['{key_for_batch_size}'] must be a Tensor or list of Tensors")


        points = scale_tensor(
            self.isosurface_helper.grid_vertices.to(self.device),
            self.isosurface_helper.points_range,
            [-1, 1], # Assuming geometry expects queries in [-1, 1] bbox
        )

        mesh_list = []
        # Process each item in the batch separately for SDF query if needed,
        # although geometry might support batched queries directly.
        # Let's assume geometry supports batched queries for efficiency.

        # Query SDF and deformation (if available) for the grid points using the point-based cache
        # Geometry needs to handle the space_cache format internally (e.g., using KNN)
        sdf_batch, deformation_batch = self.geometry.forward_field(
            points.unsqueeze(0).expand(batch_size, -1, -1), # (B, N_points, 3)
            space_cache=space_cache, # Pass the full cache
            query_sdf=True,
            query_deformation=True # Request deformation if geometry supports it
        )

        # Apply gradient shrinking
        if self.sdf_grad_shrink != 1.0:
            sdf_batch = self.sdf_grad_shrink * sdf_batch + (1.0 - self.sdf_grad_shrink) * sdf_batch.detach()
        # else: keep original grad

        if deformation_batch is not None and self.def_grad_shrink != 1.0:
            deformation_batch = self.def_grad_shrink * deformation_batch + (1.0 - self.def_grad_shrink) * deformation_batch.detach()
        # else: keep original grad or it's None

        # Extract mesh for each item in the batch
        for index in range(batch_size):
            sdf = sdf_batch[index]
            deformation = deformation_batch[index] if deformation_batch is not None else None

            # Handle empty SDF case (all positive or all negative)
            if torch.all(sdf > 0) or torch.all(sdf < 0):
                threestudio.info(f"SDF for batch item {index} is uniform sign, attempting fallback.")
                self.empty_flag = self.cfg.allow_empty_flag # Signal forward pass

                if self.cfg.allow_empty_flag and self.center_indices is not None and self.border_indices is not None:
                    # Try InstantMesh-style SDF forcing
                    update_sdf = torch.zeros_like(sdf)
                    max_sdf = sdf.max().item() # Use .item() if using detach below is intended
                    min_sdf = sdf.min().item()
                    # Make center negative, border positive
                    update_sdf[self.center_indices] += (-0.1 - max_sdf) # Target slightly negative
                    update_sdf[self.border_indices] += (0.1 - min_sdf)  # Target slightly positive

                    # Apply additive update (consider detach if causing instability)
                    new_sdf = sdf + update_sdf.detach() # Detach update to avoid influencing original SDF grad

                    # Prevent exact zeros if possible, reuse original sdf values at boundary
                    # update_mask = (new_sdf * sdf < 0).float() # Mask where sign changed
                    # sdf = new_sdf * update_mask + sdf * (1 - update_mask)
                    sdf = new_sdf # Use the modified SDF for extraction

                else:
                    # If not allowed or indices not available, create an empty mesh or skip
                    threestudio.warn(f"Cannot create fallback mesh for batch item {index}.")
                    # Create a dummy empty mesh to maintain list structure
                    mesh_list.append(Mesh(v_pos=torch.empty(0,3, device=self.device),
                                         t_pos_idx=torch.empty(0,3, dtype=torch.long, device=self.device)))
                    continue # Skip to next item


            # Choose isosurface helper (handle multi-batch DiffMC if needed)
            if index > 0 and self.cfg.isosurface_method == "diffmc":
                helper_name = f"isosurface_helper_{index}"
                if not hasattr(self, helper_name):
                    from threestudio.models.isosurface import DiffMarchingCubeHelper
                    setattr(self, helper_name, DiffMarchingCubeHelper(
                        self.cfg.isosurface_resolution,
                    ))
                mesh = getattr(self, helper_name)(sdf, deformation)
            else:
                mesh = self.isosurface_helper(sdf, deformation)

            # Scale mesh vertices back to the original range (e.g., [-1, 1])
            mesh.v_pos = scale_tensor(
                mesh.v_pos,
                self.isosurface_helper.points_range,
                [-1, 1], # Target bbox
            )

            # Remove outliers if configured
            if self.cfg.isosurface_remove_outliers:
                 # Ensure mesh has faces before attempting removal
                 if mesh.t_pos_idx.shape[0] > 0:
                     mesh = mesh.remove_outlier(self.cfg.isosurface_outlier_n_faces_threshold)
                 else:
                     threestudio.debug("Skipping outlier removal on mesh with no faces.")


            mesh_list.append(mesh)

        return mesh_list


    # Keep train/eval methods for consistency
    def train(self, mode=True):
        if hasattr(self.geometry, "train"):
            self.geometry.train(mode)
        return super().train(mode=mode)

    def eval(self):
        if hasattr(self.geometry, "eval"):
            self.geometry.eval()
        return super().eval()


