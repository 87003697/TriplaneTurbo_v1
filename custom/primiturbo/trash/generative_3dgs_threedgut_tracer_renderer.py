import math
from dataclasses import dataclass, field

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *

# Import directly from the tracer submodule where classes are defined
# from threedgut_tracer import Tracer, SensorPose3D, SensorPose3DModel
from threedgut_tracer.tracer import Tracer, SensorPose3D, SensorPose3DModel

import torch
from threestudio.utils.ops import get_cam_info_gaussian
from torch.cuda.amp import autocast

class Camera(NamedTuple):
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor

    
@threestudio.register("generative-3dgs-threedgut-tracer-renderer")
class Generative3dgsThreedgutTracerRenderer(Rasterizer):
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False
        invert_bg_prob: float = 0.5
        back_ground_color: Tuple[float, float, float] = (0.6, 0.6, 0.6)

        # for rendering the normal
        normal_direction: str = "camera"  # "front" or "camera" or "world"

        # Add configuration for threedgut_tracer if needed
        tracer_config: dict = field(default_factory=dict) # Example: Path to plugin, settings etc.

    cfg: Config
    tracer: Optional[Tracer] = None # Add tracer instance

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        threestudio.info(
            "[Note] Using ThreeDGUT Tracer. Material and background support might differ."
        )
        super().configure(geometry, material, background)
        self.background_tensor = torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32, device=self.device
        )
        # Initialize the Tracer
        # Assuming the Tracer class takes a config dict
        # Pass the relevant part of the config to the tracer
        self.tracer = Tracer(OmegaConf.create(self.cfg.tracer_config))
        threestudio.info("ThreeDGUT Tracer initialized successfully.")


    def _forward(
        self,
        viewpoint_camera,
        pc_data: Dict[str, Tensor],
        bg_color: torch.Tensor,
        frame_id: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Render the scene using ThreeDGUT Tracer.
        """
        if self.tracer is None:
            threestudio.warn("ThreeDGUT Tracer is not initialized, returning empty output.")
            # Return empty tensors matching expected output structure
            h, w = viewpoint_camera.image_height, viewpoint_camera.image_width
            return {
                "render": torch.zeros((1, 3, h, w), device=self.device),
                "depth": torch.zeros((1, 1, h, w), device=self.device),
                "mask": torch.zeros((1, 1, h, w), device=self.device),
                # Add other expected outputs if needed
            }

        # --- Prepare inputs for self.tracer.render ---
        # This part requires knowledge of the exact signature of tracer.render or tracer._Autograd.forward
        # Based on tracer.py: _Autograd.forward takes:
        # tracer_wrapper, frame_id, n_active_features, ray_ori, ray_dir,
        # mog_pos, mog_rot, mog_scl, mog_dns, mog_sph,
        # sensor_params, sensor_poses

        # 1. Get Gaussian parameters (already available in `pc`)
        # Access data directly from the input dictionary pc_data
        mog_pos = pc_data['xyz'].contiguous()
        mog_rot = pc_data['rotation'].contiguous()
        mog_scl = pc_data['scale'].contiguous()
        mog_dns = pc_data['opacity'].contiguous() # Assuming 'density' maps to opacity
        mog_sph = pc_data['rgb'].contiguous() # Assuming this maps to 'particle_radiance' / shs

        n_active_features = mog_pos.shape[0] # Number of Gaussians

        # 2. Prepare camera/sensor information
        # This needs careful mapping from viewpoint_camera or batch to what Tracer expects.
        # Tracer.py uses SensorPose3DModel and SensorPose3D.
        # Let's try to construct SensorPose3D from viewpoint_camera

        # Need R (rotation matrix) and T (translation vector) for SensorPose3DModel
        # world_view_transform is W2C. We need C2W (inverse).
        c2w = torch.inverse(viewpoint_camera.world_view_transform)
        R = c2w[:3, :3].cpu().numpy() # Convert to numpy for SensorPose3DModel
        T = c2w[:3, 3].cpu().numpy() # Convert to numpy

        # Use SensorPose3DModel to get the required pose format
        # Assuming default scale and translation are fine
        sensor_model = SensorPose3DModel(R=R, T=T)
        sensor_poses: SensorPose3D = sensor_model.get_sensor_pose()

        # Need Sensor Parameters (intrinsics) - Tracer.__create_camera_parameters might be useful?
        # Let's derive from FoV and image size.
        focal_y = viewpoint_camera.image_height / (2 * math.tan(viewpoint_camera.FoVy * 0.5))
        focal_x = viewpoint_camera.image_width / (2 * math.tan(viewpoint_camera.FoVx * 0.5))
        # Assuming principal point is center
        cx = viewpoint_camera.image_width / 2
        cy = viewpoint_camera.image_height / 2

        # Construct sensor_params tensor [fx, fy, cx, cy] - Shape/order might need adjustment
        sensor_params = torch.tensor([[focal_x, focal_y, cx, cy]], dtype=torch.float32, device=self.device)
        # TODO: Verify the exact format Tracer expects for sensor_params

        # 3. Prepare rays (might not be needed if tracer works directly with camera params)
        # Check if Tracer.render expects rays or camera parameters.
        # The _Autograd function takes ray_ori, ray_dir. Let's assume they are needed.
        # We can generate rays using the camera parameters.
        # This might be redundant if the tracer does it internally based on sensor_params/poses.
        # Let's assume for now tracer handles ray generation internally based on sensor info.
        # If explicit rays are needed, we'd generate them here.
        # ray_ori = viewpoint_camera.camera_center.expand(viewpoint_camera.image_height, viewpoint_camera.image_width, 3)
        # ... generate ray_dir ...

        # --- Call the tracer ---
        # The public method seems to be render(), let's try that first.
        # It might internally call the _Autograd function.
        # We need to construct the `gpu_batch` argument for `render`.
        # This requires knowing the structure of `threedgrut.datasets.protocols.Batch`.
        # Let's *assume* a simpler call structure for now, possibly directly using _Autograd's logic
        # or assuming render() can be called differently.
        # This part is highly speculative without the exact API definition.

        # --- Alternative: Try mimicking _Autograd call structure ---
        # This requires access to the internal tracer_wrapper object used in tracer.py,
        # which might not be exposed directly.
        # Let's assume `self.tracer.trace(...)` exists mirroring the C++ binding call
        try:
            # This call structure matches the C++ trace function arguments seen in tracer.py Autograd.forward
            # Note: ray_ori, ray_dir, ray_time might need generation if not handled internally
            #       Let's provide dummy/placeholder values first to check structure.
            dummy_rays_o = torch.zeros((viewpoint_camera.image_height, viewpoint_camera.image_width, 3), device=self.device)
            dummy_rays_d = torch.zeros((viewpoint_camera.image_height, viewpoint_camera.image_width, 3), device=self.device)
            dummy_ray_time = torch.ones((viewpoint_camera.image_height, viewpoint_camera.image_width, 1), device=self.device, dtype=torch.long) * sensor_poses.timestamps_us[0]

            # Direct call to the trace functionality (assuming it's accessible like this)
            # Need to handle the tracer_wrapper object - let's assume self.tracer holds it or provides access
            # This is a strong assumption. The intended use might be self.tracer.render(batch_info)
            if hasattr(self.tracer, 'tracer_wrapper'): # Check if internal object exists
                tracer_wrapper = self.tracer.tracer_wrapper
                ray_radiance_density, ray_hit_distance, ray_hit_count, mog_visibility = tracer_wrapper.trace(
                    frame_id,
                    n_active_features,
                    torch.cat([mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)], dim=1).contiguous(), # particle_density
                    mog_sph.contiguous(), # particle_radiance
                    dummy_rays_o.contiguous(), # ray_ori
                    dummy_rays_d.contiguous(), # ray_dir
                    dummy_ray_time.contiguous(), # ray_time
                    sensor_params.contiguous(), # sensor_params (intrinsics)
                    sensor_poses.timestamps_us[0], # ts_begin_us
                    sensor_poses.timestamps_us[1], # ts_end_us
                    sensor_poses.T_world_sensors[0].contiguous(), # T_world_sensor_begin
                    sensor_poses.T_world_sensors[1].contiguous(), # T_world_sensor_end
                )
            else:
                # Fallback: Assume render method handles this internally
                # This needs the 'Batch' object properly constructed.
                # Placeholder call - likely incorrect structure
                threestudio.warn("Tracer wrapper not accessible, attempting fallback with dummy batch.")
                # Construct a minimal dictionary mimicking Batch if needed
                dummy_gpu_batch = {
                    'intrinsics': sensor_params,
                    'poses': torch.stack([torch.from_numpy(sensor_model.T_world_sensors[0]), torch.from_numpy(sensor_model.T_world_sensors[1])]), # Example structure
                    'timestamps_us': torch.tensor(sensor_poses.timestamps_us),
                    'height': viewpoint_camera.image_height,
                    'width': viewpoint_camera.image_width,
                    # Add other fields based on threedgrut.datasets.protocols.Batch if possible
                }
                # This render call signature is a guess based on common patterns
                render_output = self.tracer.render(
                    gaussians=pc_data, # Pass the GaussianModel object
                    gpu_batch=dummy_gpu_batch, # Pass the constructed batch info
                    train=self.training,
                    frame_id=frame_id
                )
                # Unpack results from render_output - structure unknown
                ray_radiance_density = render_output.get("radiance_density", torch.zeros_like(dummy_rays_o)) # Guess key
                ray_hit_distance = render_output.get("hit_distance", torch.zeros_like(dummy_rays_o[..., :1])) # Guess key

            # --- Process outputs ---
            # ray_radiance_density seems to contain [R, G, B, Density] (4 channels)
            # ray_hit_distance is the depth
            rendered_image = ray_radiance_density[..., :3] # Extract RGB
            rendered_alpha = ray_radiance_density[..., 3:4] # Extract Density/Alpha
            rendered_depth = ray_hit_distance

            # Reshape to expected output format [B, C, H, W] - Assume batch size 1 for now
            # Tracer output might be [H, W, C]
            rendered_image = rendered_image.permute(2, 0, 1).unsqueeze(0) # H, W, 3 -> 1, 3, H, W
            rendered_alpha = rendered_alpha.permute(2, 0, 1).unsqueeze(0) # H, W, 1 -> 1, 1, H, W
            rendered_depth = rendered_depth.permute(2, 0, 1).unsqueeze(0) # H, W, 1 -> 1, 1, H, W

            # TODO: Handle normals if tracer provides them

            return {
                "render": rendered_image,
                "depth": rendered_depth,
                "mask": rendered_alpha,
                # "normal": rendered_normal, # Add if available
            }

        except Exception as e:
            threestudio.error(f"Error during ThreeDGUT trace: {e}")
            # Return empty/default tensors on error
            h, w = viewpoint_camera.image_height, viewpoint_camera.image_width
            return {
                "render": torch.zeros((1, 3, h, w), device=self.device),
                "depth": torch.zeros((1, 1, h, w), device=self.device),
                "mask": torch.zeros((1, 1, h, w), device=self.device),
            }

    def _space_cache_to_pc(
        self,
        space_cache: Float[Tensor, "B ..."],
    ) -> List[Dict[str, Tensor]]:
        pc_list = []
        for i in range(len(space_cache)):
            _space_cache = space_cache[i]
            pc_data = {
                "xyz": _space_cache["gs_xyz"],
                "rgb": _space_cache["gs_rgb"],
                "scale": _space_cache["gs_scale"],
                "rotation": _space_cache["gs_rotation"],
                "opacity": _space_cache["gs_opacity"],
            }
            pc_list.append(pc_data)
        return pc_list

    def forward(
        self, 
        batch
    ):
        space_cache = batch['space_cache']

        batch_size = batch["c2w"].shape[0]
        batch_size_space_cache = len(space_cache)
        num_views_per_batch = batch_size // batch_size_space_cache

        pc_list = self._space_cache_to_pc(space_cache)

        renders = []
        normals = []
        depths = []
        masks = []

        w2cs = []
        projs = []
        cam_ps = []
        # 在批处理循环中添加
        for pc_index, pc_data in enumerate(pc_list):
            # 释放其他点云的内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 只处理当前点云
            for batch_idx in range(pc_index * num_views_per_batch, (pc_index + 1) * num_views_per_batch):                
                # print(f"batch_idx: {batch_idx}")
                batch["batch_idx"] = batch_idx
                fovy = batch["fovy"][batch_idx]
                fovx = batch["fovx"][batch_idx] if "fovx" in batch else fovy
                c2w = batch["c2w"][batch_idx]           
                # TODO: check if this is correct
                w2c, proj, cam_p = get_cam_info_gaussian(
                    c2w=c2w, 
                    fovx=fovx, 
                    fovy=fovy, 
                    znear=0.1, 
                    zfar=100
                )
                # # TODO: check if this is correct
                # w2c = torch.inverse(c2w)
                # proj = batch["mvp_mtx"][batch_idx]
                # cam_p = batch["camera_positions"][batch_idx]

                viewpoint_cam = Camera(
                    FoVx=fovx,
                    FoVy=fovy,
                    image_width=batch["width"],
                    image_height=batch["height"],
                    world_view_transform=w2c,
                    full_proj_transform=proj,
                    camera_center=cam_p,
                )

                with autocast(enabled=False):
                    render_pkg = self._forward(
                        viewpoint_cam, 
                        pc_data,
                        self.background_tensor,
                        **batch
                    )
                    # 立即释放渲染结果的内存
                    with torch.cuda.stream(torch.cuda.Stream()):
                        # 处理渲染结果
                        if "render" in render_pkg:
                            renders.append(render_pkg["render"])
                        if "normal" in render_pkg:
                            normals.append(render_pkg["normal"])
                        if "depth" in render_pkg:
                            depths.append(render_pkg["depth"])
                        if "mask" in render_pkg:
                            masks.append(render_pkg["mask"])

                w2cs.append(w2c)
                projs.append(proj)
                cam_ps.append(cam_p)

                
        height = batch["height"]
        width = batch["width"]

        outputs = {
            "comp_rgb": torch.stack(renders, dim=0).permute(0, 2, 3, 1),
        }
        if len(masks) > 0:
            opacity = torch.stack(masks, dim=0).permute(0, 2, 3, 1)
            outputs.update(
                {
                    "opacity": opacity,
                }
            )
        if len(normals) > 0:
            comp_normal = torch.stack(normals, dim=0).permute(0, 2, 3, 1)
            comp_normal = F.normalize(comp_normal, dim=-1)
            outputs.update(
                {
                    "comp_normal": comp_normal,
                }
            )

            if self.cfg.normal_direction == "camera":
                # for compatibility with RichDreamer #############
                bg_normal = 0.5 * torch.ones_like(comp_normal)
                bg_normal[:, 2] = 1.0 # for a blue background
                bg_normal_white = torch.ones_like(comp_normal)

                # # convert_normal_to_cam_space
                # # TODO: check if this is correct
                # w2c: Float[Tensor, "B 4 4"] = torch.stack(w2cs, dim=0)
                # rot: Float[Tensor, "B 3 3"] = w2c[:, :3, :3]
                # # TODO: check if this is correct
                # w2c: Float[Tensor, "B 4 4"] = torch.inverse(batch["c2w"])
                # rot: Float[Tensor, "B 3 3"] = w2c[:, :3, :3]

                # comp_normal_cam = comp_normal.view(batch_size, -1, 3) @ rot.permute(0, 2, 1)
                comp_normal_cam = comp_normal.view(batch_size, -1, 3)
                flip_x = torch.eye(3, device=comp_normal_cam.device) #  pixel space flip axis so we need built negative y-axis normal
                # flip_x[0, 0] = -1
                # flip_x[1, 1] = -1
                flip_x[2, 2] = -1
                comp_normal_cam = comp_normal_cam @ flip_x[None, :, :]
                comp_normal_cam = comp_normal_cam.view(batch_size, height, width, 3)
                # comp_normal_cam = comp_normal * -1
                


                comp_normal_cam_vis = (comp_normal_cam + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal
                comp_normal_cam_vis_white = (comp_normal_cam + 1.0) / 2.0 * opacity + (1 - opacity) * bg_normal_white

                outputs.update(
                    {
                        "comp_normal_cam_vis": comp_normal_cam_vis.view(batch_size, height, width, 3),
                        "comp_normal_cam_vis_white": comp_normal_cam_vis_white.view(batch_size, height, width, 3),
                    }
                )
            else:
                raise ValueError(f"Unknown normal direction: {self.cfg.normal_direction}")


        if len(depths) > 0:
            depth = torch.stack(depths, dim=0).permute(0, 2, 3, 1)
            # TODO: check if this is correct
            camera_distances = torch.stack(cam_ps, dim=0).norm(dim=-1, p=2)[:, None, None, None]  # 2-norm of camera_positions
            far = camera_distances + torch.sqrt(3 * torch.ones(1, 1, 1, 1, device=camera_distances.device))
            near = camera_distances - torch.sqrt(3 * torch.ones(1, 1, 1, 1, device=camera_distances.device))
            # # TODO: check if this is correct
            # far = camera_distances + torch.sqrt(3 * torch.ones(1, 1, 1, 1, device=batch["camera_distances"].device))
            # near = batch["camera_distances"].reshape(-1, 1, 1, 1) - torch.sqrt(3 * torch.ones(1, 1, 1, 1, device=batch["camera_distances"].device))
            disparity_tmp = depth * opacity + (1.0 - opacity) * far
            disparity_norm = (far - disparity_tmp) / (far - near)
            disparity_norm = torch.clamp(disparity_norm, 0.0, 1.0)
            outputs.update(
                {
                    "depth": depth,
                    "disparity": disparity_norm,
                }
            )
        return outputs 