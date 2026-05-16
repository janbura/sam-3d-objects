# Image-evidence guidance functions
import torch
import torch.nn.functional as F
from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.flexicubes.flexicubes import FlexiCubes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    SoftSilhouetteShader, PerspectiveCameras, BlendParams,
)


def make_mask_guidance_fn(target_mask: torch.Tensor, flexicubes: FlexiCubes, device: str = "cuda"):
    """
    Returns a guidance_fn(x_t, t) -> loss that penalises
    silhouette mismatch between the predicted mesh and the SAM mask.

    target_mask: (H, W) binary float tensor, values in [0, 1]
    """
    target = target_mask.to(device).float()
    H, W = target.shape

    raster_settings = RasterizationSettings(image_size=(H, W), blur_radius=1e-4, faces_per_pixel=50)
    cameras = PerspectiveCameras(device=device)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4)),
    )

    def guidance_fn(x_t, t):
        # x_t: (1, N, C) shape latent — decode to mesh via flexicubes
        try:
            mesh = flexicubes(x_t)
            silhouette = renderer(mesh)[..., 3]          # (1, H, W) alpha channel
            silhouette = silhouette.squeeze(0)           # (H, W)
            loss = F.binary_cross_entropy(silhouette.clamp(1e-6, 1 - 1e-6), target)
        except Exception:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        return loss

    return guidance_fn