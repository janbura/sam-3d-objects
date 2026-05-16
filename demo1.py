import argparse
import sys
import os

os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
os.environ["LIDRA_SKIP_INIT"] = "true"
sys.path.append("notebook")

import torch
import torch.nn.functional as F
import numpy as np

from inference import Inference, load_image, load_single_mask
from sam3d_objects.model.backbone.generator.flow_matching.solver import Euler


class GuidedEuler(Euler):
    def __init__(self, guidance_fn, guidance_scale=5.0, start_t=0.8, end_t=0.1):
        super().__init__()
        self.guidance_fn = guidance_fn
        self.guidance_scale = guidance_scale
        self.start_t = start_t
        self.end_t = end_t

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        x_tp1 = super().step(dynamics_fn, x_t, t, dt, *args, **kwargs)
        t_val = t.item() if torch.is_tensor(t) else float(t)
        if self.end_t < t_val < self.start_t:
            x_tp1 = self._guide(x_tp1)
        return x_tp1

    def _guide(self, x_t):
        if isinstance(x_t, dict):
            lat = x_t["shape"].detach().requires_grad_(True)
            with torch.enable_grad():
                loss = self.guidance_fn(lat)
                grad = torch.autograd.grad(loss, lat)[0]
            return {**x_t, "shape": (lat - self.guidance_scale * grad).detach()}
        else:
            lat = x_t.detach().requires_grad_(True)
            with torch.enable_grad():
                loss = self.guidance_fn(lat)
                grad = torch.autograd.grad(loss, lat)[0]
            return (lat - self.guidance_scale * grad).detach()


def make_guidance_fn(mask_np, ss_decoder, device="cuda"):
    target = torch.from_numpy(mask_np.astype(np.float32)).to(device)
    H, W = target.shape

    def guidance_fn(shape_latent):
        decoded = ss_decoder(
            shape_latent.permute(0, 2, 1).contiguous().view(shape_latent.shape[0], 8, 16, 16, 16)
        )
        occ = torch.sigmoid(decoded.squeeze(0).squeeze(0))
        sil_small = occ.max(dim=2).values
        sil = F.interpolate(
            sil_small.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze()
        # use logits-safe loss
        return F.binary_cross_entropy_with_logits(sil, target)

    return guidance_fn


# patch — ShortCut wraps a FlowMatching, solver is on the inner model
ss_gen = inference._pipeline.models["ss_generator"]
guided_solver = GuidedEuler(guidance_fn=guidance_fn, guidance_scale=5.0, start_t=0.8, end_t=0.1)

# try both locations
if hasattr(ss_gen, '_solver'):
    ss_gen._solver = guided_solver
if hasattr(ss_gen, 'flow_matching') and hasattr(ss_gen.flow_matching, '_solver'):
    ss_gen.flow_matching._solver = guided_solver
if hasattr(ss_gen, 'reverse_fn') and hasattr(ss_gen.reverse_fn, '_solver'):
    ss_gen.reverse_fn._solver = guided_solver


tag = "hf"
inference = Inference(f"checkpoints/{tag}/pipeline.yaml", compile=False)

image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask  = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

ss_decoder = inference._pipeline.models["ss_decoder"]
guidance_fn = make_guidance_fn(mask, ss_decoder, device="cuda")

inference._pipeline.models["ss_generator"]._solver = GuidedEuler(
    guidance_fn=guidance_fn,
    guidance_scale=5.0,
    start_t=0.8,
    end_t=0.1,
)

output = inference(image, mask, seed=42)
output["gs"].save_ply("splat_guided.ply")
print("Saved to splat_guided.ply")