# Guided Euler solver — injects image-evidence gradient after each flow step
import torch
import torch.nn.functional as F
from sam3d_objects.model.backbone.generator.flow_matching.solver import Euler, linear_approximation_step
from sam3d_objects.data.utils import tree_tensor_map


class GuidedEuler(Euler):
    """
    Drop-in replacement for Euler that applies image-evidence guidance
    after each ODE step via Universal Guidance (Bansal et al., CVPR 2023).

    Args:
        guidance_fn:   callable(x_t, t) -> scalar loss tensor (with grad)
        guidance_scale: float, gradient step size
        start_t:       float in [0,1], guidance active when t < start_t
        end_t:         float in [0,1], guidance active when t > end_t
    """
    def __init__(self, guidance_fn=None, guidance_scale=1.0, start_t=0.8, end_t=0.0, **kwargs):
        super().__init__(**kwargs)
        self.guidance_fn = guidance_fn
        self.guidance_scale = guidance_scale
        self.start_t = start_t
        self.end_t = end_t

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        # 1. standard Euler step
        x_tp1 = super().step(dynamics_fn, x_t, t, dt, *args, **kwargs)

        # 2. guidance: only apply within [end_t, start_t]
        t_val = t.item() if isinstance(t, torch.Tensor) else float(t)
        if self.guidance_fn is not None and self.end_t < t_val < self.start_t:
            x_tp1 = self._apply_guidance(x_tp1, t_val)

        return x_tp1

    def _apply_guidance(self, x_t, t):
        # x_t is a dict {"shape": tensor} for MM-DiT or a plain tensor
        if isinstance(x_t, dict):
            shape_latent = x_t["shape"].detach().requires_grad_(True)
            loss = self.guidance_fn(shape_latent, t)
            grad = torch.autograd.grad(loss, shape_latent)[0]
            guided = {**x_t, "shape": (shape_latent - self.guidance_scale * grad).detach()}
        else:
            x = x_t.detach().requires_grad_(True)
            loss = self.guidance_fn(x, t)
            grad = torch.autograd.grad(loss, x)[0]
            guided = (x - self.guidance_scale * grad).detach()
        return guided