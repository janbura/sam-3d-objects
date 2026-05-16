from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F

from sam3d_objects.model.backbone.generator.flow_matching.solver import ODESolver


_kaolin = None
_p3d = None


def _lazy_kaolin():
    global _kaolin
    if _kaolin is None:
        import kaolin
        _kaolin = kaolin
    return _kaolin


def _lazy_p3d():
    global _p3d
    if _p3d is None:
        from pytorch3d.renderer import (
            PerspectiveCameras,
            RasterizationSettings,
            MeshRasterizer,
            SoftSilhouetteShader,
            BlendParams,
        )
        from pytorch3d.structures import Meshes

        _p3d = {
            "PerspectiveCameras": PerspectiveCameras,
            "RasterizationSettings": RasterizationSettings,
            "MeshRasterizer": MeshRasterizer,
            "SoftSilhouetteShader": SoftSilhouetteShader,
            "BlendParams": BlendParams,
            "Meshes": Meshes,
        }

    return _p3d


def build_mask_target(mask, target_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    import numpy as np

    if isinstance(mask, np.ndarray):
        m = torch.from_numpy(mask)
    elif torch.is_tensor(mask):
        m = mask
    else:
        raise TypeError(f"unsupported mask type: {type(mask)}")

    if m.dtype == torch.bool:
        m = m.float()
    elif m.dtype == torch.uint8:
        m = m.float() / 255.0
    else:
        m = m.float()
        if m.max() > 1.5:
            m = m / 255.0

    if m.ndim == 3:
        m = m.squeeze()

    if m.ndim != 2:
        raise ValueError(f"mask must be 2D after squeezing; got {tuple(m.shape)}")

    m = m.to(device)[None, None]
    m = F.interpolate(m, size=target_hw, mode="bilinear", align_corners=False)
    return m[0, 0].clamp(0.0, 1.0)


class FlexiCubesExtractor:
    def __init__(self, grid_res: int, device: torch.device):
        self.grid_res = grid_res
        self.device = device

        kaolin = _lazy_kaolin()

        if hasattr(kaolin, "non_commercial") and hasattr(kaolin.non_commercial, "FlexiCubes"):
            FC = kaolin.non_commercial.FlexiCubes
        elif (
            hasattr(kaolin, "ops")
            and hasattr(kaolin.ops, "conversions")
            and hasattr(kaolin.ops.conversions, "FlexiCubes")
        ):
            FC = kaolin.ops.conversions.FlexiCubes
        else:
            raise RuntimeError("kaolin FlexiCubes class not found.")

        self._fc = FC(device=device)
        self._x_nx3, self._cube_fx8 = self._fc.construct_voxel_grid(grid_res)

    def extract(self, occ_logits: torch.Tensor):
        D = self.grid_res
        if occ_logits.shape != (D, D, D):
            raise ValueError(f"expected occ_logits shape {(D, D, D)}, got {tuple(occ_logits.shape)}")

        print(f"[FLEXI DEBUG] _x_nx3 shape={tuple(self._x_nx3.shape)} "
              f"range=[{self._x_nx3.min().item():.3f}, {self._x_nx3.max().item():.3f}] "
              f"dtype={self._x_nx3.dtype}")
        print(f"[FLEXI DEBUG] _cube_fx8 shape={tuple(self._cube_fx8.shape)}")

        scaled = occ_logits / 50.0
        p_inside = torch.sigmoid(scaled)
        sdf_cell = -(p_inside - 0.5)
        print(f"[FLEXI DEBUG] sdf_cell range=[{sdf_cell.min().item():.4f}, {sdf_cell.max().item():.4f}] "
              f"frac_negative={(sdf_cell < 0).float().mean().item():.3f}")

        sdf_vert = _voxel_centers_to_grid_vertices(sdf_cell)
        sdf_flat = sdf_vert.reshape(-1).contiguous()
        print(f"[FLEXI DEBUG] sdf_vert shape={tuple(sdf_vert.shape)} "
              f"range=[{sdf_vert.min().item():.4f}, {sdf_vert.max().item():.4f}]")
        print(f"[FLEXI DEBUG] sdf_flat shape={tuple(sdf_flat.shape)} "
              f"expected=N_voxelgrid_vertices={self._x_nx3.shape[0]}")

        verts, faces, _ = self._fc(
            voxelgrid_vertices=self._x_nx3,
            scalar_field=sdf_flat,
            cube_idx=self._cube_fx8,
            resolution=self.grid_res,
            training=True,
        )

        print(f"[FLEXI DEBUG] raw verts shape={tuple(verts.shape)} "
              f"X=[{verts[:,0].min().item():.3f}, {verts[:,0].max().item():.3f}] "
              f"Y=[{verts[:,1].min().item():.3f}, {verts[:,1].max().item():.3f}] "
              f"Z=[{verts[:,2].min().item():.3f}, {verts[:,2].max().item():.3f}]")
        print(f"[FLEXI DEBUG] raw faces shape={tuple(faces.shape)}")

        # verts are already in [-0.5, 0.5] from kaolin's construct_voxel_grid
        faces = faces.long()
        return verts, faces





def _voxel_centers_to_grid_vertices(field_cell: torch.Tensor) -> torch.Tensor:
    D = field_cell.shape[0]

    x = field_cell[None, None]
    x = F.interpolate(
        x,
        size=(D + 1, D + 1, D + 1),
        mode="trilinear",
        align_corners=True,
    )

    return x[0, 0]


def _opencv_intrinsics_to_p3d_focal_pp(K: torch.Tensor):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    focal = torch.stack([fx, fy])[None]
    pp = torch.stack([cx, cy])[None]

    return focal, pp


def build_p3d_camera(K: torch.Tensor, image_hw: Tuple[int, int], device: torch.device):
    p3d = _lazy_p3d()

    H, W = image_hw
    K = K.float().to(device)

    focal, pp = _opencv_intrinsics_to_p3d_focal_pp(K)

    return p3d["PerspectiveCameras"](
        focal_length=focal,
        principal_point=pp,
        in_ndc=False,
        image_size=torch.tensor([[H, W]], device=device),
        device=device,
    )


def transform_mesh_to_camera_frame(
    verts_canonical: torch.Tensor,
    R_obj: torch.Tensor,
    t_obj: torch.Tensor,
    scale_obj: torch.Tensor,
) -> torch.Tensor:
    scaled = verts_canonical * scale_obj[None, :]
    rotated = scaled @ R_obj.T
    in_cam = rotated + t_obj[None, :]

    in_p3d = in_cam.clone()
    in_p3d[:, 0] = -in_p3d[:, 0]
    in_p3d[:, 1] = -in_p3d[:, 1]

    return in_p3d


def render_silhouette(
    verts_cam: torch.Tensor,
    faces: torch.Tensor,
    cameras,
    image_hw: Tuple[int, int],
    sigma: float = 1e-4,
    gamma: float = 1e-4,
    faces_per_pixel: int = 50,
) -> torch.Tensor:
    p3d = _lazy_p3d()

    H, W = image_hw

    mesh = p3d["Meshes"](verts=[verts_cam], faces=[faces])

    blur_radius = math.log(1.0 / 1e-4 - 1.0) * sigma
    raster_settings = p3d["RasterizationSettings"](
        image_size=(H, W),
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=0,
    )

    rasterizer = p3d["MeshRasterizer"](
        cameras=cameras,
        raster_settings=raster_settings,
    )

    fragments = rasterizer(mesh)

    shader = p3d["SoftSilhouetteShader"](
        blend_params=p3d["BlendParams"](
            sigma=sigma,
            gamma=gamma,
            background_color=(0.0, 0.0, 0.0),
        )
    )

    silhouette_rgba = shader(fragments, mesh)
    return silhouette_rgba[0, ..., 3]


def silhouette_iou_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inter = (rendered * target).sum()
    union = (rendered + target - rendered * target).sum().clamp(min=1e-8)
    return 1.0 - inter / union


def _sixd_to_rotation_matrix(six: torch.Tensor) -> torch.Tensor:
    a1 = six[..., 0:3]
    a2 = six[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def extract_pose_from_state(x_t: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    six = x_t["6drotation_normalized"][0, 0]
    R = _sixd_to_rotation_matrix(six)

    t = x_t["translation"][0, 0]
    s = x_t["scale"][0, 0]

    if "translation_scale" in x_t:
        ts = x_t["translation_scale"][0, 0, 0]
        t = t * ts

    return R, t, s


@dataclass
class GuidedEulerConfig:
    guidance_scale: float = 0.1
    guide_t_lo: float = 0.50
    guide_t_hi: float = 0.65
    ramp: str = "triangle"
    grid_res: int = 64
    render_hw: Tuple[int, int] = (256, 256)
    silhouette_sigma: float = 1e-4
    silhouette_gamma: float = 1e-4
    faces_per_pixel: int = 50
    verbose: bool = False


class GuidedEuler(ODESolver):
    def __init__(
        self,
        ss_decoder: torch.nn.Module,
        target_mask: torch.Tensor,
        K: torch.Tensor,
        config: GuidedEulerConfig = GuidedEulerConfig(),
        log_fn: Callable = print,
    ):
        super().__init__()

        self.ss_decoder = ss_decoder
        self.target_mask = target_mask
        self.K = K
        self.cfg = config
        self.log_fn = log_fn
        self._flexi = None

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        print(f"t={float(t):.3f} dt={float(dt):.4f}")
        velocity = dynamics_fn(x_t, t, *args, **kwargs)
        x_new = _tree_add(x_t, _tree_scale(velocity, dt))

        t_val = float(t)
        scale = self._scale_at(t_val)

        if scale > 0:
            x_new = self._apply_guidance(x_new, t_val, scale, velocity)

        return x_new

    def solve_iter(self, dynamics_fn, x_0, t_seq, *args, **kwargs):
        x_t = x_0

        for i in range(len(t_seq) - 1):
            t = t_seq[i]
            dt = t_seq[i + 1] - t_seq[i]
            x_t = self.step(dynamics_fn, x_t, t, dt, *args, **kwargs)
            yield x_t, t_seq[i + 1]

    def solve(self, dynamics_fn, x_0, t_seq, *args, **kwargs):
        x_t = x_0

        for x_t, _ in self.solve_iter(dynamics_fn, x_0, t_seq, *args, **kwargs):
            pass

        return x_t

    def _scale_at(self, t_val: float) -> float:
        if not (self.cfg.guide_t_lo <= t_val <= self.cfg.guide_t_hi):
            return 0.0

        u = (t_val - self.cfg.guide_t_lo) / max(self.cfg.guide_t_hi - self.cfg.guide_t_lo, 1e-8)

        if self.cfg.ramp == "constant":
            return self.cfg.guidance_scale

        if self.cfg.ramp == "triangle":
            return self.cfg.guidance_scale * (1.0 - 2.0 * abs(u - 0.5))

        if self.cfg.ramp == "cosine":
            return self.cfg.guidance_scale * 0.5 * (1.0 - math.cos(2.0 * math.pi * u))

        raise ValueError(f"unknown ramp: {self.cfg.ramp}")



    def _ensure_flexi(self, device: torch.device):
        if self._flexi is None:
            self._flexi = FlexiCubesExtractor(
                grid_res=self.cfg.grid_res,
                device=device,
            )



    def _apply_guidance(self, x_t: dict, t_val: float, scale: float, velocity) -> dict:
        if not isinstance(x_t, dict) or "shape" not in x_t:
            return x_t

        x_clean = _tree_add(x_t, _tree_scale(velocity, 1.0 - t_val))

        device = x_t["shape"].device
        self._ensure_flexi(device)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            shape_clean = x_clean["shape"].detach().float().clone().requires_grad_(True)

            decoded_input = (
                shape_clean.permute(0, 2, 1)
                .contiguous()
                .view(shape_clean.shape[0], 8, 16, 16, 16)
            )

            occ_logits = self.ss_decoder(decoded_input).float()

            if occ_logits.ndim != 5:
                if self.cfg.verbose:
                    self.log_fn(f"[GUIDED t={t_val:.3f}] bad decoder output shape {tuple(occ_logits.shape)}")
                return x_t

            occ = occ_logits[0, 0]

            if self.cfg.verbose:
                self.log_fn(
                    f"[GUIDED t={t_val:.3f}] occ stats: "
                    f"min={occ.min().item():.3f} max={occ.max().item():.3f} "
                    f"mean={occ.mean().item():.3f} "
                    f"frac_positive={(occ > 0).float().mean().item():.3f} "
                    f"frac_negative={(occ < 0).float().mean().item():.3f}"
                )

            try:
                verts_canon, faces = self._flexi.extract(occ)
            except Exception as e:
                if self.cfg.verbose:
                    self.log_fn(f"[GUIDED t={t_val:.3f}] FlexiCubes failed: {e}. Skipping.")
                return x_t

            if verts_canon.numel() == 0 or faces.numel() == 0:
                if self.cfg.verbose:
                    self.log_fn(f"[GUIDED t={t_val:.3f}] empty mesh. Skipping.")
                return x_t

            # Use canonical pose: render mesh in its own frame.
            # Use canonical pose with a side view: rotate 90° around X
            # so Z (canonical vertical) becomes Y (screen vertical),
            # and original Y becomes -Z (depth into screen).
            verts_cam = verts_canon.clone()
            # Apply rotation: (x, y, z) -> (x, z, -y)
            new_verts = torch.zeros_like(verts_cam)
            new_verts[:, 0] = verts_cam[:, 0]
            new_verts[:, 1] = verts_cam[:, 2]
            new_verts[:, 2] = -verts_cam[:, 1]
            verts_cam = new_verts
            # PyTorch3D camera convention flips
            verts_cam[:, 0] = -verts_cam[:, 0]
            verts_cam[:, 1] = -verts_cam[:, 1]
            # Push in front of camera
            verts_cam[:, 2] = verts_cam[:, 2] + 2.0

            z = verts_cam[:, 2]

            if self.cfg.verbose:
                self.log_fn(
                    f"[GUIDED t={t_val:.3f}] verts_cam (canonical): "
                    f"X=[{verts_cam[:,0].min().item():.2f},{verts_cam[:,0].max().item():.2f}] "
                    f"Y=[{verts_cam[:,1].min().item():.2f},{verts_cam[:,1].max().item():.2f}] "
                    f"Z=[{verts_cam[:,2].min().item():.2f},{verts_cam[:,2].max().item():.2f}]"
                )

            # Canonical camera
            H_rdr, W_rdr = self.cfg.render_hw
            f = float(H_rdr)
            K_canonical = torch.tensor([
                [f,   0.0, W_rdr / 2.0],
                [0.0, f,   H_rdr / 2.0],
                [0.0, 0.0, 1.0],
            ], device=device, dtype=torch.float32)
            cameras = build_p3d_camera(
                K=K_canonical,
                image_hw=self.cfg.render_hw,
                device=device,
            )

            try:
                rendered = render_silhouette(
                    verts_cam=verts_cam,
                    faces=faces,
                    cameras=cameras,
                    image_hw=self.cfg.render_hw,
                    sigma=self.cfg.silhouette_sigma,
                    gamma=self.cfg.silhouette_gamma,
                    faces_per_pixel=self.cfg.faces_per_pixel,
                )
            except Exception as e:
                if self.cfg.verbose:
                    self.log_fn(f"[GUIDED t={t_val:.3f}] render failed: {e}. Skipping.")
                return x_t

            if self.cfg.verbose:
                self.log_fn(
                    f"[GUIDED t={t_val:.3f}] rendered: "
                    f"sum={rendered.sum().item():.1f} "
                    f"max={rendered.max().item():.3f}"
                )

            if self.target_mask.shape != rendered.shape:
                tgt = F.interpolate(
                    self.target_mask[None, None],
                    size=rendered.shape,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            else:
                tgt = self.target_mask

            # DEBUG: dump rendered vs target PNGs on first guided step
            if 0.80 <= t_val < 0.85:
                from PIL import Image
                import numpy as np
                r_img = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
                t_img = (tgt.detach().cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(r_img).save(f"debug_rendered_t{t_val:.2f}.png")
                Image.fromarray(t_img).save(f"debug_target_t{t_val:.2f}.png")

            loss = silhouette_iou_loss(rendered, tgt)

            if not torch.isfinite(loss):
                if self.cfg.verbose:
                    self.log_fn(f"[GUIDED t={t_val:.3f}] non-finite loss. Skipping.")
                return x_t

            grad, = torch.autograd.grad(
                loss,
                shape_clean,
                retain_graph=False,
                create_graph=False,
            )

            if not torch.isfinite(grad).all():
                if self.cfg.verbose:
                    bad = (~torch.isfinite(grad)).float().mean().item()
                    self.log_fn(f"[GUIDED t={t_val:.3f}] {bad * 100:.1f}% bad grads. Skipping.")
                return x_t

            grad_norm = grad.norm()

            if grad_norm < 1e-12:
                if self.cfg.verbose:
                    self.log_fn(f"[GUIDED t={t_val:.3f}] zero grad. Skipping. |grad|={grad_norm.item():.4e}")
                return x_t

            grad_unit = grad / (grad_norm + 1e-8)

        x_out = dict(x_t)
        x_out["shape"] = (x_t["shape"] - scale * grad_unit.to(x_t["shape"].dtype)).detach()

        if self.cfg.verbose:
            self.log_fn(
                f"[GUIDED t={t_val:.3f}] "
                f"loss={loss.item():.4f} "
                f"|grad|={grad_norm.item():.4e} "
                f"scale={scale:.4f}"
            )

        return x_out


def _tree_add(a, b):
    if isinstance(a, dict):
        return {k: _tree_add(a[k], b[k]) for k in a}

    if isinstance(a, list):
        return [_tree_add(ai, bi) for ai, bi in zip(a, b)]

    if isinstance(a, tuple):
        return tuple(_tree_add(ai, bi) for ai, bi in zip(a, b))

    return a + b


def _tree_scale(a, s):
    if isinstance(a, dict):
        return {k: _tree_scale(v, s) for k, v in a.items()}

    if isinstance(a, list):
        return [_tree_scale(v, s) for v in a]

    if isinstance(a, tuple):
        return tuple(_tree_scale(v, s) for v in a)

    return a * s