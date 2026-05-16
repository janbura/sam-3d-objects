import argparse
import sys
import numpy as np
import torch
import os
from PIL import Image

sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

from guided_solver import GuidedEuler, GuidedEulerConfig, build_mask_target



def compute_intrinsics(pipeline, image, mask, device):
    """
    Run MoGe to get pointmap + intrinsics for the input image.
    """
    image_merged = pipeline.merge_image_and_mask(image, mask)
    pointmap_dict = pipeline.compute_pointmap(image_merged, None)

    # Inspect everything available
    print(f"[compute_intrinsics] pointmap_dict keys: {list(pointmap_dict.keys())}")
    for k, v in pointmap_dict.items():
        if torch.is_tensor(v):
            print(f"  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v).__name__}")

    # Get image size from the pointmap
    pointmap = pointmap_dict["pointmap"]
    if pointmap.ndim == 3:
        if pointmap.shape[0] == 3:
            _, H, W = pointmap.shape
        else:
            H, W, _ = pointmap.shape
    elif pointmap.ndim == 4:
        # (B, 3, H, W) or (B, H, W, 3)
        if pointmap.shape[1] == 3:
            _, _, H, W = pointmap.shape
        else:
            _, H, W, _ = pointmap.shape
    else:
        raise ValueError(f"unexpected pointmap ndim {pointmap.ndim}, shape {tuple(pointmap.shape)}")

    # Look for intrinsics already in the dict (different names depending on
    # the pipeline version)
    K = None
    for key in ["intrinsics", "K", "camera_intrinsics", "intrinsic"]:
        if key in pointmap_dict:
            K = pointmap_dict[key]
            print(f"[compute_intrinsics] using pointmap_dict[{key!r}]")
            break

    if K is None:
        # Last resort: estimate K from pointmap. MoGe pointmap is (3, H, W) of
        # 3D points in camera frame; pixel (u, v) views point (X, Y, Z) under
        # pinhole projection: u = fx * X/Z + cx, v = fy * Y/Z + cy. We can
        # solve for fx, fy, cx, cy from valid pixels by least squares.
        K = _estimate_K_from_pointmap(pointmap, (H, W))
        print(f"[compute_intrinsics] estimated K from pointmap geometry")

    if torch.is_tensor(K):
        K = K.to(device).float()
    else:
        K = torch.tensor(K, dtype=torch.float32, device=device)
    if K.ndim == 3:
        K = K[0]  # drop batch
    if K.shape != (3, 3):
        raise ValueError(f"K should be 3x3, got {tuple(K.shape)}")

    # MoGe returns K in normalized image coords (image spans [0, 1]).
    # Detect this and convert to pixel coords so PyTorch3D (in_ndc=False)
    # interprets it correctly. Heuristic: if cx, cy are around 0.5, normalize.
    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy = K[0, 2].item(), K[1, 2].item()
    if cx < 2.0 and cy < 2.0:
        # Normalized coords. Convert: K_px[0,:] *= W, K_px[1,:] *= H
        print(f"[compute_intrinsics] K is normalized (cx={cx:.3f}, cy={cy:.3f}); "
              f"scaling to pixels for HxW={H}x{W}")
        K_px = K.clone()
        K_px[0, 0] *= W; K_px[0, 2] *= W
        K_px[1, 1] *= H; K_px[1, 2] *= H
        K = K_px

    return K, (H, W)


def _estimate_K_from_pointmap(pointmap, image_hw):
    """
    Fit pinhole intrinsics to a pointmap by least squares on valid pixels.
    pointmap: (3, H, W) or (H, W, 3) -- 3D points in camera frame (X, Y, Z).
    Solves for fx, fy, cx, cy such that
        u = fx * X/Z + cx,   v = fy * Y/Z + cy.
    """
    if pointmap.ndim == 3 and pointmap.shape[0] == 3:
        pm = pointmap  # (3, H, W)
    elif pointmap.ndim == 3 and pointmap.shape[-1] == 3:
        pm = pointmap.permute(2, 0, 1)  # -> (3, H, W)
    elif pointmap.ndim == 4:
        # batch dim, take first
        if pointmap.shape[1] == 3:
            pm = pointmap[0]
        else:
            pm = pointmap[0].permute(2, 0, 1)
    else:
        raise ValueError(f"can't interpret pointmap shape {tuple(pointmap.shape)}")

    H, W = image_hw
    X, Y, Z = pm[0], pm[1], pm[2]  # each (H, W)
    valid = (Z > 1e-3) & torch.isfinite(Z)
    if valid.sum() < 100:
        # Fallback: assume FOV ~ 60 deg
        f = max(H, W) * 0.9
        return torch.tensor([[f, 0, W/2.0], [0, f, H/2.0], [0, 0, 1]],
                            dtype=torch.float32, device=pm.device)

    vs, us = torch.meshgrid(
        torch.arange(H, device=pm.device, dtype=pm.dtype),
        torch.arange(W, device=pm.device, dtype=pm.dtype),
        indexing="ij",
    )
    u = us[valid]; v = vs[valid]
    XoZ = X[valid] / Z[valid]
    YoZ = Y[valid] / Z[valid]

    # Solve [XoZ, 1] @ [fx, cx]^T = u   (and the same for y)
    A_x = torch.stack([XoZ, torch.ones_like(XoZ)], dim=1)
    sol_x = torch.linalg.lstsq(A_x, u.unsqueeze(1)).solution.squeeze(1)
    A_y = torch.stack([YoZ, torch.ones_like(YoZ)], dim=1)
    sol_y = torch.linalg.lstsq(A_y, v.unsqueeze(1)).solution.squeeze(1)

    fx, cx = sol_x[0], sol_x[1]
    fy, cy = sol_y[0], sol_y[1]
    K = torch.zeros(3, 3, device=pm.device, dtype=pm.dtype)
    K[0, 0] = fx; K[0, 2] = cx
    K[1, 1] = fy; K[1, 2] = cy
    K[2, 2] = 1.0
    return K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--mask-index", type=int, default=14)
    p.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    p.add_argument("--out", default="splat_guided.ply")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vanilla", action="store_true",
                   help="Skip guidance entirely (matched baseline through the same script).")
    # Guidance knobs
    p.add_argument("--guidance-scale", type=float, default=10.0)
    p.add_argument("--guide-t-lo", type=float, default=0.5)
    p.add_argument("--guide-t-hi", type=float, default=0.65)
    p.add_argument("--ramp", default="constant", choices=["constant", "triangle", "cosine"])
    p.add_argument("--render-h", type=int, default=256)
    p.add_argument("--render-w", type=int, default=256)
    p.add_argument("--sigma", type=float, default=1e-2)
    p.add_argument("--gamma", type=float, default=1e-4)
    args = p.parse_args()

    print(f"[demo_guided] loading model from {args.config}")
    inference = Inference(args.config, compile=False)
    pipeline = inference._pipeline
    device = torch.device("cuda")

    print(f"[demo_guided] loading image + mask")
    image = load_image(args.image)
    if os.path.isfile(args.mask_dir):
        mask = np.array(Image.open(args.mask_dir).convert("L")).astype(np.float32) / 255.0
    else:
        mask = load_single_mask(args.mask_dir, index=args.mask_index)

    if not args.vanilla:
        # Get intrinsics by running MoGe (the pipeline does this internally
        # at sampling time anyway; we just need K up front).
        print(f"[demo_guided] computing MoGe pointmap + intrinsics...")
        K, image_hw = compute_intrinsics(pipeline, image, mask, device)
        print(f"[demo_guided] K=\n{K.cpu().numpy()}\nimage HxW={image_hw}")

        # The render resolution can be smaller than the full image -- guidance
        # is comparing silhouettes, doesn't need full resolution.
        # Rescale K to match the render size.
        H_img, W_img = image_hw
        H_rdr, W_rdr = args.render_h, args.render_w
        sx, sy = W_rdr / W_img, H_rdr / H_img
        K_rdr = K.clone()
        K_rdr[0, 0] *= sx; K_rdr[0, 2] *= sx
        K_rdr[1, 1] *= sy; K_rdr[1, 2] *= sy

        target_mask = build_mask_target(mask, target_hw=(H_rdr, W_rdr), device=device)
        print(f"[demo_guided] target_mask {tuple(target_mask.shape)} sum={target_mask.sum().item():.1f}")

        cfg = GuidedEulerConfig(
            guidance_scale=args.guidance_scale,
            guide_t_lo=args.guide_t_lo,
            guide_t_hi=args.guide_t_hi,
            ramp=args.ramp,
            grid_res=64,
            render_hw=(H_rdr, W_rdr),
            silhouette_sigma=args.sigma,
            silhouette_gamma=args.gamma,
            verbose=True,
        )
        pipeline.guided_solver = GuidedEuler(
            ss_decoder=pipeline.models["ss_decoder"],
            target_mask=target_mask,
            K=K_rdr,
            config=cfg,
        )
        print(f"[demo_guided] guidance ON  scale={args.guidance_scale}  "
              f"window=[{args.guide_t_lo},{args.guide_t_hi}]  ramp={args.ramp}")
    else:
        if hasattr(pipeline, "guided_solver"):
            delattr(pipeline, "guided_solver")
        print("[demo_guided] guidance OFF (vanilla)")

    print("[demo_guided] sampling...")
    output = inference(image, mask, seed=args.seed)
    output["gs"].save_ply(args.out)
    print(f"[demo_guided] saved {args.out}")


if __name__ == "__main__":
    main()