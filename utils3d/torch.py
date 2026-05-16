import torch

def intrinsics_from_focal_center(fx, fy, cx, cy):
    K = torch.zeros((3, 3), device=fx.device, dtype=fx.dtype)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1.0
    return K

def depth_to_points(depth, intrinsics):
    B, H, W = depth.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device),
        torch.arange(W, device=depth.device),
        indexing="ij",
    )
    x = x.float() / W
    y = y.float() / H

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    Z = depth
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    return torch.stack([X, Y, Z], dim=-1)
