import torch

def intrinsics_from_focal_center(fx, fy, cx, cy):
    K = torch.zeros((3, 3), device=fx.device, dtype=fx.dtype)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1.0
    return K
