import numpy as np

def depth_edge(depth, thresh=0.01):
    gx = np.gradient(depth, axis=1)
    gy = np.gradient(depth, axis=0)
    edge = np.sqrt(gx**2 + gy**2)
    return edge > thresh




def normals_edge(normals, thresh=0.1):
    gx = np.gradient(normals, axis=1)
    gy = np.gradient(normals, axis=0)
    edge = np.sqrt(sum(g**2 for g in gx) + sum(g**2 for g in gy))
    return edge > thresh



def points_to_normals(points):
    # points: (H, W, 3)
    dx = np.gradient(points, axis=1)
    dy = np.gradient(points, axis=0)

    # cross product to get normals
    normals = np.cross(dx, dy)

    # normalize
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals = normals / norm

    return normals





def image_uv(h, w):
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    uv = np.stack([xs, ys], axis=-1)
    return uv




def image_mesh(points):
    # points: (H, W, 3)
    H, W, _ = points.shape

    vertices = points.reshape(-1, 3)

    faces = []
    for i in range(H - 1):
        for j in range(W - 1):
            idx = i * W + j
            faces.append([idx, idx + 1, idx + W])
            faces.append([idx + 1, idx + W + 1, idx + W])

    return vertices, np.array(faces)
