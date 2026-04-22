# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import hashlib
import os
import tempfile

import numpy as np
import torch
import trimesh
import xatlas


def _has_valid_uv(mesh):
    return hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) == len(mesh.vertices)


def _uv_backend():
    return os.environ.get('HY3D_UV_BACKEND', 'xatlas').strip().lower()


def _auto_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _mesh_cache_key(mesh, backend):
    h = hashlib.blake2b(digest_size=16)
    h.update(backend.encode('utf-8'))
    h.update(np.asarray(mesh.vertices, dtype=np.float32).tobytes())
    h.update(np.asarray(mesh.faces, dtype=np.int32).tobytes())
    return h.hexdigest()


def _cache_dir():
    return os.environ.get('HY3D_UV_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'hy3d_uv_cache'))


def _load_cached_mesh(mesh, backend):
    cache_root = _cache_dir()
    os.makedirs(cache_root, exist_ok=True)
    cache_path = os.path.join(cache_root, f'{_mesh_cache_key(mesh, backend)}.npz')
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        mesh.vertices = cached['vertices']
        mesh.faces = cached['faces']
        mesh.visual.uv = cached['uv']
        return mesh, cache_path
    return None, cache_path


def _save_cached_mesh(mesh, cache_path):
    np.savez_compressed(cache_path, vertices=mesh.vertices, faces=mesh.faces, uv=mesh.visual.uv)


def _mesh_uv_wrap_xatlas(mesh):
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs
    return mesh


def _mesh_uv_wrap_cube_gpu(mesh):
    device = _auto_device()

    verts_np = np.asarray(mesh.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh.faces, dtype=np.int64)

    verts = torch.as_tensor(verts_np, device=device)
    faces = torch.as_tensor(faces_np, device=device)
    tri = verts[faces]  # [F, 3, 3]

    v01 = tri[:, 1] - tri[:, 0]
    v02 = tri[:, 2] - tri[:, 0]
    normals = torch.cross(v01, v02, dim=-1)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    axis = normals.abs().argmax(dim=-1)
    sign = (normals.gather(1, axis[:, None]).squeeze(1) < 0).to(torch.int64)
    chart = axis * 2 + sign  # 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z

    x = tri[..., 0]
    y = tri[..., 1]
    z = tri[..., 2]
    uv_local = torch.empty((tri.shape[0], 3, 2), dtype=tri.dtype, device=device)

    mask = chart == 0
    uv_local[mask] = torch.stack([-z[mask], y[mask]], dim=-1)
    mask = chart == 1
    uv_local[mask] = torch.stack([z[mask], y[mask]], dim=-1)
    mask = chart == 2
    uv_local[mask] = torch.stack([x[mask], -z[mask]], dim=-1)
    mask = chart == 3
    uv_local[mask] = torch.stack([x[mask], z[mask]], dim=-1)
    mask = chart == 4
    uv_local[mask] = torch.stack([x[mask], y[mask]], dim=-1)
    mask = chart == 5
    uv_local[mask] = torch.stack([-x[mask], y[mask]], dim=-1)

    uv_flat = uv_local.reshape(-1, 2)
    uv_min = uv_flat.min(dim=0).values
    uv_max = uv_flat.max(dim=0).values
    uv_local = (uv_local - uv_min) / (uv_max - uv_min).clamp_min(1e-8)

    # Pack six cube charts into a 3x2 atlas with a little padding.
    pad = 0.02
    cell_w = (1.0 - pad * 4) / 3.0
    cell_h = (1.0 - pad * 3) / 2.0
    cell_xy = torch.tensor([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
    ], dtype=tri.dtype, device=device)
    cell = cell_xy[chart]
    offset = torch.stack([
        pad + cell[:, 0] * (cell_w + pad),
        pad + cell[:, 1] * (cell_h + pad),
    ], dim=-1)
    scale = torch.tensor([cell_w, cell_h], dtype=tri.dtype, device=device)
    uvs = uv_local * scale + offset[:, None, :]

    new_vertices = tri.reshape(-1, 3).detach().cpu().numpy()
    new_faces = np.arange(new_vertices.shape[0], dtype=np.int64).reshape(-1, 3)
    new_uvs = uvs.reshape(-1, 2).detach().cpu().numpy().astype(np.float32)

    out = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=new_uvs)
    return out


def mesh_uv_wrap(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if _has_valid_uv(mesh):
        return mesh

    if len(mesh.faces) > 500000000:
        raise ValueError("The mesh has more than 500,000,000 faces, which is not supported.")

    backend = _uv_backend()

    cache_path = None
    try:
        cached_mesh, cache_path = _load_cached_mesh(mesh, backend)
        if cached_mesh is not None:
            return cached_mesh
    except Exception:
        cache_path = None

    if backend == 'xatlas':
        mesh = _mesh_uv_wrap_xatlas(mesh)
    elif backend in ('cube_gpu', 'gpu_cube', 'cube'):
        mesh = _mesh_uv_wrap_cube_gpu(mesh)
    else:
        raise ValueError(f'Unsupported HY3D_UV_BACKEND={backend}')

    if cache_path is not None:
        try:
            _save_cached_mesh(mesh, cache_path)
        except Exception:
            pass

    return mesh
