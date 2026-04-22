from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import trimesh
import mtldiffrast.torch as dr
from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

mesh = trimesh.load('outputs/demo/demo_shape_mps.glb', force='mesh')
r = MeshRender(default_resolution=512, texture_size=512, device='mps', raster_mode='mtl')
r.load_mesh(mesh)
_, pos_clip = r.get_pos_from_mvp(0, 0, None, None)
ctx = dr.MtlRasterizeContext()

rast, _ = dr.rasterize(ctx, pos_clip[0], r.pos_idx, resolution=[512,512])
print('orig hits', int((rast[..., -1] > 0).sum().cpu()), 'z range', float(pos_clip[0,:,2].min().cpu()), float(pos_clip[0,:,2].max().cpu()))

pos2 = pos_clip[0].clone()
pos2[:,2] = pos2[:,2] * 0.5 + 0.5
rast2, _ = dr.rasterize(ctx, pos2, r.pos_idx, resolution=[512,512])
print('mapped hits', int((rast2[..., -1] > 0).sum().cpu()), 'z range', float(pos2[:,2].min().cpu()), float(pos2[:,2].max().cpu()))
