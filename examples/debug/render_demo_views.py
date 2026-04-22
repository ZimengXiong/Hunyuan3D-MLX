from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import trimesh
from PIL import Image
from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

mesh = trimesh.load('outputs/demo/demo_textured_mps.glb', force='mesh')
out_dir = Path('outputs/debug/rendered_views')
out_dir.mkdir(parents=True, exist_ok=True)
r = MeshRender(default_resolution=768, texture_size=1024, device='mps', raster_mode='mtl')
r.load_mesh(mesh)
tex = Image.open('outputs/demo/demo_textured_mps.png').convert('RGB')
r.set_texture(tex)
for az in [0, 90, 180, 270]:
    img = r.render(0, az, keep_alpha=False, return_type='np')
    arr = np.clip(img * 255, 0, 255).astype(np.uint8)
    out = out_dir / f'view_{az}.png'
    Image.fromarray(arr).save(out)
    print('saved', out, 'mean', float(arr.mean()), 'std', float(arr.std()))
