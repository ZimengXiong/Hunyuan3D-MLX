from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import numpy as np
import torch
import trimesh
from PIL import Image

from hy3dgen.texgen import Hunyuan3DPaintPipeline


def save_pil_list(imgs, prefix, out_dir):
    for i, im in enumerate(imgs):
        im.save(os.path.join(out_dir, f"{prefix}_{i}.png"))


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    out_dir = 'outputs/debug/stages'
    os.makedirs(out_dir, exist_ok=True)
    mesh = trimesh.load('outputs/demo/demo_shape_mps.glb', force='mesh')
    image = Image.open('assets/demo.png').convert('RGBA')

    try:
        painter = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2.1', subfolder='hunyuan3d-paintpbr-v2-1')
        print('using 2.1 paint')
    except Exception as e:
        print('2.1 failed, fallback 2.0 turbo:', e)
        painter = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2', subfolder='hunyuan3d-paint-v2-0-turbo')

    painter.config.render_size = 1024
    painter.config.texture_size = 1024
    painter.render.set_default_render_resolution(1024)
    painter.render.set_default_texture_resolution(1024)

    images_prompt = [painter.recenter_image(image)]
    images_prompt = [painter.models['delight_model'](im) for im in images_prompt]
    save_pil_list(images_prompt, 'stage_prompt', out_dir)

    mesh = painter.models and mesh
    from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap
    mesh = mesh_uv_wrap(mesh)
    painter.render.load_mesh(mesh)

    elevs = painter.config.candidate_camera_elevs
    azims = painter.config.candidate_camera_azims
    weights = painter.config.candidate_view_weights

    normal_maps = painter.render_normal_multiview(elevs, azims, use_abs_coor=True)
    position_maps = painter.render_position_multiview(elevs, azims)
    save_pil_list(normal_maps, 'stage_normal', out_dir)
    save_pil_list(position_maps, 'stage_position', out_dir)

    camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in zip(azims, elevs)]

    multiviews = painter.models['multiview_model'](images_prompt, normal_maps + position_maps, camera_info)
    for i, mv in enumerate(multiviews):
        mv = mv.resize((painter.config.render_size, painter.config.render_size))
        mv.save(os.path.join(out_dir, f'stage_multiview_{i}.png'))
    print('saved multiviews')

    texture, mask = painter.bake_from_multiview(multiviews, elevs, azims, weights, method=painter.config.merge_method)
    tex_np = (texture.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    mask_np = (mask.squeeze(-1).detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(tex_np).save(os.path.join(out_dir, 'stage_baked_texture.png'))
    Image.fromarray(mask_np).save(os.path.join(out_dir, 'stage_baked_mask.png'))
    print('saved baked texture/mask')

    texture2 = painter.texture_inpaint(texture, mask_np)
    Image.fromarray(texture2.astype(np.uint8)).save(os.path.join(out_dir, 'stage_inpaint_texture.png'))
    print('saved inpaint texture')


if __name__ == '__main__':
    main()
