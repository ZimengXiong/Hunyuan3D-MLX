from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import time
import torch
import trimesh
from PIL import Image

from hy3dgen.texgen import Hunyuan3DPaintPipeline


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    out_dir = "outputs/demo"
    os.makedirs(out_dir, exist_ok=True)
    mesh_path = f"{out_dir}/demo_shape_mps.glb"
    image_path = "assets/demo.png"
    out_path = f"{out_dir}/demo_textured_mps.glb"

    mesh = trimesh.load(mesh_path, force='mesh')
    image = Image.open(image_path).convert("RGBA")

    paint_model = "tencent/Hunyuan3D-2.1"
    paint_subfolder = "hunyuan3d-paintpbr-v2-1"
    try:
        painter = Hunyuan3DPaintPipeline.from_pretrained(paint_model, subfolder=paint_subfolder)
        print(f"Loaded paint model: {paint_model}/{paint_subfolder}")
    except Exception as e:
        print(f"2.1 paint load failed: {e}")
        paint_model = "tencent/Hunyuan3D-2"
        paint_subfolder = "hunyuan3d-paint-v2-0-turbo"
        painter = Hunyuan3DPaintPipeline.from_pretrained(paint_model, subfolder=paint_subfolder)
        print(f"Fallback paint model: {paint_model}/{paint_subfolder}")

    painter.config.render_size = 1024
    painter.config.texture_size = 1024
    painter.render.set_default_render_resolution(1024)
    painter.render.set_default_texture_resolution(1024)

    print(f"Paint backend: raster={painter.render.raster_mode}, device={painter.config.device}")
    t0 = time.time()
    textured = painter(mesh, image=image)
    textured.export(out_path)
    print(f"Done in {time.time() - t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
