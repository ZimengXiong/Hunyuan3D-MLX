from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import time
import torch
from PIL import Image

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_path = "assets/demo.png"
    out_dir = "outputs/demo"
    os.makedirs(out_dir, exist_ok=True)

    image = Image.open(image_path).convert("RGBA")

    print("Loading shape pipeline (Hunyuan3D-2mini)...")
    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini",
        variant="fp16",
        device=device,
    )

    t0 = time.time()
    mesh = shape(
        image=image,
        num_inference_steps=30,
        octree_resolution=256,
        num_chunks=12000,
        generator=torch.manual_seed(12345),
        output_type="trimesh",
    )[0]
    shape_path = os.path.join(out_dir, "demo_shape_mps.glb")
    mesh.export(shape_path)
    print(f"Shape generated in {time.time() - t0:.1f}s -> {shape_path}")

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("Loading paint pipeline...")
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

    # Keep memory bounded on MPS
    painter.config.render_size = 1024
    painter.config.texture_size = 1024
    painter.render.set_default_render_resolution(1024)
    painter.render.set_default_texture_resolution(1024)
    print(f"Paint backend: raster={painter.render.raster_mode}, device={painter.config.device}")

    t1 = time.time()
    textured = painter(mesh, image=image)
    tex_path = os.path.join(out_dir, "demo_textured_mps.glb")
    textured.export(tex_path)
    print(f"Texture generated in {time.time() - t1:.1f}s -> {tex_path}")


if __name__ == "__main__":
    main()
