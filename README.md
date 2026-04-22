# Hunyuan3D-2 — cleaned Apple-Silicon workbench

This repository is a reorganized working copy of Hunyuan3D-2 with extra Apple Silicon and texture-pipeline work. The goal here is not to mirror Tencent's original README one-for-one. The goal is to make this repo understandable, runnable, and hackable without having to mentally decode fifty checkpoint names first.

If you only remember one thing, remember this:

**DiT makes shape. Paint makes texture. Delight cleans the input image.**

Everything else in the model zoo is mostly a question of size, speed, or whether the model expects one image or many.

---

## What this repo is for

This fork is mainly being used for three things:

1. **Generating meshes from images** on Apple Silicon.
2. **Painting / texturing meshes** on Apple Silicon.
3. **Profiling and optimizing the texture pipeline**, especially UV unwrap, rasterization, and multiview paint performance.

The texture side has already been modified locally to improve real performance on Mac:

- compiled mesh inpaint is enabled instead of the slow Python fallback
- UV unwrap can be cached
- UV unwrap can optionally overlap with diffusion
- an experimental GPU cube-projection UV backend exists
- output artifacts are now organized into folders instead of being dumped into one flat `outputs/` directory

---

## Repo layout

Run commands from the repo root.

```text
Hunyuan3D-2/
├── assets/                     # input images, templates, static assets
├── docs/                       # upstream docs build files
├── examples/
│   ├── debug/                  # probes, extraction tools, render checks
│   ├── mps/                    # Apple Silicon / MPS-focused scripts
│   ├── shape/                  # shape-only generation examples
│   └── texture/                # texture / textured-shape examples
├── hy3dgen/                    # library code
├── outputs/
│   ├── benchmark/              # speed-test artifacts and optimized outputs
│   ├── compare/                # side-by-side comparison images
│   ├── custom/                 # custom one-off generation results
│   ├── debug/
│   │   ├── extracted/          # textures extracted from GLBs
│   │   ├── rendered_views/     # rendered view checks
│   │   └── stages/             # intermediate paint-stage dumps
│   └── demo/                   # canonical demo mesh + textured demo outputs
├── minimal_demo.py
├── gradio_app.py
├── api_server.py
└── README.md
```

There are two small companion READMEs now as well:

- [`examples/README.md`](examples/README.md)
- [`outputs/README.md`](outputs/README.md)

---

## Model names, in plain English

Hunyuan names are noisy, but the structure is simple.

| Name piece | Meaning |
|---|---|
| `DiT` | shape generator |
| `Paint` | texture / material generator |
| `Delight` | preprocess input image to remove lighting / highlights |
| `mini` | smaller shape model |
| `mv` | multiview shape model; expects multiple input views |
| `2.0` / `2.1` | model generation / release family |
| `Fast` | guidance-distilled faster version |
| `Turbo` | step-distilled faster version |
| `paintpbr` | newer 2.1 paint/material model with PBR-oriented output |

The easiest way to decode a checkpoint name is:

- `hunyuan3d-dit-*` → **shape**
- `hunyuan3d-paint*` → **texture**
- `hunyuan3d-delight-*` → **image cleanup before texture**

---

## Model zoo cheat sheet

### The Hugging Face repos

Tencent spread the checkpoints across a few Hugging Face repos. Think of each repo as a container, and the `subfolder` as the actual model you choose.

| HF repo | What lives there | Link |
|---|---|---|
| `tencent/Hunyuan3D-2` | original 2.0 shape, paint, delight, turbo variants | https://huggingface.co/tencent/Hunyuan3D-2 |
| `tencent/Hunyuan3D-2.1` | newer 2.1 shape + paint PBR model | https://huggingface.co/tencent/Hunyuan3D-2.1 |
| `tencent/Hunyuan3D-2mini` | smaller single-image shape models | https://huggingface.co/tencent/Hunyuan3D-2mini |
| `tencent/Hunyuan3D-2mv` | multiview shape models | https://huggingface.co/tencent/Hunyuan3D-2mv |

### The actual checkpoints

| HF repo | Subfolder | Stage | Input style | Size | What it is | When to use it | Link |
|---|---|---:|---|---:|---|---|---|
| `tencent/Hunyuan3D-2.1` | `hunyuan3d-dit-v2-1` | shape | single image | 3.0B | newer high-capacity 2.1 shape model | use when you want the newest single-image shape model and can afford it | https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main/hunyuan3d-dit-v2-1 |
| `tencent/Hunyuan3D-2.1` | `hunyuan3d-paintpbr-v2-1` | texture | mesh + image | 1.3B | newer 2.1 PBR-aware paint/material model | use when you want the best newer paint model and have the 2.1 assets set up correctly | https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main/hunyuan3d-paintpbr-v2-1 |
| `tencent/Hunyuan3D-2mini` | `hunyuan3d-dit-v2-mini` | shape | single image | 0.6B | small single-image shape model | best practical starting point on Apple Silicon | https://huggingface.co/tencent/Hunyuan3D-2mini/tree/main/hunyuan3d-dit-v2-mini |
| `tencent/Hunyuan3D-2mini` | `hunyuan3d-dit-v2-mini-fast` | shape | single image | 0.6B | faster guidance-distilled mini model | use when you want more speed than base mini | https://huggingface.co/tencent/Hunyuan3D-2mini/tree/main/hunyuan3d-dit-v2-mini-fast |
| `tencent/Hunyuan3D-2mini` | `hunyuan3d-dit-v2-mini-turbo` | shape | single image | 0.6B | step-distilled faster mini model | use when you want the fastest mini-ish shape path | https://huggingface.co/tencent/Hunyuan3D-2mini/tree/main/hunyuan3d-dit-v2-mini-turbo |
| `tencent/Hunyuan3D-2mv` | `hunyuan3d-dit-v2-mv` | shape | multiview images | 1.1B | multiview shape model | use when you actually have several views of the object | https://huggingface.co/tencent/Hunyuan3D-2mv/tree/main/hunyuan3d-dit-v2-mv |
| `tencent/Hunyuan3D-2mv` | `hunyuan3d-dit-v2-mv-fast` | shape | multiview images | 1.1B | faster multiview shape model | use for faster multiview experiments | https://huggingface.co/tencent/Hunyuan3D-2mv/tree/main/hunyuan3d-dit-v2-mv-fast |
| `tencent/Hunyuan3D-2mv` | `hunyuan3d-dit-v2-mv-turbo` | shape | multiview images | 1.1B | turbo multiview shape model | fastest multiview shape path | https://huggingface.co/tencent/Hunyuan3D-2mv/tree/main/hunyuan3d-dit-v2-mv-turbo |
| `tencent/Hunyuan3D-2` | `hunyuan3d-dit-v2-0` | shape | single image | 1.1B | original 2.0 single-image shape model | use if you specifically want 2.0 base behavior | https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-dit-v2-0 |
| `tencent/Hunyuan3D-2` | `hunyuan3d-dit-v2-0-fast` | shape | single image | 1.1B | faster 2.0 shape model | use for faster 2.0 shape inference | https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-dit-v2-0-fast |
| `tencent/Hunyuan3D-2` | `hunyuan3d-dit-v2-0-turbo` | shape | single image | 1.1B | turbo 2.0 shape model | use for fastest 2.0-style shape inference | https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-dit-v2-0-turbo |
| `tencent/Hunyuan3D-2` | `hunyuan3d-paint-v2-0` | texture | mesh + image | 1.3B | original 2.0 paint model | use if you want base 2.0 paint behavior | https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-paint-v2-0 |
| `tencent/Hunyuan3D-2` | `hunyuan3d-paint-v2-0-turbo` | texture | mesh + image | 1.3B | faster turbo paint model | currently the most practical paint model in this repo on Mac | https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-paint-v2-0-turbo |
| `tencent/Hunyuan3D-2` | `hunyuan3d-delight-v2-0` | preprocess | image | 1.3B | input-image delight / de-lighting model | optional helper before paint; often skipped on MPS | https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-delight-v2-0 |

---

## Which models should you actually use?

For this repo, the practical choices are:

| Use case | Recommended choice | Why |
|---|---|---|
| single-image shape on Apple Silicon | `tencent/Hunyuan3D-2mini` + `hunyuan3d-dit-v2-mini` | best simple starting point for Mac work |
| fastest single-image shape experiments | `tencent/Hunyuan3D-2mini` + `hunyuan3d-dit-v2-mini-turbo` | smaller and more speed-oriented |
| multiview shape from several photos | `tencent/Hunyuan3D-2mv` + `hunyuan3d-dit-v2-mv` or `-turbo` | built for multiple input views |
| texture on Apple Silicon right now | `tencent/Hunyuan3D-2` + `hunyuan3d-paint-v2-0-turbo` | currently the most reliable path in this repo |
| newer texture/material model | `tencent/Hunyuan3D-2.1` + `hunyuan3d-paintpbr-v2-1` | the one to use when you specifically want 2.1 paint and the local setup is correct |

Current local reality: the MPS paint scripts in this repo try **2.1 paint first** and then fall back to **2.0 turbo paint** if 2.1 fails to load.

---

## Example scripts

All examples now live in subfolders and have shorter names.

### Apple Silicon / MPS examples

| Script | What it does |
|---|---|
| `examples/mps/demo_shape_paint_mps.py` | generate a demo shape and then texture it on MPS |
| `examples/mps/paint_demo_mps.py` | texture an already-generated demo mesh on MPS |
| `examples/mps/custom_shape_paint_mps.py` | generate and texture a custom image on MPS |

### Shape examples

| Script | What it does |
|---|---|
| `examples/shape/shape_from_image.py` | standard single-image shape generation |
| `examples/shape/shape_from_image_mini.py` | shape generation with the mini model |
| `examples/shape/shape_from_image_v21.py` | shape generation with the 2.1 model |
| `examples/shape/shape_from_multiview.py` | multiview shape generation |
| `examples/shape/shape_multiview_fast.py` | faster multiview shape example |
| `examples/shape/shape_flashvdm_fast.py` | shape example using FlashVDM |
| `examples/shape/shape_mini_turbo_flashvdm.py` | fast mini turbo FlashVDM shape example |

### Texture / textured-shape examples

| Script | What it does |
|---|---|
| `examples/texture/textured_shape_from_image.py` | end-to-end textured shape generation |
| `examples/texture/textured_shape_from_image_mini.py` | end-to-end textured shape generation with mini shape model |
| `examples/texture/textured_shape_from_multiview.py` | multiview textured-shape example |
| `examples/texture/texture_multiview_fast.py` | faster multiview texture example |

### Debug / inspection examples

| Script | What it does |
|---|---|
| `examples/debug/paint_stage_dump.py` | saves intermediate paint stages |
| `examples/debug/extract_glb_textures.py` | extracts embedded textures from GLBs |
| `examples/debug/render_demo_views.py` | renders views from the demo textured mesh |
| `examples/debug/mtl_render_probe.py` | checks MTL raster normal/position rendering |
| `examples/debug/raster_direct_probe.py` | low-level mtldiffrast raster probe |

---

## Quick start

Generate the demo shape and texture it on Apple Silicon:

```bash
python examples/mps/demo_shape_paint_mps.py
```

Texture the existing demo mesh only:

```bash
python examples/mps/paint_demo_mps.py
```

Generate and texture your own image:

```bash
python examples/mps/custom_shape_paint_mps.py --image path/to/image.png --prefix my_object
```

Extract embedded textures from a GLB:

```bash
python examples/debug/extract_glb_textures.py outputs/demo/demo_textured_mps.glb
```

---

## Output folders

Generated assets are now grouped by purpose instead of dumped into one flat folder.

| Folder | What goes there |
|---|---|
| `outputs/demo/` | canonical demo outputs |
| `outputs/custom/` | one-off custom runs |
| `outputs/benchmark/` | speed-test outputs, optimized variants, profile artifacts |
| `outputs/compare/` | visual comparison images |
| `outputs/debug/extracted/` | extracted textures |
| `outputs/debug/rendered_views/` | rendered inspection views |
| `outputs/debug/stages/` | intermediate paint-stage images |

---

## Apple Silicon notes

This repo has local texture-pipeline work focused on MPS performance. The honest version is:

- the renderer itself was not the main bottleneck
- CPU UV unwrap and texture-pipeline overhead mattered a lot
- the big remaining cost is the multiview paint diffusion step
- `hunyuan3d-paint-v2-0-turbo` is currently the most practical Mac paint path in this repo
- `hunyuan3d-paintpbr-v2-1` is the 2.1 paint model we want to use when the local 2.1 setup is fully clean

---

## Original upstream project

This repository is derived from Tencent's Hunyuan3D-2 work. If you want the original upstream presentation, papers, and announcements, see the Tencent repo and Hugging Face pages linked above.
