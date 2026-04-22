# Examples

Run examples from the repo root.

The folder is now grouped by purpose instead of being flat.

## `examples/mps/`

Apple Silicon / MPS-focused scripts.

- `demo_shape_paint_mps.py` — generate the demo shape, then texture it
- `paint_demo_mps.py` — texture the existing demo mesh only
- `custom_shape_paint_mps.py` — generate and texture a custom image

## `examples/shape/`

Shape-only generation examples.

- `shape_from_image.py`
- `shape_from_image_mini.py`
- `shape_from_image_v21.py`
- `shape_from_multiview.py`
- `shape_multiview_fast.py`
- `shape_flashvdm_fast.py`
- `shape_mini_turbo_flashvdm.py`

## `examples/texture/`

Texture or textured-shape examples.

- `textured_shape_from_image.py`
- `textured_shape_from_image_mini.py`
- `textured_shape_from_multiview.py`
- `texture_multiview_fast.py`

## `examples/debug/`

Inspection and profiling helpers.

- `paint_stage_dump.py`
- `extract_glb_textures.py`
- `render_demo_views.py`
- `mtl_render_probe.py`
- `raster_direct_probe.py`
