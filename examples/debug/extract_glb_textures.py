from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygltflib import GLTF2
from PIL import Image
import io
import numpy as np

input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('outputs/demo/demo_textured_mps.glb')
out_dir = Path('outputs/debug/extracted')
out_dir.mkdir(parents=True, exist_ok=True)

g = GLTF2().load(str(input_path))
blob = g.binary_blob()
print('images:', len(g.images or []))
for i, img in enumerate(g.images or []):
    if img.bufferView is None:
        print(i, 'no bufferView')
        continue
    bv = g.bufferViews[img.bufferView]
    data = blob[bv.byteOffset:bv.byteOffset + bv.byteLength]
    im = Image.open(io.BytesIO(data)).convert('RGB')
    arr = np.array(im)
    print(i, im.size, 'min', int(arr.min()), 'max', int(arr.max()), 'mean', float(arr.mean()), 'std', float(arr.std()))
    out = out_dir / f'{input_path.stem}_tex_{i}.png'
    im.save(out)
    print('saved', out)
