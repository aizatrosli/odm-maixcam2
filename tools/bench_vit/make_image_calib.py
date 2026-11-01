"""Pack calibration images for Pulsar2 (one float32 .npy per sample, NCHW or NCTHW).

    python3 make_image_calib.py --res 160 --out calib.tar                 # random images
    python3 make_image_calib.py --res 160 --images frames/ --out calib.tar  # real camera frames

Random images are fine when you only want timing. For a real encoder, use frames
from the duck's camera so the quantiser sees realistic ranges.
"""

import argparse
import io
import os
import tarfile

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--res", type=int, required=True)
p.add_argument("--frames", type=int, default=1)
p.add_argument("--images", default=None, help="folder of .jpg/.png (needs Pillow)")
p.add_argument("--count", type=int, default=32)
p.add_argument("--out", required=True)
a = p.parse_args()

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def load_images():
    if not a.images:
        rng = np.random.default_rng(0)
        for _ in range(a.count):
            yield rng.random((3, a.res, a.res), dtype=np.float32)
        return
    from PIL import Image

    files = sorted(f for f in os.listdir(a.images) if f.lower().endswith((".jpg", ".jpeg", ".png")))[: a.count * a.frames]
    for f in files:
        im = Image.open(os.path.join(a.images, f)).convert("RGB").resize((a.res, a.res))
        yield np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0


imgs = [(im - MEAN) / STD for im in load_images()]
samples = []
if a.frames == 1:
    samples = [im[None] for im in imgs]
else:
    for i in range(0, len(imgs) - a.frames + 1, a.frames):
        samples.append(np.stack(imgs[i : i + a.frames], axis=1)[None])  # [1,3,T,H,W]

with tarfile.open(a.out, "w") as tar:
    for i, s in enumerate(samples):
        buf = io.BytesIO()
        np.save(buf, s.astype(np.float32))
        info = tarfile.TarInfo(f"img_{i:05d}.npy")
        info.size = buf.tell()
        buf.seek(0)
        tar.addfile(info, buf)
print(f"wrote {len(samples)} samples to {a.out}")
