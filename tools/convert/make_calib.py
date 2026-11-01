"""Pack observations into a Pulsar2 calibration archive (one [1,101] float32 .npy per sample).

    python3 make_calib.py obs.npy calib_obs.tar [--max 512]

obs.npy comes from ``scripts/walk.py --save_obs obs.npy`` (robot running the
ONNX policy on CPU) or from recording observations in MuJoCo. Real walking
data matters: the quantizer sets each layer's range from it.
"""

import argparse
import io
import tarfile

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("obs")
p.add_argument("out")
p.add_argument("--max", type=int, default=512)
args = p.parse_args()

obs = np.load(args.obs).astype(np.float32).reshape(-1, 101)
if len(obs) > args.max:
    idx = np.linspace(0, len(obs) - 1, args.max).astype(int)
    obs = obs[idx]

with tarfile.open(args.out, "w") as tar:
    for i, o in enumerate(obs):
        buf = io.BytesIO()
        np.save(buf, o.reshape(1, 101))
        info = tarfile.TarInfo(f"obs_{i:05d}.npy")
        info.size = buf.tell()
        buf.seek(0)
        tar.addfile(info, buf)
print(f"wrote {len(obs)} samples to {args.out}")
