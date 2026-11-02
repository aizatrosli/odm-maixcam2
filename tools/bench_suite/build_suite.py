"""Build the frozen MaixCAM2 encoder benchmark suite (runs on a PC with Docker + Pulsar2).

    pip install torch timm onnx onnxsim onnxruntime safetensors pillow
    VJEPA21_DIR=/path/to/vjepa2.1-vit-base-384 \\
      python3 tools/bench_suite/build_suite.py --images path/to/64_images --out suite

Produces:

    suite/
      manifest.json                 every model build with sha256, shapes, precision, source, license
      inputs/eval_<res>.npy         32 fixed, normalised evaluation images per input size
      refs/<model>_<res>.npy        FP32 (onnxruntime) embeddings of those images
      models/<model>_<res>_<prec>/  <name>.mud + <name>_npu.axmodel + <name>_vnpu.axmodel
      onnx/<model>_<res>.onnx       the exported FP32 graphs (kept for re-builds)

Images 0-31 (sorted by filename) calibrate the quantiser, images 32-63 are the
evaluation set, so calibration never sees the evaluation images. Copy the suite
to the board and run scripts/bench_suite.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models as M  # noqa: E402

SCHEMA = 1
UNSUPPORTED_AX620E = {"Conv3d", "Einsum", "CumSum", "ScatterND", "Range", "Cos", "Sin"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_images(paths, res, mean, std):
    from PIL import Image

    out = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((res, res), Image.BILINEAR)
        a = np.asarray(im, np.float32) / 255.0
        a = (a - np.array(mean, np.float32)) / np.array(std, np.float32)
        out.append(a.transpose(2, 0, 1))
    return np.stack(out).astype(np.float32)  # [N, 3, H, W]


def export_onnx(model, res, path):
    import onnx
    import torch
    from onnxsim import simplify

    x = torch.zeros(1, 3, res, res)
    torch.onnx.export(model, (x,), path, input_names=["image"], output_names=["embedding"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    m, ok = simplify(onnx.load(path))
    if not ok:
        raise RuntimeError(f"onnxsim could not validate {path}")
    onnx.save(m, path)
    ops = sorted({n.op_type for n in m.graph.node})
    bad = sorted(set(ops) & UNSUPPORTED_AX620E)
    return ops, bad


def write_calib(images, path):
    with tarfile.open(path, "w") as tar:
        for i, im in enumerate(images):
            buf = io.BytesIO()
            np.save(buf, im[None])
            info = tarfile.TarInfo(f"img_{i:03d}.npy")
            info.size = buf.tell()
            buf.seek(0)
            tar.addfile(info, buf)


def pulsar_config(npu_mode, precision, family):
    quant = {
        "input_configs": [{"tensor_name": "image", "calibration_dataset": "./calib.tar",
                           "calibration_size": -1, "calibration_format": "Numpy"}],
        "calibration_method": "MinMax",
        "precision_analysis": False,
    }
    if family == "vit":
        # recommended level per the Pulsar2 docs (validated on DeiT / ViT / Swin)
        quant["transformer_opt_level"] = 1
    if precision == "mixed":
        quant["layer_configs"] = [{"op_types": ["LayerNormalization", "Softmax"], "data_type": "U16"}]
    return {
        "model_type": "ONNX",
        "npu_mode": npu_mode,
        "quant": quant,
        "input_processors": [{"tensor_name": "image", "src_dtype": "FP32"}],
        "compiler": {"check": 0},
    }


def pulsar_build(image, onnx_path, calib, family, precision, outdir, name):
    """Compile NPU2 + NPU1 builds and write the .mud. Returns per-build info."""
    os.makedirs(outdir, exist_ok=True)
    info = {}
    for tag, npu_mode in (("npu", "NPU2"), ("vnpu", "NPU1")):
        work = tempfile.mkdtemp(prefix=f"{name}_{tag}_")
        try:
            shutil.copy(onnx_path, os.path.join(work, "model.onnx"))
            shutil.copy(calib, os.path.join(work, "calib.tar"))
            with open(os.path.join(work, "config.json"), "w") as f:
                json.dump(pulsar_config(npu_mode, precision, family), f, indent=2)
            t = time.time()
            log = subprocess.run(
                ["docker", "run", "--rm", "-v", f"{work}:/data", "-w", "/data", image,
                 "pulsar2", "build", "--target_hardware", "AX620E", "--input", "model.onnx",
                 "--output_dir", "build", "--config", "config.json"],
                capture_output=True, text=True)
            out = os.path.join(work, "build", "compiled.axmodel")
            if log.returncode != 0 or not os.path.exists(out):
                with open(os.path.join(outdir, f"{tag}_build.log"), "w") as f:
                    f.write(log.stdout[-20000:] + log.stderr[-20000:])
                raise RuntimeError(f"pulsar2 build failed for {name} ({tag}), see {outdir}/{tag}_build.log")
            dst = os.path.join(outdir, f"{name}_{tag}.axmodel")
            shutil.copy(out, dst)
            info[tag] = {"file": os.path.basename(dst), "sha256": sha256(dst),
                         "bytes": os.path.getsize(dst), "compile_s": round(time.time() - t, 1)}
        finally:
            shutil.rmtree(work, ignore_errors=True)
    with open(os.path.join(outdir, f"{name}.mud"), "w") as f:
        f.write(f"[basic]\ntype = axmodel\nmodel_npu = {name}_npu.axmodel\nmodel_vnpu = {name}_vnpu.axmodel\n\n"
                f"[extra]\nmodel_type = bench_encoder\ninput_type = float32\n")
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="folder with at least 64 .jpg/.png images")
    p.add_argument("--out", default="suite")
    p.add_argument("--models", nargs="*", default=None, help="subset of candidate names")
    p.add_argument("--image", default=os.environ.get("PULSAR2_IMAGE", "pulsar2:6.0"))
    p.add_argument("--jobs", type=int, default=2, help="parallel Pulsar2 builds")
    p.add_argument("--skip_build", action="store_true", help="export + refs only")
    a = p.parse_args()

    import onnxruntime as ort
    import torch

    files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if len(files) < 64:
        raise SystemExit(f"need 64 images, found {len(files)}")
    calib_files, eval_files = files[:32], files[32:64]
    for d in ("inputs", "refs", "models", "onnx", "calib"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    cands = [M.by_name(n) for n in a.models] if a.models else M.CANDIDATES
    manifest = {"schema": SCHEMA, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "pulsar2_image": a.image,
                "target_hardware": "AX620E", "eval_images": [os.path.basename(f) for f in eval_files],
                "calib_images": [os.path.basename(f) for f in calib_files], "builds": []}
    jobs = []
    for c in cands:
        for res in c.resolutions:
            key = f"{c.name}_{res}"
            onnx_path = os.path.join(a.out, "onnx", key + ".onnx")
            print(f"export {key}", flush=True)
            with torch.no_grad():
                model = c.build(res)
            ops, bad = export_onnx(model, res, onnx_path)
            if bad:
                print(f"  WARNING: {key} uses ops not listed for AX620E: {bad}")
            ev_path = os.path.join(a.out, "inputs", f"eval_{res}.npy")
            ev = load_images(eval_files, res, c.mean, c.std)
            if not os.path.exists(ev_path):
                np.save(ev_path, ev)
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            ref = np.concatenate([sess.run(None, {"image": x[None]})[0] for x in ev])
            np.save(os.path.join(a.out, "refs", key + ".npy"), ref.astype(np.float32))
            with torch.no_grad():
                torch_ref = np.concatenate([model(torch.from_numpy(e[None])).numpy() for e in ev[:4]])
            drift = float(np.abs(torch_ref - ref[:4]).max())
            calib = os.path.join(a.out, "calib", key + ".tar")
            write_calib(load_images(calib_files, res, c.mean, c.std), calib)
            for prec in c.precisions:
                name = f"{key}_{prec}"
                entry = {"name": name, "model": c.name, "family": c.family, "res": res, "precision": prec,
                         "embed_dim": int(ref.shape[1]), "params_m": c.params_m, "why": c.why,
                         "source": c.source, "license": c.license, "notes": c.notes,
                         "onnx_ops": ops, "unsupported_ops": bad, "onnx_vs_torch_max_abs": drift,
                         "input": {"name": "image", "shape": [1, 3, res, res], "dtype": "float32",
                                   "mean": list(c.mean), "std": list(c.std)},
                         "eval_inputs": f"inputs/eval_{res}.npy", "ref": f"refs/{key}.npy",
                         "mud": f"models/{name}/{name}.mud"}
                manifest["builds"].append(entry)
                jobs.append((entry, onnx_path, calib, c.family, prec, os.path.join(a.out, "models", name), name))

    def run(job):
        entry, onnx_path, calib, fam, prec, outdir, name = job
        try:
            entry["axmodels"] = pulsar_build(a.image, onnx_path, calib, fam, prec, outdir, name)
            entry["build_ok"] = True
            print(f"built {name}", flush=True)
        except Exception as e:  # keep the rest of the suite
            entry["build_ok"] = False
            entry["build_error"] = str(e)
            print(f"FAILED {name}: {e}", flush=True)

    if not a.skip_build:
        with cf.ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(run, jobs))
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    ok = sum(1 for b in manifest["builds"] if b.get("build_ok"))
    print(f"suite at {os.path.abspath(a.out)}: {ok}/{len(manifest['builds'])} builds ok")


if __name__ == "__main__":
    main()
