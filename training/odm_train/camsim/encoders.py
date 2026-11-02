"""Frozen image encoders (V-JEPA 2.1, DINOv2, any ONNX) and the small projection to 16-32 values.

    enc = make_encoder("vjepa2_1_base", res=160)          # torch, weights downloaded on first use
    enc = make_encoder("dinov2_small", res=168)
    enc = make_encoder("onnx:out/vjepa_base_160.onnx")    # exactly the graph that goes to the NPU
    enc = make_encoder("standin_small", res=160)          # random-weight ViT, no download (tests, timing)

    feats = enc(frames)                                    # uint8 (N, H, W, 3) or (H, W, 3) -> (N, D) float32
    proj = Projection.fit(feats, dim=16); proj.save("proj.npz")
    emb = proj(enc(frame))                                 # (16,) -> policy extra_obs emb_0..emb_15

Every encoder takes RGB frames as the camera delivers them, resizes them to its
input size (squashing 4:3 to square, the same on the robot), scales to 0..1 and
applies its own normalization. The torch encoders and their ONNX exports
(``export_onnx``) agree, so features recorded in sim match the NPU model.

Command line (from the Open_Duck_Playground checkout, PYTHONPATH as in CAMSIM.md):

    python -m odm_train.camsim.encoders embed follow.npz --encoder dinov2_small --dim 16 --out follow_emb.npz
    python -m odm_train.camsim.encoders export --encoder vjepa2_1_base --res 160 --projection proj.npz --out vjepa_b160.onnx
    python -m odm_train.camsim.encoders time --encoder vjepa2_1_base --res 160
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# facebookresearch/vjepa2 at a fixed commit (torch.hub), and the official checkpoints
VJEPA2_REPO = "facebookresearch/vjepa2:204698b45b3712590f06245fbfba32d3be539812"
VJEPA2_WEIGHTS = {
    # name: (hub entry point, checkpoint url, checkpoint key)
    "vjepa2_1_base": ("vjepa2_1_vit_base_384", "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt", "ema_encoder"),
    "vjepa2_1_large": ("vjepa2_1_vit_large_384", "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt", "ema_encoder"),
}
DINOV2_REPO = "facebookresearch/dinov2"
DINOV2 = {"dinov2_small": "dinov2_vits14", "dinov2_base": "dinov2_vitb14"}
STANDIN_SIZES = ("tiny", "small", "base", "large")
DEFAULT_RES = {"vjepa2_1_base": 160, "vjepa2_1_large": 160, "dinov2_small": 168, "dinov2_base": 168}


def resize(frames: np.ndarray, res: int) -> np.ndarray:
    """uint8 (N, H, W, 3) -> float32 (N, 3, res, res) in 0..1, bilinear, aspect squashed to square."""
    from odm_train.camsim.lens import bilinear

    frames = np.asarray(frames)
    if frames.ndim == 3:
        frames = frames[None]
    n, h, w, _ = frames.shape
    ys = (np.arange(res) + 0.5) * h / res - 0.5
    xs = (np.arange(res) + 0.5) * w / res - 0.5
    mx, my = np.meshgrid(np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1))
    out = np.stack([bilinear(f.astype(np.float32), mx, my) for f in frames]) / 255.0
    return out.transpose(0, 3, 1, 2).astype(np.float32)


class Encoder:
    """Base class: ``__call__(frames) -> (N, D)`` float32 features."""

    name = "encoder"
    res = 160
    dim = 0

    def __call__(self, frames: np.ndarray, batch: int = 32) -> np.ndarray:
        x = resize(frames, self.res)
        return np.concatenate([self.forward(x[i : i + batch]) for i in range(0, len(x), batch)]).astype(np.float32)

    def forward(self, x01: np.ndarray) -> np.ndarray:  # (N, 3, res, res) in 0..1
        raise NotImplementedError


class TorchEncoder(Encoder):
    """A torch module wrapped so it takes 0..1 images and returns pooled features.

    ``module`` maps a normalized (N, 3, res, res) tensor to (N, D) features."""

    def __init__(self, name, module, res, device="cpu"):
        import torch

        self.torch, self.name, self.res, self.device = torch, name, res, device
        self.module = module.to(device).eval()
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)
        with torch.no_grad():
            self.dim = int(self.module((torch.zeros(1, 3, res, res, device=device) - self.mean) / self.std).shape[-1])

    def forward(self, x01):
        t = self.torch
        with t.no_grad():
            x = (t.from_numpy(x01).to(self.device) - self.mean) / self.std
            return self.module(x).float().cpu().numpy()

    def wrapped(self, projection: Optional["Projection"] = None):
        """nn.Module: 0..1 image (1, 3, res, res) -> features (or projected embedding). For ONNX export."""
        t = self.torch
        mean, std, inner = self.mean.cpu(), self.std.cpu(), self.module.cpu()
        P = None if projection is None else projection

        class Wrapped(t.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = inner
                self.register_buffer("mean", mean)
                self.register_buffer("std", std)
                if P is not None:
                    self.register_buffer("p_mean", t.tensor(P.mean))
                    self.register_buffer("p_w", t.tensor(P.components.T.copy()))

            def forward(self, image):
                f = self.inner((image - self.mean) / self.std)
                if P is not None:
                    f = (f - self.p_mean) @ self.p_w
                return f

        return Wrapped().eval()


def _vjepa_module(name, res, cache_dir=None):
    """V-JEPA 2.1 encoder at ``res`` px, single image through the 2D patch embedding, mean-pooled tokens."""
    import torch

    entry, url, key = VJEPA2_WEIGHTS[name]
    enc, _pred = torch.hub.load(VJEPA2_REPO, entry, pretrained=False, trust_repo=True)
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu", model_dir=cache_dir, weights_only=True)
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state[key].items()}
    enc.load_state_dict(sd, strict=True)
    del state, _pred

    class Pooled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc

        def forward(self, x):
            tokens = self.enc(x.unsqueeze(2))  # (N, 3, T=1, H, W) selects the image patch embedding
            return tokens.mean(dim=1)

    return Pooled()


def _dinov2_module(name):
    import torch

    return torch.hub.load(DINOV2_REPO, DINOV2[name], trust_repo=True)  # returns the CLS embedding


class OnnxEncoder(Encoder):
    """Any ONNX encoder with one image input (1, 3, res, res) in 0..1 (normalization inside the graph)."""

    def __init__(self, path, threads: int = 0):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input = inp.name
        self.res = int(inp.shape[-1])
        self.name = "onnx:" + os.path.basename(path)
        self.dim = int(self.sess.get_outputs()[0].shape[-1])

    def forward(self, x01):
        return np.concatenate([self.sess.run(None, {self.input: x01[i : i + 1]})[0].reshape(1, -1) for i in range(len(x01))])


class StandinEncoder(OnnxEncoder):
    """Random-weight ViT from tools/bench_vit (same graph family the NPU is timed with). No download."""

    def __init__(self, size="small", res=160, emb=16):
        import tempfile

        import onnx

        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.normpath(os.path.join(here, "..", "..", "..", "tools", "bench_vit")))
        from make_vit_onnx import build

        model, _ = build(size, res, 1, emb)
        path = os.path.join(tempfile.mkdtemp(prefix="standin_"), f"vit_{size}_{res}.onnx")
        onnx.save_model(model, path)
        super().__init__(path)
        self.name = f"standin_{size}"


def make_encoder(name: str, res: Optional[int] = None, device: str = "cpu", cache_dir: Optional[str] = None) -> Encoder:
    if name.startswith("onnx:"):
        return OnnxEncoder(name[5:])
    if name.startswith("standin_"):
        return StandinEncoder(name[8:], res or 160)
    res = res or DEFAULT_RES[name]
    if name in VJEPA2_WEIGHTS:
        assert res % 16 == 0, "V-JEPA patch size is 16: use a multiple of 16"
        return TorchEncoder(name, _vjepa_module(name, res, cache_dir), res, device)
    if name in DINOV2:
        assert res % 14 == 0, "DINOv2 patch size is 14: use a multiple of 14"
        return TorchEncoder(name, _dinov2_module(name), res, device)
    raise ValueError(f"unknown encoder {name!r}")


@dataclass
class Projection:
    """PCA from encoder features to a few values: emb = (f - mean) @ components.T, components (k, D)."""

    mean: np.ndarray
    components: np.ndarray
    explained: np.ndarray

    @classmethod
    def fit(cls, feats: np.ndarray, dim: int = 16) -> "Projection":
        feats = np.asarray(feats, np.float64)
        mean = feats.mean(0)
        _, s, vt = np.linalg.svd(feats - mean, full_matrices=False)
        var = s**2
        return cls(mean.astype(np.float32), vt[:dim].astype(np.float32), (var[:dim] / var.sum()).astype(np.float32))

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        """(N, D) -> (N, k), or (D,) -> (k,)."""
        return (np.asarray(feats, np.float32) - self.mean) @ self.components.T

    def save(self, path):
        np.savez(path, mean=self.mean, components=self.components, explained=self.explained)

    @classmethod
    def load(cls, path) -> "Projection":
        z = np.load(path)
        return cls(z["mean"], z["components"], z["explained"])


class EmbeddingPipeline:
    """frame -> encoder -> projection: the callable FollowSim takes as ``encoder=``."""

    def __init__(self, encoder: Encoder, projection: Optional[Projection] = None):
        self.encoder, self.projection = encoder, projection

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        f = self.encoder(frame)
        return (self.projection(f) if self.projection is not None else f)[0]


def export_onnx(encoder: TorchEncoder, out: str, projection: Optional[Projection] = None, opset: int = 17):
    """Write encoder (+ projection) as one fixed-size ONNX: image (1, 3, res, res) in 0..1 -> embedding."""
    import torch

    model = encoder.wrapped(projection)
    x = torch.rand(1, 3, encoder.res, encoder.res)
    torch.onnx.export(model, (x,), out, input_names=["image"], output_names=["embedding"], opset_version=opset, dynamo=False)
    # check: onnxruntime matches torch
    ref = model(x).detach().numpy()
    got = OnnxEncoder(out).forward(x.numpy())
    err = float(np.abs(ref - got).max())
    return err


def _cli(argv=None):
    p = argparse.ArgumentParser(description="Frozen image encoders for the camera simulator")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed", help="add encoder features and a fitted projection to a follow_sim --record dataset")
    e.add_argument("dataset")
    e.add_argument("--encoder", required=True)
    e.add_argument("--res", type=int, default=None)
    e.add_argument("--dim", type=int, default=16)
    e.add_argument("--projection", default=None, help="use this projection instead of fitting one")
    e.add_argument("--out", required=True)
    x = sub.add_parser("export", help="torch encoder (+ projection) -> one ONNX for Pulsar2")
    x.add_argument("--encoder", required=True)
    x.add_argument("--res", type=int, default=None)
    x.add_argument("--projection", default=None)
    x.add_argument("--out", required=True)
    t = sub.add_parser("time", help="CPU latency of an encoder on camera-sized frames")
    t.add_argument("--encoder", required=True)
    t.add_argument("--res", type=int, default=None)
    t.add_argument("-n", type=int, default=20)
    for s in (e, x, t):
        s.add_argument("--device", default="cpu")
    a = p.parse_args(argv)

    enc = make_encoder(a.encoder, a.res, a.device)
    if a.cmd == "embed":
        z = dict(np.load(a.dataset))
        feats = enc(z["frame"])
        proj = Projection.load(a.projection) if a.projection else Projection.fit(feats, a.dim)
        z["feat"] = feats
        z["emb"] = proj(feats).astype(np.float32)
        np.savez_compressed(a.out, **z)
        proj_path = a.out.rsplit(".", 1)[0] + "_projection.npz"
        if not a.projection:
            proj.save(proj_path)
        print(f"{enc.name} @ {enc.res}px: {feats.shape[1]} features -> {z['emb'].shape[1]} values "
              f"({100 * proj.explained.sum():.0f}% of variance) for {len(feats)} frames -> {a.out}"
              + ("" if a.projection else f", projection {proj_path}"))
    elif a.cmd == "export":
        assert isinstance(enc, TorchEncoder), "export takes a torch encoder (vjepa2_1_*, dinov2_*)"
        proj = Projection.load(a.projection) if a.projection else None
        err = export_onnx(enc, a.out, proj)
        print(f"wrote {a.out}: {enc.name} @ {enc.res}px -> {proj.components.shape[0] if proj else enc.dim} values, "
              f"max |torch - onnxruntime| = {err:.2e}")
    else:
        frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
        enc(frame)
        t0 = time.perf_counter()
        for _ in range(a.n):
            enc(frame)
        print(f"{enc.name} @ {enc.res}px: {1000 * (time.perf_counter() - t0) / a.n:.1f} ms per frame on {a.device}, {enc.dim} features")


if __name__ == "__main__":
    _cli()
