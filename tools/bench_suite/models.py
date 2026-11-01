"""Candidate camera encoders for the MaixCAM2 benchmark suite.

Every entry builds a torch module that takes one normalised RGB image
[1, 3, H, W] (float32) and returns a pooled embedding [1, D]. The board only
ever sees the compiled model, so all candidates are compared on the same terms.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, List

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Candidate:
    name: str
    family: str  # "vit" or "cnn"
    why: str
    source: str
    license: str
    resolutions: List[int]
    precisions: List[str]  # "int8", "mixed" (U16 LayerNorm + Softmax)
    build: Callable[[int], "object"]
    params_m: float = 0.0
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- timm models
def _timm(model_name: str, pool_tokens: bool):
    def build(res: int):
        import timm
        import torch

        kw = {"img_size": res} if pool_tokens else {}
        m = timm.create_model(model_name, pretrained=True, num_classes=0, **kw).eval()

        class Wrap(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                if pool_tokens:
                    t = self.inner.forward_features(x)  # [1, prefix + N, D]
                    n_prefix = getattr(self.inner, "num_prefix_tokens", 0)
                    return t[:, n_prefix:].mean(1)
                return self.inner(x)  # timm CNN with num_classes=0 -> pooled [1, D]

        return Wrap(m).eval()

    return build


# --------------------------------------------------------------------------- V-JEPA 2.1 ViT-B
def _vjepa21_vitb(res: int):
    """V-JEPA 2.1 ViT-B/16 encoder in image mode, re-implemented here.

    Only the weights are taken from the download (VJEPA21_DIR pointing at
    apiantonio/vjepa2.1-vit-base-384, model.safetensors); no code from that repo
    runs. Architecture, following its modeling file: tubelet-1 patch embedding +
    image modality embedding, 12 pre-norm blocks with 3-D RoPE attention
    (20 dims each for time/height/width per 64-dim head, last 4 dims unrotated,
    positions interpolated to the 16x16 pre-training grid), exact-GELU MLP, final
    LayerNorm (norms_block[-1]). Output: mean over patch tokens.
    Not checked against the reference implementation; fine for timing and for
    quantisation error relative to this FP32 graph.
    """
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    d = os.environ.get("VJEPA21_DIR", "")
    w = load_file(os.path.join(d, "model.safetensors"))
    W = {k[len("encoder."):]: v for k, v in w.items() if k.startswith("encoder.")}

    D, depth, heads, patch = 768, 12, 12, 16
    hd = D // heads
    sub = 2 * ((hd // 3) // 2)  # 20
    grid = res // patch
    n = grid * grid
    pre_grid = 256 // patch

    # constant RoPE tables for this input size
    ids = torch.arange(n)
    hpos = (ids // grid).float() * (pre_grid - 1) / max(grid - 1, 1)
    wpos = (ids % grid).float() * (pre_grid - 1) / max(grid - 1, 1)
    dpos = torch.zeros(n)
    omega = 1.0 / (10000.0 ** (torch.arange(sub // 2, dtype=torch.float32) / (sub / 2.0)))

    def tables(pos):
        f = pos[:, None] * omega[None]
        return f.cos().repeat_interleave(2, -1), f.sin().repeat_interleave(2, -1)

    cos = torch.cat([tables(p)[0] for p in (dpos, hpos, wpos)] + [torch.ones(n, hd - 3 * sub)], -1)
    sin = torch.cat([tables(p)[1] for p in (dpos, hpos, wpos)] + [torch.zeros(n, hd - 3 * sub)], -1)

    def lin(prefix):
        m = torch.nn.Linear(W[prefix + ".weight"].shape[1], W[prefix + ".weight"].shape[0])
        m.weight.data, m.bias.data = W[prefix + ".weight"], W[prefix + ".bias"]
        return m

    def ln(prefix):
        m = torch.nn.LayerNorm(D, eps=1e-6)
        m.weight.data, m.bias.data = W[prefix + ".weight"], W[prefix + ".bias"]
        return m

    class Block(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            p = f"layer.{i}"
            self.norm1, self.norm2 = ln(p + ".norm1"), ln(p + ".norm2")
            self.q, self.k, self.v = lin(p + ".attention.query"), lin(p + ".attention.key"), lin(p + ".attention.value")
            self.proj = lin(p + ".attention.proj")
            self.fc1, self.fc2 = lin(p + ".mlp.fc1"), lin(p + ".mlp.fc2")

        @staticmethod
        def rope(x):  # x [1, H, N, hd]
            pairs = x.reshape(1, heads, n, hd // 2, 2)
            rot = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).reshape(1, heads, n, hd)
            return x * cos + rot * sin

        def forward(self, x):
            h = self.norm1(x)
            q = self.rope(self.q(h).reshape(1, n, heads, hd).transpose(1, 2))
            k = self.rope(self.k(h).reshape(1, n, heads, hd).transpose(1, 2))
            v = self.v(h).reshape(1, n, heads, hd).transpose(1, 2)
            a = torch.softmax(q @ k.transpose(-1, -2) * hd ** -0.5, -1)
            x = x + self.proj((a @ v).transpose(1, 2).reshape(1, n, D))
            return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            pw = W["embeddings.patch_embeddings_img.proj.weight"][:, :, 0]
            self.patch = torch.nn.Conv2d(3, D, patch, patch)
            self.patch.weight.data, self.patch.bias.data = pw, W["embeddings.patch_embeddings_img.proj.bias"]
            self.mod = torch.nn.Parameter(W["embeddings.img_mod_embed"])
            self.blocks = torch.nn.ModuleList([Block(i) for i in range(depth)])
            last = max(int(k.split(".")[1]) for k in W if k.startswith("norms_block."))
            self.norm = ln(f"norms_block.{last}")

        def forward(self, x):
            x = self.patch(x).flatten(2).transpose(1, 2) + self.mod
            for b in self.blocks:
                x = b(x)
            return self.norm(x).mean(1)

    return Encoder().eval()


CANDIDATES = [
    Candidate(
        name="vjepa21_vitb16",
        family="vit",
        why="Smallest official JEPA (V-JEPA 2.1, distilled from ViT-G); teacher candidate",
        source="Meta V-JEPA 2.1 ViT-B weights via huggingface.co/apiantonio/vjepa2.1-vit-base-384",
        license="MIT",
        resolutions=[224, 160],
        precisions=["int8", "mixed"],
        build=_vjepa21_vitb,
        params_m=86.0,
        notes=["encoder re-implemented from the weights; RoPE tables constant-folded per input size"],
    ),
    Candidate(
        name="dinov2_vits14",
        family="vit",
        why="Most-used frozen encoder for robots (DINO-WM)",
        source="timm vit_small_patch14_dinov2.lvd142m",
        license="Apache-2.0",
        resolutions=[224, 168],
        precisions=["int8", "mixed"],
        build=_timm("vit_small_patch14_dinov2.lvd142m", pool_tokens=True),
        params_m=22.1,
    ),
    Candidate(
        name="dinov3_convnext_tiny",
        family="cnn",
        why="Strong DINOv3 features in a CNN, which quantises well",
        source="timm convnext_tiny.dinov3_lvd1689m",
        license="DINOv3 License (check before commercial use)",
        resolutions=[224, 160],
        precisions=["int8"],
        build=_timm("convnext_tiny.dinov3_lvd1689m", pool_tokens=False),
        params_m=28.6,
    ),
    Candidate(
        name="deit_tiny16",
        family="vit",
        why="LeWorldModel-sized ViT; Axera measured 3.35 ms on AX630C at 224",
        source="timm deit_tiny_patch16_224.fb_in1k",
        license="Apache-2.0",
        resolutions=[224, 160],
        precisions=["int8", "mixed"],
        build=_timm("deit_tiny_patch16_224.fb_in1k", pool_tokens=True),
        params_m=5.7,
    ),
    Candidate(
        name="resnet18",
        family="cnn",
        why="Small CNN baseline for a distilled student",
        source="timm resnet18.a1_in1k",
        license="Apache-2.0",
        resolutions=[224, 160],
        precisions=["int8"],
        build=_timm("resnet18.a1_in1k", pool_tokens=False),
        params_m=11.7,
    ),
]


def by_name(name: str) -> Candidate:
    for c in CANDIDATES:
        if c.name == name:
            return c
    raise KeyError(name)


def vit_gop(params_m: float, tokens: int, width: int, depth: int) -> float:
    """Rough operation count (GOP) for one ViT forward, for the manifest."""
    return depth * (24 * tokens * width * width + 4 * tokens * tokens * width) / 1e9


_ = math  # keep import for callers using models.math
