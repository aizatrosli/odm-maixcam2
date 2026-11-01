"""Write a plain ViT encoder as ONNX, for timing on the MaixCAM2 NPU.

No PyTorch needed: the graph is built with the onnx helper and random weights.
Latency depends only on the architecture (width, depth, tokens), not on the
weights, so this is enough to measure what a JEPA encoder of each size costs.
The output is the mean-pooled patch embedding projected to --emb values, the
same shape the walking policy would read.

    python3 make_vit_onnx.py --size small --res 160 --out vit_small_160.onnx
    python3 make_vit_onnx.py --size large --res 224 --frames 2 --out vit_large_224_f2.onnx

Sizes follow the standard ViT configs (V-JEPA 2 / I-JEPA use large, huge, giant).
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SIZES = {
    #        width, depth, heads
    "tiny": (192, 12, 3),
    "small": (384, 12, 6),
    "base": (768, 12, 12),
    "large": (1024, 24, 16),
    "huge": (1280, 32, 16),
    "giant": (1408, 40, 16),
}
PATCH = 16
TUBELET = 2  # frames per token when --frames > 1 (V-JEPA style)


class Graph:
    def __init__(self, seed: int = 0):
        self.nodes, self.inits = [], []
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.params = 0

    def name(self, base: str) -> str:
        self.n += 1
        return f"{base}_{self.n}"

    def weight(self, shape, std=0.02, value=None) -> str:
        arr = np.full(shape, value, np.float32) if value is not None else (self.rng.standard_normal(shape) * std).astype(np.float32)
        nm = self.name("w")
        self.inits.append(numpy_helper.from_array(arr, nm))
        self.params += arr.size
        return nm

    def const(self, arr, dtype=np.int64) -> str:
        nm = self.name("c")
        self.inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), nm))
        return nm

    def op(self, op_type: str, inputs, **attrs) -> str:
        out = self.name(op_type.lower())
        self.nodes.append(helper.make_node(op_type, list(inputs), [out], **attrs))
        return out

    # building blocks
    def linear(self, x, d_in, d_out):
        return self.op("Add", [self.op("MatMul", [x, self.weight((d_in, d_out))]), self.weight((d_out,), value=0.0)])

    def layernorm(self, x, d):
        return self.op("LayerNormalization", [x, self.weight((d,), value=1.0), self.weight((d,), value=0.0)], axis=-1, epsilon=1e-6)

    def gelu(self, x):
        # 0.5 * x * (1 + erf(x / sqrt(2))); Pulsar2 fuses this pattern
        e = self.op("Erf", [self.op("Div", [x, self.const(math.sqrt(2.0), np.float32)])])
        return self.op("Mul", [self.op("Mul", [x, self.op("Add", [e, self.const(1.0, np.float32)])]), self.const(0.5, np.float32)])


def build(size: str, res: int, frames: int, emb: int) -> tuple[onnx.ModelProto, dict]:
    d, depth, heads = SIZES[size]
    dh = d // heads
    g = Graph()
    grid = res // PATCH
    if frames > 1:
        t = frames // TUBELET
        n_tok = grid * grid * t
        x_in = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, frames, res, res])
        x = g.op("Conv", ["image", g.weight((d, 3, TUBELET, PATCH, PATCH)), g.weight((d,), value=0.0)],
                 kernel_shape=[TUBELET, PATCH, PATCH], strides=[TUBELET, PATCH, PATCH])
    else:
        n_tok = grid * grid
        x_in = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, res, res])
        x = g.op("Conv", ["image", g.weight((d, 3, PATCH, PATCH)), g.weight((d,), value=0.0)],
                 kernel_shape=[PATCH, PATCH], strides=[PATCH, PATCH])
    x = g.op("Reshape", [x, g.const([1, d, n_tok])])
    x = g.op("Transpose", [x], perm=[0, 2, 1])  # [1, N, D]
    x = g.op("Add", [x, g.weight((1, n_tok, d))])  # position embedding

    scale = g.const(1.0 / math.sqrt(dh), np.float32)
    split_shape = g.const([1, n_tok, heads, dh])
    merge_shape = g.const([1, n_tok, d])
    for _ in range(depth):
        h = g.layernorm(x, d)
        q = g.op("Transpose", [g.op("Reshape", [g.linear(h, d, d), split_shape])], perm=[0, 2, 1, 3])  # [1,H,N,dh]
        k = g.op("Transpose", [g.op("Reshape", [g.linear(h, d, d), split_shape])], perm=[0, 2, 3, 1])  # [1,H,dh,N]
        v = g.op("Transpose", [g.op("Reshape", [g.linear(h, d, d), split_shape])], perm=[0, 2, 1, 3])
        att = g.op("Softmax", [g.op("Mul", [g.op("MatMul", [q, k]), scale])], axis=-1)
        o = g.op("Reshape", [g.op("Transpose", [g.op("MatMul", [att, v])], perm=[0, 2, 1, 3]), merge_shape])
        x = g.op("Add", [x, g.linear(o, d, d)])
        h = g.layernorm(x, d)
        x = g.op("Add", [x, g.linear(g.gelu(g.linear(h, d, 4 * d)), 4 * d, d)])

    x = g.layernorm(x, d)
    x = g.op("ReduceMean", [x], axes=[1], keepdims=0)  # [1, D]
    out = g.linear(x, d, emb)
    g.nodes.append(helper.make_node("Identity", [out], ["embedding"]))
    y = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, emb])

    graph = helper.make_graph(g.nodes, f"vit_{size}_{res}", [x_in], [y], g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="odm-maixcam2")
    model.ir_version = 8
    ops = depth * (24 * n_tok * d * d + 4 * n_tok * n_tok * d)
    info = {"size": size, "res": res, "frames": frames, "tokens": n_tok, "width": d, "depth": depth,
            "params": g.params, "gop": ops / 1e9}
    return model, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=SIZES, default="small")
    p.add_argument("--res", type=int, default=160, help="input side in pixels, multiple of 16")
    p.add_argument("--frames", type=int, default=1, help="1 = single image, else a multiple of 2")
    p.add_argument("--emb", type=int, default=16, help="output embedding size")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    assert a.res % PATCH == 0, "--res must be a multiple of 16"
    assert a.frames == 1 or a.frames % TUBELET == 0, "--frames must be 1 or even"
    model, info = build(a.size, a.res, a.frames, a.emb)
    onnx.checker.check_model(model) if info["params"] * 4 < 1.8e9 else None
    big = info["params"] * 4 > 1.8e9  # protobuf limit is 2 GB
    onnx.save_model(model, a.out, save_as_external_data=big, all_tensors_to_one_file=True,
                    location=a.out.rsplit("/", 1)[-1] + ".data" if big else None)
    print(f"{a.out}: ViT-{a.size} {a.res}px x{a.frames} -> {info['tokens']} tokens, "
          f"{info['params'] / 1e6:.1f}M params, {info['gop']:.1f} GOP per run")


if __name__ == "__main__":
    main()
