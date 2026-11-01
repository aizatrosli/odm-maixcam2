"""Prepare the walking policy ONNX for Pulsar2.

* fixes the batch size to 1 (input [1, 101 + extra inputs], output [1, 14])
* drops the unused ``ai.onnx.ml`` opset import (Pulsar2 rejects unknown domains)
* lists the operators so you can check they are all NPU-supported
  (BEST_WALK_ONNX_2.onnx uses Gemm, Mul, Sigmoid, Split, Sub, Tanh)

    python3 prepare_onnx.py BEST_WALK_ONNX_2.onnx walk_policy.onnx
"""

import sys

import onnx
from onnx import checker, shape_inference

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph

used_domains = {n.domain for n in g.node}
keep = [o for o in m.opset_import if o.domain in ("", "ai.onnx") or o.domain in used_domains]
del m.opset_import[:]
m.opset_import.extend(keep)

for vi, size in ((g.input[0], 101), (g.output[0], 14)):
    dims = vi.type.tensor_type.shape.dim
    dims[0].dim_value = 1
    if not dims[1].dim_value:  # keep a fixed size (policies with extra inputs are > 101)
        dims[1].dim_value = size

m = shape_inference.infer_shapes(m)
checker.check_model(m)
onnx.save(m, dst)
print("input ", g.input[0].name, [d.dim_value for d in g.input[0].type.tensor_type.shape.dim])
print("output", g.output[0].name, [d.dim_value for d in g.output[0].type.tensor_type.shape.dim])
print("ops   ", sorted({n.op_type for n in g.node}))
print("saved ", dst)
