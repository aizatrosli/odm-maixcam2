import importlib.util
import os
import sys

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("make_vit_onnx", os.path.join(HERE, "..", "tools", "bench_vit", "make_vit_onnx.py"))
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)


@pytest.mark.parametrize("frames,shape", [(1, (1, 3, 64, 64)), (2, (1, 3, 2, 64, 64))])
def test_tiny_vit_runs(tmp_path, frames, shape):
    model, info = mv.build("tiny", 64, frames, 16)
    onnx.checker.check_model(model)
    path = str(tmp_path / "m.onnx")
    onnx.save_model(model, path)
    y = ort.InferenceSession(path).run(None, {"image": np.random.rand(*shape).astype(np.float32)})[0]
    assert y.shape == (1, 16)
    assert np.isfinite(y).all()
    assert info["tokens"] == 16
    assert 5e6 < info["params"] < 6e6


def test_large_matches_vjepa_size():
    d, depth, _ = mv.SIZES["large"]
    params = depth * 12 * d * d
    assert 290e6 < params < 310e6
