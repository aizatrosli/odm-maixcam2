"""Runs scripts/bench_suite.py end to end against a fake `maix` module and a tiny suite."""

import hashlib
import json
import os
import subprocess
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

FAKE_MAIX = '''
import numpy as np
class _T:
    def __init__(self, a): self.a = np.asarray(a, np.float32)
class Tensors(dict):
    def add_tensor(self, name, t, copy, auto_delete): self[name] = t
    def keys(self): return list(super().keys())
def tensor_from_numpy_float32(a, copy=True): return _T(a)
def tensor_to_numpy_float32(t): return t.a
class _Info:
    def __init__(self, name, shape): self.name, self.shape, self.dtype = name, shape, "float32"
class NN:
    def __init__(self, mud, dual_buff=False):
        self.w = np.load(open(mud).read().split("weights=")[1].strip())
    def inputs_info(self): return [_Info("image", [1, 3, 8, 8])]
    def outputs_info(self): return [_Info("embedding", [1, 4])]
    def forward(self, ts):
        x = ts["image"].a.reshape(-1)
        out = Tensors(); out["embedding"] = _T((x @ self.w)[None]); return out
'''


def make_suite(d):
    rng = np.random.default_rng(0)
    w = rng.normal(size=(3 * 8 * 8, 4)).astype(np.float32)
    ev = rng.normal(size=(32, 3, 8, 8)).astype(np.float32)
    os.makedirs(os.path.join(d, "inputs"))
    os.makedirs(os.path.join(d, "refs"))
    mdir = os.path.join(d, "models", "toy_8_int8")
    os.makedirs(mdir)
    np.save(os.path.join(d, "inputs", "eval_8.npy"), ev)
    np.save(os.path.join(d, "refs", "toy_8.npy"), ev.reshape(32, -1) @ w)
    np.save(os.path.join(mdir, "w.npy"), w)
    ax = os.path.join(mdir, "toy_8_int8_npu.axmodel")
    open(ax, "wb").write(b"fake")
    open(os.path.join(mdir, "toy_8_int8.mud"), "w").write("weights=" + os.path.join(mdir, "w.npy"))
    man = {"schema": 1, "created": "now", "pulsar2_image": "x", "builds": [{
        "name": "toy_8_int8", "model": "toy", "family": "cnn", "res": 8, "precision": "int8", "embed_dim": 4,
        "input": {"mean": [0, 0, 0], "std": [1, 1, 1]}, "eval_inputs": "inputs/eval_8.npy", "ref": "refs/toy_8.npy",
        "mud": "models/toy_8_int8/toy_8_int8.mud", "build_ok": True,
        "axmodels": {"npu": {"file": "toy_8_int8_npu.axmodel", "sha256": hashlib.sha256(b"fake").hexdigest()}}}]}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))


def test_suite_runner_and_compare(tmp_path):
    suite = tmp_path / "suite"
    make_suite(str(suite))
    fake = tmp_path / "fake" / "maix"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("from . import nn, tensor\n")
    (fake / "tensor.py").write_text(FAKE_MAIX)
    (fake / "nn.py").write_text("from .tensor import NN\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "fake"))
    out = tmp_path / "results"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "bench_suite.py"), str(suite),
                        "--scenarios", "idle", "-n", "20", "--rounds", "2", "--warmup", "2", "--out", str(out)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    files = list(out.iterdir())
    res = json.load(open(files[0]))
    row = res["results"][0]
    assert "error" not in row, row
    assert row["fidelity"]["cos_min"] > 0.9999
    assert row["idle"]["median_ms"] > 0 and len(row["idle"]["rounds"]) == 2
    c = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "bench_suite", "compare.py"), str(files[0])],
                       capture_output=True, text=True)
    assert c.returncode == 0 and "toy_8_int8" in c.stdout, c.stderr


def test_hash_mismatch_is_reported(tmp_path):
    suite = tmp_path / "suite"
    make_suite(str(suite))
    open(suite / "models" / "toy_8_int8" / "toy_8_int8_npu.axmodel", "wb").write(b"changed")
    fake = tmp_path / "fake" / "maix"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("from . import nn, tensor\n")
    (fake / "tensor.py").write_text(FAKE_MAIX)
    (fake / "nn.py").write_text("from .tensor import NN\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "fake"))
    out = tmp_path / "results"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "bench_suite.py"), str(suite), "--out", str(out)],
                   env=env, capture_output=True, text=True, check=True)
    res = json.load(open(next(out.iterdir())))
    assert "manifest hash" in res["results"][0]["error"]


_ = types
