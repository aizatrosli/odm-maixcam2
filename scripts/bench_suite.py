"""Run the frozen encoder benchmark suite on the MaixCAM2 and write one results file.

    python3 scripts/bench_suite.py ~/suite --all --policy ~/out/walk_policy.mud
    python3 scripts/bench_suite.py ~/suite --scenarios idle --models dinov2_vits14   # quick subset

The suite comes from tools/bench_suite/build_suite.py. Every model is timed the
same way (20 warm-up runs, then 3 rounds of 200), its output is checked against
the FP32 reference on 32 fixed images, and the board state (temperatures, CMM,
memory, versions) is recorded so runs can be compared across days and boards.

Scenarios:
  idle    each model alone, camera off (latency, load time, CMM, accuracy)
  camera  same timing while the camera streams in another process (AI-ISP on)
  e2e     camera frame -> preprocessing -> embedding, per-stage timing
  loop    the 50 Hz control loop while the model runs flat out in another process
  stack   YOLO11n + model alternating flat out, plus the 50 Hz loop

Results: results/<host>_<date>_<time>.json (schema 1). Compare runs on a PC with
tools/bench_suite/compare.py.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_npu import bench_loop, cmm_used_mb, linux_available_mb, stats  # noqa: E402

SCHEMA = 1
HOT_C = 70.0


# ---------------------------------------------------------------- board state
def temps_c():
    out = {}
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            name = open(os.path.join(z, "type")).read().strip()
            out[name] = int(open(os.path.join(z, "temp")).read().strip()) / 1000.0
        except (OSError, ValueError):
            pass
    return out


def max_temp():
    t = temps_c()
    return max(t.values()) if t else None


def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


def board_info():
    info = {
        "host": platform.node(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu_governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_freq_khz": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
        "cmm_used_mb": cmm_used_mb(),
        "linux_available_mb": linux_available_mb(),
        "temps_c": temps_c(),
    }
    try:
        import maix

        info["maixpy"] = getattr(maix, "__version__", None)
    except Exception:
        info["maixpy"] = None
    return info


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cool_down(limit=HOT_C, timeout=120):
    """Wait (up to timeout) until the SoC is below `limit` so runs start comparable."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = max_temp()
        if t is None or t < limit:
            return t
        time.sleep(5)
    return max_temp()


# ---------------------------------------------------------------- model runner
class Model:
    def __init__(self, mud, input_name="image"):
        from maix import nn, tensor

        self.t = tensor
        self.input_name = input_name
        self.nn = nn.NN(mud)
        try:
            self.out_name = self.nn.outputs_info()[0].name
        except Exception:
            self.out_name = None

    def tensor(self, x):
        return self.t.tensor_from_numpy_float32(np.ascontiguousarray(x, dtype=np.float32), True)

    def forward_tensor(self, tx):
        ts = self.t.Tensors()
        ts.add_tensor(self.input_name, tx, False, False)
        return self.nn.forward(ts)

    def forward(self, x):
        out = self.forward_tensor(self.tensor(x))
        key = self.out_name or out.keys()[0]
        return np.asarray(self.t.tensor_to_numpy_float32(out[key]), np.float32).reshape(-1)


def fidelity(model, inputs, ref):
    outs = np.stack([model.forward(x[None]) for x in inputs])
    ref = ref.reshape(outs.shape)
    cos = np.sum(outs * ref, 1) / (np.linalg.norm(outs, axis=1) * np.linalg.norm(ref, axis=1) + 1e-9)
    rel = np.linalg.norm(outs - ref, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-9)
    return {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
            "rel_err_mean": float(rel.mean()), "rel_err_max": float(rel.max())}


def time_model(model, x, warmup, rounds, n):
    tx = model.tensor(x)
    for _ in range(warmup):
        model.forward_tensor(tx)
    bare = []
    for _ in range(rounds):
        ms = []
        for _ in range(n):
            t = time.perf_counter()
            model.forward_tensor(tx)
            ms.append((time.perf_counter() - t) * 1000)
        bare.append(stats(ms))
    ms = []
    for _ in range(max(20, n // 4)):  # includes numpy -> tensor conversion and output copy
        t = time.perf_counter()
        model.forward(x)
        ms.append((time.perf_counter() - t) * 1000)
    conv = stats(ms)
    med = lambda k: float(np.median([r[k] for r in bare]))  # noqa: E731
    return {
        "median_ms": med("p50_ms"), "mean_ms": med("mean_ms"), "p99_ms": med("p99_ms"), "hz": 1000.0 / med("p50_ms"),
        "round_spread_pct": float(100 * (max(r["p50_ms"] for r in bare) - min(r["p50_ms"] for r in bare)) / med("p50_ms")),
        "rounds": bare, "with_conversion": conv,
    }


# ---------------------------------------------------------------- background workers
def _camera_worker(stop, frames, w, h):
    from maix import camera, image

    cam = camera.Camera(w, h, image.Format.FMT_RGB888)
    while not stop.is_set():
        cam.read()
        with frames.get_lock():
            frames.value += 1


def _model_worker(mud, stop, counter, yolo=None):
    m = Model(mud)
    spec = None
    x = None
    try:
        info = m.nn.inputs_info()[0]
        spec = [int(s) for s in info.shape]
    except Exception:
        pass
    x = m.tensor(np.random.rand(*(spec or [1, 3, 224, 224])).astype(np.float32))
    det = img = None
    if yolo:
        from maix import image, nn

        det = nn.YOLO11(model=yolo, dual_buff=False)
        img = image.Image(det.input_width(), det.input_height(), det.input_format())
    while not stop.is_set():
        if det is not None:
            det.detect(img)
        m.forward_tensor(x)
        with counter.get_lock():
            counter.value += 1


# ---------------------------------------------------------------- end to end
def e2e(model, build, frames=100):
    from maix import camera, image

    res = build["res"]
    mean = np.array(build["input"]["mean"], np.float32)
    std = np.array(build["input"]["std"], np.float32)
    cam = camera.Camera(res, res, image.Format.FMT_RGB888)
    for _ in range(10):
        cam.read()
    read_ms, prep_ms, fwd_ms, total = [], [], [], []
    for _ in range(frames):
        t0 = time.perf_counter()
        img = cam.read()
        t1 = time.perf_counter()
        a = image.image2cv(img, False, False).astype(np.float32) / 255.0
        x = ((a - mean) / std).transpose(2, 0, 1)[None]
        t2 = time.perf_counter()
        model.forward(x)
        t3 = time.perf_counter()
        read_ms.append((t1 - t0) * 1000)
        prep_ms.append((t2 - t1) * 1000)
        fwd_ms.append((t3 - t2) * 1000)
        total.append((t3 - t0) * 1000)
    del cam
    return {"camera_read": stats(read_ms), "preprocess": stats(prep_ms), "forward": stats(fwd_ms),
            "total": stats(total)}


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite")
    p.add_argument("--all", action="store_true", help="every scenario")
    p.add_argument("--scenarios", default="idle", help="comma list: idle,camera,e2e,loop,stack")
    p.add_argument("--models", nargs="*", default=None, help="filter by model or build name")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--loop_s", type=float, default=20)
    p.add_argument("--policy", default=None, help="walking policy .mud for loop/stack (default: policy on CPU skipped)")
    p.add_argument("--yolo", default="/root/models/yolo11n.mud")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    scen = ["idle", "camera", "e2e", "loop", "stack"] if a.all else a.scenarios.split(",")
    man = json.load(open(os.path.join(a.suite, "manifest.json")))
    builds = [b for b in man["builds"] if b.get("build_ok")]
    if a.models:
        builds = [b for b in builds if b["model"] in a.models or b["name"] in a.models]

    res = {"schema": SCHEMA, "suite_created": man["created"], "pulsar2_image": man["pulsar2_image"],
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "scenarios": scen,
           "protocol": {"warmup": a.warmup, "rounds": a.rounds, "n": a.n, "loop_s": a.loop_s},
           "board_start": board_info(), "results": []}
    print(f"{len(builds)} builds, scenarios {scen}", flush=True)

    for b in builds:
        r = {"name": b["name"], "model": b["model"], "res": b["res"], "precision": b["precision"],
             "family": b["family"], "embed_dim": b["embed_dim"]}
        mud = os.path.join(a.suite, b["mud"])
        d = os.path.dirname(mud)
        try:
            for tag, info in b["axmodels"].items():
                if sha256(os.path.join(d, info["file"])) != info["sha256"]:
                    raise RuntimeError(f"{info['file']} does not match the manifest hash")
            r["temp_start_c"] = cool_down()
            x = np.load(os.path.join(a.suite, b["eval_inputs"]))
            ref = np.load(os.path.join(a.suite, b["ref"]))
            c0 = cmm_used_mb()
            t = time.perf_counter()
            model = Model(mud)
            r["load_ms"] = (time.perf_counter() - t) * 1000
            c1 = cmm_used_mb()
            r["cmm_mb"] = (c1 - c0) if (c0 is not None and c1 is not None) else None
            if "idle" in scen:
                r["fidelity"] = fidelity(model, x, ref)
                r["idle"] = time_model(model, x[:1], a.warmup, a.rounds, a.n)
            if "camera" in scen:
                with _CameraBg(b["res"]) as bg:
                    r["camera"] = time_model(model, x[:1], a.warmup, 1, a.n)
                    r["camera"]["camera_fps"] = bg.rate()
            if "e2e" in scen:
                r["e2e"] = e2e(model, b)
            del model
            if "loop" in scen:
                r["loop"] = bench_loop(a.loop_s, a.policy, mud)
            if "stack" in scen and os.path.exists(a.yolo):
                r["stack"] = _stack(mud, a.yolo, a.policy, a.loop_s)
            r["temp_end_c"] = max_temp()
            r["hot"] = bool(r["temp_end_c"] and r["temp_end_c"] >= HOT_C)
            msg = f"{b['name']}: "
            if "idle" in r:
                msg += f"{r['idle']['median_ms']:.2f} ms ({r['idle']['hz']:.0f} Hz), cos {r['fidelity']['cos_mean']:.4f}"
            if "loop" in r:
                msg += f", loop overruns {r['loop']['overruns']}/{r['loop']['steps']}"
            print(msg, flush=True)
        except Exception as e:
            r["error"] = repr(e)
            print(f"{b['name']}: FAILED {e!r}", flush=True)
        res["results"].append(r)

    res["board_end"] = board_info()
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{platform.node()}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {os.path.abspath(path)}")


class _CameraBg:
    """Keeps the camera (and AI-ISP) streaming in another process."""

    def __init__(self, res):
        self.res = res
        self.stop = mp.Event()
        self.frames = mp.Value("i", 0)
        self.p = mp.Process(target=_camera_worker, args=(self.stop, self.frames, self.res, self.res), daemon=True)

    def __enter__(self):
        self.p.start()
        time.sleep(3)
        with self.frames.get_lock():
            self.frames.value = 0
        self.t0 = time.time()
        return self

    def rate(self):
        return self.frames.value / max(time.time() - self.t0, 1e-6)

    def __exit__(self, *exc):
        self.stop.set()
        self.p.join(timeout=10)
        if self.p.is_alive():
            self.p.terminate()


def _stack(mud, yolo, policy, seconds):
    stop, counter = mp.Event(), mp.Value("i", 0)
    proc = mp.Process(target=_model_worker, args=(mud, stop, counter, yolo), daemon=True)
    proc.start()
    time.sleep(4)
    with counter.get_lock():
        counter.value = 0
    r = bench_loop(seconds, policy, None)
    r["yolo_plus_encoder_hz"] = counter.value / seconds
    stop.set()
    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
    return r


if __name__ == "__main__":
    main()
