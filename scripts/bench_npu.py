"""Time NPU models and the 50 Hz control loop on the MaixCAM2.

Run on the board. Prints a table and writes everything to a JSON file you can
send back, so the encoder estimates can be replaced with real numbers.

    # 1. latency of any compiled models (.mud), e.g. the ViTs from tools/bench_vit
    python3 scripts/bench_npu.py out/*/*.mud

    # 2. reference point: YOLO11n through MaixPy (Sipeed quotes up to 113 FPS for YOLO11n at 640x640, model only;
    #    the stock 640x480 model timed through detect() includes pre/post-processing)
    python3 scripts/bench_npu.py --yolo /root/models/yolo11n.mud

    # 3. control-loop jitter at 50 Hz with the walking policy, while a ViT
    #    hammers the NPU in another process (no servos are moved)
    python3 scripts/bench_npu.py --loop 30 --policy walk_policy.mud --load out/vit_small_160_f1/vit_small_160_f1.mud

All three can be combined in one run. Results go to bench_results.json (--json).
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import platform
import re
import time

import numpy as np


# ---------------------------------------------------------------- memory
def cmm_used_mb():
    try:
        txt = open("/proc/ax_proc/mem_cmm_info").read()
        m = re.search(r"used=(\d+)KB", txt)
        return int(m.group(1)) / 1024 if m else None
    except OSError:
        return None


def linux_available_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def stats(ms):
    ms = np.asarray(ms)
    return {
        "mean_ms": float(ms.mean()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p99_ms": float(np.percentile(ms, 99)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "hz": float(1000.0 / ms.mean()),
    }


# ---------------------------------------------------------------- generic models
class Runner:
    """Feeds random inputs of the right shape to a maix.nn.NN model."""

    def __init__(self, mud):
        from maix import nn, tensor

        self.t = tensor
        self.model = nn.NN(mud)
        self.inputs = []
        for info in self.model.inputs_info():
            shape = [int(s) for s in info.shape]
            dt = str(info.dtype).lower()
            if "uint8" in dt:
                arr = np.random.randint(0, 255, size=shape, dtype=np.uint8)
                conv = tensor.tensor_from_numpy_uint8
            else:
                arr = np.random.rand(*shape).astype(np.float32)
                conv = tensor.tensor_from_numpy_float32
            self.inputs.append((info.name, shape, dt, conv(arr)))

    def describe(self):
        return [{"name": n, "shape": s, "dtype": d} for n, s, d, _ in self.inputs]

    def forward(self):
        ts = self.t.Tensors()
        for name, _, _, x in self.inputs:
            ts.add_tensor(name, x, False, False)
        return self.model.forward(ts)


def bench_model(mud, n, warmup):
    before = cmm_used_mb()
    r = Runner(mud)
    after = cmm_used_mb()
    for _ in range(warmup):
        r.forward()
    ms = []
    for _ in range(n):
        t = time.perf_counter()
        r.forward()
        ms.append((time.perf_counter() - t) * 1000)
    out = {"model": mud, "inputs": r.describe(), **stats(ms)}
    if before is not None and after is not None:
        out["cmm_mb"] = after - before
    del r
    gc.collect()
    return out


def bench_yolo(mud, n, warmup):
    from maix import image, nn

    det = nn.YOLO11(model=mud, dual_buff=False)
    img = image.Image(det.input_width(), det.input_height(), det.input_format())
    for _ in range(warmup):
        det.detect(img)
    ms = []
    for _ in range(n):
        t = time.perf_counter()
        det.detect(img)
        ms.append((time.perf_counter() - t) * 1000)
    return {"model": mud, "input": [det.input_width(), det.input_height()], **stats(ms)}


# ---------------------------------------------------------------- control loop
def _load_worker(mud, stop, counter):
    r = Runner(mud)
    while not stop.is_set():
        r.forward()
        with counter.get_lock():
            counter.value += 1


def bench_loop(seconds, policy_mud, load_mud, freq=50.0):
    stop, counter, proc = None, None, None
    if load_mud:
        stop, counter = mp.Event(), mp.Value("i", 0)
        proc = mp.Process(target=_load_worker, args=(load_mud, stop, counter), daemon=True)
        proc.start()
        time.sleep(3)  # let it load
        with counter.get_lock():
            counter.value = 0

    policy = Runner(policy_mud) if policy_mud else None
    obs = np.random.rand(101).astype(np.float32)
    period = 1.0 / freq
    lateness, work, infer = [], [], []
    overruns = 0
    gc_was = gc.isenabled()
    t0 = time.monotonic()
    next_t = t0
    while time.monotonic() - t0 < seconds:
        now = time.monotonic()
        lateness.append((now - next_t) * 1000)
        ts = time.perf_counter()
        _ = np.concatenate([obs, obs[:16]])  # stand-in for building the observation
        if policy:
            ti = time.perf_counter()
            policy.forward()
            infer.append((time.perf_counter() - ti) * 1000)
        work.append((time.perf_counter() - ts) * 1000)
        next_t += period
        delay = next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            overruns += 1
            next_t = time.monotonic()
    load_hz = None
    if proc:
        load_hz = counter.value / seconds
        stop.set()
        proc.join(timeout=5)
    res = {
        "seconds": seconds,
        "steps": len(work),
        "overruns": overruns,
        "wake_late": stats(np.abs(lateness)),
        "step_work": stats(work),
        "gc_enabled": gc_was,
        "load_model": load_mud,
        "load_model_hz": load_hz,
    }
    if infer:
        res["policy_infer"] = stats(infer)
    return res


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", help=".mud files to time")
    p.add_argument("-n", type=int, default=100, help="timed runs per model")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--yolo", default=None, help="YOLO11 .mud for a reference point")
    p.add_argument("--loop", type=float, default=0, help="seconds of 50 Hz loop test")
    p.add_argument("--policy", default=None, help="walking policy .mud for the loop test")
    p.add_argument("--load", default=None, help="model to run flat out in another process during the loop test")
    p.add_argument("--json", default="bench_results.json")
    a = p.parse_args()

    results = {
        "host": platform.node(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cmm_used_mb_start": cmm_used_mb(),
        "linux_available_mb_start": linux_available_mb(),
        "models": [],
    }

    for mud in a.models:
        print(f"timing {mud} ...", flush=True)
        try:
            r = bench_model(mud, a.n, a.warmup)
            print(f"  {r['mean_ms']:.2f} ms mean, p99 {r['p99_ms']:.2f} ms, {r['hz']:.1f} Hz"
                  + (f", CMM {r['cmm_mb']:.0f} MB" if r.get("cmm_mb") is not None else ""))
        except Exception as e:  # keep going so one bad model doesn't lose the rest
            r = {"model": mud, "error": repr(e)}
            print(f"  failed: {e!r}")
        results["models"].append(r)

    if a.yolo:
        print(f"timing YOLO {a.yolo} ...", flush=True)
        r = bench_yolo(a.yolo, a.n, a.warmup)
        print(f"  {r['mean_ms']:.2f} ms mean ({r['hz']:.1f} FPS, detect() incl. pre/post)")
        results["yolo"] = r

    if a.loop:
        print(f"50 Hz loop for {a.loop:.0f} s"
              + (f" with policy {a.policy}" if a.policy else "")
              + (f", NPU load {a.load}" if a.load else "") + " ...", flush=True)
        r = bench_loop(a.loop, a.policy, a.load)
        print(f"  overruns {r['overruns']}/{r['steps']}, step work p99 {r['step_work']['p99_ms']:.2f} ms, "
              f"wake-up late p99 {r['wake_late']['p99_ms']:.2f} ms")
        if "policy_infer" in r:
            print(f"  policy on NPU: {r['policy_infer']['mean_ms']:.2f} ms mean, p99 {r['policy_infer']['p99_ms']:.2f} ms")
        if r["load_model_hz"]:
            print(f"  background model ran at {r['load_model_hz']:.1f} Hz")
        results["loop"] = r

    results["cmm_used_mb_end"] = cmm_used_mb()
    with open(a.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {os.path.abspath(a.json)}")


if __name__ == "__main__":
    main()
