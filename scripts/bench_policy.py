"""Compare the NPU policy with the ONNX reference and time both.

    python3 scripts/bench_policy.py --onnx BEST_WALK_ONNX_2.onnx --mud walk_policy.mud --obs obs.npy

``--obs`` should be observations recorded on the robot (walk.py --save_obs);
without it random observations are used, which only checks timing.
"""

import argparse
import time

import numpy as np
import _common  # noqa: F401  (sets sys.path)

from odm_maix.policy import OBS_SIZE, NpuPolicy, OnnxPolicy


def timeit(policy, obs, n):
    for o in obs[:10]:
        policy.infer(o)
    t = time.perf_counter()
    outs = [policy.infer(obs[i % len(obs)]) for i in range(n)]
    return (time.perf_counter() - t) / n * 1000, np.array(outs)


p = argparse.ArgumentParser()
p.add_argument("--onnx", required=True)
p.add_argument("--mud", default=None)
p.add_argument("--obs", default=None)
p.add_argument("-n", type=int, default=1000)
args = p.parse_args()

if args.obs:
    obs = np.load(args.obs).astype(np.float32)
else:
    obs = np.random.default_rng(0).normal(0, 0.5, size=(500, OBS_SIZE)).astype(np.float32)
n = min(args.n, len(obs))

ms, ref = timeit(OnnxPolicy(args.onnx), obs, n)
print(f"ONNX (CPU): {ms:.3f} ms per step")

if args.mud:
    ms, out = timeit(NpuPolicy(args.mud), obs, n)
    print(f"NPU       : {ms:.3f} ms per step")
    err = np.abs(out - ref)
    cos = np.sum(out * ref, 1) / (np.linalg.norm(out, axis=1) * np.linalg.norm(ref, axis=1) + 1e-9)
    print(f"max abs error {err.max():.4f}  mean abs error {err.mean():.4f}  min cosine {cos.min():.4f}")
    # actions are scaled by 0.25 rad, so 0.04 here is ~0.6 degrees at the joint
    print("OK for walking" if err.max() < 0.04 and cos.min() > 0.99 else "CHECK: NPU output differs noticeably")
