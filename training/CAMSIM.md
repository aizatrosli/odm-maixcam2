# Camera simulator (MaixCAM2 + follow mode in MuJoCo)

`odm_train/camsim/` adds the MaixCAM2 camera, a person to follow and a randomized room to Open_Duck_Playground's MuJoCo duck. It then runs the robot runtime's own loop at 50 Hz:

```
camera frame (15 Hz, 66 ms late) -> person box -> odm_maix FollowController -> 7 commands -> walking policy (ONNX)
```

Use it to:

- watch follow mode end to end before trying it on the robot (`--video`),
- record datasets of camera frames with ground truth for the JEPA encoder work (`--record`),
- run policies whose profile lists camera inputs (`target_*`, `emb_<i>`) the way `walk.py` would.

## Run

The quickest way is Docker: `docker compose run --rm sim` from the repo root, or the shared notebook [`camsim.ipynb`](camsim.ipynb) in the JupyterLab service. See [docker/README.md](../docker/README.md).

By hand:

From an Open_Duck_Playground checkout, set up as in [README.md](README.md). Both `training/` and the repo root go on `PYTHONPATH`, because the controller is `odm_maix.vision.follow`:

```bash
cd Open_Duck_Playground
export PYTHONPATH=$PYTHONPATH:$PWD:/path/to/odm-maixcam2/training:/path/to/odm-maixcam2
pip install imageio imageio-ffmpeg           # for --video (falls back to GIF without ffmpeg)

python -m odm_train.camsim.follow_sim ../Open_Duck_Mini/BEST_WALK_ONNX_2.onnx --episodes 3 --video follow.mp4
python -m odm_train.camsim.follow_sim policy.onnx --episodes 20 --record follow.npz   # dataset
python -m odm_train.camsim.follow_sim policy.onnx --episodes 50 --no_render           # fast, boxes from geometry only
python -m odm_train.camsim.follow_sim policy.onnx --mount trunk                      # camera_on_head: false
```

Each episode prints whether the duck fell, how often the person was tracked (`target_seen`), the share of time spent 0.6 to 2 m from the person (`in_follow_range`), the mean distance, and the mean absolute bearing of the person from the duck's heading.

Rendering uses `MUJOCO_GL=egl` when an NVIDIA device is visible and OSMesa (CPU) otherwise. Set `MUJOCO_GL` yourself to override. On CPU, a 10 s episode with video takes about 30 s, and about 8 s with `--no_render`.

## What is simulated

| Part | Model | Default |
|---|---|---|
| Mount | `maixcam2` camera on `head_assembly` (moves with head yaw/pitch) or `trunk_assembly`; optional module weight (`--camera_mass_g`) | head, lens 7.5 cm ahead of the head centre, no added weight |
| Lens | Pinhole set by the horizontal FOV, plus radial distortion `k1`, `k2`. Rendered wider and resampled | 70 deg, 320x240, k1 = -0.08 |
| Sensor | Exposure and white-balance offsets per episode, Gaussian noise, row blur from the camera's turn rate x exposure | ±15 %, ±6 %, σ 2.5, 8 ms |
| Timing | Frames every `1/hz`, delivered `latency_ms` later on the 50 Hz clock | 15 Hz, 66 ms |
| Person | 1.55 to 1.9 m, random clothes and skin. Walks between waypoints 0.9 to 3 m from the duck, pauses, starts in front | 0.15 to 0.45 m/s |
| Distractor | A second person wandering the room. The controller's `pick_target` has to stay on the first one | on |
| Detector | Box from projecting the person's geoms into the distorted image. Misses, jitter, no detection below 6 % of frame height | 5 % misses |
| Room | 8 m square, walls, 8 props, 5 floor textures, 2 lights + headlight, all randomized per episode | |
| Dynamics | Same randomization as `sim_eval.py` (floor friction, servo kp, torso mass) and 0 to 2 steps of action delay | |

The added geoms are visual only. The duck doesn't collide with walls, props or people, and occlusion between people isn't modelled in the boxes.

## Dataset (`--record`)

One row per delivered camera frame, in a compressed `.npz`:

| Key | Shape | Meaning |
|---|---|---|
| `frame` | N x 240 x 320 x 3, uint8 | the delivered image |
| `box`, `box_valid` | N x 4, N | true person box (cx, cy, w, h; NaN when out of view) |
| `dist`, `bearing` | N | ground truth: metres and radians (left positive) from the duck's heading |
| `cmd` | N x 7 | commands sent on the delivery step |
| `obs` | N x 101+ | policy observation on that step |
| `t`, `episode` | N | time in the episode (s), episode index |
| `emb` | N x E | projected encoder output, with `--encoder` (and `feat`, the raw features, after `encoders embed`) |

## Interactive (`app.py`)

```bash
pip install viser==1.1.1 trimesh && pip install --no-deps mjviser==0.0.16   # mjviser asks for mujoco>=3.6; it works on 3.3.7
python -m odm_train.camsim.app policy.onnx        # http://localhost:8080, gamepad and keyboard page on :8081
```

A browser 3D view (Viser, through mjviser) with the camera image, live plots, follow-controller and detector sliders, mouse pushes, and three modes: follow the person, drive the duck, or walk the person yourself. See [docker/README.md](../docker/README.md#interactive-simulator), where it is the `sim` service.

## JEPA encoders (`encoders.py`)

The encoder runs on each delivered frame, and its projected output fills the policy inputs `emb_0..emb_<k-1>`, the same way `walk.py` would feed them on the robot.

| `--encoder` | What | Input (default) | Features |
|---|---|---|---|
| `vjepa2_1_base`, `vjepa2_1_large` | V-JEPA 2.1 via `torch.hub` (pinned commit), official weights, one frame through the image patch embedding, tokens mean-pooled | 160 px (multiple of 16) | 768 / 1024 |
| `dinov2_small`, `dinov2_base` | DINOv2 CLS embedding | 168 px (multiple of 14) | 384 / 768 |
| `onnx:<path>` | any exported encoder: exactly the graph that goes to Pulsar2 | from the graph | from the graph |
| `standin_<size>` | random-weight ViT from `tools/bench_vit`, no download; for tests and timing only | 160 px | 16 |

Frames are squashed from 4:3 to square and normalized inside the encoder, the same as on the robot. A PCA projection (`Projection`) takes the features down to 16 to 32 values.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU torch is enough for a few frames per second

# 1. record frames, 2. fit the projection on them, 3. drive the sim with encoder + projection
python -m odm_train.camsim.follow_sim policy.onnx --episodes 20 --record follow.npz
python -m odm_train.camsim.encoders embed follow.npz --encoder vjepa2_1_base --dim 16 --out follow_emb.npz   # also writes follow_emb_projection.npz
python -m odm_train.camsim.follow_sim policy.onnx --encoder vjepa2_1_base --projection follow_emb_projection.npz --video emb.mp4

# one ONNX (normalization + encoder + projection) for Pulsar2, checked against onnxruntime
python -m odm_train.camsim.encoders export --encoder vjepa2_1_base --res 160 --projection follow_emb_projection.npz --out vjepa_b160_emb16.onnx
python -m odm_train.camsim.encoders time --encoder vjepa2_1_base --res 160
```

`--encoder onnx:vjepa_b160_emb16.onnx` then runs the exported graph (projection included, so leave out `--projection`). The stand-in takes about 28 ms per frame on a laptop CPU at 160 px. The V-JEPA and DINOv2 paths load code from `torch.hub`; they weren't run in this repo's CI, so check the first export's printed `max |torch - onnxruntime|` before converting.

## Camera weight (`--camera_mass_g`)

The stock duck model has no camera, so by default the sim adds none of its weight. `CameraConfig.mass_g` (CLI `--camera_mass_g`, in grams) adds the camera module as its own small body, welded just behind the lens on the mount (head or trunk). Weigh your module with its mount and cable and pass that number.

```bash
python -m odm_train.camsim.follow_sim policy.onnx --camera_mass_g 120 --episodes 8 --no_render
```

Stock policy, 8 episodes of 15 s per row (same seeds), servo limit 3.23 N·m:

| Mount | Added | Duck total | Falls | Head-pitch torque, mean / p99 (N·m) | Head-yaw p99 (N·m) |
|---|---|---|---|---|---|
| head | 0 g | 2.11 kg | 0/8 | 0.48 / 2.13 | 2.88 |
| head | 60 g | 2.17 kg | 0/8 | 0.72 / 2.49 | 2.95 |
| head | 120 g | 2.23 kg | 0/8 | 0.58 / 2.21 | 2.78 |
| head | 250 g | 2.36 kg | 0/8 | 0.54 / 2.22 | 2.70 |
| head | 400 g | 2.51 kg | 5/8 | 0.66 / 2.82 | 3.23 (at the limit) |
| trunk | 120 g | 2.23 kg | 0/8 | 0.21 / 1.08 | 1.56 |
| trunk | 250 g | 2.36 kg | 2/8 | 0.22 / 1.30 | 2.63 |
| trunk | 400 g | 2.51 kg | 8/8 | 0.43 / 2.93 | 3.23 |

- Holding a module still costs little: 120 g at about 6 cm in front of the head-pitch axis is 0.12 × 9.81 × 0.06 ≈ 0.07 N·m, 2 % of the servo's limit. Most of the head torque comes from tracking the person (head yaw is near the limit even without a camera), not from the weight.
- Up to about 250 g on the head the stock policy didn't fall. On the trunk, 250 g already caused falls, because the trunk mount is 10 cm ahead of the trunk and moves the duck's centre of mass forward more than the head mount does.
- Training only randomizes the trunk mass (`torso_mass_add` in `randomize.py`). If your module is heavy, retrain with it in the model.
- 8 episodes is a small sample: rerun with more for a decision.

## Findings so far

- **head_pitch sign.** In the MuJoCo model, positive `head_pitch` tilts the head up. `FollowParams.head_pitch_sign` defaults to +1 ("positive tilts down"), so the sim uses -1. Check which way your real head moves before enabling `camera_on_head` on the robot.
- **sim_eval friction bug, fixed.** `Sim.randomize` set the friction of geom 0, which is part of the duck's trunk, not the floor. It now randomizes the floor.

## Code

| File | What |
|---|---|
| `camsim/scene.py` | `CameraConfig`, `SceneConfig`, `build_scene` (MjSpec), `VisualRandomizer` |
| `camsim/lens.py` | Lens model and `FrameClock`, pure numpy, unit-tested in `tests/test_camsim.py` |
| `camsim/camera.py` | `MaixCam2`: render, project, person box |
| `camsim/person.py` | `PersonWalker` (with `manual` mode for the interactive app) |
| `camsim/follow_sim.py` | `FollowSim` (one control step at a time, for the notebook and the app), `Recorder`, `VideoWriter`, CLI |
| `camsim/encoders.py` | V-JEPA 2.1, DINOv2, ONNX and stand-in encoders, `Projection` (PCA), `export_onnx`, CLI |
| `camsim/app.py`, `camsim/pad.py` | Interactive Viser app; gamepad/keyboard page |
| `camsim/gl.py` | Picks EGL or OSMesa before MuJoCo is imported |
