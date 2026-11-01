# Training a better walking policy (step 3)

This folder extends [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) (MuJoCo MJX + Brax PPO). It trains a walking policy that matches how the MaixCAM2 duck is actually driven in follow mode.

## Why retrain

The stock policy was trained with head commands that **only appear in the observation**: the training env never moves the head. On the robot, the runtime adds those commands to the head joints. Follow mode moves the head all the time, so the duck meets a disturbance it never saw in training.

`sim_eval.py` reproduces this in MuJoCo. These are 40 randomized 20 s episodes per row, with follow-style commands, up to 2 steps of action latency, and randomized friction, kp and torso mass:

| Stock `BEST_WALK_ONNX_2.onnx` | falls | vx RMS error (m/s) | wz RMS error (rad/s) |
|---|---|---|---|
| head still | 0/40 | 0.030 | 0.247 |
| head tracking (follow mode) | 2/40 | 0.055 | 0.536 |

Head tracking roughly doubles the tracking error, and the duck starts falling.

## What `FollowJoystick` changes

`odm_train/follow_env.py` subclasses the stock `Joystick` task:

1. **Head commands move the head joints**, as on the robot, and the observed `motor_targets` include the head offsets.
2. **Follow-style commands.** A new goal comes every 1 to 4 s and the command ramps toward it at the follow controller's rate limits. 15% of goals stand still, 55% walk or turn with the head leading the turn, and 30% are uniform (so gamepad driving still works).
3. **Wider randomization** (`odm_train/randomize.py`). This covers floor friction 0.4 to 1.2, servo kp ±20%, a heavier torso (the MaixCAM2 and its mount), and larger calibration offsets. Pushes go up to 1.5 m/s.
4. **The stand-still cost ignores the head**, because the head is commanded now.

The observation is still the same 101 values. That means the result is a drop-in replacement on the MaixCAM2, including the NPU conversion.

## Policy profiles

Every exported checkpoint gets a `<name>.json` next to `<name>.onnx`, and the runtime reads it (`odm_maix/policy_profile.py`). It records how the policy was trained: action scale, whether head commands are applied, the motor speed limit (training clips target changes to 5.24 rad/s, which the Pi runtime never did), and any **extra inputs**.

Extra inputs leave room for camera-aware policies. A profile can list `extra_obs`, such as `target_bearing_sin` or `target_size` (already provided by follow mode), or the 16 to 32 values of a small vision-encoder embedding. The runtime appends them after the 101 base values and checks the model's input size at start-up. Training with such inputs needs a matching feature in simulation (a virtual target is easy; image embeddings need rendering or a teacher-student setup), so that is left for later.

## Setup (on a Linux PC with an NVIDIA GPU)

```bash
git clone https://github.com/apirrone/Open_Duck_Playground
cd Open_Duck_Playground
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv pip install "playground==0.0.5"   # newer mujoco_playground moved collision helpers the env imports
export PYTHONPATH=$PYTHONPATH:/path/to/odm-maixcam2/training
```

## Train

```bash
uv run python -m odm_train.train --num_timesteps 300000000 --output_dir runs/follow_v1
# rough floors:
uv run python -m odm_train.train --task rough_terrain_backlash --output_dir runs/follow_v1_rough
uv run tensorboard --logdir runs
```

Expect a few hours on one recent GPU for 300M steps. Each checkpoint writes `<date>_<step>.onnx` and `.json`.

## Evaluate before it touches the robot

```bash
uv run python -m odm_train.sim_eval ../Open_Duck_Mini/BEST_WALK_ONNX_2.onnx runs/follow_v1/<ckpt>.onnx -n 40
uv run python -m odm_train.sim_eval runs/follow_v1/<ckpt>.onnx -n 40 --no_head   # still fine without head motion?
uv run python -m odm_train.sim_eval runs/follow_v1/<ckpt>.onnx --mode straight   # plain walking
uv run playground/open_duck_mini_v2/mujoco_infer.py -o runs/follow_v1/<ckpt>.onnx  # watch it
```

Only move on when the new checkpoint has fewer falls and lower `wz_rms` than the stock policy with head tracking on, and is no worse with the head still.

## Deploy

Copy `<ckpt>.onnx` and `<ckpt>.json` to the duck (same folder, same name):

```bash
python3 scripts/walk.py --policy ~/follow_v1.onnx --follow
```

For the NPU, `tools/convert/convert_policy.sh <ckpt>.onnx obs.npy` carries the profile into `out/walk_policy.json`. Record `obs.npy` with the new policy running on the CPU, because its observations differ slightly from the stock policy's.

## Status

- The env is smoke-tested on CPU. Reset and step run, the commands are slew-limited, and the head offsets are applied.
- `sim_eval.py` runs the baseline above.
- **No full training run has been done yet** (it needs a GPU). The reward scales are the stock ones, so the first run may need tuning.
