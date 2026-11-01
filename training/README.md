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

## Easiest: the notebook (Colab or your own GPU)

[`colab_train.ipynb`](colab_train.ipynb) does everything below in one place: it installs a tested set of pinned packages, trains the flavors you pick, saves checkpoints to Google Drive (or `./duck_runs` locally), resumes after a disconnect, compares each result with the stock policy in MuJoCo, and zips `ONNX + JSON` per flavor for the robot.

Open it in Colab from GitHub (File → Open notebook → GitHub → `aizatrosli/odm-maixcam2`), pick a GPU runtime and run all cells. It also runs in local Jupyter on any NVIDIA GPU. Use `NUM_ENVS = 4096` on 8 GB cards and `2048` on 4 GB cards.

## Flavors

`python -m odm_train.train --list_flavors` prints them. They are defined in `odm_train/flavors.py`:

| Flavor | Env | Scene | Randomization |
|---|---|---|---|
| `follow_flat_backlash` (default, recommended) | FollowJoystick | flat, leg backlash | strong |
| `follow_rough_backlash` | FollowJoystick | rough, leg backlash | strong |
| `follow_flat` | FollowJoystick | flat | strong |
| `stock_flat`, `stock_flat_backlash`, `stock_rough_backlash` | upstream Joystick | as named | stock |
| `standing_flat` (experimental, 85-value obs, not runnable by `walk.py` yet) | upstream Standing | flat | stock |

## Setup by hand (Linux PC with an NVIDIA GPU)

```bash
git clone https://github.com/apirrone/Open_Duck_Playground
cd Open_Duck_Playground
git checkout b9be205ac64488c23504ca42e5ec790337adeec3
python3.12 -m venv .venv && . .venv/bin/activate
pip install "jax[cuda12]==0.8.0" mujoco==3.3.7 mujoco-mjx==3.3.7 brax==0.13.0 playground==0.0.5 \
  flax==0.12.0 orbax-checkpoint==0.12.8 ml-collections==1.1.0 tensorflow==2.21.0 tf2onnx==1.17.0 \
  onnx==1.23.2 onnxruntime==1.31.1 tensorboardX==2.6.5 jaxtyping mediapy
export PYTHONPATH=$PYTHONPATH:$PWD:/path/to/odm-maixcam2/training
```

Newer `playground` releases removed modules the env imports, and unpinned installs pull a newer MuJoCo and JAX, so keep these pins.

## Train

```bash
python -m odm_train.train --flavor follow_flat_backlash --output_dir runs/follow_v1
python -m odm_train.train --flavor follow_rough_backlash --output_dir runs/follow_v1_rough
# small GPU:            --num_envs 2048
# continue after a stop: --resume   (same --output_dir; --num_timesteps is the total)
tensorboard --logdir runs
```

Expect a few hours on one recent GPU for 300M steps. Each checkpoint writes `<date>_<step>.onnx` and `.json`, and `progress.json` records the step count and newest ONNX.

## Evaluate before it touches the robot

```bash
python -m odm_train.sim_eval ../Open_Duck_Mini/BEST_WALK_ONNX_2.onnx runs/follow_v1/<ckpt>.onnx -n 40
python -m odm_train.sim_eval runs/follow_v1/<ckpt>.onnx -n 40 --no_head   # still fine without head motion?
python -m odm_train.sim_eval runs/follow_v1/<ckpt>.onnx --mode straight   # plain walking
python playground/open_duck_mini_v2/mujoco_infer.py -o runs/follow_v1/<ckpt>.onnx  # watch it
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
- The notebook and trainer were run end to end on CPU with tiny settings (install pins, train two flavors, resume, evaluate, package). Real GPU timings are still unmeasured.
- **No full training run has been done yet** (it needs a GPU). The reward scales are the stock ones, so the first run may need tuning.
