# Plan

Goal: replace the Raspberry Pi Zero 2W in Open Duck Mini v2 with a Sipeed MaixCAM2, run the ML on its NPU, and use the built-in camera to walk better.

The camera steers the walk (where to go, when to stop) and the RL policy keeps handling balance and gait. Feeding raw images straight into the policy is out of scope.

| # | Step | Done when |
|---|------|-----------|
| 1 | **Port the runtime and walk with the existing policy.** Servo, IMU and foot-switch drivers for the MaixCAM2, the 50 Hz control loop, and ONNX to NPU conversion of `BEST_WALK_ONNX_2.onnx`. | The duck walks as well as it did on the Pi, with the policy on the NPU and no loop overruns. |
| 2 | **Camera perception on the NPU.** Run a detector (person, face, obstacles) on the built-in camera and turn its output into walking commands (turn toward a target, stop before an obstacle). | The duck follows a person and stops before obstacles. |
| 3 | **Train a better policy.** Retrain in Open_Duck_Playground with stronger domain randomization and rewards, plus the camera-derived inputs (target direction, obstacle distance). Export to the NPU. | It beats the stock policy in simulation and on the robot. |
| 4 | **Tune on the robot.** Compare with the stock policy and tune gains and latency. | Measured improvement on the real duck. |

## Step 1 status

Code is in this repo (see the README). It still needs testing on hardware:

- [ ] Pick the foot-switch GPIO pins and the I2C pins (`scripts/find_pins.py`)
- [ ] `scripts/check_motors.py` finds all 14 servos
- [ ] `scripts/test_imu.py` shows gravity on +Z when upright, then `--calibrate`
- [ ] `scripts/test_feet.py` shows both switches
- [ ] Walk on CPU (`scripts/walk.py`) and record `--save_obs obs.npy`
- [ ] Convert to NPU (`tools/convert/convert_policy.sh`) and check with `scripts/bench_policy.py`
- [ ] Walk on NPU (`--backend npu`)

## Step 2 status

Follow mode code is in (`--follow`, `odm_maix/vision/`). It still needs testing on hardware:

- [ ] `scripts/test_vision.py --show` detects people at a usable frame rate
- [ ] Turn direction and head signs are right, and `hfov_deg` matches the lens
- [ ] Walk and follow, then tune `target_height`, `k_turn` and `k_forward`
- [ ] Obstacle stop is still to do. The plan is to use `maix.nn.DepthAnything` (relative depth) on the NPU, or a detector trained on obstacles.

## Step 3 status

Training code is in `training/` (see [training/README.md](../training/README.md)).

- [x] Found a sim/robot mismatch: training never moves the head, but the robot applies head commands. In MuJoCo, head tracking doubles the stock policy's tracking error and causes falls (2/40 episodes).
- [x] `FollowJoystick` training env, wider randomization, policy profiles, `sim_eval.py`
- [x] Runtime reads policy profiles (head commands, motor speed limit, extra camera inputs)
- [ ] First 300M-step training run on a GPU, then compare it with stock in `sim_eval.py`
- [ ] Try it on the robot (CPU first, then NPU)
- [ ] Later: camera-input policy (target bearing, or a small vision-encoder embedding) through `extra_obs`
