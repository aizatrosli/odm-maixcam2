# First connection: diagnostics and calibration

Two tools take the duck from "MaixCAM2 wired in" to "ready for the first walk":

- `scripts/diagnose.py` checks the config, the board, every servo, the IMU, the foot switches, the camera, NPU and audio, and the 50 Hz loop. **No check turns servo torque on.**
- `scripts/calibrate.py` sets up new servos, measures joint offsets and ranges, and calibrates the IMU. It writes `duck_config.json` only after you confirm, and keeps a timestamped `.bak` copy.

Both run on a PC with `--dry-run`, against a simulated duck that speaks the real Feetech protocol and BNO055 registers, so the same driver code runs in both cases. `--fault` injects a problem (missing servo, flat battery, IMU upside down, swapped feet, ...) so you can see what the report looks like before it happens for real.

```bash
python3 scripts/diagnose.py --dry-run -i                              # healthy simulated duck
python3 scripts/diagnose.py --dry-run -i --fault missing:23,upside_down,swapped_feet
python3 scripts/diagnose.py --list-faults
python3 scripts/calibrate.py --dry-run offsets --nudge                # writes /tmp/duck_config.dryrun.json, never ~/
```

## Order on the robot

Put the duck on a stand with its feet off the ground, battery charged, and SSH into the MaixCAM2.

| # | Command | What it settles |
|---|---|---|
| 1 | `python3 scripts/diagnose.py --only config,board` | Config file, offsets, policy/gait/IMU files present, pin conflicts, MaixPy version, SoC temperature, governor, memory |
| 2 | `python3 scripts/diagnose.py --only servo --scan` | All 14 servos answer, no stray IDs, servo mode and return delay, battery voltage and spread, temperatures and error flags, encoder wrap, bus latency |
| 3 | `python3 scripts/calibrate.py servo-id --id N` | Only if step 2 found a servo on ID 1 or a missing one: connect that servo alone, give it its joint ID. `--center` drives it to 0 rad to fit the horn centred |
| 4 | `python3 scripts/calibrate.py offsets` | Hold the duck in its init pose with torque off; offsets = measured - init pose |
| 5 | `python3 scripts/calibrate.py offsets --nudge-only` | Fine-tune joint by joint with low torque (one joint powered at a time) |
| 6 | `python3 scripts/calibrate.py ranges --save-limits` | Move each joint end stop to end stop by hand. Flags joints that cross the encoder wrap and saves soft limits (`maixcam2.joint_limits`) that `HWI` clamps targets to. Rerun after changing offsets |
| 7 | `python3 scripts/calibrate.py -i imu` | BNO055 calibration saved to `imu_offsets_path`, then gravity, gyro bias and the tilt/turn axis signs |
| 8 | `python3 scripts/diagnose.py -i` | Everything again, with the guided steps: init pose, tilt and turn, press each foot, hear the speaker, watch the antennas and eye LEDs |

Between steps, `apps/duck_dash` on the MaixCAM2 screen shows the same parts live (servos answering, battery, IMU tilt, feet, camera, NPU policy) without a laptop: plug a servo in and its row turns green. It only reads, so it is safe to leave running on the stand. See [apps/duck_dash/README.md](../apps/duck_dash/README.md).

Exit code is 0 when everything passed, 1 with warnings, 2 with failures, and every number goes to `diag_report.json` so two runs can be compared (for example before and after the thermal soak).

## What each check looks at

| Check | Pass | Typical fix when it fails |
|---|---|---|
| `config.offsets` | non-zero, all under 0.2 rad | above 0.2 rad a horn is probably one spline tooth off: re-seat it |
| `config.pins` | no pin used twice, pinmap matches each part | see docs/WIRING.md |
| `servo.ping` / `servo.scan` | 14 answer, nothing on other IDs | ID 1 is a factory servo: `calibrate.py servo-id` |
| `servo.eeprom` | mode 0, return delay 0, no limits around the init pose | `calibrate.py servo-id --id N --current N` rewrites them |
| `servo.power` | 7.0 to 8.5 V, under 0.4 V spread, under 55 C, no error flags | charge; a big spread is a loose lead in the chain |
| `servo.pose` | every joint at least 0.8 rad from raw 0/4095 | a joint that crosses the wrap jumps by 2 pi in the observation: re-seat the horn |
| `servo.timing` | 14-servo position+velocity read p99 under 4 ms, no errors | baud rate, cable, ground between boards, USB adapter latency |
| `imu.chip` | BNO055 self test passes for accel, mag, gyro and MCU | power-cycle; replace the board if it stays |
| `imu.still` | gravity 9.81 +- 0.3, gyro bias under 0.05 rad/s, no I2C errors | I2C errors: drop I2C7 to 100 kHz (clock stretching) |
| `imu.upright` / `imu.axes` | upright reads +z; nose down gives -x, right side down +y, turning left +z gyro | flip `imu_upside_down`, or fix the mounting |
| `feet.stable` / `feet.press` | inputs steady; each switch changes its own side | flicker: fit the 10 kOhm pull-ups; swapped: swap `left_foot_gpio` / `right_foot_gpio` |
| `camera.frames`, `npu.*`, `audio.*`, `expression.outputs` | frames not black and at least 20 fps, NPU policy within 0.04 of ONNX, YOLO under 40 ms, speaker heard, mic not silent, antennas and LEDs move | per message |
| `loop.timing` | the walk loop's own read + infer path at 50 Hz, torque off: p99 under 12 ms, no overruns | look at `servo.timing` and `imu.still`; set the governor to performance |

## Worth doing before the first walk

These came out of reading the runtime with a first connection in mind. The soft joint limits are in this change; the rest are still open.

1. **A safety stop in the walk loop.** Today a fall, a run of failed bus reads, or a sagging battery does not stop anything: a failed read just skips the step and the servos keep their last target, and `RLWalk.shutdown()` leaves torque on. Torque off (or a soft crouch) on tilt beyond ~45 degrees, on N failed reads in a row, and on servo voltage under ~6.6 V would protect the servos and the gears on the first runs.
2. **Soft joint limits** (done here, optional): measure them with `calibrate.py ranges --save-limits`, and the HWI clamps every target to them.
3. **First torque on a stand with low gains**: `walk.py -p 8 --no_gamepad` before the default `-p 30`, feet off the ground, hand on the switch.
4. **Commit the local work** (thermal soak script and the fixes to `expression.py`, `test_interact.py`, `test_vision.py`) so the board and the repo match before the duck depends on it.
5. **Repeat the thermal soak with the servos loaded.** The 58 C plateau was the MaixCAM2 alone; with the UBEC and servo driver inside the body it will run warmer. `diagnose.py --only board,loop` before and after gives comparable numbers.
6. **Record real observations** (`walk.py --save_obs`) on the first CPU walk and re-run the NPU conversion with them; the current NPU calibration uses simulated observations.
