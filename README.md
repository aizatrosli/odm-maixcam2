# Open Duck Mini on MaixCAM2

A runtime for [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini) that replaces the Raspberry Pi Zero 2W with a [Sipeed MaixCAM2](https://wiki.sipeed.com/hardware/en/maixcam/maixcam2.html) (AX630C, 2x A53, 3.2 TOPS INT8 NPU, built-in camera). The hardware is the standard v2 BOM; only the brain changes.

The roadmap is in [docs/PLAN.md](docs/PLAN.md). This first part (step 1) ports the [Pi runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) so the pretrained walking policy runs on the MaixCAM2, first on the CPU and then on the NPU.

## What changed from the Pi runtime

| Pi runtime | Here | Why |
|---|---|---|
| `rustypot` (Rust extension) | `odm_maix/feetech.py`, pure Python + pyserial | Nothing to compile for the board; reads position and velocity in one bus transaction |
| `adafruit_bno055` through Blinka | `odm_maix/bno055.py` over `/dev/i2c-N` (smbus2) | Blinka has no MaixCAM2 board definition |
| `board`/`digitalio` foot switches | `maix.gpio` | MaixCAM2 GPIO |
| `onnxruntime` only | `onnx` (CPU) or `npu` (`maix.nn` + Pulsar2 `.axmodel`) backends | NPU inference |
| calibration in `.pkl` | `~/imu_offsets.json` | Plain file |

These stay exactly the same, because the policy depends on them: the 101-value observation layout, the action scale (0.25), servo IDs, init pose, gains, the gait phase clock, and the rustypot unit conversions. `tests/` checks them.

## Wiring

The full pin table, connector map and power layout are in [docs/WIRING.md](docs/WIRING.md). In short (all MaixCAM2 IO is **3.3 V**):

![MaixCAM2 to Open Duck Mini v2 wiring](docs/images/maixcam2-duck-wiring.png)

<sub>Base pinout drawing: [Sipeed MaixCAM2 pins](https://wiki.sipeed.com/hardware/en/maixcam/maixcam2.html). Duck labels added for this project.</sub>


| Part | MaixCAM2 |
|---|---|
| UBEC 5 V | 5V + GND on the right header |
| Servo driver board | Type-C in USB host mode (`/dev/ttyACM0`), or UART4 `A21`/`A22` (`/dev/ttyS4`) |
| BNO055 | 3V3, GND, `A8` I2C7_SCL, `A9` I2C7_SDA |
| Foot switches (to GND) | left `B21`, right `B20` |
| Antenna servos | `A31` PWM7, `B2` PWM4 |
| Eye LEDs, projector | `B3` PWM5, `A30` PWM6, `A4` GPIO |
| Speaker, mic, camera | onboard, no wiring |

## Setup on the MaixCAM2

```bash
git clone https://github.com/aizatrosli/odm-maixcam2 && cd odm-maixcam2
pip3 install -r requirements.txt
cp example_duck_config.json ~/duck_config.json   # then set your joint offsets and pins
# from the Pi runtime / Open_Duck_Mini repos:
#   ~/BEST_WALK_ONNX_2.onnx
#   ~/polynomial_coefficients.pkl   (or set maixcam2.nb_steps_in_period)
```

If you already have a `duck_config.json` from the Pi, reuse it and add the `maixcam2` section.

## Bring-up, in order

```bash
python3 scripts/find_pins.py               # check the pins in docs/WIRING.md
python3 scripts/check_motors.py            # all 14 servos answer, torque off
python3 scripts/test_imu.py                # upright -> accelero ~ [0, 0, +9.8]
python3 scripts/test_imu.py --calibrate    # saves ~/imu_offsets.json
python3 scripts/test_feet.py
python3 scripts/walk.py --no_gamepad       # walk in place, policy on CPU
python3 scripts/walk.py --save_obs obs.npy # walk around a bit with the gamepad, record observations
```

`walk.py` prints the loop time, policy time and overruns every 5 seconds. The loop has 20 ms per step at 50 Hz.

## Running the policy on the NPU

On a PC with Docker and the Pulsar2 image ([Sipeed guide](https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam2.html)):

```bash
pip install -r requirements-dev.txt
tools/convert/convert_policy.sh BEST_WALK_ONNX_2.onnx obs.npy   # -> out/walk_policy.mud + .axmodel
```

The quantizer calibrates on `obs.npy`, so use observations from real walking. The config quantizes the whole network to 16-bit (`U16`), because 8-bit loses too much accuracy for a balance controller. Copy `out/` to the board, then:

```bash
python3 scripts/bench_policy.py --onnx ~/BEST_WALK_ONNX_2.onnx --mud ~/out/walk_policy.mud --obs obs.npy
python3 scripts/walk.py --backend npu --policy ~/out/walk_policy.mud
```

`bench_policy.py` compares NPU and CPU outputs and timings. Only switch to the NPU when it prints `OK for walking`.

The policy is a small MLP (101 -> 14), so the CPU handles it in well under a millisecond. Moving it to the NPU frees the CPU for the camera pipeline in step 2, and it is the same conversion path a future vision-aware policy will use. If the NPU result is worse, keep `policy_backend: "onnx"` for now and spend the NPU on vision.

## Follow a person with the camera (step 2)

`--follow` replaces the gamepad sticks with the camera. YOLO11n runs on the NPU in its own process, so camera and NPU work never stall the 50 Hz loop. A follow controller then turns the person it sees into walking commands:

- it turns toward the person, and only walks forward once it is facing them (within 20 degrees)
- it walks until the person's box fills `target_height` of the image, and stands still if they get closer than `stop_height`
- the head tracks the person (yaw and pitch)
- if it loses the person for 0.6 s, it ramps down to standing still
- every command is rate-limited, and stays inside the ranges the stock policy was trained on

The gamepad still works: A pauses, and pushing a stick overrides the camera.

```bash
python3 scripts/test_vision.py --show      # motors off: prints state, target and commands
python3 scripts/walk.py --follow           # walk and follow
```

Before walking, use `test_vision.py` to check three things. First, a person on the right of the image gives a negative `wz` (turn right). Second, `head_yaw` moves the head toward the person; if it doesn't, set `follow.head_yaw_sign` (and `head_pitch_sign`) to -1. Third, set `hfov_deg` to your lens. If the camera is mounted in the head, set `camera_on_head: true` so the head angle is taken into account. All tuning lives under `maixcam2.vision.follow` in `duck_config.json` (see `odm_maix/vision/follow.py`).

## Better walking policy (step 3)

`training/` retrains the walking policy in Open_Duck_Playground so that it matches follow mode, where the head moves while the duck walks. The stock policy never trained with that. See [training/README.md](training/README.md). A trained policy comes as `name.onnx` plus a `name.json` *policy profile*, which `walk.py` reads automatically. A policy without a profile runs exactly as before.

## Benchmark the NPU and the control loop

These numbers decide which camera encoder (step 3) the duck can afford. On a PC with Docker and Pulsar2, build ViT encoders of several sizes (random weights, since only the architecture sets the speed):

```bash
tools/bench_vit/build_vits.sh                      # tiny/small/base/large at 160 and 224 px
tools/bench_vit/build_vits.sh small:128:4          # size:res:frames, e.g. a 4-frame clip
PRECISION=int8 tools/bench_vit/build_vits.sh       # all INT8 instead of U16 LayerNorm + Softmax
```

Copy `out/` to the board, then:

```bash
python3 scripts/bench_npu.py out/*/*.mud --yolo /root/models/yolo11n.mud
python3 scripts/bench_npu.py --loop 30 --policy ~/out/walk_policy.mud --load out/vit_small_160_f1/vit_small_160_f1.mud
```

The first line times each model and records how much CMM it takes. YOLO11n is a reference point against Sipeed's published figure. The second runs the 50 Hz loop with the policy on the NPU while a ViT runs flat out in another process, and counts overruns. No servos move. Everything is saved to `bench_results.json`.

## Debloat the MaixCAM2 and boot into the duck

The stock image boots into Sipeed's launcher, runs some background services, and reserves 3 GB of the 4 GB board for the NPU/camera memory pool (CMM). Only about 62 MB of that pool is in use at idle, so Linux gets under 1 GB. `tools/system/setup_board.sh` trims this, and records every change so it can be undone:

```bash
sh tools/system/setup_board.sh audit                        # what runs, memory split, governor; changes nothing
sh tools/system/setup_board.sh apply --dry-run --cmm 1024   # preview
sh tools/system/setup_board.sh apply --cmm 1024             # debloat, install duck.service (not enabled yet)
systemctl start duck; journalctl -u duck -f                 # try it
sh tools/system/setup_board.sh apply --autostart --args "--follow"   # boot into the duck instead of the launcher
reboot
sh tools/system/setup_board.sh revert && reboot             # back to stock
```

`apply` does the following:
- Disables package-update timers and daemons the duck doesn't use, such as `apt-daily`, `man-db`, `ModemManager`, `avahi` and `cups`, but only the ones present on your image. It leaves NPU, camera, USB, SSH and Wi-Fi units alone.
- Keeps the journal in RAM (16 MB). Use `--keep-journal` to keep it on disk.
- Installs `duck.service`, which runs `scripts/walk.py` with real-time priority and sets the CPU governor to `performance` first. It never auto-restarts, because a crash loop could make the robot lurch.
- With `--autostart`, disables `launcher.service` and moves `/maixapp/auto_start.txt` aside.
- With `--cmm MIB`, sets `maix_memory_cmm` in `/boot/configs` (needs a reboot). Start at 1024, then check peak usage with the full duck running (`grep -A1 CMM_USE_INFO /proc/ax_proc/mem_cmm_info`) and keep at least 2× that. A bigger NPU model needs its size + about 512 MB.

While `duck.service` runs, MaixVision can't stop it, so run `systemctl stop duck` before developing from MaixVision. If the servo board uses the Type-C port, the USB network is gone, so make sure Wi-Fi SSH works before you enable `--autostart`. With `--autostart`, the duck starts its walk loop right after boot, so boot it on its stand or lying down.

To set up more ducks, set up one board, back up its SD card or eMMC, and flash that image to the others.

## Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests
ODM_POLICY_ONNX=path/to/BEST_WALK_ONNX_2.onnx python3 -m pytest tests   # also runs the real policy
```

## Not verified yet

This was written against the documentation and has not run on a MaixCAM2 yet. Check these first:

- the MaixCAM2 pin names and UART/I2C device numbers
- whether the Pulsar2 config (`Numpy` calibration, `FP32` input, full `U16`) builds on your Pulsar2 version
- `maix.nn.NN.forward` with a raw float tensor (the Sipeed docs only show image models)
- whether the ViT benchmark configs (`op_types` U16 override, 5-D input for multi-frame) build on your Pulsar2 version
- whether onnxruntime installs with pip on the board image
- that `/root/models/yolo11n.mud` ships on your MaixPy image (YOLO11 needs MaixPy 4.7.0 or later), and the camera FOV and head joint signs for follow mode
- `tools/system/setup_board.sh` on a real board: the systemd unit names Sipeed ships (`launcher.service`), the `/boot/configs` format, and that SCHED_FIFO is allowed for `duck.service`

## Credits

Based on [Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini) and [Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) by Antoine Pirrone and contributors. The control loop, observation layout and servo constants follow the runtime closely.
