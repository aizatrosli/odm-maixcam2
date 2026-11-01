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

All MaixCAM2 IO is **3.3 V**. Do not connect 5 V signals.

- **Power:** feed the MaixCAM2 from the duck's 5 V buck, the same one that powered the Pi.
- **Servos:** either keep the Waveshare bus-servo adapter on USB (the MaixCAM2 Type-C port in host/OTG mode, giving `/dev/ttyACM0`), or wire the adapter's UART header to a MaixCAM2 UART and set `servo_port` (for example `/dev/ttyS1`) plus the `pinmap` entries. The bus runs at 1 Mbps.
- **BNO055:** to the MaixCAM2 I2C pins (3.3 V, GND, SDA, SCL). The bus is found automatically at start-up, or set `imu_i2c_bus`.
- **Foot switches:** two GPIOs to GND, using internal pull-ups. Run `python3 scripts/find_pins.py` on the board and set `left_foot_gpio` / `right_foot_gpio`. **The defaults `GPIOA4`/`GPIOA5` are placeholders, so check them against the pinout before you use them.**

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
python3 scripts/find_pins.py               # pick I2C / GPIO pins
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
- whether onnxruntime installs with pip on the board image

## Credits

Based on [Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini) and [Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) by Antoine Pirrone and contributors. The control loop, observation layout and servo constants follow the runtime closely.
