# On-board test checklist

These are the items that the docs and source code cannot settle. Everything else in the README was checked against Sipeed, Axera, Feetech, Bosch and Open_Duck_Mini sources (see "Verification status" in the [README](../README.md)). Run these over SSH on the MaixCAM2 (`ssh maixcam`), in order. Each one says what a pass looks like.

## Board and OS

- [ ] **Stock launcher unit name.** `systemctl cat launcher.service; systemctl is-enabled launcher.service`. Pass: the unit exists and is enabled. If it has another name, `setup_board.sh apply --autostart` stops and asks you to find it.
- [ ] **Services that must stay on.** `systemctl list-unit-files --state=enabled | grep -Ei 'ax|npu|cmm|usb|gadget|maix|launcher'`. Pass: none of these are in `UNITS_TO_DISABLE` in `tools/system/setup_board.sh`.
- [ ] **Real-time priority for duck.service.** `cat /proc/sys/kernel/sched_rt_runtime_us; zcat /proc/config.gz 2>/dev/null | grep RT_GROUP_SCHED`, then `systemctl start duck && chrt -p $(systemctl show -p MainPID --value duck)`. Pass: `SCHED_FIFO` priority 50. A failure shows up as status 214/SETSCHEDULER; then delete the two `CPUScheduling*` lines from the unit.
- [ ] **CPU governor.** `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors`. Pass: the list includes `performance`.
- [ ] **CMM after shrinking.** After `apply --cmm 1024` and a reboot, run the full duck (`walk.py --follow`) and sample `grep -A1 CMM_USE_INFO /proc/ax_proc/mem_cmm_info` a few times. Pass: `used` stays under half of `total`. The file shows current use only, so take the highest sample.

## Pins and buses

- [ ] **Boot pin functions.** `python3 scripts/find_pins.py`. Pass: A8/A9 report `I2C7_SCL`/`I2C7_SDA` after the runtime sets the pinmap, and B20/B21 can be set to GPIO.
- [ ] **Foot switch pull-ups.** `maix.gpio` does not turn on an internal pull-up for inputs, so fit a 10 kΩ resistor from B20 and B21 to 3V3. Before fitting them, run `python3 scripts/test_feet.py` with the switches open. If the readings flicker, the resistors are required. Pass: with the resistors, open = no contact and pressed = contact, steady.
- [ ] **Servo adapter device node.** With USB mode set to host, `ls /dev/ttyACM* /dev/ttyUSB*`. Pass: one of them appears (CH343 → `ttyACM0`, CH340 → `ttyUSB0`), and `scripts/check_motors.py` finds all 14 servos. Set `servo_port` if it is `ttyUSB0`.
- [ ] **UART4 at 1 Mbps (only if you move the servos off USB).** Sipeed has only tested 115200. Pass: `check_motors.py` finds all 14 servos and a few minutes of `walk.py` logs no read errors. If every read fails, check whether the driver board echoes TX onto RX.
- [ ] **BNO055 on I2C7.** The BNO055 uses I2C clock stretching. `python3 scripts/test_imu.py` for a few minutes. Pass: chip found on `i2c-7` at 0x28, upright reads about `[0, 0, +9.8]`, and the error counter stays at 0. If you see timeouts or `EREMOTEIO`, the I2C7 bus speed (set in the device tree) needs to come down to 100 kHz.

## NPU and vision

- [ ] **Policy on the NPU.** `python3 scripts/bench_policy.py --onnx ~/BEST_WALK_ONNX_2.onnx --mud ~/out/walk_policy.mud --obs ~/out/obs_sim.npy`. Pass: it prints `OK for walking`. This also confirms that `nn.NN.forward` takes the raw float32 `[1,101]` tensor and returns float32. The CPU reference is 1.30 ms per step, already measured on the board.
- [ ] **YOLO11n model.** `ls /root/models/yolo11n.mud` (MaixPy 4.11.0 or later on MaixCAM2), then `python3 scripts/bench_npu.py --yolo /root/models/yolo11n.mud`. For reference, Sipeed quotes up to 113 FPS for the 640x640 model alone; the stock model is 640x480 and the timing includes pre and post-processing, so expect less.
- [ ] **Camera FOV and head signs.** `python3 scripts/test_vision.py --show`. Stand at a known angle and compare. The stock lens is 86.7° wide at 16:9, so the 4:3 frame should measure about 65–71°. Set `hfov_deg` to what you measure, and flip `head_yaw_sign`/`head_pitch_sign` if the head turns away.
- [ ] **ViT benchmark builds (PC, not board).** `tools/bench_vit/build_vits.sh` on Pulsar2. Pass: the build log shows U16 for the LayerNormalization and Softmax layers in the mixed config. Multi-frame (5-D input, 3-D patch conv) is not covered by the AX620E operator list. If it fails, fold time into channels (4-D `[1,3T,H,W]` input).

## Peripherals

- [ ] **BLE gamepad.** Pair the controller and run `walk.py` with the gamepad. Pass: sticks move the duck.
