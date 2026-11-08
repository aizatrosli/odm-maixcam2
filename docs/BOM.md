# Bill of materials: what changes from the v2 BOM

Start from the [Open Duck Mini v2 BOM](https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc/edit?usp=sharing) (linked from the [Open_Duck_Mini README](https://github.com/apirrone/Open_Duck_Mini/tree/v2#bom)). Everything on it stays (servos, servo driver board, battery, BMS, UBEC, foot switches, printed parts, screws) except the lines below.

| v2 BOM | This build | Qty | Notes |
|---|---|---|---|
| Raspberry Pi Zero 2W (+ SD card) | **Sipeed MaixCAM2** | 1 | The brain, with camera, screen, speaker, mics, WiFi 6 + BLE and the NPU on board. Get the version without the battery: the duck has its own pack. MaixPy 4.11.0 or later. |
| Adafruit BNO055 breakout | **BNO086 breakout** (for example SparkFun VR IMU SEN-22857, Qwiic). An Adafruit BNO085 (#4754) or another BNO08x board also works. | 1 | I2C on `A8`/`A9` (`/dev/i2c-7`), address 0x4B or 0x4A, the runtime scans both. **3.3 V only.** See [WIRING.md](WIRING.md). |
| | Qwiic / STEMMA QT to female Dupont cable, 100 to 150 mm | 1 | Only for a board with a Qwiic connector; otherwise solder header pins. Keep the I2C wires short. |
| | 10 kΩ resistor | 2 | Foot switch pull-ups from B20 and B21 to 3V3 (`maix.gpio` does not turn on an internal pull-up). |
| | Female-female Dupont jumpers | ~10 | MaixCAM2 headers to the IMU, feet, antenna servos and LEDs. |
| MAX98357A amplifier + speaker | not needed | 0 | Onboard amplifier and speaker. |
| Pi camera, microphone | not needed | 0 | Onboard. |
| UBEC | UBEC rated **3 A or more** | 1 | The MaixCAM2 draws more than the Pi with the NPU and camera running. |

## Software dependencies for the BNO086

None to install. `odm_maix/bno08x.py` speaks the BNO08x's SHTP/SH-2 protocol over `/dev/i2c-N` with the Python standard library, so it needs neither Adafruit Blinka/CircuitPython (no MaixCAM2 board definition) nor `smbus2`. The dashboard (`apps/duck_dash`) has the same protocol in C (`main/src/core/hw.c`).

The calibration is kept in the BNO086's own flash (`scripts/calibrate.py imu`), not in a file. `~/imu_offsets.json` only records when it was saved.

## About the BNO086

The BNO086 is CEVA's (ex Hillcrest Labs) successor to the BNO080/BNO085, on the same Bosch sensor silicon as the BNO055 but with different firmware: it is not register compatible with the BNO055. The runtime reads its calibrated accelerometer (gravity included, m/s²) and calibrated gyroscope (rad/s) reports and applies the BNO055 axis remap in software, so the walking policy sees the same 101-value observation as before. If your breakout holds the chip at a different angle from the Adafruit BNO055 board, `calibrate.py -i imu` tells you, and `maixcam2.imu_axis_remap` in `duck_config.json` fixes it.
