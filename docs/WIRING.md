# Wiring the MaixCAM2 into Open Duck Mini v2

This replaces the [Pi Zero pin table](https://github.com/apirrone/Open_Duck_Mini/blob/v2/docs/assembly_guide.md#electronics) in the v2 assembly guide. Everything else in the v2 build (battery pack, BMS, UBEC, servo driver board, foot switches) stays as it is, except the IMU: this build uses a **BNO086** breakout in place of the BNO055 (see [BOM.md](BOM.md)).

Pin names come from Sipeed's [MaixCAM2 pinout](https://wiki.sipeed.com/hardware/assets/maixcam/maixcam2_pins.jpg) and the MaixPy [pinmap](https://wiki.sipeed.com/maixpy/doc/en/peripheral/pinmap.html), [UART](https://wiki.sipeed.com/maixpy/doc/en/peripheral/uart.html), [I2C](https://wiki.sipeed.com/maixpy/doc/en/peripheral/i2c.html) and [PWM](https://wiki.sipeed.com/maixpy/doc/en/peripheral/pwm.html) pages. Every pin name, function and `/dev` path here was checked against the MaixCDK pinmap source (`components/peripheral/port/maixcam2/maix_pinmap.cpp`) and the MaixPy peripheral docs. The pin functions at boot are read from hardware, so run `python3 scripts/find_pins.py` once before you solder.

![MaixCAM2 to Open Duck Mini v2 wiring](images/maixcam2-duck-wiring.png)

## Rules

- **All MaixCAM2 IO is 3.3 V.** Never put 5 V or 7.4 V on a signal pin.
- **Power goes into the 5V header pin**, which accepts 3.7 to 5 V and is the same rail as the Type-C port. Feed it from the duck's UBEC.
- **One common ground** between the battery, UBEC, servo driver board and MaixCAM2.
- **Do not use `A28`/`A29` (UART0).** That is the Linux console and boot log.
- **`A2` cannot be used.** Sipeed blocks that hole on purpose so the header can't be plugged in the wrong way.

## The three connectors

The MaixCAM2 has two 2x6 2.54 mm headers and one 6-pin 1.25 mm connector. Pins are listed in the order of Sipeed's pinout diagram:

```
   LEFT 2x6 header (body sensors)               RIGHT 2x6 header (power + head)
   ---------------------------------            ---------------------------------
   3V3  -> BNO086 3V3/VIN                       GND  <- UBEC GND
   GND  -> BNO086 GND, foot switch common       5V   <- UBEC 5V out
   B0   (UART2_TX)  spare                       A31  PWM7 -> left antenna signal
   B1   (UART2_RX)  spare                       B2   PWM4 -> right antenna signal
   A8   I2C7_SCL -> BNO086 SCL                  A28  UART0_TX  (console, do not use)
   A9   I2C7_SDA -> BNO086 SDA                  A4   GPIO   -> projector LED
   B21  GPIO in  <- left foot switch            A2   (blocked)
   B18  (SPI2_MOSI) spare                       A29  UART0_RX  (console, do not use)
   B19  (SPI2_MISO) spare                       B3   PWM5   -> left eye LED
   B20  GPIO in  <- right foot switch           A30  PWM6   -> right eye LED
   GND                                          5V   (second 5V pin, same rail)
   3V3                                          GND

   6-pin 1.25 mm connector (optional servo UART)
   ---------------------------------
   GND, 5V, A1 (I2C6_SCL), A0 (I2C6_SDA), A21 UART4_TX, A22 UART4_RX
```

The header silkscreen shows the same pins in two rows, so match by name, not by the row order above.

## Wiring table

| Part | Part pin | MaixCAM2 pin | Function (`maix.pinmap`) | Notes |
|---|---|---|---|---|
| **Power (UBEC 5 V)** | 5V out | 5V (right header) | power in | UBEC rated 3 A or more. Also feeds the antenna servos. |
| | GND | GND (right header) | | |
| **Servo driver board** (Waveshare bus servo adapter) | USB | Type-C | USB host, `/dev/ttyACM0` | Default. Set the MaixCAM2 USB mode to **host** (Settings → USB Mode, or `maix_usb_mode=host`). A CH343-based board shows up as `/dev/ttyACM0`, a CH340-based one as `/dev/ttyUSB0`. The board keeps its own 7.4 V DC input from the battery. |
| | *or* UART RX | A21 (6-pin) | `UART4_TX`, `/dev/ttyS4` | Optional, see below. |
| | *or* UART TX | A22 (6-pin) | `UART4_RX` | |
| | *or* GND | GND (6-pin) | | |
| **BNO086** (BNO08x breakout, e.g. SparkFun VR IMU SEN-22857; an Adafruit BNO085 works too) | 3V3 / VIN | 3V3 (left header) | | **3.3 V only**: the chip's limit is 3.6 V and some BNO086 boards have no regulator. The breakout pull-ups then sit at 3.3 V, which is what the MaixCAM2 needs. A Qwiic/STEMMA QT to female Dupont cable fits the header. |
| | GND | GND (left header) | | |
| | SCL | A8 | `I2C7_SCL` | `/dev/i2c-7`, address **0x4B** (SparkFun default) or **0x4A** (ADR/DI jumper, Adafruit default). The runtime scans both. |
| | SDA | A9 | `I2C7_SDA` | |
| | INT, RST | not connected | | The driver polls, so INT is not needed. Breakouts pull RST high; if yours doesn't, tie RST to 3V3. |
| | PS0, PS1 | not connected | | Must read low for I2C; breakouts pull them down (SparkFun: leave the PS jumpers open). |
| **Left foot switch** | NO contact | B21 | `GPIOB21` | 10 kΩ pull-up from B21 to 3V3 (`maix.gpio` does not enable an internal pull-up on inputs). Closed = foot on the ground. |
| | COM | GND (left header) | | |
| **Right foot switch** | NO contact | B20 | `GPIOB20` | 10 kΩ pull-up from B20 to 3V3. |
| | COM | GND (left header) | | |
| **Left antenna servo** | signal | A31 | `PWM7`, 50 Hz | 3.3 V signal is fine for hobby servos. Power from the UBEC 5 V, not from a MaixCAM2 pin. |
| **Right antenna servo** | signal | B2 | `PWM4`, 50 Hz | |
| | 5V, GND | UBEC 5 V, GND | | |
| **Left eye LED** | anode (+) through 330 ohm | B3 | `PWM5` or `GPIOB3` | PWM gives dimming. |
| **Right eye LED** | anode (+) through 330 ohm | A30 | `PWM6` or `GPIOA30` | `PWM6` is shared with the onboard light (`B25`), so use one or the other as PWM. |
| **Projector LED** | anode (+) | A4 | `GPIOA4` | Brighter LED: switch it with an NPN or N-MOSFET from 5 V instead of driving it from the pin. |
| | common cathode (-) | GND | | |
| **MAX98357A + speaker** | | not needed | | The MaixCAM2 has an onboard amplifier and 1 W speaker (`maix.audio`). |
| **Microphone** | | not needed | | Two onboard mics. |
| **Camera, flashlight** | | not needed | | Onboard camera, and the onboard light LED on `B25`. |
| **Gamepad** | | Bluetooth | | The MaixCAM2 radio is WiFi 6 + BLE 5.4, so use a controller that pairs over BLE (recent Xbox Wireless Controller firmware does). Not tested yet. |

## Servo bus: USB or UART

**Start with USB.** It is what the v2 duck already uses, it needs no soldering, and the servo driver board handles the half-duplex bus. The Type-C port becomes the servo link, which is why the MaixCAM2 has to be powered from the 5V header pin. Use WiFi and SSH to reach the board.

**Move to UART4 later** if you want the Type-C port back, or if USB turns out flaky. The MaixCAM2 UART clock is 200 MHz and its divider has 1/16 steps, so 1 Mbps (200 MHz / (16 x 1 Mbps) = 12.5) is set exactly. Sipeed has only tested 115200 baud, so check the bus with `check_motors.py` after switching. Before you wire it, check that the driver board's UART header is 3.3 V logic and set its mode jumper to UART. Cross TX and RX, then set:

```json
"servo_port": "/dev/ttyS4",
"pinmap": {"A8": "I2C7_SCL", "A9": "I2C7_SDA", "A21": "UART4_TX", "A22": "UART4_RX"}
```

`pinmap` in `duck_config.json` replaces the default one, so keep the I2C entries in it.

## Power

```
2x 18650 -> BMS -> power switch -> 7.4 V -+-> servo driver board (DC in)
                                          |
                                          +-> UBEC -> 5 V -+-> MaixCAM2 5V pin
                                                           +-> antenna servos
```

This is the v2 layout with the Pi swapped out. The MaixCAM2 draws more than a Pi Zero 2W when the NPU and camera run, so use a UBEC rated 3 A or more. The battery version of the MaixCAM2 has its own charger. Don't fit a cell to it inside the duck, because the duck already has a pack.

## Check it

```bash
python3 scripts/find_pins.py      # every pin above lists the function in the table
python3 scripts/test_imu.py       # BNO08x on i2c-7 @0x4b (or 0x4a), firmware x.y.z
python3 scripts/test_feet.py      # press each foot, the right side changes
python3 scripts/check_motors.py   # all 14 servos answer
```
