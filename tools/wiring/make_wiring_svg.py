#!/usr/bin/env python3
"""Draw docs/images/duck-wiring.svg, the whole-duck wiring diagram.

Plain Python, no dependencies. The SVG is the file GitHub shows, and it is
text, so a change to a wire shows up as a readable diff. Edit the parts and
wires below, then run:

    python3 tools/wiring/make_wiring_svg.py

The pin names here must match the wiring table in docs/WIRING.md.
"""
from __future__ import annotations

import pathlib
from xml.sax.saxutils import escape

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "images" / "duck-wiring.svg"
W, H = 1800, 1345

# wire colours
BAT = "#d62728"     # pack +, unswitched
V74 = "#ff7f0e"     # 7.4 V switched
V5 = "#e377c2"      # 5 V rail
V33 = "#9467bd"     # 3.3 V from the MaixCAM2
GND = "#222222"
I2C = "#1f77b4"
SIG = "#2ca02c"     # GPIO / PWM
USB = "#17becf"
FPC = "#8c564b"
OPT = "#7f7f7f"

out: list[str] = []


def text(x, y, s, size=14, anchor="start", weight="normal", fill="#222", italic=False):
    style = ' font-style="italic"' if italic else ""
    out.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}"{style}>{escape(s)}</text>'
    )


def box(x, y, w, h, title, lines=(), fill="#f4f4f4", stroke="#555", dash=False, title_size=16):
    d = ' stroke-dasharray="6 4"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"{d}/>')
    text(x + w / 2, y + 24, title, size=title_size, anchor="middle", weight="bold")
    for i, line in enumerate(lines):
        text(x + w / 2, y + 46 + i * 18, line, size=13, anchor="middle", fill="#444")


def wire(points, color, width=3, dash=False):
    d = "M" + " L".join(f"{x} {y}" for x, y in points)
    da = ' stroke-dasharray="8 6"' if dash else ""
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
               f'stroke-linecap="round" stroke-linejoin="round"{da}/>')


def dot(x, y, color=GND, r=6):
    out.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')


def pin(x, y, name, side, color="#fff"):
    """A pin label on a box edge. side: 'l' (label inside, to the right of x) or 'r'."""
    w = 8 * len(name) + 14
    px = x if side == "l" else x - w
    out.append(f'<rect x="{px}" y="{y - 11}" width="{w}" height="22" rx="4" fill="{color}" stroke="#777"/>')
    text(px + w / 2, y + 5, name, size=13, anchor="middle", weight="bold")


def label(x, y, s, anchor="start", fill="#444", size=12):
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
               f'stroke="#fff" stroke-width="4" paint-order="stroke">{escape(s)}</text>')


# --------------------------------------------------------------------------
# header
text(W / 2, 40, "Open Duck Mini v2 + MaixCAM2: full wiring", size=28, anchor="middle", weight="bold")
text(W / 2, 66, "Every part on the BOM. Pin table: docs/WIRING.md.  Power details: docs/POWER.md.",
     size=15, anchor="middle", fill="#555")

# ===================== POWER (top left) ==================================
# battery pack
box(30, 100, 260, 250, "Battery pack", ["2x 18650, 2S holder", "(Molicel P30B, 20 A+ cells)", "6.0 to 8.4 V"], fill="#fdecea")
out.append('<rect x="70" y="190" width="90" height="40" rx="5" fill="#fff" stroke="#999"/>')
text(115, 215, "Cell 2", size=13, anchor="middle")
out.append('<rect x="70" y="270" width="90" height="40" rx="5" fill="#fff" stroke="#999"/>')
text(115, 295, "Cell 1", size=13, anchor="middle")
pin(290, 180, "B+", "r", "#fbd5d2")
pin(290, 250, "BM", "r")
pin(290, 320, "B-", "r", "#ddd")
label(160, 340, "optional XT30 here", anchor="middle", size=11)

# BMS
box(400, 100, 200, 250, "HX-2S-D20 BMS", ["2S protection", "~20 A trip, no balancing"], fill="#fdecea")
pin(400, 180, "B+", "l", "#fbd5d2")
pin(400, 250, "BM", "l")
pin(400, 320, "B-", "l", "#ddd")
pin(600, 180, "P+", "r", "#fbd5d2")
pin(600, 320, "P-", "r", "#ddd")
wire([(290, 180), (400, 180)], BAT, 6)
wire([(290, 250), (400, 250)], I2C, 2)
wire([(290, 320), (400, 320)], GND, 6)
label(345, 172, "18 AWG", anchor="middle")
label(345, 242, "thin", anchor="middle")

# charger
box(400, 400, 200, 160, "USB-C 2S charger", ["5 V in, 8.4 V CC/CV out", "CR = charging, OK = full"], fill="#fdecea")
pin(600, 495, "BAT", "r", "#fbd5d2")
pin(600, 530, "GND", "r", "#ddd")
pin(400, 512, "USB-C", "l")
wire([(330, 512), (400, 512)], USB, 3, dash=True)
text(320, 502, "phone charger", size=12, anchor="end", fill="#444")
text(320, 518, "5 V, no PD needed", size=12, anchor="end", fill="#444")

# power switch
box(700, 150, 110, 60, "Switch", [], fill="#fff7e6", title_size=15)
text(755, 200, "10 A DC+", size=12, anchor="middle", fill="#444")

# P+ to switch and charger BAT
wire([(600, 180), (700, 180)], BAT, 6)
wire([(650, 180), (650, 495), (600, 495)], BAT, 4)
dot(650, 180, BAT, 7)
# P- and charger GND to ground bus
wire([(600, 320), (680, 320), (680, 580)], GND, 6)
wire([(600, 530), (680, 530)], GND, 4)
dot(680, 530)

# 7.4 V bus
wire([(810, 180), (860, 180), (860, 490)], V74, 6)
dot(860, 180, V74, 7)
label(835, 172, "7.4 V", anchor="middle", fill=V74)

# servo driver board
box(940, 100, 250, 220, "Servo driver board", ["Waveshare bus servo adapter", "(CH343 or CH340 USB-UART)", "half-duplex Feetech bus"], fill="#fff7e6")
pin(940, 180, "DC+", "l", "#ffe0b3")
pin(940, 220, "DC-", "l", "#ddd")
pin(1190, 250, "BUS", "r", "#ffe0b3")
pin(1190, 290, "USB-C", "r")
pin(940, 290, "UART", "l", "#eee")
wire([(860, 180), (940, 180)], V74, 6)
label(900, 172, "barrel jack", anchor="middle", size=11)
wire([(940, 220), (900, 220), (900, 580)], GND, 6)

# buck
box(940, 400, 250, 160, "V752-755 5 V buck", ["5.0 V, 3 A (check on the bench)", "IN- and OUT- are common"], fill="#fde8f4")
pin(940, 490, "IN+", "l", "#ffe0b3")
pin(940, 525, "IN-", "l", "#ddd")
wire([(860, 490), (940, 490)], V74, 4)
wire([(940, 525), (900, 525)], GND, 4)
dot(900, 525)
# buck outputs on the bottom edge
pin(1000, 560, "OUT+", "l", "#f9c9e6")
pin(1110, 560, "OUT-", "l", "#ddd")

# ground bus
wire([(680, 580), (900, 580)], GND, 6)
dot(900, 580, GND, 7)
label(790, 600, "common GND", anchor="middle")

# ===================== SERVOS (top right) ================================
SX = 1290
chains = [
    ("Left leg", "20 hip yaw > 21 hip roll > 22 hip pitch > 23 knee > 24 ankle", 110),
    ("Right leg", "10 hip yaw > 11 hip roll > 12 hip pitch > 13 knee > 14 ankle", 210),
    ("Neck + head", "30 neck pitch > 31 head pitch > 32 head yaw > 33 head roll", 310),
]
for title, ids, y in chains:
    box(SX, y, 480, 80, f"{title}: STS3215 7.4 V", [ids], fill="#fff7e6", title_size=15)
    wire([(1240, y + 40), (SX, y + 40)], V74, 5)
wire([(1190, 250), (1240, 250)], V74, 5)
wire([(1240, 150), (1240, 350)], V74, 5)
dot(1240, 150, V74, 6)
dot(1240, 250, V74, 7)
label(1255, 412, "3-wire bus cable (7.4 V, GND, data), daisy-chained. 14 servos,", size=12)
label(1255, 428, "IDs as in odm_maix/hwi.py. Never feed the servos through the buck.", size=12)

# ===================== MAIXCAM2 (centre) =================================
MX, MY, MW, MH = 960, 680, 440, 420
box(MX, MY, MW, MH, "", [], fill="#e8f3ea", stroke="#2c7a3f")
text(MX + MW / 2 + 20, MY + 85, "Sipeed MaixCAM2", size=22, anchor="middle", weight="bold", fill="#1d4d29")
text(MX + MW / 2 + 20, MY + 106, "pins: right header top, left header left", size=12, anchor="middle", fill="#2c4a33")
onboard = [
    "Onboard, no wiring:",
    "2.4\" 640x480 touch screen (the face,",
    "  screen_face head)",
    "1 W speaker + amp, 2 mics",
    "light LED (B25), WiFi 6 + BLE 5.4",
    "AX630C NPU, TF card",
]
for i, s in enumerate(onboard):
    text(MX + 120, MY + 210 + i * 20, s, size=13, fill="#2c4a33", weight="bold" if i == 0 else "normal")

# top edge: power from the buck (right header 5V/GND), USB to servo board
pin(1000, MY + 22 + 10, "5V", "l", "#f9c9e6")
pin(1110, MY + 22 + 10, "GND", "l", "#ddd")
wire([(1022, 571), (1022, MY)], V5, 5)
wire([(1135, 571), (1135, MY)], GND, 5)
dot(1022, 600, V5)
dot(1135, 625)
# 470 uF cap across 5V/GND
out.append(f'<rect x="1050" y="640" width="58" height="22" rx="4" fill="#fff" stroke="#777"/>')
text(1079, 656, "470 µF", size=12, anchor="middle")
wire([(1022, 651), (1050, 651)], V5, 2)
wire([(1108, 651), (1135, 651)], GND, 2)
label(1080, 592, "20 to 22 AWG", anchor="middle", size=11)

pin(1290, MY + 32, "Type-C", "l")
wire([(1190, 290), (1320, 290), (1320, MY)], USB, 3, dash=True)
label(1330, 470, "USB, data only", size=11)
label(1330, 484, "MaixCAM2 = host", size=11)
label(1330, 498, "/dev/ttyACM0", size=11)

# 5 V and GND branches to the antenna servos
wire([(1022, 600), (1560, 600), (1560, 690)], V5, 3)
wire([(1135, 625), (1640, 625), (1640, 690)], GND, 3)

# left edge: left header (IMU, feet)
LP = [("3V3", 740, "#e3d7f3"), ("GND", 780, "#ddd"), ("A8 I2C7_SCL", 830, "#d4e6f6"),
      ("A9 I2C7_SDA", 870, "#d4e6f6"), ("B21 GPIO", 975, "#d6f0d6"), ("B20 GPIO", 1055, "#d6f0d6"), ("GND", 1085, "#ddd")]
for name, y, c in LP:
    pin(MX, y, name, "l", c)

# BNO086
box(420, 690, 260, 210, "BNO086 IMU", ["SparkFun SEN-22857 or", "Adafruit BNO085", "3.3 V only, /dev/i2c-7", "0x4B or 0x4A", "INT, RST, PS0/1 open"], fill="#e8f0fa")
for name, y, c in [("VIN", 740, "#e3d7f3"), ("GND", 780, "#ddd"), ("SCL", 830, "#d4e6f6"), ("SDA", 870, "#d4e6f6")]:
    pin(680, y, name, "r", c)
wire([(680, 740), (MX, 740)], V33, 3)
wire([(680, 780), (MX, 780)], GND, 3)
wire([(680, 830), (MX, 830)], I2C, 3)
wire([(680, 870), (MX, 870)], I2C, 3)
label(690, 818, "Qwiic to Dupont, short", size=11)

# feet: switch closes the GPIO to GND; a 10 k resistor pulls it up to 3V3
box(420, 950, 260, 50, "Left foot switch", [], fill="#eef7ee", title_size=15)
box(420, 1030, 260, 50, "Right foot switch", [], fill="#eef7ee", title_size=15)
pin(680, 975, "NO", "r", "#d6f0d6")
pin(680, 1055, "NO", "r", "#d6f0d6")
pin(420, 975, "COM", "l", "#ddd")
pin(420, 1055, "COM", "l", "#ddd")
wire([(680, 975), (MX, 975)], SIG, 3)
wire([(680, 1055), (MX, 1055)], SIG, 3)


def v33_flag(x, y):
    """A 3V3 tag: the wire ends at the MaixCAM2 3V3 pin, not drawn across the sheet."""
    out.append(f'<rect x="{x - 20}" y="{y - 10}" width="40" height="20" rx="4" fill="#e3d7f3" stroke="{V33}" stroke-width="2"/>')
    text(x, y + 5, "3V3", size=12, anchor="middle", weight="bold", fill="#4b2d73")


def resistor(x, y_top):
    out.append(f'<rect x="{x - 7}" y="{y_top}" width="14" height="24" fill="#fff" stroke="{V33}" stroke-width="2"/>')


# left foot: pull-up above its line
wire([(860, 975), (860, 955)], V33, 2)
resistor(860, 931)
wire([(860, 931), (860, 916)], V33, 2)
v33_flag(860, 906)
dot(860, 975, SIG, 5)
label(874, 948, "10 kΩ", size=11)
# right foot: pull-up below its line
wire([(860, 1055), (860, 1075)], V33, 2)
resistor(860, 1075)
wire([(860, 1099), (860, 1112)], V33, 2)
v33_flag(860, 1122)
dot(860, 1055, SIG, 5)
label(874, 1092, "10 kΩ", size=11)
label(725, 1126, "to the MaixCAM2 3V3 pin", anchor="middle", size=11)

# both COMs to the second GND pin on the left header
wire([(420, 975), (395, 975), (395, 1055), (420, 1055)], GND, 3)
wire([(395, 1055), (395, 1150), (930, 1150), (930, 1085), (MX, 1085)], GND, 3)
dot(395, 1055, GND, 5)
label(650, 1168, "foot COM to MaixCAM2 GND (left header)", anchor="middle", size=11)

# right edge: right header (antennas, LEDs) and camera connector
RP = [("A31 PWM7", 730, "#d6f0d6"), ("B2 PWM4", 770, "#d6f0d6"), ("B3 PWM5", 840, "#d6f0d6"),
      ("A30 PWM6", 880, "#d6f0d6"), ("A4 GPIO", 940, "#d6f0d6"), ("FPC5 camera", 1040, "#ead9cf")]
for name, y, c in RP:
    pin(MX + MW, y, name, "r", c)

# antenna servos
box(1500, 690, 270, 100, "Antenna servos (2x 9 g)", ["power from the 5 V rail", "50 Hz PWM, 3.3 V signal is fine"], fill="#eef7ee", title_size=15)
pin(1500, 730, "L sig", "l", "#d6f0d6")
pin(1500, 770, "R sig", "l", "#d6f0d6")
wire([(MX + MW, 730), (1500, 730)], SIG, 3)
wire([(MX + MW, 770), (1500, 770)], SIG, 3)

# eye LEDs, projector (case_closed only)
box(1500, 815, 270, 85, "Eye LEDs (case_closed head)", ["L on B3, R on A30, 330 Ω to anode", "cathode to GND; none with screen_face"], fill="#f7f7f7", dash=True, title_size=14)
wire([(MX + MW, 840), (1500, 840)], SIG, 3, dash=True)
wire([(MX + MW, 880), (1500, 880)], SIG, 3, dash=True)
box(1500, 915, 270, 65, "Projector LED (case_closed)", ["A4 switches an NPN / N-MOSFET"], fill="#f7f7f7", dash=True, title_size=14)
wire([(MX + MW, 940), (1500, 940)], SIG, 3, dash=True)

# camera module
box(1500, 1000, 270, 100, "Camera module", ["in the head-top pod (screen_face)", "22-pin FPC, ~100 mm, 0.5 mm pitch", "(stock short cable if case_closed)"], fill="#f6eee9", title_size=15)
wire([(MX + MW, 1040), (1500, 1040)], FPC, 7)
label(1450, 1032, "flat cable", anchor="middle", size=11)

# bottom: 6-pin connector (optional UART4)
pin(1060, MY + MH, "6-pin 1.25 mm", "l", "#eee")
box(960, 1150, 440, 70, "Optional: servo bus on UART4 instead of USB", ["A21 UART4_TX > servo board RX, A22 UART4_RX < TX, GND", "/dev/ttyS4, check the board's UART is 3.3 V"], fill="#f7f7f7", stroke=OPT, dash=True, title_size=14)
wire([(1110, MY + MH + 11), (1110, 1150)], OPT, 3, dash=True)

# gamepad
box(1500, 1140, 270, 60, "Gamepad", ["BLE, no wire"], fill="#f7f7f7", dash=True, title_size=15)
wire([(1400, 1070), (1450, 1070), (1450, 1170), (1500, 1170)], OPT, 2, dash=True)

# ===================== legend ============================================
LX, LY = 30, 1170
legend = [(BAT, "pack + (unswitched)", False), (V74, "7.4 V switched / servo bus", False),
          (V5, "5 V rail", False), (V33, "3.3 V from MaixCAM2", False), (GND, "GND", False),
          (I2C, "I2C / cell balance tap", False), (SIG, "GPIO / PWM", False), (USB, "USB", True),
          (FPC, "camera flat cable", False), (OPT, "optional / wireless", True)]
for i, (c, s, d) in enumerate(legend):
    x = LX + (i % 2) * 185
    y = LY + (i // 2) * 22 + 20
    wire([(x, y), (x + 34, y)], c, 4, dash=d)
    text(x + 42, y + 5, s, size=12, fill="#333")
text(30, 1310, "All MaixCAM2 IO is 3.3 V. One common ground. Do not use A28/A29 (console); A2 is blocked. "
     "Dashed boxes are optional or depend on the head layout.", size=13, fill="#333")
text(30, 1332, "Generated by tools/wiring/make_wiring_svg.py. Edit that script, not this file.", size=12, fill="#777")

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    f'font-family="DejaVu Sans, Arial, sans-serif">\n'
    f'<rect width="{W}" height="{H}" fill="#ffffff"/>\n' + "\n".join(out) + "\n</svg>\n"
)
OUT.write_text(svg, encoding="utf-8")
print(f"wrote {OUT}")
