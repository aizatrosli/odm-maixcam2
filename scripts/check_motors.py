"""Ping every servo, print position and voltage. Torque stays off."""

import argparse

from _common import DEFAULT_CONFIG_PATH, load_config

from odm_maix.config import JOINT_NAMES
from odm_maix.feetech import FeetechBus
from odm_maix.feetech import FeetechError
from odm_maix.hwi import JOINT_IDS

p = argparse.ArgumentParser()
p.add_argument("--duck_config_path", default=DEFAULT_CONFIG_PATH)
args = p.parse_args()
config = load_config(args.duck_config_path)
bus = FeetechBus(config.maixcam2["servo_port"], config.maixcam2["servo_baudrate"])

ok = True
for name in JOINT_NAMES:
    sid = JOINT_IDS[name]
    if not bus.ping(sid):
        print(f"{name:16s} id {sid:2d}  NOT FOUND")
        ok = False
        continue
    try:
        pos = bus.read_present_position([sid])[0]
        volt = bus.read_voltage(sid)
        print(f"{name:16s} id {sid:2d}  pos {pos:+.3f} rad  {volt:.1f} V")
    except FeetechError as e:
        print(f"{name:16s} id {sid:2d}  read error: {e}")
        ok = False
print("ALL OK" if ok else "SOME SERVOS FAILED")
