"""Print IMU data, or calibrate with --calibrate (saved on the BNO086, recorded in maixcam2.imu_offsets_path)."""

import argparse
import time

import numpy as np
from _common import DEFAULT_CONFIG_PATH, load_config

from odm_maix.imu import Imu

p = argparse.ArgumentParser()
p.add_argument("--duck_config_path", default=DEFAULT_CONFIG_PATH)
p.add_argument("--calibrate", action="store_true")
args = p.parse_args()
config = load_config(args.duck_config_path)
imu = Imu(config, calibrate=args.calibrate)
if not args.calibrate:
    print("Upright and still, accelero should read about [0, 0, +9.8]")
    while True:
        d = imu.get_data()
        print("gyro", np.around(d["gyro"], 3), "accelero", np.around(d["accelero"], 3))
        time.sleep(0.1)
