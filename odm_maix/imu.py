"""Threaded IMU reader: same output as the Pi runtime's ``raw_imu.Imu``.

``get_data()`` returns ``{"gyro": [3] rad/s, "accelero": [3] m/s^2}`` and never
blocks, so the 50 Hz control loop is not held up by I2C. The IMU is a BNO08x
(BNO085/BNO086) read through ``odm_maix.bno08x``.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import bno08x

# Report rate asked of the chip. Faster than the 50 Hz loop so the newest
# sample is at most ~10 ms old when the policy reads it.
REPORT_HZ = 100


class Imu:
    def __init__(self, config, sampling_freq: int = 50, calibrate: bool = False, dev=None):
        """``dev``: an open bno08x.BNO08x (skips the bus scan)."""
        m = config.maixcam2
        if dev is None:
            dev = bno08x.open_from_config(config)
            print(f"[IMU] BNO08x on i2c-{dev.bus_number} @0x{dev.address:02x}, firmware {dev.product['sw_version']}")
        else:
            dev.set_axis_remap(bno08x.remap_from_config(config))
        self.dev = dev
        self.sampling_freq = sampling_freq
        self.offsets_path = m["imu_offsets_path"]

        if calibrate:
            self._calibrate()
            return

        if bno08x.load_calibration_record(self.offsets_path):
            print("[IMU] using the calibration saved on the chip")
        else:
            print(f"[IMU] no calibration record at {self.offsets_path}; run scripts/test_imu.py --calibrate")
        self.dev.start(max(REPORT_HZ, 2 * sampling_freq))

        self._lock = threading.Lock()
        self._last = {"gyro": np.zeros(3), "accelero": np.zeros(3)}
        self.errors = 0
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _calibrate(self) -> None:
        self.dev.begin_calibration()
        print("Hold the duck still for a few seconds, then turn it slowly through")
        print("six faces (like a die), holding each for a second, until accel and gyro reach 3")
        while True:
            s = self.dev.calibration_status()
            print(s)
            if s["accel"] == 3 and s["gyro"] == 3:
                break
            time.sleep(0.2)
        self.dev.save_calibration()
        bno08x.save_calibration_record(self.offsets_path, s, self.dev)
        print(f"Saved calibration to the chip's flash (record in {self.offsets_path})")

    def _worker(self) -> None:
        period = 1.0 / self.sampling_freq
        while self._running:
            t = time.monotonic()
            try:
                gyro, acc = self.dev.gyro_and_acceleration()
                with self._lock:
                    self._last = {"gyro": np.array(gyro), "accelero": np.array(acc)}
            except OSError as e:
                self.errors += 1
                if self.errors % 50 == 1:
                    print("[IMU]", e)
            time.sleep(max(0.0, period - (time.monotonic() - t)))

    def get_data(self):
        with self._lock:
            return self._last

    def stop(self) -> None:
        self._running = False
