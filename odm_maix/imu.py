"""Threaded IMU reader: same output as the Pi runtime's ``raw_imu.Imu``.

``get_data()`` returns ``{"gyro": [3] rad/s, "accelero": [3] m/s^2}`` and never
blocks, so the 50 Hz control loop is not held up by I2C.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import bno055


class Imu:
    def __init__(self, config, sampling_freq: int = 50, calibrate: bool = False, dev=None):
        """``dev``: an open bno055.BNO055 (skips the bus scan)."""
        m = config.maixcam2
        bus, addr = m["imu_i2c_bus"], m["imu_address"]
        if dev is not None:
            self.dev = dev
        elif bus is None:
            found = bno055.scan_for_bno055()
            if found is None:
                raise IOError("No BNO055 found on /dev/i2c-0..7. Check wiring and the 'pinmap' config.")
            bus, addr = found
            print(f"[IMU] BNO055 found on i2c-{bus} @0x{addr:02x}")
        if dev is None:
            self.dev = bno055.BNO055(bus, addr)
        self.dev.set_axis_remap(bno055.REMAP_UPSIDE_DOWN if config.imu_upside_down else bno055.REMAP_UPRIGHT)
        self.sampling_freq = sampling_freq
        self.offsets_path = m["imu_offsets_path"]

        if calibrate:
            self._calibrate()
            return

        if self.dev.load_offsets(self.offsets_path):
            print(f"[IMU] loaded calibration from {self.offsets_path}")
        else:
            print("[IMU] no calibration file, running uncalibrated")
        # NDOF mode, as in the Pi runtime's raw_imu.py
        self.dev.mode = bno055.NDOF_MODE

        self._lock = threading.Lock()
        self._last = {"gyro": np.zeros(3), "accelero": np.zeros(3)}
        self.errors = 0
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _calibrate(self) -> None:
        self.dev.mode = bno055.NDOF_MODE
        print("Move the duck slowly through several orientations until all values reach 3")
        while True:
            s = self.dev.calibration_status()
            print(s)
            if all(v == 3 for v in s.values()):
                break
            time.sleep(0.2)
        self.dev.save_offsets(self.offsets_path)
        print(f"Saved calibration to {self.offsets_path}")

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
