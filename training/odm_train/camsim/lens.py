"""Lens and timing maths for the MaixCAM2 camera model. Pure numpy (unit-tested without MuJoCo)."""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class Lens:
    """Pinhole + radial distortion: r_d = r_u (1 + k1 r_u^2 + k2 r_u^4), radii in focal lengths.

    The pinhole part is set by the horizontal FOV of the delivered frame. With
    barrel distortion (k1 < 0) the frame shows more than the pinhole FOV, so the
    scene is rendered ``margin`` times larger and resampled (``map_x``/``map_y``).
    """

    def __init__(self, width: int, height: int, hfov_deg: float, k1: float = 0.0, k2: float = 0.0):
        self.W, self.H, self.k1, self.k2 = width, height, k1, k2
        self.f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)  # focal length, pixels
        corner = math.hypot(width / 2, height / 2) / self.f
        self.r_valid = float(self.undistort_r(corner))  # undistorted radius of the frame corner
        self.margin = max(1.0, self.r_valid / corner * 1.02)
        self.rw = int(math.ceil(width * self.margin / 2) * 2)
        self.rh = int(math.ceil(height * self.margin / 2) * 2)
        self.render_fovy_deg = math.degrees(2 * math.atan((self.rh / 2) / self.f))
        # remap grid: delivered (distorted) pixel -> rendered (pinhole) pixel
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        x, y = (u - width / 2) / self.f, (v - height / 2) / self.f
        r_d = np.hypot(x, y)
        scale = np.where(r_d > 0, self.undistort_r(r_d) / np.maximum(r_d, 1e-9), 1.0)
        self.map_x = x * scale * self.f + self.rw / 2 - 0.5
        self.map_y = y * scale * self.f + self.rh / 2 - 0.5

    def distort_r(self, r_u):
        return r_u * (1 + self.k1 * r_u**2 + self.k2 * r_u**4)

    def undistort_r(self, r_d):
        r = np.asarray(r_d, dtype=float).copy()
        for _ in range(30):  # fixed point, converges for mild distortion
            r = r_d / (1 + self.k1 * r**2 + self.k2 * r**4)
        return r

    def to_image(self, x, y):
        """Pinhole coordinates (x right, y down, in focal lengths) -> normalised image (u, v) in 0..1.

        The polynomial folds back beyond the lens's field of view; points out there are
        outside the frame anyway, so they are projected as a pinhole would (still outside).
        """
        r_u = np.hypot(x, y)
        s = np.where((r_u > 0) & (r_u <= self.r_valid), self.distort_r(r_u) / np.maximum(r_u, 1e-9), 1.0)
        return (x * s * self.f + self.W / 2) / self.W, (y * s * self.f + self.H / 2) / self.H

    def resample(self, img: np.ndarray) -> np.ndarray:
        """Rendered pinhole image (rh, rw, 3) -> delivered distorted image (H, W, 3), bilinear."""
        return bilinear(img, self.map_x, self.map_y)


class FrameClock:
    """Camera timing on the 50 Hz control clock: capture every ``period`` steps,
    deliver each result ``latency`` steps later."""

    def __init__(self, hz: float, latency_ms: float, ctrl_dt: float = 0.02):
        self.period = max(1, int(round(1 / (hz * ctrl_dt))))
        self.latency = max(0, int(round(latency_ms / 1000 / ctrl_dt)))
        self.reset()

    def reset(self, phase: int = 0):
        self.k = phase
        self.queue = deque()

    def tick(self) -> bool:
        """Advance one control step; True when a frame is captured on this step."""
        capture = self.k % self.period == 0
        self.k += 1
        return capture

    def push(self, item):
        """Queue what was captured on this step."""
        self.queue.append((self.k - 1 + self.latency, item))

    def pop_ready(self):
        """Items whose latency has elapsed on this step (oldest first)."""
        out = []
        while self.queue and self.queue[0][0] <= self.k - 1:
            out.append(self.queue.popleft()[1])
        return out


def hblur(img, n):
    """Box blur of n pixels along each row (n <= 1: unchanged)."""
    if n <= 1:
        return img
    c = np.cumsum(np.pad(img, ((0, 0), (n // 2 + 1, n - n // 2 - 1), (0, 0)), mode="edge"), axis=1)
    return (c[:, n:] - c[:, :-n]) / n


def bilinear(img, mx, my):
    h, w = img.shape[:2]
    x0 = np.clip(np.floor(mx).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(my).astype(int), 0, h - 2)
    fx = np.clip(mx - x0, 0, 1)[..., None]
    fy = np.clip(my - y0, 0, 1)[..., None]
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
