"""Camera simulator: lens/timing maths always; the full MuJoCo loop when the training stack is present.

The integration test needs mujoco, onnxruntime and an Open_Duck_Playground checkout:
    ODP_DIR=/path/to/Open_Duck_Playground STOCK_ONNX=/path/to/BEST_WALK_ONNX_2.onnx pytest tests/test_camsim.py
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from odm_train.camsim.lens import FrameClock, Lens, hblur  # noqa: E402


def test_frame_clock_rate_and_latency():
    c = FrameClock(hz=15, latency_ms=66, ctrl_dt=0.02)
    assert c.period == 3 and c.latency == 3
    captured, delivered = [], []
    for k in range(30):
        if c.tick():
            captured.append(k)
            c.push(k)
        for item in c.pop_ready():
            delivered.append((item, k))
    assert captured == list(range(0, 30, 3))
    assert all(k - item == 3 for item, k in delivered)


def test_frame_clock_zero_latency_delivers_same_step():
    c = FrameClock(hz=50, latency_ms=0)
    assert c.tick()
    c.push("a")
    assert c.pop_ready() == ["a"]


def test_pinhole_lens_matches_follow_bearing():
    from odm_maix.vision.follow import pixel_to_angle

    lens = Lens(320, 240, 70.0)
    for ang in (-0.5, -0.2, 0.0, 0.3, 0.55):
        u, _ = lens.to_image(np.tan(ang), 0.0)
        assert abs(pixel_to_angle(float(u), 70.0) - ang) < 1e-6


def test_barrel_lens_round_trip_and_margin():
    lens = Lens(320, 240, 70.0, k1=-0.08)
    assert lens.margin > 1 and lens.rw > 320 and lens.rh > 240
    r = np.linspace(0, 0.9, 50)
    assert np.allclose(lens.distort_r(lens.undistort_r(r)), r, atol=1e-6)
    # the frame corner samples inside the rendered image
    assert 0 <= lens.map_x.min() and lens.map_x.max() <= lens.rw - 1
    assert 0 <= lens.map_y.min() and lens.map_y.max() <= lens.rh - 1


def test_points_far_outside_the_fov_stay_outside():
    lens = Lens(320, 240, 70.0, k1=-0.08)
    # 85 degrees off-axis: the distortion polynomial would fold this back into the frame
    u, v = lens.to_image(np.tan(math.radians(85)), 0.0)
    assert u > 1


def test_hblur_keeps_shape_and_mean():
    img = np.random.default_rng(0).uniform(0, 255, (24, 32, 3))
    out = hblur(img, 5)
    assert out.shape == img.shape
    assert abs(out.mean() - img.mean()) < 2
    assert hblur(img, 1) is img


ODP = os.environ.get("ODP_DIR")
STOCK = os.environ.get("STOCK_ONNX")


@pytest.mark.skipif(not (ODP and STOCK), reason="needs ODP_DIR and STOCK_ONNX (training stack)")
def test_follow_sim_runs_and_sees_the_person(monkeypatch):
    from odm_train.camsim import gl

    gl.setup()  # before mujoco is imported
    pytest.importorskip("mujoco")
    monkeypatch.chdir(ODP)
    monkeypatch.syspath_prepend(ODP)
    from odm_train.camsim.follow_sim import FollowSim

    sim = FollowSim(STOCK, seed=1)
    frames = boxes = 0
    for _ in range(100):  # 2 s
        info = sim.step()
        if info["delivered"] is not None:
            frames += 1
            assert info["delivered"]["frame"].shape == (240, 320, 3)
            boxes += info["delivered"]["box"] is not None
    assert frames >= 30  # 15 Hz for 2 s, minus the start-up latency
    assert boxes > 0  # the person starts in front of the duck
    assert not sim.fallen
    sim.close()
