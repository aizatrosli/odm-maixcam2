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


# --- step 2: encoders and projection


def test_resize_squashes_to_square_in_0_1():
    from odm_train.camsim.encoders import resize

    frame = np.zeros((240, 320, 3), np.uint8)
    frame[:, 160:] = 255  # right half white
    x = resize(frame, 32)
    assert x.shape == (1, 3, 32, 32) and x.dtype == np.float32
    assert x[0, :, :, :15].max() == 0 and x[0, :, :, 17:].min() == 1


def test_projection_fit_apply_save(tmp_path):
    from odm_train.camsim.encoders import Projection

    rng = np.random.default_rng(0)
    feats = rng.normal(size=(200, 3)) @ rng.normal(size=(3, 32))  # rank 3
    p = Projection.fit(feats, dim=4)
    assert p(feats).shape == (200, 4) and p(feats[0]).shape == (4,)
    assert p.explained[:3].sum() > 0.999
    p.save(tmp_path / "p.npz")
    q = Projection.load(tmp_path / "p.npz")
    assert np.allclose(q(feats), p(feats))


def test_export_onnx_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from odm_train.camsim.encoders import EmbeddingPipeline, OnnxEncoder, Projection, TorchEncoder, export_onnx

    class Tiny(torch.nn.Module):  # stands in for V-JEPA / DINOv2: image -> pooled features
        def __init__(self):
            super().__init__()
            self.patch = torch.nn.Conv2d(3, 8, 16, 16)

        def forward(self, x):
            return self.patch(x).flatten(2).mean(-1)

    torch.manual_seed(0)
    enc = TorchEncoder("tiny", Tiny(), 64)
    frames = np.random.default_rng(0).integers(0, 255, (12, 240, 320, 3), dtype=np.uint8)
    feats = enc(frames)
    proj = Projection.fit(feats, dim=4)
    assert export_onnx(enc, str(tmp_path / "e.onnx"), proj) < 1e-4
    onnx_emb = OnnxEncoder(str(tmp_path / "e.onnx"))(frames)
    assert np.abs(onnx_emb - proj(feats)).max() < 1e-4
    assert EmbeddingPipeline(enc, proj)(frames[0]).shape == (4,)


# --- step 4: gamepad page for the interactive app


def test_pad_to_cmd_matches_robot_gamepad_ranges():
    from odm_train.camsim.pad import pad_to_cmd

    assert np.allclose(pad_to_cmd(0, -1, 0, 0), [0.15, 0, 0, 0, 0, 0, 0])  # stick up = forward
    assert np.allclose(pad_to_cmd(-1, 0, 1, 0)[[1, 2]], [0.2, -1.0])  # left = +vy, right turn = -wz
    assert not pad_to_cmd(0.05, 0.05, 0.05, 0.05).any()  # deadzone
    head = pad_to_cmd(0, 0, 1, 1, head=True)
    assert head[2] == 0 and head[5] < 0 and head[4] > 0


def test_pad_server_roundtrip_and_taps():
    import json
    import urllib.request

    from odm_train.camsim.pad import PadServer

    pad = PadServer("127.0.0.1", 0)
    pad.start()
    port = pad.httpd.server_address[1]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        assert b"Duck gamepad" in opener.open(f"http://127.0.0.1:{port}/").read()

        def post(axes, buttons):
            req = urllib.request.Request(f"http://127.0.0.1:{port}/", data=json.dumps({"axes": axes, "buttons": buttons}).encode(), method="POST")
            opener.open(req).read()

        post([0, -1, 0, 0], [True] + [False] * 9)
        post([0, -1, 0, 0], [False] * 10)
        assert pad.connected() and pad.cmd()[0] > 0.14
        assert pad.pop_buttons() == {"a"} and pad.pop_buttons() == set()  # one tap, reported once
    finally:
        pad.stop()


@pytest.mark.skipif(not (ODP and STOCK), reason="needs ODP_DIR and STOCK_ONNX (training stack)")
@pytest.mark.parametrize("mount,parent", [("head", "head_assembly"), ("trunk", "trunk_assembly")])
def test_camera_mass_is_added_to_the_mount(monkeypatch, mount, parent):
    from odm_train.camsim import gl

    gl.setup()
    pytest.importorskip("mujoco")
    monkeypatch.chdir(ODP)
    monkeypatch.syspath_prepend(ODP)
    from odm_train.camsim.scene import CameraConfig, SceneConfig, build_scene
    from odm_train.sim_eval import XML

    m0, _ = build_scene(XML, CameraConfig(mount=mount), SceneConfig())
    m1, _ = build_scene(XML, CameraConfig(mount=mount, mass_g=120), SceneConfig())
    assert abs(m1.body_subtreemass[1] - m0.body_subtreemass[1] - 0.12) < 1e-6
    assert m1.body_parentid[m1.body("maixcam2_module").id] == m1.body(parent).id
