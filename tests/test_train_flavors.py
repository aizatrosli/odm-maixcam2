"""Trainer bookkeeping that runs without JAX: flavors, resume, profiles."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from odm_train import train  # noqa: E402
from odm_train.flavors import DEPLOYABLE, FLAVORS  # noqa: E402

TASKS = {"flat_terrain", "flat_terrain_backlash", "rough_terrain_backlash"}


def test_flavors_are_well_formed():
    for name, f in FLAVORS.items():
        assert f["env"] in train.PROFILES, name
        assert f["task"] in TASKS, name
        assert f["randomization"] in {"strong", "stock", "none"}, name
    assert "follow_flat_backlash" in DEPLOYABLE
    assert "standing_flat" not in DEPLOYABLE


def test_walking_profiles_match_runtime():
    for env in ("follow", "joystick"):
        prof = train.PROFILES[env]
        assert prof["obs_layout"] == "v2_101"
        assert prof["motor_speed_limit"] == 5.24
        assert prof["action_scale"] == 0.25


def test_latest_checkpoint_and_progress(tmp_path):
    assert train.latest_checkpoint(str(tmp_path)) is None
    assert train.read_progress(str(tmp_path)) == {"done_steps": 0}
    (tmp_path / "2026_01_01_000000_0").mkdir()
    (tmp_path / "2026_01_01_000000_0.onnx").write_bytes(b"")
    time.sleep(0.01)
    newest = tmp_path / "2026_01_01_010000_1000"
    newest.mkdir()
    (tmp_path / "progress.json").write_text(json.dumps({"done_steps": 1000}))
    assert train.latest_checkpoint(str(tmp_path)) == str(newest)
    assert train.read_progress(str(tmp_path))["done_steps"] == 1000


def test_list_flavors_needs_no_jax(capsys):
    train.main(["--list_flavors"])
    out = capsys.readouterr().out
    for name in FLAVORS:
        assert name in out
