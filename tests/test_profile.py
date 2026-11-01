import json

import numpy as np
import pytest

from odm_maix.config import JOINT_NAMES, DuckConfig
from odm_maix.hwi import INIT_POS
from odm_maix.policy_profile import STOCK_PROFILE, load_profile, obs_size
from odm_maix.walk import RLWalk
from tests.test_walk import FakeCommands, FakeFeet, FakeHWI, FakeImu

INIT = np.array([INIT_POS[j] for j in JOINT_NAMES])


class ConstPolicy:
    def __init__(self, value, size=101):
        self.value = value
        self.obs = []
        self.obs_size = size

    def infer(self, obs):
        self.obs.append(obs)
        return np.ones(14) * self.value


def walker(profile, cmd=(0, 0, 0, 0, 0.2, 0, 0), value=1.0, commands=None):
    hwi = FakeHWI()
    pol = ConstPolicy(value)
    w = RLWalk(
        DuckConfig(), hwi, FakeImu(), FakeFeet(), pol, commands or FakeCommands(cmd), nb_steps_in_period=36, profile=profile
    )
    return w, hwi, pol


def test_profile_file_next_to_policy(tmp_path):
    (tmp_path / "walk.json").write_text(json.dumps({"name": "x", "motor_speed_limit": 5.24}))
    p = load_profile(str(tmp_path / "walk.onnx"))
    assert p["name"] == "x" and p["motor_speed_limit"] == 5.24
    assert p["apply_head_commands"] is True  # default kept
    assert load_profile(str(tmp_path / "other.mud")) == STOCK_PROFILE


def test_stock_profile_matches_pi_behaviour():
    w, hwi, _ = walker(None)
    w.step(0.0)
    expected = INIT + 0.25
    expected[5:9] += [0, 0.2, 0, 0]
    np.testing.assert_allclose(hwi.sent[-1], expected)


def test_motor_speed_limit_clips_target_changes():
    w, hwi, _ = walker({**STOCK_PROFILE, "motor_speed_limit": 5.0})
    w.step(0.0)
    step = 5.0 / 50
    np.testing.assert_allclose(hwi.sent[-1][:5], INIT[:5] + step)


def test_speed_limit_ignores_head_offsets():
    # the clip is on policy targets, so a constant head command doesn't eat the budget
    w, hwi, _ = walker({**STOCK_PROFILE, "motor_speed_limit": 50.0}, cmd=(0, 0, 0, 0, 0.3, 0, 0))
    for _ in range(3):
        w.step(0.0)
    np.testing.assert_allclose(hwi.sent[-1][6], INIT[6] + 0.25 + 0.3)


def test_head_commands_can_be_disabled():
    w, hwi, _ = walker({**STOCK_PROFILE, "apply_head_commands": False})
    w.step(0.0)
    np.testing.assert_allclose(hwi.sent[-1], INIT + 0.25)


class CameraCommands(FakeCommands):
    def extra_obs(self, names):
        return [0.5] * len(names)


def test_extra_obs_appended():
    prof = {**STOCK_PROFILE, "extra_obs": ["target_visible", "target_size"]}
    assert obs_size(prof) == 103
    w, _, pol = walker(prof, commands=CameraCommands((0,) * 7))
    w.step(0.0)
    assert pol.obs[-1].shape == (103,)
    np.testing.assert_allclose(pol.obs[-1][-2:], 0.5)


def test_extra_obs_without_source_fails():
    w, _, _ = walker({**STOCK_PROFILE, "extra_obs": ["target_visible"]})
    with pytest.raises(RuntimeError):
        w.step(0.0)
