import os

import numpy as np
import pytest

from odm_maix.config import JOINT_NAMES, DuckConfig
from odm_maix.hwi import INIT_POS
from odm_maix.walk import RLWalk, build_obs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_obs_size():
    z = np.zeros(14)
    obs = build_obs(np.zeros(3), np.zeros(3), np.zeros(7), z, z, z, [z, z, z], z, np.zeros(2), np.zeros(2))
    assert obs.shape == (101,)
    assert obs.dtype == np.float32


class FakeHWI:
    def __init__(self):
        self.init_pos = dict(INIT_POS)
        self.sent = []

    def get_present_positions_velocities(self):
        return np.array([INIT_POS[j] for j in JOINT_NAMES]), np.zeros(14)

    def set_position_array(self, p):
        self.sent.append(np.array(p))


class FakeImu:
    def get_data(self):
        return {"gyro": np.zeros(3), "accelero": np.array([0, 0, 9.81])}


class FakeFeet:
    def get(self):
        return [True, True]


class FakeCommands:
    def __init__(self, cmd):
        self.cmd = np.array(cmd, dtype=float)

    def get(self):
        return self.cmd.copy(), {}


class RecordingPolicy:
    def __init__(self):
        self.obs = []

    def infer(self, obs):
        self.obs.append(obs)
        return np.ones(14) * 0.4


def make_walker(cmd=(0.1, 0, 0, 0, 0.2, 0, 0)):
    cfg = DuckConfig()
    hwi = FakeHWI()
    pol = RecordingPolicy()
    w = RLWalk(cfg, hwi, FakeImu(), FakeFeet(), pol, FakeCommands(cmd), nb_steps_in_period=36)
    return w, hwi, pol


def test_step_scales_actions_and_adds_head_commands():
    w, hwi, pol = make_walker()
    assert w.step(0.0)
    init = np.array([INIT_POS[j] for j in JOINT_NAMES])
    expected = init + 0.4 * 0.25
    expected[5:9] += np.array([0, 0.2, 0, 0])
    np.testing.assert_allclose(hwi.sent[-1], expected)


def test_action_history_and_phase_feed_back():
    w, _, pol = make_walker()
    w.step(0.0)
    w.step(0.0)
    obs = pol.obs[-1]
    # layout: gyro 0:3, acc 3:6, cmd 6:13, pos 13:27, vel 27:41, last_action 41:55
    np.testing.assert_allclose(obs[41:55], 0.4)
    np.testing.assert_allclose(obs[55:69], 0.0)
    # phase advanced once before the 2nd inference
    a = 2 * np.pi / 36
    np.testing.assert_allclose(obs[-2:], [np.cos(a), np.sin(a)], atol=1e-6)
    # commands block
    np.testing.assert_allclose(obs[6:13], [0.1, 0, 0, 0, 0.2, 0, 0], atol=1e-6)


POLICY = os.environ.get("ODM_POLICY_ONNX")


@pytest.mark.skipif(not POLICY, reason="set ODM_POLICY_ONNX to the BEST_WALK_ONNX_2.onnx path")
def test_onnx_policy_runs():
    from odm_maix.policy import OnnxPolicy

    out = OnnxPolicy(POLICY).infer(np.zeros(101, dtype=np.float32))
    assert out.shape == (14,)
