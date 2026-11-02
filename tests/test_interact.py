import math

import numpy as np
import pytest

from odm_maix.behavior import HEAD_LIMITS, Behavior, BehaviorParams
from odm_maix.commands import FollowCommands
from odm_maix.config import DuckConfig
from odm_maix.expression import Expressions, antenna_duty, eye_shapes
from odm_maix.vision.detector import Face, Hand
from odm_maix.vision.follow import Detection
from odm_maix.vision.gestures import GestureTracker, classify_pose, keypoints_from_maix

# -- synthetic hands ------------------------------------------------------------
# Upright hand in image coords (y down), wrist at (0, 0), fingers pointing up.
FINGER_X = {"index": -0.3, "middle": -0.1, "ring": 0.1, "pinky": 0.3}


def make_hand(extended=("index", "middle", "ring", "pinky"), thumb="side", scale=1.0, dx=0.0, angle=0.0):
    kp = [(0.0, 0.0)] * 21
    kp[0] = (0.0, 0.0)
    for name, x in FINGER_X.items():
        mcp_i = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}[name]
        mcp = (x, -1.0)
        if name in extended:
            pip, dip, tip = (x, -1.5), (x, -1.8), (x, -2.1)
        else:  # curled back toward the palm
            pip, dip, tip = (x, -1.4), (x, -1.1), (x, -0.9)
        kp[mcp_i : mcp_i + 4] = [mcp, pip, dip, tip]
    if thumb == "side":  # out to the side
        kp[1:5] = [(-0.4, -0.3), (-0.7, -0.5), (-0.95, -0.65), (-1.2, -0.8)]
    elif thumb == "up":  # straight up, above the knuckles
        kp[1:5] = [(-0.45, -0.6), (-0.5, -1.2), (-0.5, -1.7), (-0.5, -2.2)]
    else:  # tucked across the palm
        kp[1:5] = [(-0.4, -0.3), (-0.3, -0.6), (-0.2, -0.8), (-0.1, -0.85)]
    c, s = math.cos(angle), math.sin(angle)
    return [(scale * (c * x - s * y) + dx, scale * (s * x + c * y)) for x, y in kp]


def test_classify_poses():
    assert classify_pose(make_hand()) == "open_palm"
    assert classify_pose(make_hand(extended=(), thumb="tucked")) == "fist"
    assert classify_pose(make_hand(extended=(), thumb="up")) == "thumbs_up"
    assert classify_pose(make_hand(extended=("index",), thumb="tucked")) == "point"
    assert classify_pose(make_hand(extended=("index", "middle"), thumb="tucked")) == "peace"
    assert classify_pose([]) == "none"


def test_classify_is_scale_and_rotation_invariant():
    assert classify_pose(make_hand(scale=40.0, dx=200)) == "open_palm"
    assert classify_pose(make_hand(extended=("index", "middle"), thumb="tucked", angle=0.6)) == "peace"
    assert classify_pose(make_hand(extended=(), thumb="tucked", angle=-0.8, scale=0.1)) == "fist"


def test_keypoints_from_maix_layout():
    hand = make_hand()
    pts = [0.0] * 8
    for x, y in hand:
        pts += [x, y, 0.5]
    assert keypoints_from_maix(pts) == hand
    assert keypoints_from_maix([1, 2, 3]) == []


# -- gesture events -------------------------------------------------------------
def feed(tracker, poses, t0=0.0, dt=0.066):
    events, t = [], t0
    for pose, cx in poses:
        e = tracker.update(t, pose, cx)
        if e:
            events.append((round(t, 2), e))
        t += dt
    return events, t


def test_pose_must_be_held_and_fires_once():
    g = GestureTracker()
    events, _ = feed(g, [("thumbs_up", 0.5)] * 30)
    assert [e for _, e in events] == ["thumbs_up"]
    assert events[0][0] >= 0.4
    # a pose flickering through for one frame does nothing
    g = GestureTracker()
    events, _ = feed(g, [("fist", 0.5), ("open_palm", 0.5)] * 10)
    assert events == []


def test_wave_is_detected_and_not_a_stop():
    g = GestureTracker()
    xs = [0.5 + 0.1 * math.sin(2 * math.pi * 2 * i * 0.066) for i in range(20)]
    events, _ = feed(g, [("open_palm", x) for x in xs])
    names = [e for _, e in events]
    assert "wave" in names and "stop" not in names


def test_still_open_palm_is_stop():
    g = GestureTracker()
    events, _ = feed(g, [("open_palm", 0.5 + 0.005 * (i % 2)) for i in range(20)])
    assert [e for _, e in events] == ["stop"]


def test_hand_gone_rearms_after_cooldown():
    g = GestureTracker()
    ev1, t = feed(g, [("thumbs_up", 0.5)] * 10)
    _, t = feed(g, [(None, 0.5)] * 40, t0=t)
    ev2, _ = feed(g, [("thumbs_up", 0.5)] * 10, t0=t)
    assert len(ev1) == 1 and len(ev2) == 1


# -- behavior -------------------------------------------------------------------
WALK = np.array([0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])


def run_behavior(b, seconds, t0=0.0, dt=0.02, cmd=WALK, target_h=0.3, **once):
    t, out, expr = t0, None, None
    first = True
    sounds = []
    while t < t0 + seconds - 1e-9:
        kw = once if first else {}
        out, expr = b.update(t, dt, cmd, target_h, **kw)
        if expr.sound:
            sounds.append(expr.sound)
        first = False
        t += dt
    return out, expr, t, sounds


def test_stay_mode_blocks_walking_until_wave():
    b = Behavior()
    out, _, t, _ = run_behavior(b, 1.0)
    assert b.mode == "stay" and np.allclose(out[:3], 0)
    out, expr, t, sounds = run_behavior(b, 1.0, t0=t, gesture="wave")
    assert b.mode == "follow" and expr.mood == "happy" and "happy" in sounds
    np.testing.assert_allclose(out[:3], WALK[:3])  # ramped back in within 0.5 s
    out, _, t, sounds = run_behavior(b, 1.0, t0=t, gesture="stop")
    assert b.mode == "stay" and np.allclose(out[:3], 0) and "ok" in sounds


def test_walk_fades_rather_than_jumps():
    b = Behavior(BehaviorParams(start_mode="follow"))
    out, _ = b.update(0.0, 0.02, WALK, 0.3, gesture="stop")
    assert out[0] == pytest.approx(WALK[0] * (1 - 2.0 * 0.02))


def test_head_animates_only_when_standing():
    b = Behavior(BehaviorParams(start_mode="follow"))
    out, expr, t, _ = run_behavior(b, 1.0, gesture="thumbs_up")
    assert expr.mood == "happy"
    assert np.allclose(expr.head, 0)  # walking: no head animation
    b = Behavior()
    out, expr, _, _ = run_behavior(b, 1.0, gesture="thumbs_up")
    assert expr.mood == "happy" and np.abs(expr.head).max() > 0.02
    assert np.all(np.abs(out[3:7]) <= HEAD_LIMITS + 1e-9)


def test_known_face_is_greeted_once():
    b = Behavior()
    _, expr, t, sounds = run_behavior(b, 0.5, face="aizat", face_known=True)
    assert "hello" in sounds and expr.mood == "happy"
    _, _, t, sounds = run_behavior(b, 0.5, t0=t, face="aizat", face_known=True)
    assert "hello" not in sounds


def test_new_person_makes_it_curious_then_attentive():
    b = Behavior()
    run_behavior(b, 1.0, target_h=None)
    _, expr, t, sounds = run_behavior(b, 0.5, t0=1.0)
    assert expr.mood == "curious" and "curious" in sounds
    _, expr, _, _ = run_behavior(b, 2.0, t0=t)
    assert expr.mood == "attentive"


def test_sudden_approach_startles():
    b = Behavior()
    run_behavior(b, 2.0, target_h=0.3)
    _, expr, _, sounds = run_behavior(b, 0.2, t0=2.0, target_h=0.7)
    assert expr.mood == "startled" and "startled" in sounds and expr.antennas == (1.0, 1.0)


def test_falls_asleep_alone():
    b = Behavior(BehaviorParams(sleepy_after=5.0))
    _, expr, t, sounds = run_behavior(b, 6.0, target_h=None)
    assert expr.mood == "sleepy" and sounds == ["sleepy"] and expr.eye_leds < 0.3
    _, expr, _, _ = run_behavior(b, 0.2, t0=t, target_h=0.3)
    assert expr.mood == "curious"


# -- expression outputs -----------------------------------------------------------
class FakePWM:
    def __init__(self, pid, freq, duty, enable):
        self.pid, self.freq, self.duties = pid, freq, [duty]

    def duty(self, d):
        self.duties.append(d)

    def disable(self):
        pass


class FakePWMModule:
    def __init__(self):
        self.outs = []

    def PWM(self, *a, **kw):
        o = FakePWM(*a, **kw)
        self.outs.append(o)
        return o


class FakePinmap:
    def __init__(self):
        self.pins = {}

    def set_pin_function(self, pin, func):
        self.pins[pin] = func


class Slot:
    value = 0


def test_expressions_drive_pwm_and_mood_slot():
    cfg = DuckConfig()
    cfg.expression_features = {"antennas": True, "eyes": True, "screen_eyes": True}
    pwm, pm, slot = FakePWMModule(), FakePinmap(), Slot()
    ex = Expressions(cfg, mood_slot=slot, pwm_module=pwm, pinmap_module=pm)
    assert pm.pins == {"A31": "PWM7", "B2": "PWM4", "B3": "PWM5", "A30": "PWM6"}
    b = Behavior()
    _, expr = b.update(0.0, 0.02, WALK, 0.7)
    _, expr = b.update(0.02, 0.02, WALK, 0.7)
    expr.mood, expr.antennas, expr.eye_leds = "startled", (1.0, 1.0), 1.0
    ex.apply(expr, now=1.0)
    assert slot.value == 4
    left, right, leye, reye = pwm.outs
    assert left.duties[-1] == 10.0 and right.duties[-1] == 5.0 and leye.duties[-1] == 100.0
    n = len(left.duties)
    ex.apply(expr, now=1.01)  # rate-limited
    ex.apply(expr, now=2.0)  # unchanged value: no write
    assert len(left.duties) == n


def test_antenna_duty_range():
    assert antenna_duty(-1, [5, 10]) == 5 and antenna_duty(1, [5, 10]) == 10 and antenna_duty(0, [5, 10]) == 7.5
    assert antenna_duty(3, [5, 10]) == 10


def test_eye_shapes_inside_screen():
    for mood in ("neutral", "attentive", "happy", "curious", "startled", "sleepy"):
        for t in (0.0, 1.0):
            shapes = eye_shapes(mood, t, 640, 480)
            assert sum(k == "ellipse" for k, _ in shapes) == 2
            for kind, a in shapes:
                if kind != "rect":
                    assert 0 <= a[0] - a[2] and a[0] + a[2] <= 640


# -- wired into FollowCommands --------------------------------------------------
class FakeVision:
    def __init__(self):
        self.dets = [Detection(0.5, 0.5, 0.1, 0.25)]
        self.hand = None
        self.face = None
        self.mood = Slot()

    def latest(self):
        return True, 0.0, list(self.dets)

    def latest_hand(self):
        return True, self.hand

    def latest_face(self):
        return (self.face is not None), self.face

    def stop(self):
        pass


def test_follow_commands_interact_wave_then_follow():
    clock = iter(np.arange(0, 100, 0.02))
    v = FakeVision()
    fc = FollowCommands(DuckConfig().maixcam2["vision"], detector=v, clock=lambda: next(clock), interact=True)
    for _ in range(100):
        cmd, _ = fc.get()
    assert cmd[0] == pytest.approx(0.0)  # waits in "stay"
    for i in range(30):  # wave
        v.hand = Hand(0.5 + 0.1 * math.sin(2 * math.pi * 2 * i * 0.02 * 3), 0.4, 0.2, "open_palm", 0.9)
        for _ in range(3):
            cmd, _ = fc.get()
    v.hand = None
    assert fc.behavior.mode == "follow"
    for _ in range(100):
        cmd, _ = fc.get()
    assert cmd[0] > 0
    assert fc.state.startswith("following/follow/")


def test_follow_commands_interact_greets_face():
    clock = iter(np.arange(0, 100, 0.02))
    v = FakeVision()
    v.face = Face(0.5, 0.3, 0.1, 0.15, "aizat", True, 0.9)
    fc = FollowCommands(DuckConfig().maixcam2["vision"], detector=v, clock=lambda: next(clock), interact=True)
    fc.get()
    assert fc.expression.sound == "hello" and fc.last_face[1] == "aizat"
