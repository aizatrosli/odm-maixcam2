"""Compare walking policies in MuJoCo (CPU) the way the MaixCAM2 runtime drives them.

The control loop mirrors odm_maix/walk.py: the same 101-value observation,
head commands added to the head targets, an optional motor speed limit from
the policy profile, and 0 to 2 steps of action latency. Commands come from the
same follow-style process used in training. Each episode randomizes floor
friction, servo kp and torso mass.

    cd Open_Duck_Playground
    PYTHONPATH=$PYTHONPATH:/path/to/odm-maixcam2/training \\
      uv run python -m odm_train.sim_eval BEST_WALK_ONNX_2.onnx runs/follow_v1/<ckpt>.onnx -n 20

Prints, per policy: falls, mean time upright, and RMS tracking error for
forward speed (vx) and turn rate (wz).
"""

import argparse
import json
import os

import mujoco
import numpy as np
import onnxruntime
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase

from odm_train.profile import STOCK_PROFILE

XML = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml"
PRM_PATH = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
HEAD = slice(5, 9)
SLEW = np.array([0.3, 0.3, 3.0, 1.0, 2.0, 2.0, 1.0])
CTRL_DT = 0.02


def load_profile(onnx_path):
    p = onnx_path[:-5] + ".json"
    if os.path.exists(p):
        return json.load(open(p))
    return dict(STOCK_PROFILE)


class CommandProcess:
    """numpy twin of FollowJoystick's command process."""

    def __init__(self, rng, head: bool, mode: str):
        self.rng = rng
        self.head = head
        self.mode = mode
        self.cmd = np.zeros(7)
        self.goal = np.zeros(7)
        self.timer = 0

    def _goal(self):
        r = self.rng
        if self.mode == "straight":
            g = np.array([0.12, 0, 0, 0, 0, 0, 0.0])
        elif r.uniform() < 0.15:
            g = np.zeros(7)
        else:
            vx = r.uniform(0, 0.15)
            wz = r.uniform(-1, 1)
            vx = 0.0 if abs(wz) > 0.6 else vx
            yaw = np.clip(0.5 * wz + 0.15 * r.normal(), -0.6, 0.6)
            g = np.array([vx, 0, wz, 0, r.uniform(-0.5, 0.5), yaw, 0])
        if not self.head:
            g[3:] = 0
        return g

    def step(self):
        if self.timer <= 0:
            self.goal = self._goal()
            self.timer = int(round(self.rng.uniform(1, 4) / CTRL_DT))
        self.timer -= 1
        self.cmd += np.clip(self.goal - self.cmd, -SLEW * CTRL_DT, SLEW * CTRL_DT)
        return self.cmd.copy()


class Sim(MJInferBase):
    def __init__(self):
        super().__init__(XML)
        self.base_friction = self.model.geom_friction[0, 0]
        self.base_kp = self.model.actuator_gainprm[:, 0].copy()
        self.torso = self.model.body("trunk_assembly").id
        self.base_torso_mass = self.model.body_mass[self.torso]
        self.linvel_adr = self.model.sensor_adr[self.model.sensor("local_linvel").id]

    def randomize(self, rng):
        self.model.geom_friction[0, 0] = rng.uniform(0.4, 1.2)
        f = rng.uniform(0.8, 1.2, size=self.model.nu)
        self.model.actuator_gainprm[:, 0] = self.base_kp * f
        self.model.actuator_biasprm[:, 1] = -self.base_kp * f
        self.model.body_mass[self.torso] = self.base_torso_mass + rng.uniform(-0.15, 0.25)

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.keyframe("home").id)
        self.data.ctrl[:] = self.default_actuator
        mujoco.mj_forward(self.model, self.data)

    def upright(self):
        # z component of the torso's up axis; below 0.3 means the duck has fallen over
        return self.data.xmat[self.torso][8] > 0.3


def run_episode(sim, policy, profile, prm_steps, rng, seconds, head, mode, latency):
    sim.randomize(rng)
    sim.reset()
    cmds = CommandProcess(rng, head, mode)
    default = sim.default_actuator.copy()
    last = [np.zeros(14)] * 3
    targets = default.copy()  # observed (includes head offsets)
    policy_targets = default.copy()
    pending = []
    phase_i = 0.0
    phase = np.zeros(2)
    vmax = profile.get("motor_speed_limit")
    err_vx, err_wz = [], []
    steps = int(seconds / CTRL_DT)
    for k in range(steps):
        cmd = cmds.step()
        contacts = np.array(sim.get_feet_contacts(sim.data), dtype=np.float32)
        obs = np.concatenate(
            [
                sim.get_gyro(sim.data),
                sim.get_accelerometer(sim.data),
                cmd,
                sim.get_actuator_joints_qpos(sim.data.qpos) - default,
                sim.get_actuator_joints_qvel(sim.data.qvel) * 0.05,
                last[0],
                last[1],
                last[2],
                targets,
                contacts,
                phase,
            ]
        ).astype(np.float32)
        phase_i = (phase_i + 1) % prm_steps
        a = phase_i / prm_steps * 2 * np.pi
        phase = np.array([np.cos(a), np.sin(a)])

        action = policy.run(None, {"obs": obs[None]})[0][0]
        last = [action.copy(), last[0], last[1]]
        pending.append(action)
        delay = rng.integers(0, latency + 1)
        applied = pending[max(0, len(pending) - 1 - delay)]
        pending = pending[-(latency + 1) :]

        new_targets = default + applied * profile["action_scale"]
        if vmax:
            new_targets = np.clip(new_targets, policy_targets - vmax * CTRL_DT, policy_targets + vmax * CTRL_DT)
        policy_targets = new_targets
        targets = policy_targets.copy()
        if profile.get("apply_head_commands", True):
            targets[HEAD] += cmd[3:7]
        sim.data.ctrl[:] = targets
        for _ in range(sim.decimation):
            mujoco.mj_step(sim.model, sim.data)

        if not sim.upright():
            return False, (k + 1) * CTRL_DT, err_vx, err_wz
        if k * CTRL_DT > 1.0:  # skip the start
            lv = sim.data.sensordata[sim.linvel_adr : sim.linvel_adr + 3]
            err_vx.append(lv[0] - cmd[0])
            err_wz.append(sim.get_gyro(sim.data)[2] - cmd[2])
    return True, seconds, err_vx, err_wz


def evaluate(path, n, seconds, head, mode, latency, seed):
    profile = load_profile(path)
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = 1
    policy = onnxruntime.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    prm = PolyReferenceMotion(PRM_PATH)
    sim = Sim()
    rng = np.random.default_rng(seed)
    falls, alive, evx, ewz = 0, [], [], []
    for _ in range(n):
        ok, t, ex, ew = run_episode(sim, policy, profile, prm.nb_steps_in_period, rng, seconds, head, mode, latency)
        falls += not ok
        alive.append(t)
        evx += ex
        ewz += ew
    rms = lambda e: float(np.sqrt(np.mean(np.square(e)))) if e else float("nan")  # noqa: E731
    return dict(
        policy=os.path.basename(path),
        profile=profile.get("name"),
        falls=f"{falls}/{n}",
        mean_upright_s=round(float(np.mean(alive)), 1),
        vx_rms=round(rms(evx), 3),
        wz_rms=round(rms(ewz), 3),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("policies", nargs="+")
    p.add_argument("-n", type=int, default=20, help="episodes per policy")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--mode", choices=["follow", "straight"], default="follow")
    p.add_argument("--no_head", action="store_true", help="keep head commands at zero")
    p.add_argument("--latency", type=int, default=2, help="max action delay in control steps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for path in args.policies:
        r = evaluate(path, args.n, args.seconds, not args.no_head, args.mode, args.latency, args.seed)
        print(json.dumps(r))


if __name__ == "__main__":
    main()
