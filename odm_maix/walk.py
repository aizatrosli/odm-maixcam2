"""50 Hz RL walking loop (port of v2_rl_walk_mujoco.py).

The observation layout, action scaling, phase clock and head-command handling
are kept identical to the Pi runtime, because the pretrained policy expects
exactly this input.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .config import JOINT_NAMES
from .policy_profile import STOCK_PROFILE
from .rl_utils import GaitPhase, LowPassActionFilter

NUM_DOFS = 14
HEAD_SLICE = slice(5, 9)  # neck_pitch, head_pitch, head_yaw, head_roll


def build_obs(
    gyro, accelero, commands, dof_pos, init_pos, dof_vel, last_actions, motor_targets, feet, phase, extra=()
):
    """101 floats in the Pi runtime's order, then any extra inputs from the policy profile."""
    a1, a2, a3 = last_actions
    return np.concatenate(
        [
            gyro,  # 3
            accelero,  # 3
            commands,  # 7
            dof_pos - init_pos,  # 14
            dof_vel * 0.05,  # 14
            a1,  # 14
            a2,  # 14
            a3,  # 14
            motor_targets,  # 14
            feet,  # 2
            phase,  # 2
            np.asarray(extra, dtype=np.float32).reshape(-1),  # profile extra_obs, usually empty
        ]
    ).astype(np.float32)


def phase_frequency_factor(x_velocity: float) -> float:
    return 1.0 + (abs(x_velocity) / 0.15) * (1.2 - 1.0)


class RLWalk:
    def __init__(
        self,
        config,
        hwi,
        imu,
        feet_contacts,
        policy,
        commands,
        nb_steps_in_period: int,
        control_freq: float = 50,
        pid=(30, 0, 0),
        action_scale: float = 0.25,
        cutoff_frequency: Optional[float] = None,
        save_obs_path: Optional[str] = None,
        profile: Optional[dict] = None,
    ):
        self.config = config
        self.hwi = hwi
        self.imu = imu
        self.feet_contacts = feet_contacts
        self.policy = policy
        self.commands = commands
        self.control_freq = control_freq
        self.pid = pid
        self.profile = dict(profile or STOCK_PROFILE)
        # an explicit profile wins over the CLI default
        self.action_scale = self.profile.get("action_scale", action_scale) if profile else action_scale
        self.motor_speed_limit = self.profile.get("motor_speed_limit")
        self.apply_head_commands = self.profile.get("apply_head_commands", True)
        self.extra_obs_names = list(self.profile.get("extra_obs", []))
        self.save_obs_path = save_obs_path
        self.saved_obs = []

        self.action_filter = LowPassActionFilter(control_freq, cutoff_frequency) if cutoff_frequency else None
        self.phase = GaitPhase(nb_steps_in_period)
        self.phase_frequency_factor = 1.0
        self.phase_frequency_factor_offset = config.phase_frequency_factor_offset

        self.init_pos = np.array([hwi.init_pos[j] for j in JOINT_NAMES])
        self.last_actions = [np.zeros(NUM_DOFS) for _ in range(3)]
        self.motor_targets = self.init_pos.copy()
        self.policy_targets = self.init_pos.copy()  # before head offsets, for the speed limit
        self.last_commands = np.zeros(7)
        self.paused = config.start_paused

        # timing stats
        self.loop_times = []
        self.infer_times = []
        self.overruns = 0

    def start(self):
        kps = [self.pid[0]] * NUM_DOFS
        kds = [self.pid[2]] * NUM_DOFS
        kps[5:9] = [8, 8, 8, 8]  # softer head
        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()
        time.sleep(2)

    def extra_obs(self) -> np.ndarray:
        """Extra policy inputs (camera features), asked from the command source by name."""
        if not self.extra_obs_names:
            return np.zeros(0, dtype=np.float32)
        get = getattr(self.commands, "extra_obs", None)
        if get is None:
            raise RuntimeError(
                f"policy needs extra inputs {self.extra_obs_names} but the command source has no extra_obs()"
            )
        x = np.asarray(get(self.extra_obs_names), dtype=np.float32).reshape(-1)
        if len(x) != len(self.extra_obs_names):
            raise RuntimeError(f"extra_obs() returned {len(x)} values, policy expects {len(self.extra_obs_names)}")
        return x

    def get_obs(self):
        imu = self.imu.get_data()
        dof_pos, dof_vel = self.hwi.get_present_positions_velocities()
        if dof_pos is None or len(dof_pos) != NUM_DOFS:
            return None
        return build_obs(
            imu["gyro"],
            imu["accelero"],
            self.last_commands,
            dof_pos,
            self.init_pos,
            dof_vel,
            self.last_actions,
            self.motor_targets,
            np.array(self.feet_contacts.get(), dtype=np.float32),
            self.phase.phase,
            self.extra_obs(),
        )

    def step(self, t_start: float) -> bool:
        cmd, events = self.commands.get()
        self.last_commands = cmd
        if events.get("A"):
            self.paused = not self.paused
            print("PAUSE" if self.paused else "UNPAUSE")
        self.phase_frequency_factor = 1.3 if events.get("LB_held") else 1.0

        if self.paused:
            return False

        obs = self.get_obs()
        if obs is None:
            return False

        self.phase.step(self.phase_frequency_factor + self.phase_frequency_factor_offset)

        if self.save_obs_path:
            self.saved_obs.append(obs)

        ti = time.perf_counter()
        action = np.asarray(self.policy.infer(obs), dtype=np.float64)
        self.infer_times.append(time.perf_counter() - ti)

        self.last_actions = [action.copy(), self.last_actions[0], self.last_actions[1]]
        targets = self.init_pos + action * self.action_scale

        if self.motor_speed_limit:
            step = self.motor_speed_limit / self.control_freq
            targets = np.clip(targets, self.policy_targets - step, self.policy_targets + step)

        if self.action_filter is not None:
            self.action_filter.push(targets)
            filtered = self.action_filter.get_filtered_action()
            if time.monotonic() - t_start > 1:  # let the filter settle
                targets = filtered

        self.policy_targets = np.array(targets, dtype=np.float64)
        self.motor_targets = self.policy_targets.copy()
        if self.apply_head_commands:
            self.motor_targets[HEAD_SLICE] += self.last_commands[3:]
        self.hwi.set_position_array(self.motor_targets)
        return True

    def run(self):
        period = 1.0 / self.control_freq
        t_start = time.monotonic()
        next_t = t_start
        last_report = t_start
        print("Starting")
        try:
            while True:
                t = time.monotonic()
                if self.paused:
                    self.step(t_start)
                    time.sleep(0.1)
                    next_t = time.monotonic()
                    continue
                self.step(t_start)
                took = time.monotonic() - t
                self.loop_times.append(took)
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    self.overruns += 1
                    next_t = time.monotonic()  # don't try to catch up
                if t - last_report > 5:
                    self.report()
                    last_report = t
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def report(self):
        if not self.loop_times:
            return
        lt = np.array(self.loop_times[-250:]) * 1000
        it = np.array(self.infer_times[-250:]) * 1000
        print(
            f"loop mean {lt.mean():.1f} ms  p99 {np.percentile(lt, 99):.1f} ms  "
            f"policy mean {it.mean():.2f} ms  overruns {self.overruns}"
        )

    def shutdown(self):
        print("Stopping")
        self.report()
        for part in (self.commands, self.feet_contacts, self.imu):
            try:
                part.stop()
            except Exception:
                pass
        if self.save_obs_path and self.saved_obs:
            np.save(self.save_obs_path, np.stack(self.saved_obs))
            print(f"saved {len(self.saved_obs)} observations to {self.save_obs_path}")
