# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
# Copyright 2026 odm-maixcam2 contributors
#
# Licensed under the Apache License, Version 2.0. Derived from
# Open_Duck_Playground playground/open_duck_mini_v2/joystick.py.
"""FollowJoystick: the Open Duck Mini v2 joystick task, made to match how the
MaixCAM2 robot is actually driven in follow mode.

Differences from Open_Duck_Playground's ``Joystick``:

1. **Head commands move the head.** The robot runtime adds ``commands[3:7]``
   to the head joint targets, but the stock training env only puts them in
   the observation. Follow mode moves the head constantly, so the policy
   now trains with that disturbance, and the ``motor_targets`` it observes
   include the head offsets, the same as on the robot.
2. **Follow-style commands.** Instead of a new random command every 10 s,
   the goal changes every 1 to 4 s and the command ramps toward it at the
   follow controller's rate limits. Head yaw tends to lead the turn, as when
   tracking a person. Some episodes keep the original uniform sampling so
   gamepad driving still works.
3. **Stand-still cost ignores the head**, because the head is now commanded.
4. Used with ``randomize.domain_randomize_strong`` for wider dynamics randomization.

The 101-value observation is unchanged, so the result is a drop-in
replacement for BEST_WALK_ONNX_2.onnx on the MaixCAM2 (including the NPU path).
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from playground.common.rewards import cost_stand_still
from playground.open_duck_mini_v2 import joystick as base_joystick

HEAD = slice(5, 9)


def default_config() -> config_dict.ConfigDict:
    cfg = base_joystick.default_config()
    cfg.follow = config_dict.create(
        # resample the command goal every U(min, max) seconds
        goal_interval_s=[1.0, 4.0],
        # probability of: standing still / follow-style goal / uniform goal
        p_zero=0.15,
        p_follow=0.55,
        # command slew rates (per second): vx, vy, wz, neck, head_pitch, head_yaw, head_roll
        slew_rates=[0.3, 0.3, 3.0, 1.0, 2.0, 2.0, 1.0],
        # head ranges actually applied to the joints (smaller than the obs-only ranges)
        neck_pitch_range=[-0.3, 0.6],
        head_pitch_range=[-0.5, 0.5],
        head_yaw_range=[-0.6, 0.6],
        head_roll_range=[-0.3, 0.3],
        head_yaw_lead=0.5,  # follow goals: head_yaw ~= lead * wz + noise
    )
    # stronger pushes than stock (people bump the duck while it follows them)
    cfg.push_config.magnitude_range = [0.1, 1.5]
    return cfg


class FollowJoystick(base_joystick.Joystick):
    def __init__(
        self,
        task: str = "flat_terrain_backlash",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(task=task, config=config, config_overrides=config_overrides)
        self._slew = jp.array(self._config.follow.slew_rates)

    # --- commands ------------------------------------------------------------

    def _uniform_goal(self, rng):
        f = self._config.follow
        c = self._config
        k = jax.random.split(rng, 7)
        u = lambda key, r: jax.random.uniform(key, minval=r[0], maxval=r[1])  # noqa: E731
        return jp.hstack(
            [
                u(k[0], c.lin_vel_x),
                u(k[1], c.lin_vel_y),
                u(k[2], c.ang_vel_yaw),
                u(k[3], f.neck_pitch_range),
                u(k[4], f.head_pitch_range),
                u(k[5], f.head_yaw_range),
                u(k[6], f.head_roll_range),
            ]
        )

    def _follow_goal(self, rng):
        """Forward walking and turning, head looking toward the turn."""
        f = self._config.follow
        c = self._config
        k = jax.random.split(rng, 5)
        vx = jax.random.uniform(k[0], minval=0.0, maxval=c.lin_vel_x[1])
        wz = jax.random.uniform(k[1], minval=c.ang_vel_yaw[0], maxval=c.ang_vel_yaw[1])
        yaw = jp.clip(
            f.head_yaw_lead * wz + 0.15 * jax.random.normal(k[2]),
            f.head_yaw_range[0],
            f.head_yaw_range[1],
        )
        pitch = jax.random.uniform(k[3], minval=f.head_pitch_range[0], maxval=f.head_pitch_range[1])
        # turning on the spot is common in follow mode: no forward speed while turning hard
        vx = jp.where(jp.abs(wz) > 0.6, 0.0, vx)
        return jp.hstack([vx, 0.0, wz, 0.0, pitch, yaw, 0.0])

    def _sample_goal(self, rng):
        f = self._config.follow
        k_mode, k_u, k_f = jax.random.split(rng, 3)
        r = jax.random.uniform(k_mode)
        goal = jp.where(r < f.p_zero + f.p_follow, self._follow_goal(k_f), self._uniform_goal(k_u))
        return jp.where(r < f.p_zero, jp.zeros(7), goal)

    def _sample_interval(self, rng):
        lo, hi = self._config.follow.goal_interval_s
        return jp.round(jax.random.uniform(rng, minval=lo, maxval=hi) / self.dt).astype(jp.int32)

    # --- reset / step ------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_goal, k_int = jax.random.split(rng, 3)
        state = super().reset(rng)
        goal = self._sample_goal(k_goal)
        state.info["command"] = goal  # start on the goal (the robot also starts from rest)
        state.info["cmd_goal"] = goal
        state.info["cmd_timer"] = self._sample_interval(k_int)
        state.info["policy_targets"] = self._default_actuator
        if base_joystick.USE_IMITATION_REWARD:
            state.info["current_reference_motion"] = self.PRM.get_reference_motion(goal[0], goal[1], goal[2], 0)
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = state.info
        use_imitation = base_joystick.USE_IMITATION_REWARD

        # command process: ramp toward the goal, new goal when the timer runs out
        info["rng"], k_goal, k_int = jax.random.split(info["rng"], 3)
        expired = info["cmd_timer"] <= 0
        info["cmd_goal"] = jp.where(expired, self._sample_goal(k_goal), info["cmd_goal"])
        info["cmd_timer"] = jp.where(expired, self._sample_interval(k_int), info["cmd_timer"] - 1)
        step_max = self._slew * self.dt
        info["command"] = info["command"] + jp.clip(info["cmd_goal"] - info["command"], -step_max, step_max)

        if use_imitation:
            info["imitation_i"] = (info["imitation_i"] + 1) % self.PRM.nb_steps_in_period
            a = info["imitation_i"] / self.PRM.nb_steps_in_period * 2 * jp.pi
            info["imitation_phase"] = jp.array([jp.cos(a), jp.sin(a)])
            info["current_reference_motion"] = self.PRM.get_reference_motion(
                info["command"][0], info["command"][1], info["command"][2], info["imitation_i"]
            )
        else:
            info["imitation_i"] = 0
            info["current_reference_motion"] = jp.zeros(0)

        info["rng"], push1_rng, push2_rng, action_delay_rng = jax.random.split(info["rng"], 4)

        # action delay (same as stock)
        action_history = jp.roll(info["action_history"], self._actuators).at[: self._actuators].set(action)
        info["action_history"] = action_history
        action_idx = jax.random.randint(
            action_delay_rng,
            (1,),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        action_w_delay = action_history.reshape((-1, self._actuators))[action_idx[0]]

        # pushes (same as stock)
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= jp.mod(info["push_step"] + 1, info["push_interval_steps"]) == 0
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        a0 = self._floating_base_qvel_addr
        qvel = qvel.at[a0 : a0 + 2].set(push * push_magnitude + qvel[a0 : a0 + 2])
        state = state.replace(data=state.data.replace(qvel=qvel))

        # policy targets, speed-limited exactly as the MaixCAM2 runtime does (profile motor_speed_limit)
        policy_targets = self._default_actuator + action_w_delay * self._config.action_scale
        prev = info["policy_targets"]
        vmax = self._config.max_motor_velocity * self.dt
        policy_targets = jp.clip(policy_targets, prev - vmax, prev + vmax)
        info["policy_targets"] = policy_targets

        # head commands are applied on top, as on the robot
        motor_targets = policy_targets.at[HEAD].add(info["command"][3:7])

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        info["motor_targets"] = motor_targets  # observed, includes the head offsets

        contact = jp.array([geoms_colliding(data, g, self._floor_geom_id) for g in self._feet_geom_id])
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0.0) * contact_filt
        info["feet_air_time"] += self.dt
        p_fz = data.site_xpos[self._feet_site_id][..., -1]
        info["swing_peak"] = jp.maximum(info["swing_peak"], p_fz)

        obs = self._get_obs(data, info, contact)
        done = self._get_termination(data)

        rewards = self._get_reward(data, action, info, state.metrics, done, first_contact, contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        info["push"] = push
        info["step"] += 1
        info["push_step"] += 1
        info["last_last_last_act"] = info["last_last_act"]
        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        info["step"] = jp.where(done, 0, info["step"])
        info["feet_air_time"] *= ~contact
        info["last_contact"] = contact
        info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            scale = self._config.reward_config.scales[k]
            if scale != 0:
                state.metrics[f"reward/{k}" if scale > 0 else f"cost/{k}"] = v if scale > 0 else -v
        state.metrics["swing_peak"] = jp.mean(info["swing_peak"])

        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype), info=info)

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        rewards["stand_still"] = cost_stand_still(
            info["command"],
            self.get_actuator_joints_qpos(data.qpos),
            self.get_actuator_joints_qvel(data.qvel),
            self._default_actuator,
            ignore_head=True,  # the head is commanded now
        )
        return rewards
