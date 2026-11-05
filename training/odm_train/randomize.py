# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
# Copyright 2026 odm-maixcam2 contributors
#
# Licensed under the Apache License, Version 2.0. Derived from
# Open_Duck_Playground playground/common/randomize.py.
"""Wider domain randomization than stock (stock ranges in comments).

Optionally adds the MaixCAM2 module's weight to the head (or trunk) with a
random mass each env, at the same place as the camera simulator puts it
(``camera_module``). Pass the result to ``domain_randomize_strong(camera=...)``.
"""

import functools
from typing import Optional

import jax
import jax.numpy as jp
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1

RANGES = dict(
    floor_friction=(0.4, 1.2),  # stock (0.5, 1.0)
    frictionloss_scale=(0.8, 1.2),  # stock (0.9, 1.1)
    armature_scale=(1.0, 1.1),  # stock (1.0, 1.05)
    com_offset=0.05,  # same as stock
    link_mass_scale=(0.9, 1.1),  # same as stock
    torso_mass_add=(-0.15, 0.25),  # stock (-0.1, 0.1); the MaixCAM2 + mount is heavier than a Pi Zero
    qpos0_jitter=0.04,  # stock 0.03 (calibration offsets)
    kp_scale=(0.8, 1.2),  # stock (0.9, 1.1); battery sag changes servo stiffness
)


def camera_module(mj_model, mount: str = "head", mass_g=(60.0, 200.0)) -> dict:
    """Where and how heavy the camera module is, for ``domain_randomize_strong(camera=...)``."""
    from odm_train.camsim.scene import CameraConfig, camera_mount, module_center

    cam = CameraConfig(mount=mount)
    bid, p_lens, R_bc = camera_mount(mj_model, cam)
    lo, hi = float(mass_g[0]), float(mass_g[1])
    assert 0 <= lo <= hi, "camera mass range must be 0 <= lo <= hi (grams)"
    return dict(body=int(bid), pos=[float(v) for v in module_center(cam, p_lens, R_bc)], mass_kg=(lo / 1000, hi / 1000), mount=mount)


def _quat_to_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def add_point_mass(mass, ipos, inertia, iquat, m, p):
    """A body (mass, centre of mass ``ipos``, diagonal ``inertia`` in its ``iquat`` frame) plus a point
    mass ``m`` at ``p`` (body frame) -> the combined (mass, ipos, inertia).

    The inertial frame keeps its orientation, so the parallel-axis terms off its diagonal are dropped.
    For a module of up to a few hundred grams a few cm from the head's centre this changes the
    mass matrix by well under 1 % (tests/test_randomize_camera.py checks it against MuJoCo)."""
    M = mass + m
    c = (mass * ipos + m * p) / M
    R = _quat_to_mat(iquat)  # columns = inertial axes in the body frame
    d = R.T @ (ipos - c)
    r = R.T @ (p - c)

    def shift(v):
        return jp.sum(v * v) - v * v

    return M, c, inertia + mass * shift(d) + m * shift(r)


def domain_randomize_strong(model: mjx.Model, rng: jax.Array, camera: Optional[dict] = None):
    dof_id = jp.array([i for i, fr in enumerate(model.dof_hasfrictionloss) if fr])
    jnt_id = model.dof_jntid[dof_id]
    dof_addr = jp.array([a for a in model.jnt_dofadr if a in dof_id])
    joint_addr = model.jnt_qposadr[jnt_id]
    R = RANGES

    @jax.vmap
    def rand(rng):
        k = jax.random.split(rng, 9 if camera is not None else 8)  # 8: same draws as before without a camera
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(k[0], minval=R["floor_friction"][0], maxval=R["floor_friction"][1])
        )
        fl = model.dof_frictionloss[dof_addr] * jax.random.uniform(
            k[1], shape=(model.nu,), minval=R["frictionloss_scale"][0], maxval=R["frictionloss_scale"][1]
        )
        dof_frictionloss = model.dof_frictionloss.at[dof_addr].set(fl)
        arm = model.dof_armature[dof_addr] * jax.random.uniform(
            k[2], shape=(model.nu,), minval=R["armature_scale"][0], maxval=R["armature_scale"][1]
        )
        dof_armature = model.dof_armature.at[dof_addr].set(arm)
        dpos = jax.random.uniform(k[3], (3,), minval=-R["com_offset"], maxval=R["com_offset"])
        body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(model.body_ipos[TORSO_BODY_ID] + dpos)
        dmass = jax.random.uniform(
            k[4], shape=(model.nbody,), minval=R["link_mass_scale"][0], maxval=R["link_mass_scale"][1]
        )
        body_mass = model.body_mass * dmass
        body_mass = body_mass.at[TORSO_BODY_ID].add(
            jax.random.uniform(k[5], minval=R["torso_mass_add"][0], maxval=R["torso_mass_add"][1])
        )
        body_inertia = model.body_inertia
        if camera is not None:
            b = camera["body"]
            m_cam = jax.random.uniform(k[8], minval=camera["mass_kg"][0], maxval=camera["mass_kg"][1])
            M, c, I = add_point_mass(
                body_mass[b], body_ipos[b], body_inertia[b], model.body_iquat[b], m_cam, jp.array(camera["pos"])
            )
            body_mass = body_mass.at[b].set(M)
            body_ipos = body_ipos.at[b].set(c)
            body_inertia = body_inertia.at[b].set(I)
        qpos0 = model.qpos0.at[joint_addr].add(
            jax.random.uniform(k[6], shape=(model.nu,), minval=-R["qpos0_jitter"], maxval=R["qpos0_jitter"])
        )
        f = jax.random.uniform(k[7], shape=(model.nu,), minval=R["kp_scale"][0], maxval=R["kp_scale"][1])
        kp = model.actuator_gainprm[:, 0]
        gainprm = model.actuator_gainprm.at[:, 0].set(kp * f)
        biasprm = model.actuator_biasprm.at[:, 1].set(-kp * f)
        return geom_friction, body_ipos, dof_frictionloss, dof_armature, body_mass, body_inertia, qpos0, gainprm, biasprm

    fields = [
        "geom_friction",
        "body_ipos",
        "dof_frictionloss",
        "dof_armature",
        "body_mass",
        "body_inertia",
        "qpos0",
        "actuator_gainprm",
        "actuator_biasprm",
    ]
    values = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({f: 0 for f in fields})
    model = model.tree_replace(dict(zip(fields, values)))
    return model, in_axes


def with_camera(mj_model, mount: str = "head", mass_g=(60.0, 200.0)):
    """``domain_randomize_strong`` with the camera module's weight on ``mount``."""
    return functools.partial(domain_randomize_strong, camera=camera_module(mj_model, mount, mass_g))
