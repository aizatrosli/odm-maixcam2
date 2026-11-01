# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
# Copyright 2026 odm-maixcam2 contributors
#
# Licensed under the Apache License, Version 2.0. Derived from
# Open_Duck_Playground playground/common/randomize.py.
"""Wider domain randomization than stock (stock ranges in comments)."""

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


def domain_randomize_strong(model: mjx.Model, rng: jax.Array):
    dof_id = jp.array([i for i, fr in enumerate(model.dof_hasfrictionloss) if fr])
    jnt_id = model.dof_jntid[dof_id]
    dof_addr = jp.array([a for a in model.jnt_dofadr if a in dof_id])
    joint_addr = model.jnt_qposadr[jnt_id]
    R = RANGES

    @jax.vmap
    def rand(rng):
        k = jax.random.split(rng, 8)
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
        qpos0 = model.qpos0.at[joint_addr].add(
            jax.random.uniform(k[6], shape=(model.nu,), minval=-R["qpos0_jitter"], maxval=R["qpos0_jitter"])
        )
        f = jax.random.uniform(k[7], shape=(model.nu,), minval=R["kp_scale"][0], maxval=R["kp_scale"][1])
        kp = model.actuator_gainprm[:, 0]
        gainprm = model.actuator_gainprm.at[:, 0].set(kp * f)
        biasprm = model.actuator_biasprm.at[:, 1].set(-kp * f)
        return geom_friction, body_ipos, dof_frictionloss, dof_armature, body_mass, qpos0, gainprm, biasprm

    fields = [
        "geom_friction",
        "body_ipos",
        "dof_frictionloss",
        "dof_armature",
        "body_mass",
        "qpos0",
        "actuator_gainprm",
        "actuator_biasprm",
    ]
    values = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({f: 0 for f in fields})
    model = model.tree_replace(dict(zip(fields, values)))
    return model, in_axes
