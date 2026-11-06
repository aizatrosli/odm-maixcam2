"""The trainer's camera weight (randomize.py) against MuJoCo's own result for a separate module body.

Needs jax, mujoco and an Open_Duck_Playground checkout:
    ODP_DIR=/path/to/Open_Duck_Playground pytest tests/test_randomize_camera.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

ODP = os.environ.get("ODP_DIR")
pytestmark = pytest.mark.skipif(not ODP, reason="needs ODP_DIR (training stack)")


@pytest.fixture
def odp(monkeypatch):
    from odm_train.camsim import gl

    gl.setup()
    pytest.importorskip("mujoco")
    pytest.importorskip("jax")
    monkeypatch.chdir(ODP)
    monkeypatch.syspath_prepend(ODP)
    from odm_train.sim_eval import XML

    return XML


def _home_mass_matrix(model):
    import mujoco

    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.keyframe("home").id)
    mujoco.mj_forward(model, d)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, d.qM)
    return M, d.subtree_com[0].copy()


@pytest.mark.parametrize("mount", ["head", "trunk"])
def test_point_mass_matches_a_separate_module_body(odp, mount):
    import mujoco

    from odm_train.camsim.scene import CameraConfig, camera_mount, module_center
    from odm_train.randomize import add_point_mass, camera_module

    m_cam = 0.15
    base = mujoco.MjModel.from_xml_path(odp)
    cam = camera_module(base, mount, (150, 150))
    b = cam["body"]

    # trainer: fold the module into the mount body
    ours = mujoco.MjModel.from_xml_path(odp)
    M, c, inertia = add_point_mass(
        ours.body_mass[b], ours.body_ipos[b], ours.body_inertia[b], ours.body_iquat[b], m_cam, np.array(cam["pos"])
    )
    ours.body_mass[b], ours.body_ipos[b], ours.body_inertia[b] = float(M), np.asarray(c), np.asarray(inertia)
    # CPU MuJoCo caches subtree masses at compile time (MJX recomputes them each step)
    mujoco.mj_setConst(ours, mujoco.MjData(ours))

    # MuJoCo: the module as its own (tiny) welded body, as the camera simulator builds it
    spec = mujoco.MjSpec.from_file(odp)
    cfg = CameraConfig(mount=mount)
    _, p_lens, R_bc = camera_mount(base, cfg)
    body = spec.body({"head": "head_assembly", "trunk": "trunk_assembly"}[mount])
    mod = body.add_body(name="module", pos=module_center(cfg, p_lens, R_bc).tolist())
    mod.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1e-3, 0, 0], mass=m_cam, contype=0, conaffinity=0)
    ref = spec.compile()

    M_ours, com_ours = _home_mass_matrix(ours)
    M_ref, com_ref = _home_mass_matrix(ref)
    assert abs(ours.body_subtreemass[0] - ref.body_subtreemass[0]) < 1e-9
    assert np.abs(com_ours - com_ref).max() < 1e-6
    # only the parallel-axis terms off the inertial frame's diagonal are dropped
    assert np.abs(M_ours - M_ref).max() / np.abs(M_ref).max() < 0.01


def test_randomizer_adds_the_mass_in_range(odp):
    import jax
    import mujoco
    from mujoco import mjx

    from odm_train.randomize import camera_module, domain_randomize_strong

    base = mujoco.MjModel.from_xml_path(odp)
    cam = camera_module(base, "head", (60, 200))
    model = mjx.put_model(base)
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    with_cam, _ = domain_randomize_strong(model, keys, camera=cam)
    without, _ = domain_randomize_strong(model, keys)
    b = cam["body"]
    head = base.body_mass[b]
    got = np.asarray(with_cam.body_mass[:, b])
    # link masses are scaled 0.9-1.1, then 60-200 g are added
    assert got.min() >= 0.9 * head + 0.06 - 1e-6 and got.max() <= 1.1 * head + 0.2 + 1e-6
    assert 0.09 < got.mean() - np.asarray(without.body_mass[:, b]).mean() < 0.17
    assert np.asarray(with_cam.body_ipos[:, b]).std(0).max() > 1e-4  # the centre of mass moves with the draw
