"""Build the camera scene: the duck + a MaixCAM2 camera + people + a room.

Everything is added to Open_Duck_Playground's MJCF through ``mujoco.MjSpec``
at load time, so no upstream file is edited. New things are visual only
(contype/conaffinity 0) and add no joints, so the duck's qpos/qvel layout,
sensors and actuators are exactly those of the walking scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import mujoco
import numpy as np

CAMERA = "maixcam2"
PERSON = "person"
DISTRACTOR = "person_2"

# camera looks along -z, y up: world frame of a camera that looks along world +x
_R_WORLD_CAM_FORWARD = np.column_stack([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


@dataclass
class CameraConfig:
    """The MaixCAM2 camera. Defaults match odm_maix/vision/follow.py FollowParams."""

    mount: str = "head"  # "head" (moves with head yaw/pitch) or "trunk" (camera_on_head: false)
    width: int = 320
    height: int = 240
    hfov_deg: float = 70.0  # horizontal FOV of the 4:3 frame, pinhole part of the lens model
    pitch_deg: float = 0.0  # extra downward tilt of the mount
    # where the lens sits, in world coordinates with the duck in its "home" pose
    head_pos: Tuple[float, float, float] = (0.075, 0.0, 0.36)
    trunk_pos: Tuple[float, float, float] = (0.10, 0.0, 0.22)
    # timing
    hz: float = 15.0  # frames per second the vision pipeline delivers
    latency_ms: float = 66.0  # capture -> detection available to the controller
    # image model
    k1: float = -0.08  # radial distortion (negative = barrel, as on wide lenses)
    k2: float = 0.0
    noise_std: float = 2.5  # sensor noise, in 0..255 units
    exposure_jitter: float = 0.15  # +- fraction per episode
    wb_jitter: float = 0.06  # +- per colour channel per episode
    exposure_ms: float = 8.0  # motion blur length = camera turn rate x exposure (0 = off)
    # The camera module's weight (board, lens, mount, cable), added as a box behind the lens on the
    # mount body. The stock duck model has no camera. 0 = leave the model's masses as they are.
    mass_g: float = 0.0
    module_size: Tuple[float, float, float] = (0.03, 0.04, 0.04)  # depth, width, height (m)

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(math.tan(math.radians(self.hfov_deg) / 2) * self.height / self.width))


@dataclass
class SceneConfig:
    room_size: float = 8.0  # square room, metres
    n_props: int = 8  # boxes and cylinders (furniture) placed per episode
    n_floor_textures: int = 5
    distractor: bool = True  # a second person who wanders around
    seed: int = 0  # build-time choices (texture colours)


@dataclass
class PersonParts:
    body: int
    mocap: int
    geoms: dict = field(default_factory=dict)  # part -> list of geom ids


@dataclass
class SceneInfo:
    camera: int
    camera_body: int
    persons: List[PersonParts]
    walls: List[int]
    props: List[int]
    floor: int
    floor_materials: List[int]
    lights: List[int]


def _home_state(spec: mujoco.MjSpec):
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.keyframe("home").id)
    mujoco.mj_forward(m, d)
    return m, d


def _add_person(spec: mujoco.MjSpec, name: str, pos) -> None:
    """A 1.7 m person made of primitives, facing +x of its own frame.

    Sizes are for a 1.7 m person; per-episode height scaling is done at run time
    by moving the geoms (see VisualRandomizer.person_height)."""
    b = spec.worldbody.add_body(name=name, mocap=True, pos=list(pos))
    cap, box, sph = mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE
    vis = dict(contype=0, conaffinity=0, group=1)
    b.add_geom(name=f"{name}_leg_l", type=cap, fromto=[0, 0.09, 0.06, 0, 0.09, 0.82], size=[0.065, 0, 0], **vis)
    b.add_geom(name=f"{name}_leg_r", type=cap, fromto=[0, -0.09, 0.06, 0, -0.09, 0.82], size=[0.065, 0, 0], **vis)
    b.add_geom(name=f"{name}_shoe_l", type=box, pos=[0.04, 0.09, 0.03], size=[0.12, 0.05, 0.03], **vis)
    b.add_geom(name=f"{name}_shoe_r", type=box, pos=[0.04, -0.09, 0.03], size=[0.12, 0.05, 0.03], **vis)
    b.add_geom(name=f"{name}_torso", type=box, pos=[0, 0, 1.13], size=[0.12, 0.2, 0.3], **vis)
    b.add_geom(name=f"{name}_arm_l", type=cap, fromto=[0, 0.25, 1.38, 0.03, 0.27, 0.85], size=[0.05, 0, 0], **vis)
    b.add_geom(name=f"{name}_arm_r", type=cap, fromto=[0, -0.25, 1.38, 0.03, -0.27, 0.85], size=[0.05, 0, 0], **vis)
    b.add_geom(name=f"{name}_neck", type=cap, fromto=[0, 0, 1.42, 0, 0, 1.5], size=[0.05, 0, 0], **vis)
    b.add_geom(name=f"{name}_head", type=sph, pos=[0, 0, 1.6], size=[0.11, 0, 0], **vis)


MOUNT_BODY = {"head": "head_assembly", "trunk": "trunk_assembly"}


def camera_mount(model: mujoco.MjModel, cam: CameraConfig):
    """Where the camera sits on its mount body: (body id, lens position and camera orientation in that body's frame).

    ``cam.head_pos`` / ``cam.trunk_pos`` are world positions in the "home" pose; the camera looks along
    world +x there, tilted by ``cam.pitch_deg``. Also used by the trainer (randomize.py) to add the
    module's weight at the same place."""
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.keyframe("home").id)
    mujoco.mj_forward(model, d)
    bid = model.body(MOUNT_BODY[cam.mount]).id
    R_wb = d.xmat[bid].reshape(3, 3)
    tilt = math.radians(cam.pitch_deg)
    # tilt down = rotate the camera about its own x axis
    R_tilt = np.array([[1, 0, 0], [0, math.cos(tilt), math.sin(tilt)], [0, -math.sin(tilt), math.cos(tilt)]])
    R_bc = R_wb.T @ _R_WORLD_CAM_FORWARD @ R_tilt.T
    p_world = np.array(cam.head_pos if cam.mount == "head" else cam.trunk_pos)
    return bid, R_wb.T @ (p_world - d.xpos[bid]), R_bc


def module_center(cam: CameraConfig, p_body: np.ndarray, R_bc: np.ndarray) -> np.ndarray:
    """Centre of the camera module (mount body frame): just behind the lens, along the optical axis."""
    return p_body + R_bc @ np.array([0.0, 0.0, cam.module_size[0] / 2])  # camera +z points backwards


def build_scene(xml_path: str, cam: CameraConfig, cfg: SceneConfig):
    """Return (model, SceneInfo) for the duck scene at ``xml_path`` plus camera, people and room."""
    spec = mujoco.MjSpec.from_file(xml_path)
    m0, _ = _home_state(spec)
    rng = np.random.default_rng(cfg.seed)

    # --- camera, rigidly attached to the head or the trunk
    body_name = MOUNT_BODY[cam.mount]
    _, p_body, R_bc = camera_mount(m0, cam)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_bc.flatten())
    spec.visual.map.znear = 0.002  # relative to the scene extent: lets a person right in front of the lens render
    spec.body(body_name).add_camera(name=CAMERA, pos=p_body.tolist(), quat=q.tolist(), fovy=cam.vfov_deg)
    if cam.mass_g > 0:
        depth, width, height = cam.module_size
        # its own welded body: the duck's bodies have explicit inertia, so a geom's mass on them would be ignored
        module = spec.body(body_name).add_body(name="maixcam2_module", pos=module_center(cam, p_body, R_bc).tolist(), quat=q.tolist())
        module.add_geom(
            name="maixcam2_module", type=mujoco.mjtGeom.mjGEOM_BOX, size=[width / 2, height / 2, depth / 2],
            mass=cam.mass_g / 1000, contype=0, conaffinity=0, rgba=[0.12, 0.12, 0.13, 1],
        )

    # --- floors: several textures to swap per episode
    floor_mats = []
    for i in range(cfg.n_floor_textures):
        kind = [mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mujoco.mjtBuiltin.mjBUILTIN_FLAT, mujoco.mjtBuiltin.mjBUILTIN_GRADIENT][i % 3]
        c1 = rng.uniform(0.15, 0.85, 3)
        c2 = np.clip(c1 + rng.uniform(-0.25, 0.25, 3), 0, 1)
        spec.add_texture(
            name=f"camsim_floor_{i}",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=kind,
            rgb1=c1.tolist(),
            rgb2=c2.tolist(),
            mark=mujoco.mjtMark.mjMARK_RANDOM if kind == mujoco.mjtBuiltin.mjBUILTIN_FLAT else mujoco.mjtMark.mjMARK_NONE,
            random=0.05,
            markrgb=(c1 * 0.8).tolist(),
            width=256,
            height=256,
        )
        mat = spec.add_material(name=f"camsim_floor_{i}", texrepeat=[rng.uniform(2, 10)] * 2, texuniform=True)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"camsim_floor_{i}"
        floor_mats.append(f"camsim_floor_{i}")

    # --- people
    _add_person(spec, PERSON, [1.5, 0, 0])
    if cfg.distractor:
        _add_person(spec, DISTRACTOR, [3.0, 2.0, 0])

    # --- room: walls and props, visual only, in one static body so geom_pos is world position
    room = spec.worldbody.add_body(name="camsim_room")
    vis = dict(contype=0, conaffinity=0, group=1)
    half = cfg.room_size / 2
    for i, (x, y, sx, sy) in enumerate([(half, 0, 0.05, half), (-half, 0, 0.05, half), (0, half, half, 0.05), (0, -half, half, 0.05)]):
        room.add_geom(name=f"camsim_wall_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, 1.2], size=[sx, sy, 1.2], **vis)
    for i in range(cfg.n_props):
        t = mujoco.mjtGeom.mjGEOM_BOX if i % 2 == 0 else mujoco.mjtGeom.mjGEOM_CYLINDER
        room.add_geom(name=f"camsim_prop_{i}", type=t, pos=[half + 5, 0, 0.3], size=[0.3, 0.3, 0.3], **vis)

    # --- lights (the scene only has the headlight)
    for i in range(2):
        spec.worldbody.add_light(name=f"camsim_light_{i}", pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.5] * 3, castshadow=i == 0)

    model = spec.compile()
    persons = []
    for name in [PERSON] + ([DISTRACTOR] if cfg.distractor else []):
        b = model.body(name).id
        parts = {}
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == b:
                part = model.geom(g).name[len(name) + 1 :].split("_")[0]  # leg, shoe, torso, arm, neck, head
                parts.setdefault(part, []).append(g)
        persons.append(PersonParts(body=b, mocap=model.body_mocapid[b], geoms=parts))
    info = SceneInfo(
        camera=model.camera(CAMERA).id,
        camera_body=model.body(body_name).id,
        persons=persons,
        walls=[model.geom(f"camsim_wall_{i}").id for i in range(4)],
        props=[model.geom(f"camsim_prop_{i}").id for i in range(cfg.n_props)],
        floor=model.geom("floor").id,
        floor_materials=[model.material(n).id for n in floor_mats],
        lights=[model.light(f"camsim_light_{i}").id for i in range(2)],
    )
    return model, info


class VisualRandomizer:
    """Per-episode appearance randomization: floor, walls, props, people, lights."""

    def __init__(self, model: mujoco.MjModel, info: SceneInfo, cfg: SceneConfig):
        self.model, self.info, self.cfg = model, info, cfg
        self.base_geom_pos = model.geom_pos.copy()
        self.base_geom_size = model.geom_size.copy()

    def person_height(self, person: PersonParts, height: float):
        """Scale a person's geoms to ``height`` metres (built at 1.7 m)."""
        s = height / 1.7
        for ids in person.geoms.values():
            for g in ids:
                self.model.geom_pos[g] = self.base_geom_pos[g] * s
                self.model.geom_size[g] = self.base_geom_size[g] * s

    def randomize(self, rng: np.random.Generator, enabled: bool = True):
        m, info = self.model, self.info
        if not enabled:
            return
        if info.floor_materials:
            m.geom_matid[info.floor] = rng.choice(info.floor_materials)
            m.geom_rgba[info.floor] = [1, 1, 1, 1]
        wall = rng.uniform(0.3, 0.95, 3)
        for g in info.walls:
            m.geom_rgba[g, :3] = np.clip(wall + rng.uniform(-0.05, 0.05, 3), 0, 1)
        half = self.cfg.room_size / 2 - 0.5
        for g in info.props:
            r = rng.uniform(1.0, half)
            a = rng.uniform(-math.pi, math.pi)
            h = rng.uniform(0.2, 0.9)
            m.geom_pos[g] = [r * math.cos(a), r * math.sin(a), h]
            m.geom_size[g] = [rng.uniform(0.15, 0.5), rng.uniform(0.15, 0.5), h]
            m.geom_rgba[g] = [*rng.uniform(0.1, 0.9, 3), 1]
        skin = np.array([0.95, 0.8, 0.65]) * rng.uniform(0.45, 1.0)
        for p in info.persons:
            self.person_height(p, rng.uniform(1.55, 1.9))
            shirt, pants, shoes = rng.uniform(0.05, 0.95, (3, 3))
            for part, rgb in [("torso", shirt), ("arm", shirt), ("leg", pants), ("shoe", shoes * 0.4), ("head", skin), ("neck", skin)]:
                for g in p.geoms.get(part, []):
                    m.geom_rgba[g] = [*rgb, 1]
        for i, light in enumerate(info.lights):
            a = rng.uniform(-math.pi, math.pi)
            m.light_pos[light] = [3 * math.cos(a), 3 * math.sin(a), rng.uniform(2.0, 3.5)]
            m.light_dir[light] = -m.light_pos[light] / np.linalg.norm(m.light_pos[light])
            m.light_diffuse[light] = [rng.uniform(0.2, 0.7)] * 3 * rng.uniform(0.85, 1.15, 3)
        m.vis.headlight.ambient[:] = [rng.uniform(0.15, 0.4)] * 3
        m.vis.headlight.diffuse[:] = [rng.uniform(0.2, 0.6)] * 3
