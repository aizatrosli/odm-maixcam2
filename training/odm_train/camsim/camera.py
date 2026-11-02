"""The MaixCAM2 camera in MuJoCo: lens, sensor and the person's box.

``MaixCam2.render`` returns what the camera would deliver (uint8 RGB at the
configured resolution). ``MaixCam2.box`` projects a person into the same
distorted image and returns the normalised box (cx, cy, w, h), so detections
and pixels agree.
"""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from odm_train.camsim.lens import FrameClock, Lens, hblur  # noqa: F401  (FrameClock re-exported)
from odm_train.camsim.scene import CameraConfig, PersonParts


class MaixCam2:
    def __init__(self, model: mujoco.MjModel, cam_id: int, cfg: CameraConfig, rng: Optional[np.random.Generator] = None):
        self.model, self.cam_id, self.cfg = model, cam_id, cfg
        self.rng = rng or np.random.default_rng(0)
        self.lens = Lens(cfg.width, cfg.height, cfg.hfov_deg, cfg.k1, cfg.k2)
        model.cam_fovy[cam_id] = self.lens.render_fovy_deg
        self.renderer = mujoco.Renderer(model, self.lens.rh, self.lens.rw)
        self.reset_episode()

    def reset_episode(self):
        c, rng = self.cfg, self.rng
        self.gain = 1 + rng.uniform(-c.exposure_jitter, c.exposure_jitter)
        self.wb = 1 + rng.uniform(-c.wb_jitter, c.wb_jitter, 3)

    # --- geometry
    def cam_pose(self, data):
        return data.cam_xpos[self.cam_id].copy(), data.cam_xmat[self.cam_id].reshape(3, 3).copy()

    def project(self, data, pts_world: np.ndarray):
        """World points (N,3) -> normalised image coords (N,2) in the delivered (distorted) image,
        and a mask of points in front of the camera."""
        p, R = self.cam_pose(data)
        pc = (np.asarray(pts_world) - p) @ R  # camera frame: x right, y up, -z forward
        depth = -pc[:, 2]
        front = depth > 0.05
        x = pc[:, 0] / np.maximum(depth, 1e-6)
        y = -pc[:, 1] / np.maximum(depth, 1e-6)  # image y points down
        u, v = self.lens.to_image(x, y)
        return np.stack([u, v], 1), front

    def box(self, data, person: PersonParts, min_visible: float = 0.25):
        """Normalised (cx, cy, w, h) box of a person, clipped to the frame, or None when
        less than ``min_visible`` of its sample points are in view (occlusion is ignored)."""
        pts = person_points(self.model, data, person)
        uv, front = self.project(data, pts)
        inside = front & (uv[:, 0] >= 0) & (uv[:, 0] <= 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= 1)
        if inside.mean() < min_visible:
            return None
        uv = uv[front]
        x0, y0 = np.clip(uv.min(0), 0, 1)
        x1, y1 = np.clip(uv.max(0), 0, 1)
        if x1 - x0 < 0.01 or y1 - y0 < 0.01:
            return None
        return float((x0 + x1) / 2), float((y0 + y1) / 2), float(x1 - x0), float(y1 - y0)

    # --- pixels
    def render(self, data, scene_option=None) -> np.ndarray:
        r = self.renderer
        r.update_scene(data, camera=self.cam_id, scene_option=scene_option)
        img = self.lens.resample(r.render().astype(np.float32))
        img = img * self.gain * self.wb
        c = self.cfg
        if c.exposure_ms > 0:
            # blur along the image rows by how far the view turns during the exposure
            omega = float(np.linalg.norm(_cam_angvel(self.model, data, self.cam_id)))
            img = hblur(img, int(round(min(25.0, omega * c.exposure_ms / 1000 * self.lens.f))))
        if c.noise_std > 0:
            img = img + self.rng.normal(0, c.noise_std, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def close(self):
        self.renderer.close()


def person_points(model, data, person: PersonParts) -> np.ndarray:
    """Corners of each of a person's geoms (world coordinates): the silhouette's extent."""
    pts = []
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    for ids in person.geoms.values():
        for g in ids:
            t, s = model.geom_type[g], model.geom_size[g]
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                ext = s
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:  # axis along local z
                ext = np.array([s[0], s[0], s[1] + s[0]])
            else:  # sphere
                ext = np.array([s[0]] * 3)
            R = data.geom_xmat[g].reshape(3, 3)
            pts.append(data.geom_xpos[g] + (signs * ext) @ R.T)
    return np.concatenate(pts)


def _cam_angvel(model, data, cam_id):
    """Angular velocity of the camera (world frame)."""
    v = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_CAMERA, cam_id, v, 0)
    return v[:3]
