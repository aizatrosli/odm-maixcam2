"""Closed-loop follow mode in simulation, with the MaixCAM2 camera.

Each 50 Hz control step does what the robot does:

    camera frame (15 Hz, delayed) -> person boxes -> FollowController -> 7 commands
    -> walking policy (ONNX) -> servos

The person boxes come from projecting the simulated person into the
simulated image (with misses and jitter), so no detector model is needed.
Frames, boxes, ground truth, commands and observations can be saved as an
.npz dataset for encoder probes, distillation and DAgger, and an MP4/GIF
shows the head camera next to a third-person view.

    cd Open_Duck_Playground
    PYTHONPATH=$PYTHONPATH:$PWD:/path/to/odm-maixcam2/training:/path/to/odm-maixcam2 \\
      python -m odm_train.camsim.follow_sim BEST_WALK_ONNX_2.onnx --episodes 3 \\
        --video follow.mp4 --record follow.npz

odm-maixcam2's repo root must be on PYTHONPATH too: the controller is the
runtime's own ``odm_maix.vision.follow``.
"""

from __future__ import annotations

from odm_train.camsim import gl

gl.setup()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from typing import Callable, List, Optional  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime  # noqa: E402
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion  # noqa: E402

from odm_maix.vision.follow import Detection, FollowController, FollowParams, pixel_to_angle  # noqa: E402
from odm_train.camsim.camera import MaixCam2  # noqa: E402
from odm_train.camsim.lens import FrameClock  # noqa: E402
from odm_train.camsim.person import PersonWalker, WalkerParams, yaw_quat  # noqa: E402
from odm_train.camsim.scene import CameraConfig, SceneConfig, VisualRandomizer, build_scene  # noqa: E402
from odm_train.sim_eval import CTRL_DT, PRM_PATH, XML, PolicyStepper, Sim, load_profile  # noqa: E402

# Joint conventions measured in the MuJoCo model: +head_yaw turns the camera left
# (matches FollowParams), but +head_pitch tilts it UP, so the sim uses head_pitch_sign = -1.
SIM_HEAD_PITCH_SIGN = -1.0


@dataclass
class DetectorNoise:
    """How the simulated person detector differs from perfect boxes."""

    miss_prob: float = 0.05  # frames where the person is not reported
    jitter: float = 0.01  # std of box centre/size noise, fraction of the image
    min_height: float = 0.06  # smaller boxes (far away) are not detected
    distractor_prob: float = 1.0  # chance the second person is reported when visible


class FollowSim:
    """One duck, one camera, one person to follow (+ an optional distractor)."""

    def __init__(
        self,
        policy_path: str,
        cam: CameraConfig = CameraConfig(),
        scene: SceneConfig = SceneConfig(),
        follow: Optional[dict] = None,
        noise: DetectorNoise = DetectorNoise(),
        walker: WalkerParams = WalkerParams(),
        seed: int = 0,
        latency: int = 2,
        xml: str = XML,
        randomize_dynamics: bool = True,
        randomize_visuals: bool = True,
        render: bool = True,
        encoder: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.cam_cfg, self.scene_cfg, self.noise = cam, scene, noise
        self.rng = np.random.default_rng(seed)
        self.randomize_dynamics, self.randomize_visuals, self.do_render = randomize_dynamics, randomize_visuals, render
        self.encoder = encoder

        # duck sim with the camera scene swapped in (same joints, sensors, actuators)
        self.sim = Sim(xml)
        model, self.info = build_scene(xml, cam, scene)
        model.opt.timestep = self.sim.sim_dt
        self.sim.model, self.sim.data = model, mujoco.MjData(model)
        self.sim._cache_model()
        self.model = model
        self.visuals = VisualRandomizer(model, self.info, scene)
        self.camera = MaixCam2(model, self.info.camera, cam, self.rng)
        self.clock = FrameClock(cam.hz, cam.latency_ms, CTRL_DT)

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        self.policy = onnxruntime.InferenceSession(policy_path, opts, providers=["CPUExecutionProvider"])
        self.profile = load_profile(policy_path)
        self.extra_names: List[str] = list(self.profile.get("extra_obs", []))
        prm = PolyReferenceMotion(PRM_PATH)
        self.stepper = PolicyStepper(self.sim, self.policy, self.profile, prm.nb_steps_in_period, self.rng, latency)

        params = dict(
            hfov_deg=cam.hfov_deg,
            vfov_deg=cam.vfov_deg,
            camera_on_head=cam.mount == "head",
            head_pitch_sign=SIM_HEAD_PITCH_SIGN,
        )
        params.update(follow or {})
        self.follow_params = FollowParams(**params)
        half = scene.room_size / 2
        self.walkers = [PersonWalker(walker, half, follow_duck=True)]
        if scene.distractor:
            self.walkers.append(PersonWalker(walker, half, follow_duck=False))
        self.reset()

    # --- episode
    def reset(self):
        rng = self.rng
        if self.randomize_dynamics:
            self.sim.randomize(rng)
        self.visuals.randomize(rng, self.randomize_visuals)
        self.camera.reset_episode()
        self.sim.reset()
        self.stepper.reset()
        self.controller = FollowController(self.follow_params)
        self.clock.reset(phase=int(rng.integers(0, self.clock.period)))
        xy, yaw = self.duck_pose()
        for w in self.walkers:
            w.reset(rng, xy, yaw)
        self._place_people()
        mujoco.mj_forward(self.model, self.sim.data)
        self.t = 0.0
        self.cmd = np.zeros(7)
        self.latest = None  # last delivered frame packet
        self.fallen = False

    def duck_pose(self):
        d = self.sim.data
        trunk = self.sim.torso
        R = d.xmat[trunk].reshape(3, 3)
        return d.xpos[trunk][:2].copy(), math.atan2(R[1, 0], R[0, 0])

    def _place_people(self):
        d = self.sim.data
        for w, p in zip(self.walkers, self.info.persons):
            x, y, yaw = w.pose
            d.mocap_pos[p.mocap] = [x, y, 0.0]
            d.mocap_quat[p.mocap] = yaw_quat(yaw)

    def truth(self):
        """Privileged target state: distance (m) and bearing (rad, left-positive) from the duck."""
        xy, yaw = self.duck_pose()
        x, y, _ = self.walkers[0].pose
        dx, dy = x - xy[0], y - xy[1]
        bearing = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
        return math.hypot(dx, dy), bearing

    def _detect(self):
        """Simulated detector output on the current frame: target box first, then the distractor."""
        n, rng, d = self.noise, self.rng, self.sim.data
        dets, target_box = [], None
        for i, p in enumerate(self.info.persons):
            b = self.camera.box(d, p)
            if i == 0:
                target_box = b
            if b is None or b[3] < n.min_height or rng.uniform() < n.miss_prob:
                continue
            if i > 0 and rng.uniform() > n.distractor_prob:
                continue
            cx, cy, w, h = np.array(b) + rng.normal(0, n.jitter, 4)
            dets.append(Detection(float(cx), float(cy), float(max(w, 0.01)), float(max(h, 0.01)), score=0.8))
        return dets, target_box

    def extra_obs(self):
        """The runtime's camera features (odm_maix FollowCommands.extra_obs) plus emb_<i>."""
        if not self.extra_names:
            return None
        t = self.controller.target
        visible = t is not None
        bearing = pixel_to_angle(t.cx, self.follow_params.hfov_deg) if visible else 0.0
        feats = {
            "target_visible": float(visible),
            "target_bearing_sin": math.sin(bearing),
            "target_bearing_cos": math.cos(bearing) if visible else 0.0,
            "target_size": t.h if visible else 0.0,
        }
        emb = self.latest["emb"] if (self.latest is not None and self.latest.get("emb") is not None) else None
        out = []
        for name in self.extra_names:
            if name.startswith("emb_"):
                out.append(float(emb[int(name[4:])]) if emb is not None else 0.0)
            else:
                out.append(feats[name])
        return np.array(out, dtype=np.float32)

    def step(self):
        """One 50 Hz control step. Returns a dict describing it."""
        d, rng = self.sim.data, self.rng
        xy, _ = self.duck_pose()
        for w in self.walkers:
            w.step(CTRL_DT, rng, xy)
        self._place_people()
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        mujoco.mj_camlight(self.model, d)

        captured = None
        if self.clock.tick():
            dets, box = self._detect()
            frame = self.camera.render(d) if self.do_render else None
            emb = self.encoder(frame) if (self.encoder is not None and frame is not None) else None
            dist, bearing = self.truth()
            captured = dict(t=self.t, frame=frame, dets=dets, box=box, emb=emb, dist=dist, bearing=bearing)
            self.clock.push(captured)
        ready = self.clock.pop_ready()
        new = ready[-1] if ready else None
        if new is not None:
            self.latest = new
        self.cmd = self.controller.update(new["dets"] if new is not None else [], self.t, CTRL_DT)
        obs = self.stepper.step(self.cmd, self.extra_obs())
        self.t += CTRL_DT
        if not self.sim.upright():
            self.fallen = True
        return dict(t=self.t, cmd=self.cmd.copy(), obs=obs, state=self.controller.state, captured=captured, delivered=new)

    def close(self):
        self.camera.close()


# ---------------------------------------------------------------- recording and video


class Recorder:
    """Collects one row per delivered camera frame and saves an .npz dataset."""

    def __init__(self):
        self.rows = {k: [] for k in ("frame", "box", "box_valid", "dist", "bearing", "cmd", "obs", "emb", "t", "episode")}

    def add(self, episode, info):
        pkt = info["delivered"]
        if pkt is None:
            return
        r = self.rows
        r["frame"].append(pkt["frame"])
        r["box"].append(pkt["box"] if pkt["box"] is not None else (np.nan,) * 4)
        r["box_valid"].append(pkt["box"] is not None)
        r["dist"].append(pkt["dist"])
        r["bearing"].append(pkt["bearing"])
        r["cmd"].append(info["cmd"])
        r["obs"].append(info["obs"])
        r["emb"].append(pkt["emb"])
        r["t"].append(pkt["t"])
        r["episode"].append(episode)

    def save(self, path):
        r = self.rows
        out = {k: np.asarray(v) for k, v in r.items() if k not in ("frame", "emb", "obs")}
        if r["frame"] and r["frame"][0] is not None:
            out["frame"] = np.stack(r["frame"])
        if r["emb"] and r["emb"][0] is not None:
            out["emb"] = np.stack(r["emb"]).astype(np.float32)
        if r["obs"]:
            out["obs"] = np.stack(r["obs"])
        np.savez_compressed(path, **out)
        return len(r["t"])


def draw_box(img, box, rgb=(255, 220, 0), width=2):
    if box is None:
        return img
    h, w = img.shape[:2]
    cx, cy, bw, bh = box
    x0, x1 = int((cx - bw / 2) * w), int((cx + bw / 2) * w)
    y0, y1 = int((cy - bh / 2) * h), int((cy + bh / 2) * h)
    x0, x1, y0, y1 = max(x0, 0), min(x1, w - 1), max(y0, 0), min(y1, h - 1)
    img[y0 : y0 + width, x0:x1] = rgb
    img[y1 - width + 1 : y1 + 1, x0:x1] = rgb
    img[y0:y1, x0 : x0 + width] = rgb
    img[y0:y1, x1 - width + 1 : x1 + 1] = rgb
    return img


class VideoWriter:
    """Head camera (with the controller's current target box) beside a third-person view."""

    def __init__(self, sim: FollowSim, path: str, fps: float):
        import imageio.v2 as iio

        self.sim, self.path = sim, path
        H = sim.cam_cfg.height
        self.view = mujoco.Renderer(sim.model, H, int(H * 4 / 3))
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = sim.sim.torso
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 1.6, -25, 150
        try:
            self.writer = iio.get_writer(path, fps=fps, macro_block_size=1)
        except Exception:  # no ffmpeg: fall back to GIF
            self.path = path.rsplit(".", 1)[0] + ".gif"
            self.writer = iio.get_writer(self.path, duration=1 / fps)

    def add(self, info):
        pkt = info["delivered"]
        if pkt is None or pkt["frame"] is None:
            return
        head = pkt["frame"].copy()
        t = self.sim.controller.target
        if t is not None:
            draw_box(head, (t.cx, t.cy, t.w, t.h))
        self.view.update_scene(self.sim.sim.data, camera=self.cam)
        third = self.view.render()
        self.writer.append_data(np.concatenate([head, third], 1))

    def close(self):
        self.writer.close()
        self.view.close()


def run(args):
    cam = CameraConfig(
        mount=args.mount, width=args.width, height=args.height, hfov_deg=args.hfov, hz=args.hz, latency_ms=args.latency_ms
    )
    scene = SceneConfig(distractor=not args.no_distractor, seed=args.seed)
    sim = FollowSim(
        args.policy,
        cam=cam,
        scene=scene,
        seed=args.seed,
        latency=args.latency,
        randomize_dynamics=not args.no_randomize,
        randomize_visuals=not args.no_randomize,
        render=bool(args.video or args.record) or not args.no_render,
    )
    rec = Recorder() if args.record else None
    video = VideoWriter(sim, args.video, cam.hz) if args.video else None
    results = []
    for ep in range(args.episodes):
        if ep:
            sim.reset()
        steps = int(args.seconds / CTRL_DT)
        seen = in_range = 0
        dists, bearings = [], []
        k = 0
        for k in range(steps):
            info = sim.step()
            if rec:
                rec.add(ep, info)
            if video and ep < args.video_episodes:
                video.add(info)
            dist, bearing = sim.truth()
            dists.append(dist)
            bearings.append(abs(bearing))
            seen += sim.controller.target is not None
            in_range += 0.6 <= dist <= 2.0
            if sim.fallen:
                break
        n = k + 1
        r = dict(
            episode=ep,
            fell=sim.fallen,
            seconds=round(n * CTRL_DT, 1),
            target_seen=round(seen / n, 2),
            in_follow_range=round(in_range / n, 2),
            mean_dist_m=round(float(np.mean(dists)), 2),
            mean_abs_bearing_deg=round(math.degrees(float(np.mean(bearings))), 1),
        )
        results.append(r)
        print(json.dumps(r))
    summary = dict(
        episodes=len(results),
        falls=sum(r["fell"] for r in results),
        target_seen=round(float(np.mean([r["target_seen"] for r in results])), 2),
        in_follow_range=round(float(np.mean([r["in_follow_range"] for r in results])), 2),
        camera=asdict(cam),
    )
    print(json.dumps(summary))
    if rec:
        n = rec.save(args.record)
        print(f"saved {n} frames to {args.record}")
    if video:
        video.close()
        print(f"wrote {video.path}")
    sim.close()
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("policy", help="walking policy .onnx (its .json profile is read if present)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--mount", choices=["head", "trunk"], default="head")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--hfov", type=float, default=70.0)
    p.add_argument("--hz", type=float, default=15.0, help="camera/detection rate")
    p.add_argument("--latency_ms", type=float, default=66.0, help="capture to detection delay")
    p.add_argument("--latency", type=int, default=2, help="max policy action delay, control steps")
    p.add_argument("--no_distractor", action="store_true")
    p.add_argument("--no_randomize", action="store_true", help="no dynamics or appearance randomization")
    p.add_argument("--no_render", action="store_true", help="boxes from geometry only, no pixels (fast)")
    p.add_argument("--record", default=None, help="save frames, boxes, truth, commands, obs to this .npz")
    p.add_argument("--video", default=None, help="write head camera + third-person view (.mp4, or .gif without ffmpeg)")
    p.add_argument("--video_episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
