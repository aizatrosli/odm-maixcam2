"""Interactive follow-mode simulator in the browser (Viser + mjviser).

    python -m odm_train.camsim.app policy.onnx            # http://localhost:8080, gamepad page on :8081

- 3D view of the duck, the person and the room. Drag a body with the mouse to
  push it (mjviser's perturbation), pause, step, change speed.
- The MaixCAM2 image as the robot gets it (15 Hz, 66 ms late), with the
  controller's target box, and live plots of distance, bearing and commands.
- Modes: follow the person with the camera, drive the duck yourself, or walk
  the person around yourself and watch the duck follow.
- Follow-controller and detector settings as sliders, applied immediately.

The gamepad and held keys come through a small page on its own port (see
pad.py), because Viser only reports single key presses.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque

from odm_train.camsim import gl

gl.setup()  # before mujoco is imported

import numpy as np  # noqa: E402

from odm_train.camsim.follow_sim import FollowSim, draw_box  # noqa: E402
from odm_train.camsim.pad import PadServer  # noqa: E402
from odm_train.camsim.scene import CameraConfig, SceneConfig  # noqa: E402
from odm_train.sim_eval import CTRL_DT  # noqa: E402

MODES = ("Follow the person", "Drive the duck", "Walk the person")
PERSON_SPEED = 0.8  # m/s at full stick in "Walk the person"
PERSON_TURN = 2.0  # rad/s
PLOT_SECONDS = 10.0


class FollowApp:
    def __init__(self, sim: FollowSim, pad: PadServer, host: str = "0.0.0.0", port: int = 8080, pad_url: str = ""):
        import viser
        from mjviser import Viewer

        self.sim, self.pad, self.pad_url = sim, pad, pad_url
        self.server = viser.ViserServer(host=host, port=port)
        self.server.gui.configure_theme(control_width="large")
        self.substeps = int(round(CTRL_DT / sim.model.opt.timestep))
        self._n = 0
        self.mode = MODES[0]
        self.auto_reset = True
        self._fell_at = None
        n = int(PLOT_SECONDS / CTRL_DT)
        self.hist = {k: deque(maxlen=n) for k in ("t", "dist", "bearing", "vx", "wz", "seen")}
        self._last_plot = 0.0
        self._build_gui(viser)
        self.viewer = Viewer(sim.model, sim.sim.data, step_fn=self._physics_step, reset_fn=self._reset, server=self.server)
        # start close behind the duck, looking past it toward the person
        self.server.initial_camera.position = (-1.3, -0.9, 0.9)
        self.server.initial_camera.look_at = (0.6, 0.0, 0.15)

    # --- GUI
    def _build_gui(self, viser):
        gui, sim = self.server.gui, self.sim
        cam = sim.cam_cfg
        with gui.add_folder("MaixCAM2 camera"):
            self.image = gui.add_image(np.zeros((cam.height, cam.width, 3), np.uint8), label=None, format="jpeg", jpeg_quality=85)
            self.readout = gui.add_html("")
        with gui.add_folder("Plots", expand_by_default=True):
            self.plot_dist = gui.add_uplot(
                data=(np.zeros(1), np.zeros(1), np.zeros(1)),
                series=(viser.uplot.Series(label="t"), viser.uplot.Series(label="distance (m)", stroke="#2f7d4f", width=2),
                        viser.uplot.Series(label="bearing (rad)", stroke="#c2662d", width=2)),
                title="Person", aspect=2.2, scales={"x": {"time": False}},
            )
            self.plot_cmd = gui.add_uplot(
                data=(np.zeros(1), np.zeros(1), np.zeros(1)),
                series=(viser.uplot.Series(label="t"), viser.uplot.Series(label="vx (m/s)", stroke="#3b6fb6", width=2),
                        viser.uplot.Series(label="wz (rad/s)", stroke="#9a4fb0", width=2)),
                title="Commands", aspect=2.2, scales={"x": {"time": False}},
            )
        with gui.add_folder("Mode"):
            self.mode_dd = gui.add_dropdown("Mode", options=MODES, initial_value=MODES[0])
            self.mode_dd.on_update(lambda _: self._set_mode(self.mode_dd.value))
            gui.add_markdown(
                f"Gamepad and keyboard: open port {self.pad.port} on this same address in another tab ([{self.pad_url}]({self.pad_url}) on this machine). "
                "**A** / Space switches follow and drive, **Start** / Enter resets."
            )
            self.pad_status = gui.add_html("")
            auto = gui.add_checkbox("Reset after a fall", initial_value=True)
            auto.on_update(lambda _: setattr(self, "auto_reset", auto.value))
            stop = gui.add_checkbox("Person stands still", initial_value=False)
            stop.on_update(lambda _: setattr(sim.walkers[0], "manual", stop.value or self.mode == MODES[2]))
            self.stop_cb = stop
        p = sim.follow_params
        with gui.add_folder("Follow controller", expand_by_default=False):
            self._param_slider("Box height to keep", "target_height", 0.2, 0.9, 0.01)
            self._param_slider("Stop above box height", "stop_height", 0.4, 1.0, 0.01)
            self._param_slider("Turn gain", "k_turn", 0.0, 4.0, 0.1)
            self._param_slider("Forward gain", "k_forward", 0.0, 2.0, 0.05)
            self._param_slider("Walk only within (deg)", "walk_bearing_deg", 5, 90, 1)
            self._param_slider("Head share of bearing", "k_head", 0.0, 1.0, 0.05)
            ht = gui.add_checkbox("Head tracks the person", initial_value=p.head_track)
            ht.on_update(lambda _: setattr(p, "head_track", ht.value))
        with gui.add_folder("Detector and person", expand_by_default=False):
            n = sim.noise
            miss = gui.add_slider("Missed detections", 0.0, 0.9, 0.05, n.miss_prob)
            miss.on_update(lambda _: setattr(n, "miss_prob", miss.value))
            jit = gui.add_slider("Box jitter", 0.0, 0.05, 0.002, n.jitter)
            jit.on_update(lambda _: setattr(n, "jitter", jit.value))
            w = sim.walkers[0].p
            sp = gui.add_multi_slider("Person speed (m/s)", 0.0, 1.2, 0.05, w.speed)
            sp.on_update(lambda _: setattr(w, "speed", tuple(sp.value)))

    def _param_slider(self, label, field, lo, hi, step):
        p = self.sim.follow_params
        s = self.server.gui.add_slider(label, lo, hi, step, float(getattr(p, field)))
        s.on_update(lambda _: setattr(p, field, s.value))

    def _set_mode(self, mode):
        self.mode = mode
        self.sim.walkers[0].manual = mode == MODES[2] or self.stop_cb.value
        if self.mode_dd.value != mode:
            self.mode_dd.value = mode

    # --- simulation (called by mjviser once per physics step, under its lock)
    def _physics_step(self, model, data):
        self._n += 1
        if self._n % self.substeps:
            return
        self._control_step()

    def _reset(self, model=None, data=None):
        self.sim.reset()
        self._n = 0
        self._fell_at = None
        for h in self.hist.values():
            h.clear()

    def _control_step(self):
        sim, pad = self.sim, self.pad
        pressed = pad.pop_buttons()
        if "start" in pressed:
            self._reset()
            return
        if "a" in pressed:
            self._set_mode(MODES[1] if self.mode == MODES[0] else MODES[0])

        if self.mode == MODES[1]:
            sim.cmd_override = pad.cmd()
        else:
            sim.cmd_override = None
        if self.mode == MODES[2]:
            lx, ly, rx, _ = pad.axes()
            w = sim.walkers[0]
            x, y, yaw = w.pose
            yaw -= rx * PERSON_TURN * CTRL_DT
            fwd, side = -ly * PERSON_SPEED * CTRL_DT, -lx * PERSON_SPEED * CTRL_DT
            x += fwd * math.cos(yaw) - side * math.sin(yaw)
            y += fwd * math.sin(yaw) + side * math.cos(yaw)
            lim = w.half
            w.set_pose(np.clip(x, -lim, lim), np.clip(y, -lim, lim), yaw)

        info = sim.step()
        dist, bearing = sim.truth()
        h = self.hist
        h["t"].append(sim.t)
        h["dist"].append(dist)
        h["bearing"].append(bearing)
        h["vx"].append(info["cmd"][0])
        h["wz"].append(info["cmd"][2])
        h["seen"].append(sim.controller.target is not None)

        pkt = info["delivered"]
        if pkt is not None and pkt["frame"] is not None:
            frame = pkt["frame"].copy()
            t = sim.controller.target
            if t is not None:
                draw_box(frame, (t.cx, t.cy, t.w, t.h))
            self.image.image = frame
            self._update_readout(info, dist, bearing)
        now = time.monotonic()
        if now - self._last_plot > 0.2:
            self._last_plot = now
            self._update_plots()

        if sim.fallen and self.auto_reset:
            if self._fell_at is None:
                self._fell_at = sim.t
            elif sim.t - self._fell_at > 1.0:
                self._reset()

    def _update_readout(self, info, dist, bearing):
        c = info["cmd"]
        seen = np.mean(self.hist["seen"]) if self.hist["seen"] else 0.0
        fell = ' <b style="color:#c0392b">fell</b>' if self.sim.fallen else ""
        self.readout.content = (
            f'<div style="font-size:0.85em;line-height:1.4;padding:0 1em">'
            f"<b>{self.mode}</b>, controller: {info['state']}{fell}<br/>"
            f"person {dist:.2f} m, {math.degrees(bearing):+.0f}&deg; &middot; tracked {100 * seen:.0f}% of the last {PLOT_SECONDS:.0f} s<br/>"
            f"vx {c[0]:+.2f} &middot; vy {c[1]:+.2f} &middot; wz {c[2]:+.2f} &middot; "
            f"head pitch {c[4]:+.2f} yaw {c[5]:+.2f}</div>"
        )
        on = self.pad.connected()
        self.pad_status.content = (
            f'<div style="font-size:0.85em;padding:0 1em">Gamepad page: '
            f"{'connected, ' + self.pad.source if on else 'not connected'}</div>"
        )

    def _update_plots(self):
        h = self.hist
        if len(h["t"]) < 2:
            return
        t = np.asarray(h["t"])
        self.plot_dist.data = (t, np.asarray(h["dist"]), np.asarray(h["bearing"]))
        self.plot_cmd.data = (t, np.asarray(h["vx"]), np.asarray(h["wz"]))

    def run(self):
        self.viewer.run()


def main(argv=None):
    p = argparse.ArgumentParser(description="Interactive follow-mode simulator (Viser)")
    p.add_argument("policy", help="walking policy .onnx")
    p.add_argument("--mount", choices=["head", "trunk"], default="head")
    p.add_argument("--hfov", type=float, default=70.0)
    p.add_argument("--hz", type=float, default=15.0)
    p.add_argument("--latency_ms", type=float, default=66.0)
    p.add_argument("--no_distractor", action="store_true")
    p.add_argument("--no_randomize", action="store_true")
    p.add_argument("--encoder", default=None, help="as in follow_sim: fills emb_<i> for camera-aware policies")
    p.add_argument("--encoder_res", type=int, default=None)
    p.add_argument("--projection", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--pad_port", type=int, default=8081)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    encoder = None
    if a.encoder:
        from odm_train.camsim.encoders import EmbeddingPipeline, Projection, make_encoder

        encoder = EmbeddingPipeline(make_encoder(a.encoder, a.encoder_res), Projection.load(a.projection) if a.projection else None)
    sim = FollowSim(
        a.policy,
        cam=CameraConfig(mount=a.mount, hfov_deg=a.hfov, hz=a.hz, latency_ms=a.latency_ms),
        scene=SceneConfig(distractor=not a.no_distractor, seed=a.seed),
        seed=a.seed,
        randomize_dynamics=not a.no_randomize,
        randomize_visuals=not a.no_randomize,
        encoder=encoder,
    )
    pad = PadServer(a.host, a.pad_port).start()
    pad_url = f"http://localhost:{a.pad_port}/"
    print(f"simulator: http://localhost:{a.port}/   gamepad and keyboard: {pad_url}", flush=True)
    app = FollowApp(sim, pad, a.host, a.port, pad_url)
    try:
        app.run()
    finally:
        pad.stop()
        sim.close()


if __name__ == "__main__":
    main()
