"""Camera + YOLO on the NPU in a separate process.

The detector runs in its own process so that camera capture and NPU calls never
hold the Python GIL inside the 50 Hz control loop. The latest result is shared
through a small lock-protected array:

    [seq, t, n, (cx, cy, w, h, score) * MAX_DETS]   (normalised 0..1)
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import List, Tuple

from .follow import Detection

MAX_DETS = 8
HEADER = 3
FIELDS = 5


def _detector_main(shared, stop, model_path: str, class_name: str, conf: float, show: bool):
    from maix import app, camera, display, image, nn

    det = nn.YOLO11(model=model_path, dual_buff=True)
    labels = list(det.labels)
    if class_name not in labels:
        raise ValueError(f"{class_name!r} not in model labels {labels[:10]}...")
    cls = labels.index(class_name)
    W, H = det.input_width(), det.input_height()
    cam = camera.Camera(W, H, det.input_format())
    disp = display.Display() if show else None
    seq = 0
    while not stop.is_set() and not app.need_exit():
        img = cam.read()
        objs = [o for o in det.detect(img, conf_th=conf, iou_th=0.45) if o.class_id == cls]
        objs = sorted(objs, key=lambda o: -o.w * o.h)[:MAX_DETS]
        seq += 1
        with shared.get_lock():
            shared[0] = seq
            shared[1] = time.monotonic()
            shared[2] = len(objs)
            for i, o in enumerate(objs):
                b = HEADER + i * FIELDS
                shared[b : b + FIELDS] = [(o.x + o.w / 2) / W, (o.y + o.h / 2) / H, o.w / W, o.h / H, o.score]
        if disp is not None:
            for o in objs:
                img.draw_rect(o.x, o.y, o.w, o.h, color=image.COLOR_RED)
            disp.show(img)


class DetectorProcess:
    def __init__(self, model_path="/root/models/yolo11n.mud", class_name="person", conf=0.5, show=False):
        # spawn, not fork: the parent may already hold maix/camera state
        ctx = mp.get_context("spawn")
        self.shared = ctx.Array("d", HEADER + MAX_DETS * FIELDS)
        self.stop_event = ctx.Event()
        self.proc = ctx.Process(
            target=_detector_main,
            args=(self.shared, self.stop_event, model_path, class_name, conf, show),
            daemon=True,
        )
        self.proc.start()
        self._last_seq = 0
        self._frames = 0
        self._t0 = time.monotonic()

    def latest(self) -> Tuple[bool, float, List[Detection]]:
        """(is_new, timestamp, detections)."""
        with self.shared.get_lock():
            vals = list(self.shared[:])
        seq, t, n = int(vals[0]), vals[1], int(vals[2])
        new = seq != self._last_seq
        if new:
            self._frames += seq - self._last_seq
            self._last_seq = seq
        dets = [Detection(*vals[HEADER + i * FIELDS : HEADER + (i + 1) * FIELDS]) for i in range(n)]
        return new, t, dets

    def fps(self) -> float:
        return self._frames / max(1e-6, time.monotonic() - self._t0)

    def alive(self) -> bool:
        return self.proc.is_alive()

    def stop(self) -> None:
        self.stop_event.set()
        self.proc.join(timeout=2)
        if self.proc.is_alive():
            self.proc.terminate()
