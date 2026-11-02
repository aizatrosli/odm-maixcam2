"""Camera + YOLO on the NPU in a separate process.

The detector runs in its own process so that camera capture and NPU calls never
hold the Python GIL inside the 50 Hz control loop. The latest result is shared
through a small lock-protected array:

    [seq, t, n, (cx, cy, w, h, score) * MAX_DETS,       people (normalised 0..1)
     hand_seq, hand_n, hcx, hcy, hsize, pose_id, hscore, best hand (interaction mode)
     face_seq, face_n, fcx, fcy, fw, fh, face_id, fscore] biggest face (interaction mode)

In interaction mode the same process also runs hand landmarks (every
``hands_every`` frames) and face recognition (every ``faces_every`` frames)
on the same camera frame, since only one process can own the camera. Hand
poses are classified here (it is cheap); turning poses into gestures over
time happens in the control process. Face names are sent once at start-up
through a queue. The control process writes the current mood into a shared
value, and this process draws matching eyes on the screen when ``show`` is off.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import List, Optional, Tuple

from .follow import Detection
from .gestures import POSE_ID, POSES, classify_pose, keypoints_from_maix

MAX_DETS = 8
HEADER = 3
FIELDS = 5
HAND_OFF = HEADER + MAX_DETS * FIELDS
HAND_FIELDS = 7
FACE_OFF = HAND_OFF + HAND_FIELDS
FACE_FIELDS = 8
SIZE = FACE_OFF + FACE_FIELDS


def _detector_main(shared, stop, mood, names_q, model_path, class_name, conf, show, interact):
    from maix import app, camera, display, image, nn

    det = nn.YOLO11(model=model_path, dual_buff=not interact)
    labels = list(det.labels)
    if class_name not in labels:
        raise ValueError(f"{class_name!r} not in model labels {labels[:10]}...")
    cls = labels.index(class_name)

    hands = faces = None
    if interact:
        hands = nn.HandLandmarks(model=interact.get("hand_model", "/root/models/hand_landmarks.mud"))
        if interact.get("faces", True):
            faces = nn.FaceRecognizer(
                detect_model=interact.get("face_detect_model", "/root/models/yolov8n_face.mud"),
                feature_model=interact.get("face_feature_model", "/root/models/insghtface_webface_r50.mud"),
                dual_buff=False,
            )
            db = interact.get("faces_path", "/root/faces.bin")
            try:
                faces.load_faces(db)
            except Exception as e:
                print(f"[vision] no face database at {db} ({e}); every face is a stranger")
            names_q.put(list(faces.labels))

    if interact and interact.get("camera_size"):
        W, H = interact["camera_size"]
    else:
        W, H = det.input_width(), det.input_height()
    cam = camera.Camera(W, H, det.input_format())
    disp = display.Display() if (show or (interact and interact.get("screen_eyes"))) else None
    hands_every = max(1, int(interact.get("hands_every", 2))) if interact else 0
    faces_every = max(1, int(interact.get("faces_every", 5))) if interact else 0
    face_cmp = interact.get("face_threshold", 0.85) if interact else 0.85
    t0 = time.monotonic()
    seq = hand_seq = face_seq = 0
    while not stop.is_set() and not app.need_exit():
        img = cam.read()
        iw, ih = img.width(), img.height()
        objs = [o for o in det.detect(img, conf_th=conf, iou_th=0.45) if o.class_id == cls]
        objs = sorted(objs, key=lambda o: -o.w * o.h)[:MAX_DETS]
        seq += 1

        hand_vals = face_vals = None
        if hands is not None and seq % hands_every == 0:
            hs = hands.detect(img, conf_th=0.7, iou_th=0.45, conf_th2=0.8)
            hand_seq += 1
            if hs:
                h = max(hs, key=lambda o: o.w * o.h)
                pose = classify_pose(keypoints_from_maix(h.points))
                hand_vals = [hand_seq, 1, (h.x + h.w / 2) / iw, (h.y + h.h / 2) / ih, h.h / ih, POSE_ID[pose], h.score]
            else:
                hand_vals = [hand_seq, 0, 0, 0, 0, 0, 0]
        if faces is not None and seq % faces_every == 0:
            fs = faces.recognize(img, 0.5, 0.45, face_cmp)
            face_seq += 1
            if fs:
                f = max(fs, key=lambda o: o.w * o.h)
                fx, fy = (f.x + f.w / 2) / iw, (f.y + f.h / 2) / ih
                face_vals = [face_seq, 1, fx, fy, f.w / iw, f.h / ih, f.class_id, f.score]
            else:
                face_vals = [face_seq, 0, 0, 0, 0, 0, 0, 0]

        with shared.get_lock():
            shared[0] = seq
            shared[1] = time.monotonic()
            shared[2] = len(objs)
            for i, o in enumerate(objs):
                b = HEADER + i * FIELDS
                shared[b : b + FIELDS] = [(o.x + o.w / 2) / iw, (o.y + o.h / 2) / ih, o.w / iw, o.h / ih, o.score]
            if hand_vals is not None:
                shared[HAND_OFF : HAND_OFF + HAND_FIELDS] = hand_vals
            if face_vals is not None:
                shared[FACE_OFF : FACE_OFF + FACE_FIELDS] = face_vals

        if disp is not None:
            if show:
                for o in objs:
                    img.draw_rect(o.x, o.y, o.w, o.h, color=image.COLOR_RED)
                disp.show(img)
            elif seq % 3 == 0:  # eyes at ~10 fps is plenty
                from ..behavior import MOODS
                from ..expression import draw_eyes

                eyes = image.Image(disp.width(), disp.height(), image.Format.FMT_RGB888)
                m = MOODS[int(mood.value)] if 0 <= int(mood.value) < len(MOODS) else "neutral"
                draw_eyes(eyes, m, time.monotonic() - t0, image)
                disp.show(eyes)


class Hand:
    __slots__ = ("cx", "cy", "size", "pose", "score")

    def __init__(self, cx, cy, size, pose, score):
        self.cx, self.cy, self.size, self.pose, self.score = cx, cy, size, pose, score


class Face:
    __slots__ = ("cx", "cy", "w", "h", "name", "known", "score")

    def __init__(self, cx, cy, w, h, name, known, score):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.name, self.known, self.score = name, known, score


class DetectorProcess:
    def __init__(
        self,
        model_path="/root/models/yolo11n.mud",
        class_name="person",
        conf=0.5,
        show=False,
        interact: Optional[dict] = None,
    ):
        # spawn, not fork: the parent may already hold maix/camera state
        ctx = mp.get_context("spawn")
        self.shared = ctx.Array("d", SIZE)
        self.mood = ctx.Value("i", 0)  # written by the control process, read here for screen eyes
        self.stop_event = ctx.Event()
        self.names_q = ctx.Queue()
        self.proc = ctx.Process(
            target=_detector_main,
            args=(self.shared, self.stop_event, self.mood, self.names_q, model_path, class_name, conf, show, interact),
            daemon=True,
        )
        self.proc.start()
        self._last_seq = 0
        self._last_hand_seq = 0
        self._last_face_seq = 0
        self._face_names: List[str] = ["unknown"]
        self._frames = 0
        self._t0 = time.monotonic()

    def latest(self) -> Tuple[bool, float, List[Detection]]:
        """(is_new, timestamp, detections)."""
        with self.shared.get_lock():
            vals = list(self.shared[:HAND_OFF])
        seq, t, n = int(vals[0]), vals[1], int(vals[2])
        new = seq != self._last_seq
        if new:
            self._frames += seq - self._last_seq
            self._last_seq = seq
        dets = [Detection(*vals[HEADER + i * FIELDS : HEADER + (i + 1) * FIELDS]) for i in range(n)]
        return new, t, dets

    def latest_hand(self) -> Tuple[bool, Optional[Hand]]:
        """(is_new, best hand or None). Only new when the hand model ran since the last call."""
        with self.shared.get_lock():
            v = list(self.shared[HAND_OFF : HAND_OFF + HAND_FIELDS])
        seq = int(v[0])
        if seq == self._last_hand_seq:
            return False, None
        self._last_hand_seq = seq
        if int(v[1]) == 0:
            return True, None
        pid = int(v[5])
        return True, Hand(v[2], v[3], v[4], POSES[pid] if 0 <= pid < len(POSES) else "none", v[6])

    def latest_face(self) -> Tuple[bool, Optional[Face]]:
        """(is_new, biggest face or None)."""
        while not self.names_q.empty():
            try:
                self._face_names = self.names_q.get_nowait()
            except Exception:
                break
        with self.shared.get_lock():
            v = list(self.shared[FACE_OFF : FACE_OFF + FACE_FIELDS])
        seq = int(v[0])
        if seq == self._last_face_seq:
            return False, None
        self._last_face_seq = seq
        if int(v[1]) == 0:
            return True, None
        cid = int(v[6])
        known = cid > 0 and cid < len(self._face_names)
        name = self._face_names[cid] if known else "unknown"
        return True, Face(v[2], v[3], v[4], v[5], name, known, v[7])

    def set_mood(self, mood_id: int) -> None:
        self.mood.value = int(mood_id)

    def fps(self) -> float:
        return self._frames / max(1e-6, time.monotonic() - self._t0)

    def alive(self) -> bool:
        return self.proc.is_alive()

    def stop(self) -> None:
        self.stop_event.set()
        self.proc.join(timeout=2)
        if self.proc.is_alive():
            self.proc.terminate()
