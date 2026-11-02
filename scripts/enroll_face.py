"""Teach the duck a face, so interaction mode greets that person by name.

    python3 scripts/enroll_face.py --name aizat     # look at the camera for a few seconds
    python3 scripts/enroll_face.py --list
    python3 scripts/enroll_face.py --remove aizat

Faces are stored in the face database used by interaction mode
(maixcam2.vision.interact.faces_path, /root/faces.bin by default). Enroll
in the light the duck will usually see you in. Run it while walk.py is
stopped: only one process can own the camera.
"""

import argparse
import time

from _common import DEFAULT_CONFIG_PATH, load_config

from odm_maix.commands import DEFAULT_INTERACT

p = argparse.ArgumentParser()
p.add_argument("--duck_config_path", default=DEFAULT_CONFIG_PATH)
p.add_argument("--name")
p.add_argument("--list", action="store_true")
p.add_argument("--remove")
p.add_argument("--seconds", type=float, default=5.0, help="how long to look for a clear face")
args = p.parse_args()

config = load_config(args.duck_config_path, apply_pinmap=False)
cfg = {**DEFAULT_INTERACT, **config.maixcam2["vision"].get("interact", {})}

from maix import camera, display, image, nn  # noqa: E402

rec = nn.FaceRecognizer(detect_model=cfg["face_detect_model"], feature_model=cfg["face_feature_model"], dual_buff=False)
try:
    rec.load_faces(cfg["faces_path"])
except Exception:
    print(f"starting a new face database at {cfg['faces_path']}")

if args.list:
    print("known faces:", list(rec.labels)[1:])
    raise SystemExit
if args.remove:
    rec.remove_face(label=args.remove)
    rec.save_faces(cfg["faces_path"])
    print(f"removed {args.remove}")
    raise SystemExit
if not args.name:
    raise SystemExit("give --name, --list or --remove")

cam = camera.Camera(rec.input_width(), rec.input_height(), rec.input_format())
disp = display.Display()
best = None
t_end = time.monotonic() + args.seconds
print(f"look at the camera for {args.seconds:.0f} s...")
while time.monotonic() < t_end:
    img = cam.read()
    faces = rec.recognize(img, 0.5, 0.45, cfg["face_threshold"], True)
    if len(faces) == 1:  # only enroll when exactly one person is in view
        f = faces[0]
        img.draw_rect(f.x, f.y, f.w, f.h, color=image.COLOR_GREEN)
        if best is None or f.score > best.score:
            best = f
    elif faces:
        img.draw_string(4, 4, "only one face please", image.COLOR_RED)
    disp.show(img)

if best is None:
    raise SystemExit("no single clear face seen, try again with better light")
rec.add_face(best, args.name)
rec.save_faces(cfg["faces_path"])
print(f"enrolled {args.name}; known faces: {list(rec.labels)[1:]}")
