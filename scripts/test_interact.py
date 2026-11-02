"""Try interaction mode without moving the legs.

Prints what the camera sees (person, hand pose, face), the gesture events and
the duck's mode and mood. Antennas, eye LEDs, sounds and screen eyes react if
they are turned on in expression_features, so you can check them here first.

    python3 scripts/test_interact.py           # screen shows eyes (if screen_eyes is on)
    python3 scripts/test_interact.py --show    # screen shows the camera and person boxes

Try: wave (the duck switches to "follow"), hold an open palm still for a
second ("stay"), thumbs up or peace sign (happy), and a face you enrolled
with scripts/enroll_face.py (greeting).
"""

import argparse
import time

from _common import DEFAULT_CONFIG_PATH, load_config, make_interact_commands

p = argparse.ArgumentParser()
p.add_argument("--duck_config_path", default=DEFAULT_CONFIG_PATH)
p.add_argument("--show", action="store_true", help="camera view instead of eyes on the screen")
args = p.parse_args()

config = load_config(args.duck_config_path)
cmds = make_interact_commands(config, show=args.show)
last_events = None
try:
    t_print = 0.0
    while True:
        cmd, _ = cmds.get()  # 50 Hz, like the walk loop
        b = cmds.behavior
        if b.events and b.events != last_events:
            print("   ", " ".join(b.events))
        last_events = b.events
        now = time.monotonic()
        if now - t_print > 0.25:
            t_print = now
            g = cmds.gestures.pose
            face = cmds.last_face[1] if cmds.last_face and now - cmds.last_face[0] < 1.0 else "-"
            e = cmds.expression
            print(
                f"{cmds.state:32s} hand {g:9s} face {face:10s} vx {cmd[0]:+.3f} wz {cmd[2]:+.3f} "
                f"head {cmd[4]:+.2f}/{cmd[5]:+.2f}/{cmd[6]:+.2f} ant {e.antennas[0]:+.1f},{e.antennas[1]:+.1f} "
                f"det {cmds.detector.fps():.1f} fps"
            )
        time.sleep(0.02)
except KeyboardInterrupt:
    pass
finally:
    cmds.stop()
