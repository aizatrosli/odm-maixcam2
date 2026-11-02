"""Write placeholder duck sounds to ~/sounds so expressions have something to play.

    python3 scripts/make_sounds.py              # ~/sounds/{happy,curious,startled,sleepy,hello,ok}.wav
    python3 scripts/make_sounds.py --out /root/sounds

Each is a short synthesized quack or chirp (48 kHz, 16-bit mono WAV). Replace
any file with a real recording of the same name whenever you like.
"""

import argparse
import os
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from odm_maix.sounds import SOUNDS, synth  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", default="~/sounds")
args = p.parse_args()

out = os.path.expanduser(args.out)
os.makedirs(out, exist_ok=True)
for name in SOUNDS:
    pcm = synth(name)
    path = os.path.join(out, f"{name}.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(pcm.tobytes())
    print(f"{path}  {len(pcm) / 48000:.2f} s")
