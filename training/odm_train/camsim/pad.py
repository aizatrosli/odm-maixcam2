"""Gamepad and keyboard for the interactive app, through the browser.

Viser only reports key presses, not held keys or gamepad sticks, so a small
page (served here on its own port) reads the browser Gamepad API and the
keyboard and posts the state about 30 times a second. Standard libraries only.

    pad = PadServer(port=8081); pad.start()
    lx, ly, rx, ry = pad.axes()      # -1..1, zero when nothing was received for 0.5 s
    pad.pop_buttons()                # buttons pressed since the last call, e.g. {"start"}
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Set, Tuple

import numpy as np

# same ranges and scaling as odm_maix.commands.GamepadCommands (the robot's gamepad)
X_RANGE = (-0.15, 0.15)
Y_RANGE = (-0.2, 0.2)
YAW_RANGE = (-1.0, 1.0)
HEAD_YAW = 0.8  # rad at full right-stick deflection with LB held
HEAD_PITCH = 0.5
DEADZONE = 0.08
STALE_S = 0.5

# standard-mapping button index -> name
BUTTONS = {0: "a", 1: "b", 2: "x", 3: "y", 4: "lb", 5: "rb", 8: "back", 9: "start"}


def _scale(v: float, rng) -> float:
    return v * abs(rng[1] if v >= 0 else rng[0])


def _dead(v: float) -> float:
    return 0.0 if abs(v) < DEADZONE else float(np.clip(v, -1, 1))


def pad_to_cmd(lx: float, ly: float, rx: float, ry: float, head: bool = False) -> np.ndarray:
    """Sticks (browser convention: right and down positive) -> the 7 walking commands.

    Left stick walks (vx, vy), right stick x turns. With ``head`` (LB held), the
    right stick points the head instead, like looking around on the robot."""
    lx, ly, rx, ry = (_dead(v) for v in (lx, ly, rx, ry))
    cmd = np.zeros(7)
    cmd[0] = _scale(-ly, X_RANGE)
    cmd[1] = _scale(-lx, Y_RANGE)
    if head:
        cmd[5] = -rx * HEAD_YAW
        cmd[4] = ry * HEAD_PITCH
    else:
        cmd[2] = _scale(-rx, YAW_RANGE)
    return np.round(cmd, 3)


class PadServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8081):
        self.host, self.port = host, port
        self._lock = threading.Lock()
        self._axes = (0.0, 0.0, 0.0, 0.0)
        self._held: Set[str] = set()
        self._pressed: Set[str] = set()
        self._t = 0.0
        self.source = "none"
        self.httpd = None

    # --- state
    def update(self, axes, buttons, source="gamepad"):
        axes = [float(a) for a in list(axes)[:4]] + [0.0] * max(0, 4 - len(axes))
        held = {BUTTONS[i] for i, b in enumerate(buttons) if b and i in BUTTONS}
        with self._lock:
            self._pressed |= held - self._held
            self._held = held
            self._axes = tuple(axes[:4])
            self._t = time.monotonic()
            self.source = source

    def connected(self) -> bool:
        return time.monotonic() - self._t < STALE_S

    def axes(self) -> Tuple[float, float, float, float]:
        with self._lock:
            return self._axes if self.connected() else (0.0, 0.0, 0.0, 0.0)

    def held(self) -> Set[str]:
        with self._lock:
            return set(self._held) if self.connected() else set()

    def pop_buttons(self) -> Set[str]:
        with self._lock:
            out, self._pressed = self._pressed, set()
            return out

    def cmd(self) -> np.ndarray:
        return pad_to_cmd(*self.axes(), head="lb" in self.held())

    # --- http
    def start(self):
        pad = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    msg = json.loads(self.rfile.read(n) or b"{}")
                    pad.update(msg.get("axes", []), msg.get("buttons", []), msg.get("source", "gamepad"))
                    self.send_response(204)
                except (ValueError, TypeError):
                    self.send_response(400)
                self.end_headers()

            def log_message(self, *a):  # quiet
                pass

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duck gamepad</title>
<style>
 :root { --bg:#f6f6f4; --fg:#1d1d1b; --muted:#6b6b66; --line:#d9d9d4; --on:#2f7d4f; }
 @media (prefers-color-scheme: dark) { :root { --bg:#191918; --fg:#ecece8; --muted:#a3a39c; --line:#3a3a37; --on:#5fbf86; } }
 body { background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; margin:0; padding:24px 16px; }
 main { max-width:560px; margin:auto; }
 h1 { font-size:20px; margin:0 0 4px; } p { color:var(--muted); margin:4px 0 16px; }
 .row { display:flex; gap:16px; flex-wrap:wrap; }
 .stick { width:120px; height:120px; border:1px solid var(--line); border-radius:50%; position:relative; }
 .dot { width:18px; height:18px; border-radius:50%; background:var(--fg); position:absolute; left:51px; top:51px; }
 #status { font-weight:600; } #status.on { color:var(--on); }
 kbd { border:1px solid var(--line); border-radius:4px; padding:0 5px; font-size:13px; }
 table { border-collapse:collapse; margin-top:16px; } td { padding:2px 12px 2px 0; color:var(--muted); }
</style></head><body><main>
<h1>Duck gamepad</h1>
<p>Keep this tab open next to the simulator. <span id="status">No input yet: press a button on the gamepad, or use the keyboard.</span></p>
<div class="row"><div class="stick"><div class="dot" id="l"></div></div><div class="stick"><div class="dot" id="r"></div></div></div>
<table>
<tr><td>Left stick, <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd></td><td>walk forward, back and sideways</td></tr>
<tr><td>Right stick x, <kbd>Q</kbd><kbd>E</kbd></td><td>turn</td></tr>
<tr><td>Hold LB, <kbd>Shift</kbd></td><td>right stick (or <kbd>Q</kbd><kbd>E</kbd><kbd>R</kbd><kbd>F</kbd>) moves the head</td></tr>
<tr><td>A, <kbd>Space</kbd></td><td>switch between following and driving</td></tr>
<tr><td>Start, <kbd>Enter</kbd></td><td>reset the episode</td></tr>
</table>
</main><script>
const keys = new Set();
let tapA = false, tapStart = false;  // a tap shorter than one poll still counts
addEventListener('keydown', e => { keys.add(e.code); if (e.code === 'Space') tapA = true; if (e.code === 'Enter') tapStart = true; if (['Space','ArrowUp','ArrowDown'].includes(e.code)) e.preventDefault(); });
addEventListener('keyup', e => keys.delete(e.code));
addEventListener('blur', () => keys.clear());
const k = c => keys.has(c) ? 1 : 0;
function place(id, x, y) { const d = document.getElementById(id); d.style.left = (51 + 45*x) + 'px'; d.style.top = (51 + 45*y) + 'px'; }
let busy = false;
function tick() {
  let axes = [0,0,0,0], buttons = new Array(10).fill(false), source = 'keyboard';
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  const gp = [...pads].find(p => p && p.connected);
  if (gp) { source = 'gamepad: ' + gp.id; axes = [0,1,2,3].map(i => gp.axes[i] || 0); buttons = gp.buttons.slice(0,10).map(b => b.pressed); }
  const shift = keys.has('ShiftLeft') || keys.has('ShiftRight');
  const kx = k('KeyD') - k('KeyA'), ky = k('KeyS') - k('KeyW'), kr = k('KeyE') - k('KeyQ'), kp = k('KeyF') - k('KeyR');
  if (kx || ky || kr || kp || shift || tapA || tapStart) {
    source = gp ? source + ' + keyboard' : 'keyboard';
    axes = [axes[0] || kx, axes[1] || ky, axes[2] || kr, axes[3] || kp];
    buttons[0] = buttons[0] || tapA; buttons[4] = buttons[4] || shift; buttons[9] = buttons[9] || tapStart;
  }
  place('l', axes[0], axes[1]); place('r', axes[2], axes[3]);
  const st = document.getElementById('status');
  if (!busy) {
    busy = true; tapA = false; tapStart = false;
    fetch(location.pathname, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({axes, buttons, source})})
      .then(() => { st.textContent = 'Connected (' + source + ')'; st.className = 'on'; })
      .catch(() => { st.textContent = 'Simulator not reachable'; st.className = ''; })
      .finally(() => { busy = false; });
  }
}
setInterval(tick, 33);
</script></body></html>
"""
