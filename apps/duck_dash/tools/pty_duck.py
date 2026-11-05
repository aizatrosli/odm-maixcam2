"""Serve the simulated duck's servo bus on a pseudo-terminal.

The dashboard then opens the pty like the real USB adapter, so its C Feetech
driver is tested against the same wire-level simulator as scripts/diagnose.py
--dry-run::

    python3 apps/duck_dash/tools/pty_duck.py --fault missing:23,hot:24
    # prints e.g. /dev/pts/5
    apps/duck_dash/build/duck_dash --servo-port /dev/pts/5 --no-imu --no-feet
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import threading
import tty

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from odm_maix.diag.simhw import SimDuck  # noqa: E402


class PtyServoBus:
    """Answers Feetech packets written to the pty slave, until stop()."""

    def __init__(self, faults=None):
        self.duck = SimDuck(faults)
        self.master, self.slave = os.openpty()
        tty.setraw(self.slave)
        attrs = termios.tcgetattr(self.slave)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(self.slave, termios.TCSANOW, attrs)
        self.path = os.ttyname(self.slave)
        # close our end so the slave looks free, like an unplugged-then-idle adapter;
        # the dashboard refuses a port another process holds open
        os.close(self.slave)
        self.slave = None
        self.packets = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _serve(self):
        buf = b""
        ser = self.duck.serial
        while not self._stop.is_set():
            r, _, _ = select.select([self.master], [], [], 0.05)
            if not r:
                continue
            try:
                buf += os.read(self.master, 4096)
            except OSError:  # EIO while nobody has the slave open
                self._stop.wait(0.02)
                continue
            while True:
                i = buf.find(b"\xff\xff")
                if i < 0:
                    buf = b""
                    break
                buf = buf[i:]
                if len(buf) < 4 or len(buf) < 4 + buf[3]:
                    break
                n = 4 + buf[3]
                pkt, buf = buf[:n], buf[n:]
                ser.reset_input_buffer()
                ser.write(pkt)
                for s in self.duck.servos:
                    s.update()
                reply = ser.read(4096)
                self.packets += 1
                if reply:
                    os.write(self.master, reply)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)
        os.close(self.master)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fault", action="append", help="simhw faults, e.g. missing:23,hot:24")
    args = p.parse_args()
    bus = PtyServoBus(args.fault).start()
    print(bus.path, flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    bus.stop()


if __name__ == "__main__":
    main()
