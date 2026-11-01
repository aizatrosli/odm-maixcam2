"""Foot contact switches on MaixCAM2 GPIO (active low).

maix.gpio accepts Pull.PULL_UP but on MaixCAM2 it does not set a bias on input
lines (it uses the GPIO v1 line-handle API without bias flags), so each switch
needs an external 10k pull-up to 3V3. PULL_UP is still passed in case a later
MaixPy honours it.
"""

from __future__ import annotations


class FeetContacts:
    def __init__(self, config):
        from maix import gpio, pinmap

        m = config.maixcam2
        self._pins = []
        for name in (m["left_foot_gpio"], m["right_foot_gpio"]):
            pin = name.replace("GPIO", "")  # "GPIOA4" -> pin "A4"
            try:
                pinmap.set_pin_function(pin, name)
            except Exception as e:  # some pins are GPIO by default
                print(f"[feet] pinmap {pin}->{name}: {e}")
            self._pins.append(gpio.GPIO(name, gpio.Mode.IN, gpio.Pull.PULL_UP))

    def get(self):
        # switch closed -> pin pulled to GND -> contact
        return [not bool(p.value()) for p in self._pins]

    def stop(self) -> None:
        pass


class NoFeetContacts:
    """Fallback when the switches are not wired yet: reports both feet down."""

    def get(self):
        return [True, True]

    def stop(self) -> None:
        pass
