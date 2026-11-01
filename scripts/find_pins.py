"""List MaixCAM2 pins, their current function and the functions they support.

Use it to pick the GPIO names for the foot switches and the I2C/UART pins.
"""

from maix import pinmap

for pin in pinmap.get_pins():
    try:
        current = pinmap.get_pin_function(pin)
    except Exception:
        current = "?"
    print(f"{pin:6s} now {current:14s} can be {', '.join(pinmap.get_pin_functions(pin))}")
