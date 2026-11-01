import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from odm_maix.config import DEFAULT_CONFIG_PATH, DuckConfig  # noqa: E402


def load_config(path=DEFAULT_CONFIG_PATH, apply_pinmap=True):
    config = DuckConfig.load(path)
    if config.is_default:
        print(f"WARNING: {path} not found, using defaults (joint offsets all 0).")
    if apply_pinmap and config.maixcam2.get("pinmap"):
        from maix import pinmap

        for pin, func in config.maixcam2["pinmap"].items():
            pinmap.set_pin_function(pin, func)
            print(f"pinmap {pin} -> {func}")
    return config
