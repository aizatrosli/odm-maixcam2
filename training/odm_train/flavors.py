"""Named training recipes ("flavors") for the Open Duck Mini v2 policies.

Each flavor picks an env, a scene and a randomization level. ``train.py
--flavor <name>`` and ``training/colab_train.ipynb`` both read this table.
No JAX imports here, so the notebook can list flavors before installing JAX.
"""

FLAVORS = {
    # Recommended. Follow-mode commands, head moves, leg backlash, strong randomization.
    "follow_flat_backlash": dict(env="follow", task="flat_terrain_backlash", randomization="strong"),
    # Same, on rough ground (heightfield). Slower to learn, sturdier on carpet and tiles.
    "follow_rough_backlash": dict(env="follow", task="rough_terrain_backlash", randomization="strong"),
    # Follow mode without backlash joints (closest to how the stock policy was trained).
    "follow_flat": dict(env="follow", task="flat_terrain", randomization="strong"),
    # Upstream Joystick task, retrained as-is: a baseline to compare the follow flavors against.
    "stock_flat": dict(env="joystick", task="flat_terrain", randomization="stock"),
    "stock_flat_backlash": dict(env="joystick", task="flat_terrain_backlash", randomization="stock"),
    "stock_rough_backlash": dict(env="joystick", task="rough_terrain_backlash", randomization="stock"),
    # Upstream standing/balance task. Its observation has 85 values, not 101,
    # so scripts/walk.py cannot run it yet. Experimental.
    "standing_flat": dict(env="standing", task="flat_terrain", randomization="stock"),
}

DEPLOYABLE = [name for name, f in FLAVORS.items() if f["env"] != "standing"]
