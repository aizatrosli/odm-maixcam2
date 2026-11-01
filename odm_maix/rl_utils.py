import numpy as np


class LowPassActionFilter:
    def __init__(self, control_freq, cutoff_frequency=30.0):
        self.last_action = 0
        self.current_action = 0
        self.alpha = (1.0 / cutoff_frequency) / (1.0 / control_freq + 1.0 / cutoff_frequency)

    def push(self, action):
        self.current_action = action

    def get_filtered_action(self):
        self.last_action = self.alpha * self.last_action + (1 - self.alpha) * self.current_action
        return self.last_action


class GaitPhase:
    """The two-value gait clock fed to the policy (cos, sin)."""

    def __init__(self, nb_steps_in_period: int):
        self.nb_steps_in_period = nb_steps_in_period
        self.i = 0.0
        self.phase = np.array([0.0, 0.0])

    def step(self, frequency_factor: float = 1.0) -> np.ndarray:
        self.i = (self.i + frequency_factor) % self.nb_steps_in_period
        a = self.i / self.nb_steps_in_period * 2 * np.pi
        self.phase = np.array([np.cos(a), np.sin(a)])
        return self.phase


def nb_steps_from_config(config) -> int:
    m = config.maixcam2
    if m.get("nb_steps_in_period"):
        return int(m["nb_steps_in_period"])
    # Same file the Pi runtime reads. Only "period" and "fps" are used.
    import pickle

    path = m["polynomial_coefficients_path"]
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{path} not found. Copy polynomial_coefficients.pkl from the Pi runtime's scripts/ "
            "folder, or set maixcam2.nb_steps_in_period in duck_config.json."
        ) from None
    first = next(iter(data.values()))
    return int(first["period"] * first["fps"])
