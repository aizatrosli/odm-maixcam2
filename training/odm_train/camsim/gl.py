"""Pick an offscreen OpenGL backend for mujoco.Renderer before mujoco is imported."""

import os


def setup(prefer: str = "") -> str:
    """Set MUJOCO_GL (and PYOPENGL_PLATFORM) if the user has not.

    EGL when an NVIDIA device node is visible (Docker with the graphics
    capability, Colab, a Linux desktop), OSMesa (pure CPU, always works)
    otherwise. WSL2 exposes the GPU as /dev/dxg, where EGL inside containers
    is unreliable, so it gets OSMesa too.
    Must run before the first ``import mujoco``.
    """
    gl = os.environ.get("MUJOCO_GL") or prefer
    if not gl:
        gl = "egl" if os.path.exists("/dev/nvidia0") else "osmesa"
    os.environ["MUJOCO_GL"] = gl
    if gl in ("egl", "osmesa"):
        os.environ.setdefault("PYOPENGL_PLATFORM", gl)
    return gl
