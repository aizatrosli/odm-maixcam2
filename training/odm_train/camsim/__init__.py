"""MaixCAM2 camera simulator for the Open Duck Mini v2.

Adds the MaixCAM2 camera (on the head or the trunk), a virtual person to
follow, and a randomized room to Open_Duck_Playground's MuJoCo model, then
runs the same loop as the robot runtime: camera frame -> person detection ->
FollowController -> walking policy. See ``follow_sim.py`` for the command line.

Pure CPU MuJoCo (``mujoco.Renderer``): EGL on an NVIDIA GPU, OSMesa elsewhere.
Call ``camsim.gl.setup()`` before importing mujoco if MUJOCO_GL is not set.
"""
