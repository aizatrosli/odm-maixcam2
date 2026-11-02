# Self-hosting with Docker Compose

One image runs the camera simulator, JupyterLab with live collaborative editing, and TensorBoard. A separate service runs Pulsar2 for converting models for the MaixCAM2 NPU. It works with or without an NVIDIA GPU.

| Service | What | Port |
|---|---|---|
| `lab` | JupyterLab 4.6 with `jupyter-collaboration`: everyone with the token edits the same notebooks live, with shared cursors | 8888 |
| `sim` | the interactive simulator: 3D view, the MaixCAM2 image, live plots, sliders; plus a gamepad/keyboard page | 8080, 8081 |
| `tensorboard` | training curves from `work/runs/` | 6006 |
| `pulsar2` (profile `convert`) | `tools/convert/convert_policy.sh` and `tools/bench_vit/build_vits.sh` without Docker-in-Docker | |

Your checkout is mounted at `/workspace/odm-maixcam2`, so notebook and code edits in the lab land in your working tree. Outputs, datasets and caches go to `work/` (ignored by git).

## Start

```bash
cp docker/.env.example .env        # set JUPYTER_TOKEN; set UID/GID to `id -u` / `id -g`
mkdir -p work                      # before the first `up`, so it's owned by you and not root
docker compose build               # 10-20 min the first time; ~1.5 GB CPU-only, much larger with CUDA jax
docker compose up -d lab sim tensorboard
```

Open http://localhost:8080 for the interactive simulator (below), and http://localhost:8888/lab?token=YOUR_TOKEN for the notebooks. Start with `training/camsim.ipynb`. Anyone else you give the URL and token to sees the same notebook and edits it with you. If a file changes on disk while it's open (a `git pull`, for example), the open copy reloads from disk within a second or so.

## Interactive simulator

http://localhost:8080 shows the duck, the person and the room in 3D. Next to it are the MaixCAM2 image as the robot gets it (15 Hz, 66 ms late, with the controller's target box) and live plots of distance, bearing and commands.

- **Mode:** *Follow the person* (the camera drives the duck), *Drive the duck* (you send the walking commands), or *Walk the person* (you move the person and watch the duck follow).
- **Gamepad and keyboard:** open http://localhost:8081 in another tab. It reads any browser gamepad (left stick walks, right stick turns, LB + right stick moves the head, A switches follow/drive, Start resets) or the keyboard (WASD, Q/E, Shift, Space, Enter). A separate page is needed because Viser only reports single key presses, not held keys or sticks.
- **Mouse:** drag a body to push it, for example shove the duck sideways while it follows. Pause, step and change speed in the Controls tab.
- **Sliders:** follow-controller gains and thresholds, missed detections and box jitter, person speed. They apply immediately.
- Choose the policy and mount with `SIM_POLICY` and `SIM_MOUNT` in `.env`. On a CPU the simulation runs at roughly real time; the status box shows the actual rate.

## One-off runs

```bash
docker compose run --rm sim python -m odm_train.camsim.follow_sim /opt/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx \
    --episodes 2 --video /work/camsim/follow.mp4               # video -> work/camsim/
docker compose run --rm sim python -m odm_train.camsim.follow_sim /opt/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx \
    --episodes 20 --record /work/camsim/follow.npz             # any follow_sim / encoders / sim_eval command
docker compose run --rm sim python -m pytest /workspace/odm-maixcam2/tests -q
```

The stock walking policy is at `/opt/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx` and Open_Duck_Playground (pinned) at `/opt/Open_Duck_Playground`, which is the working directory of `sim`. Without `--service-ports`, `run` doesn't publish ports, so it doesn't clash with the running app.

## NVIDIA GPU

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then add the GPU file to every command, or put `COMPOSE_FILE=compose.yaml:compose.gpu.yaml` in `.env`:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d lab
```

MuJoCo then renders with EGL on the GPU, and training (`python -m odm_train.train`, see `training/README.md`) uses CUDA JAX. On a machine without a GPU, set `JAX=jax==0.8.0` in `.env` before building for a smaller image; rendering uses OSMesa on the CPU. On Windows (WSL2), rendering always uses OSMesa, because EGL inside containers there is unreliable; training still uses the GPU.

PyTorch (only for the V-JEPA and DINOv2 encoders) is the CPU build, so it can't break JAX's CUDA libraries. A few frames per second on the CPU is enough for recording datasets.

## Reaching it from other devices

Ports listen on 127.0.0.1 by default. To use the lab from a laptop or tablet, the simplest safe way is [Tailscale](https://tailscale.com): set `BIND` in `.env` to the machine's Tailscale address (`tailscale ip -4`) and run `docker compose up -d` again. Don't set `BIND=0.0.0.0` on a machine reachable from the internet: the token is the only protection.

## Pulsar2 (model conversion)

Pulsar2 isn't on Docker Hub. Download the image tarball (6.0, the version the configs here were made for) from the [Sipeed guide](https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam2.html) or [Hugging Face](https://huggingface.co/AXERA-TECH/Pulsar2), then:

```bash
docker load -i ax_pulsar2_6.0.tar.gz          # check the tag with `docker images`; set PULSAR2_IMAGE if it isn't pulsar2:6.0
docker compose --profile convert run --rm pulsar2 \
    /workspace/odm-maixcam2/tools/convert/convert_policy.sh /work/runs/follow_v1/policy.onnx /work/obs.npy
docker compose --profile convert run --rm pulsar2 /workspace/odm-maixcam2/tools/bench_vit/build_vits.sh small:160
```

Results land in `work/convert/out/`. Inside this service the scripts call `pulsar2` directly (`PULSAR2_RUNNER=local`) instead of starting another container, and use the image's own Python for their `onnx`/`numpy` preprocessing. Without Compose the scripts still work as before (`docker run` per build).

## Files

| File | What |
|---|---|
| `compose.yaml`, `compose.gpu.yaml` | the services; the GPU file adds the NVIDIA device |
| `docker/Dockerfile` | CUDA 12.9 base (works without a GPU), pinned Python packages, Open_Duck_Playground and the stock policy |
| `docker/requirements.txt` | the pins, the same ones `training/colab_train.ipynb` uses |
| `docker/lab.sh` | JupyterLab with collaboration; refuses to start without a token |
| `docker/.env.example` | settings |
