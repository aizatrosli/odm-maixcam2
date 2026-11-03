# One image for the lab (JupyterLab), sim and tensorboard services in compose.yaml.
# Works with and without an NVIDIA GPU: MuJoCo renders with EGL on the GPU and OSMesa otherwise.
ARG BASE=nvcr.io/nvidia/cuda:12.9.1-base-ubuntu24.04
FROM ${BASE}

ARG ODP_COMMIT=b9be205ac64488c23504ca42e5ec790337adeec3
# "jax[cuda12]==0.8.0" for NVIDIA training, "jax==0.8.0" for CPU only
ARG JAX="jax[cuda12]==0.8.0"
# torch is only for the V-JEPA / DINOv2 encoders; the CPU build keeps jax's CUDA libraries intact
ARG TORCH="torch==2.15.0"
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev git ca-certificates curl ffmpeg \
        libegl1 libgl1 libgles2 libosmesa6 libglfw3 \
    && rm -rf /var/lib/apt/lists/*
# EGL vendor file so the NVIDIA container runtime's libEGL_nvidia is found
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install "${JAX}" -r /tmp/requirements.txt \
    && pip install "${TORCH}" --index-url "${TORCH_INDEX}" \
    && pip install --no-deps mjviser==0.0.16
# (mjviser asks for mujoco>=3.6, but the training stack pins 3.3.7; the viewer works on 3.3.7)
# mujoco_playground fetches MuJoCo Menagerie into site-packages on first import; do it now,
# because the services run as your user and can't write there
RUN MUJOCO_GL=osmesa python -c "import mujoco_playground" \
    && chmod -R a+rX /opt/venv

# Open_Duck_Playground (the MuJoCo duck) and the stock walking policy, pinned
RUN git clone -q https://github.com/apirrone/Open_Duck_Playground /opt/Open_Duck_Playground \
    && git -C /opt/Open_Duck_Playground checkout -q ${ODP_COMMIT} \
    && git clone -q --depth 1 --filter=blob:none --sparse https://github.com/apirrone/Open_Duck_Mini /opt/Open_Duck_Mini \
    && git -C /opt/Open_Duck_Mini sparse-checkout set --no-cone /BEST_WALK_ONNX_2.onnx \
    && rm -rf /opt/Open_Duck_Mini/.git \
    && chmod -R a+rX /opt/Open_Duck_Playground /opt/Open_Duck_Mini

# The repo is bind-mounted here by compose, so edits in JupyterLab change your checkout.
ENV REPO=/workspace/odm-maixcam2 \
    ODP_DIR=/opt/Open_Duck_Playground \
    STOCK_ONNX=/opt/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx \
    HOME=/work/.home \
    TORCH_HOME=/work/.cache/torch \
    XDG_CACHE_HOME=/work/.cache \
    MPLCONFIGDIR=/work/.cache/matplotlib
ENV PYTHONPATH=/opt/Open_Duck_Playground:/workspace/odm-maixcam2/training:/workspace/odm-maixcam2

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 755 /usr/local/bin/entrypoint.sh
WORKDIR /workspace/odm-maixcam2
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
