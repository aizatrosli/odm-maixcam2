#!/usr/bin/env bash
# Train the walking policy on a CPU-only x86 server (no GPU).
#
#   ODP_DIR=~/Open_Duck_Playground ./training/train_cpu.sh --flavor follow_flat_backlash \
#       --camera_mass_g 60,200 --output_dir ~/duck_runs/follow_cpu
#
# Any odm_train.train option can be passed. What this adds for CPUs:
#  - JAX sees one device per core (XLA_FLAGS), so brax's pmap spreads the envs over all
#    cores. Without it one process uses about two cores.
#  - The device count is the largest power of two <= the core count (brax needs num_envs
#    to divide its 8192-sample batch, so odd factors don't work). Override with CPU_DEVICES.
#  - --num_envs defaults to 64 per device (min 512, max 8192). The PPO batch per update is
#    the same as on a GPU; fewer envs only means fewer run at once.
#  - A persistent compile cache, so a restart doesn't recompile for minutes.
# Run it inside tmux/screen or with nohup; add --resume to continue a stopped run.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ODP_DIR=${ODP_DIR:-$HERE/../../Open_Duck_Playground}
[ -d "$ODP_DIR/playground" ] || { echo "Set ODP_DIR to your Open_Duck_Playground checkout" >&2; exit 1; }
ODP_DIR=$(cd "$ODP_DIR" && pwd)

# Open_Duck_Playground loads its data by relative path and writes .tmp/ in the working
# directory, so the run must start there. A read-only copy (the Docker image's /opt one)
# is copied to a writable place first.
if [ ! -w "$ODP_DIR" ]; then
  COPY=${ODP_COPY:-/work/Open_Duck_Playground}
  [ -d "$COPY/playground" ] || cp -r "$ODP_DIR" "$COPY"
  ODP_DIR=$COPY
fi

CORES=${CPU_CORES:-$(nproc)}
DEV=1
while [ $((DEV * 2)) -le "$CORES" ]; do DEV=$((DEV * 2)); done
DEV=${CPU_DEVICES:-$DEV}
export XLA_FLAGS="--xla_force_host_platform_device_count=$DEV ${XLA_FLAGS:-}"
export JAX_PLATFORMS=cpu

EXTRA=()
if [[ " $* " != *" --num_envs "* ]]; then
  ENVS=$((DEV * 64)); [ $ENVS -lt 512 ] && ENVS=512; [ $ENVS -gt 8192 ] && ENVS=8192
  EXTRA+=(--num_envs "$ENVS")
fi
if [[ " $* " != *" --jax_cache "* ]]; then
  EXTRA+=(--jax_cache "${JAX_CACHE:-$HOME/.cache/odm_jax}")
fi

export PYTHONPATH="$ODP_DIR:$HERE${PYTHONPATH:+:$PYTHONPATH}"
cd "$ODP_DIR"
echo "CPU training: $CORES cores, $DEV JAX devices, ${EXTRA[*]}"
exec python -u -m odm_train.train "${EXTRA[@]}" "$@"
