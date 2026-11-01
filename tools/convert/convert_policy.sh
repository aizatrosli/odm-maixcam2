#!/usr/bin/env bash
# Convert the walking policy ONNX to a MaixCAM2 NPU model (.axmodel + .mud).
# Runs on a PC (x86_64) with Docker and the Pulsar2 image loaded
# (see https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam2.html).
#
#   ./convert_policy.sh BEST_WALK_ONNX_2.onnx obs.npy
#
# Output: out/walk_policy.mud + out/walk_policy_{npu,vnpu}.axmodel
# Copy the out/ folder to the MaixCAM2 and point maixcam2.policy_path at the .mud.
set -euo pipefail

ONNX=${1:?usage: $0 policy.onnx obs.npy}
OBS=${2:?usage: $0 policy.onnx obs.npy}
IMAGE=${PULSAR2_IMAGE:-pulsar2:6.0}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

python3 "$HERE/prepare_onnx.py" "$ONNX" "$WORK/walk_policy.onnx"
python3 "$HERE/make_calib.py" "$OBS" "$WORK/calib_obs.tar"
cp "$HERE"/walk_policy.npu.json "$HERE"/walk_policy.vnpu.json "$WORK/"

for mode in npu vnpu; do
  docker run --rm -v "$WORK":/data -w /data "$IMAGE" \
    pulsar2 build --target_hardware AX620E \
      --input walk_policy.onnx \
      --output_dir build_$mode \
      --config walk_policy.$mode.json
done

mkdir -p out
cp "$WORK/build_npu/compiled.axmodel" out/walk_policy_npu.axmodel
cp "$WORK/build_vnpu/compiled.axmodel" out/walk_policy_vnpu.axmodel
cp "$HERE/walk_policy.mud" out/
# the policy profile travels with the model (walk_policy.mud -> walk_policy.json)
if [ -f "${ONNX%.onnx}.json" ]; then cp "${ONNX%.onnx}.json" out/walk_policy.json; fi
echo "done: $(pwd)/out"
