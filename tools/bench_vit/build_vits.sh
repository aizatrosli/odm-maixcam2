#!/usr/bin/env bash
# Build a set of ViT encoders for the MaixCAM2 NPU so their latency can be timed on the board.
# Runs on a PC (x86_64) with Docker and the Pulsar2 image loaded, same setup as
# tools/convert/convert_policy.sh
# (see https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam2.html).
#
#   ./build_vits.sh                          # default set below
#   ./build_vits.sh small:160 large:224:1    # size:res[:frames] list
#   PRECISION=int8 ./build_vits.sh           # all-INT8 (default: mixed, U16 LayerNorm + Softmax)
#
# Output: out/vit_<size>_<res>_f<frames>/ with .axmodel + .mud. Copy out/ to the
# MaixCAM2 and run: python3 scripts/bench_npu.py out/*/*.mud
set -euo pipefail

IMAGE=${PULSAR2_IMAGE:-pulsar2:6.0}
PRECISION=${PRECISION:-mixed}
HERE=$(cd "$(dirname "$0")" && pwd)
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(tiny:160 small:160 small:224 base:160 large:160 large:224)
fi

mkdir -p out
for cfg in "${CONFIGS[@]}"; do
  IFS=: read -r size res frames <<<"$cfg"
  frames=${frames:-1}
  name="vit_${size}_${res}_f${frames}"
  WORK=$(mktemp -d)
  echo "=== $name ($PRECISION) ==="
  python3 "$HERE/make_vit_onnx.py" --size "$size" --res "$res" --frames "$frames" --out "$WORK/model.onnx"
  python3 "$HERE/make_image_calib.py" --res "$res" --frames "$frames" --count 16 --out "$WORK/calib.tar"
  cp "$HERE/vit.$PRECISION.json" "$WORK/config.json"
  for mode in npu vnpu; do
    if [ $mode = npu ]; then npu_mode=NPU2; else npu_mode=NPU1; fi
    sed "s/\"NPU2\"/\"$npu_mode\"/" "$WORK/config.json" >"$WORK/config.$mode.json"
    docker run --rm -v "$WORK":/data -w /data "$IMAGE" \
      pulsar2 build --target_hardware AX620E \
        --input model.onnx \
        --output_dir build_$mode \
        --config config.$mode.json
  done
  mkdir -p "out/$name"
  cp "$WORK/build_npu/compiled.axmodel" "out/$name/${name}_npu.axmodel"
  cp "$WORK/build_vnpu/compiled.axmodel" "out/$name/${name}_vnpu.axmodel"
  cat >"out/$name/$name.mud" <<MUD
[basic]
type = axmodel
model_npu = ${name}_npu.axmodel
model_vnpu = ${name}_vnpu.axmodel

[extra]
model_type = vit_bench
input_type = float32
MUD
  rm -rf "$WORK"
done
echo "done: $(pwd)/out"
