#!/bin/sh
# Runs the dashboard headless against every duck_stress.py scenario and fails
# on a crash, a sanitizer report, a stalled sampler or memory that keeps growing.
#
#   tools/stress.sh [BINARY] [SECONDS_PER_SCENARIO]
#
# `make stress` builds BINARY with AddressSanitizer + UBSan first. On the board,
# point it at the installed binary (no sanitizers there):
#   sh stress.sh /maixapp/apps/duck_dash/duck_dash 60
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${1:-$HERE/../build/duck_dash}
SECS=${2:-20}
SCENARIOS=${SCENARIOS:-"nominal faults disconnect extreme rate chaos soak"}
PY=${PYTHON:-python3}
FPS=${FPS:-30}
OUT=${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/duck_stress.XXXXXX")}
FEED=$OUT/feed
export UBSAN_OPTIONS=${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=0}
export ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=1}
fail=0

echo "duck_dash stress: $BIN, ${SECS}s per scenario, logs in $OUT"
for sc in $SCENARIOS; do
    for page in 0 1; do
        name=$sc.page$page
        "$PY" "$HERE/duck_stress.py" -s "$sc" --feed "$FEED" --duration $((SECS + 2)) --seed 7 -q &
        gen=$!
        sleep 0.5
        frames=$((SECS * FPS))
        "$BIN" --feed "$FEED" --camera off --servo-port /dev/null --no-imu --no-feet --config "$OUT/none.json" \
            --fps "$FPS" --frames "$frames" --page "$page" --stats=5 --dump >"$OUT/$name.out" 2>"$OUT/$name.err"
        rc=$?
        wait $gen
        total=$(grep '^stats_total' "$OUT/$name.out")
        verdict=ok
        if [ $rc -ne 0 ]; then verdict="exit $rc"; fi
        if grep -q "runtime error\|ERROR: AddressSanitizer\|ERROR: LeakSanitizer" "$OUT/$name.err"; then verdict="sanitizer"; fi
        hz=$(echo "$total" | sed -n 's/.*sampler_hz=\([0-9.]*\).*/\1/p')
        if [ -n "$hz" ] && [ "${hz%.*}" -lt 15 ]; then verdict="sampler ${hz} Hz"; fi
        # memory: the last 5 s report must not be more than 2 MB above the first
        # (RSS_CHECK=0 under ASan, whose allocator quarantine grows on purpose)
        r0=$(grep '^stats t=' "$OUT/$name.err" | head -1 | sed -n 's/.*rss_kb=\([0-9]*\).*/\1/p')
        r1=$(grep '^stats t=' "$OUT/$name.err" | tail -1 | sed -n 's/.*rss_kb=\([0-9]*\).*/\1/p')
        if [ "${RSS_CHECK:-1}" = 1 ] && [ -n "$r0" ] && [ -n "$r1" ] && [ $((r1 - r0)) -gt 2048 ]; then verdict="rss grew ${r0}->${r1} kB"; fi
        [ "$verdict" = ok ] || fail=1
        printf '%-20s %-10s %s\n' "$name" "$verdict" "$(echo "$total" | sed 's/^stats_total //')"
        if [ "$verdict" != ok ]; then grep -m5 "runtime error\|ERROR" "$OUT/$name.err"; fi
    done
done
rm -f "$FEED"
[ $fail -eq 0 ] && echo "all scenarios passed" || echo "FAILED (logs in $OUT)"
exit $fail
