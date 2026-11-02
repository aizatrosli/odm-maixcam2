#!/usr/bin/env bash
# Make the writable folders under /work (it is a bind mount owned by the host user).
set -e
mkdir -p "$HOME" "$XDG_CACHE_HOME" /work/runs /work/camsim /work/notebooks 2>/dev/null || true
exec "$@"
