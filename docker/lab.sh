#!/usr/bin/env bash
# JupyterLab with real-time collaboration: everyone with the token edits the same notebook live.
set -euo pipefail
if [ -z "${JUPYTER_TOKEN:-}" ] || [ "$JUPYTER_TOKEN" = change-me ]; then
  echo "Set JUPYTER_TOKEN in .env (see docker/.env.example)" >&2
  exit 1
fi
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \
  --ServerApp.root_dir=/workspace/odm-maixcam2 \
  --IdentityProvider.token="$JUPYTER_TOKEN" \
  --SQLiteYStore.db_path=/work/.jupyter_ystore.db \
  "$@"
