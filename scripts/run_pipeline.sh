#!/usr/bin/env bash
# One-command end-to-end run of the Van Gogh neural rendering pipeline.
# Usage: bash scripts/run_pipeline.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

if [ ! -f "checkpoints/sam_vit_b_01ec64.pth" ]; then
  echo "Checkpoints missing — running scripts/download_models.sh first"
  bash scripts/download_models.sh
fi

python3 -m inference.run_pipeline "$@"

echo
echo "Done. View the result:"
echo "  cd renderer/web && python3 -m http.server 8000"
echo "  open http://localhost:8000"
