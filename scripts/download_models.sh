#!/usr/bin/env bash
# =====================================================================
# Download the lightweight model checkpoints for the Van Gogh neural
# rendering pipeline (SAM ViT-B + MiDaS-small + RAFT-small).
#
# Total download: ~470 MB. Full-quality alternatives (SAM ViT-H 2.4 GB,
# ZoeDepth-NK 1.3 GB, RAFT-large) are supported by the code — see the
# README — but not required by the default lightweight build.
# =====================================================================
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="$HERE/checkpoints"
mkdir -p "$CKPT_DIR"

dl () {  # dl <url> <dest> <md5>
  local url="$1" dest="$2" want="$3"
  if [ -f "$dest" ]; then
    echo "[ok] $(basename "$dest") already present"
    return 0
  fi
  echo "[..] downloading $(basename "$dest")"
  if ! curl -fSL --retry 3 --progress-bar -o "$dest" "$url"; then
    echo "[!!] FAILED: $url" >&2
    rm -f "$dest"
    return 1
  fi
  if [ -n "$want" ]; then
    local got
    got="$(md5sum "$dest" | cut -d' ' -f1)"
    if [ "$got" != "$want" ]; then
      echo "[!!] checksum mismatch for $(basename "$dest")" >&2
      echo "     want $want"
      echo "     got  $got"
      return 1
    fi
  fi
  echo "[ok] $(basename "$dest")"
}

fail=0

# ---- 1. Segment Anything, ViT-B (375 MB) ----------------------------
dl "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
   "$CKPT_DIR/sam_vit_b_01ec64.pth" \
   "01ec64" || fail=1
# (md5 above is the abbreviated hash embedded in the official filename;
#  a byte-accurate check is done at load time by the SAM loader.)

# ---- 2. MiDaS-small v2.1 (86 MB) + vendored model code ---------------
dl "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt" \
   "$CKPT_DIR/midas_v21_small_256.pt" "" || fail=1

if [ ! -d "$CKPT_DIR/MiDaS_repo" ]; then
  echo "[..] vendoring MiDaS model code (no torch.hub clone at runtime)"
  curl -fSL --retry 3 --progress-bar \
       -o /tmp/midas.zip \
       "https://github.com/isl-org/MiDaS/archive/refs/heads/master.zip" \
    && unzip -q -o /tmp/midas.zip -d /tmp \
    && mv /tmp/MiDaS-master "$CKPT_DIR/MiDaS_repo" \
    && rm /tmp/midas.zip \
    && echo "[ok] MiDaS_repo vendored" || fail=1
fi

if [ ! -d "$CKPT_DIR/gen_effnet" ]; then
  echo "[..] vendoring gen-efficientnet-pytorch (MiDaS-small backbone)"
  curl -fSL --retry 3 --progress-bar \
       -o /tmp/geneff.zip \
       "https://github.com/rwightman/gen-efficientnet-pytorch/archive/refs/heads/master.zip" \
    && unzip -q -o /tmp/geneff.zip -d /tmp \
    && mv /tmp/gen-efficientnet-pytorch-master "$CKPT_DIR/gen_effnet" \
    && rm /tmp/geneff.zip \
    && echo "[ok] gen_effnet vendored" || fail=1
fi

# ---- 3. RAFT-small (3.8 MB, torchvision CDN) -------------------------
# Weights are fetched automatically by torchvision on first use; this
# pre-warms the torch hub cache so the pipeline never needs the network.
python3 - <<'PY' || fail=1
try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    raft_small(weights=Raft_Small_Weights.DEFAULT)
    print("[ok] raft_small weights cached")
except Exception as exc:  # noqa: BLE001
    print(f"[!!] RAFT pre-download failed: {exc}")
    raise
PY

echo
if [ "$fail" -ne 0 ]; then
  echo "Some downloads failed — re-run this script when the network is back."
  exit 1
fi
echo "All lightweight checkpoints ready under $CKPT_DIR"
echo "Run the pipeline:  python -m inference.run_pipeline"
