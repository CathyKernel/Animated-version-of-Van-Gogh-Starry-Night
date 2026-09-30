"""Stage 2 — Monocular depth estimation with semantic-prior fusion.

The specification names ZoeDepth; this lightweight build defaults to
MiDaS-small (66 MB, runs comfortably on CPU) and can switch to full
ZoeDepth with ``--model zoedepth`` when the checkpoint is available
(see scripts/download_models.sh).

A raw neural depth map of a *painting* is only plausible — networks trained
on photographs rank brush-stroke luminance as much as scene geometry. This
stage therefore fuses two signals, the way paper-level pipelines do:

    depth = alpha * normalize(MiDaS) + (1 - alpha) * layer_prior

where ``layer_prior`` is the SAM-derived semantic ordering of the five
layers (sky farthest ... cypress nearest — the art-historical reading of
the composition). The fused map is smoothed with an edge-aware joint
bilateral filter so parallax displacements never tear the brushwork.

Outputs:
    output/depth/depth.png         16-bit grayscale (renderer input)
    output/depth/depth_colored.png turbo colormap visualization
    output/depth/depth_raw.npy     float32 raw map (debugging / QA)
    output/elements.json           gains per-layer depth statistics

Usage:
    python -m inference.depth_estimate [--image PATH] [--model midas|zoedepth]
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import common as C  # noqa: E402


def load_layer_prior() -> np.ndarray | None:
    """Build the semantic depth prior from stage-1 layer masks."""
    prior = None
    for name in C.LAYER_NAMES:
        p = C.LAYERS_DIR / f"{name}.png"
        if not p.exists():
            return None
        rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            return None
        m = (rgba[..., 3] > 127).astype(np.float32)
        if prior is None:
            prior = np.zeros(m.shape, np.float32)
        prior += m * C.LAYER_DEPTH_PRIOR[name]
    # Sky pixels keep prior 0.08 (sky is the base value) — already set.
    return prior


def run_midas(bgr: np.ndarray, model: str = "midas") -> np.ndarray:
    """Neural monocular depth, near = 1.0, far = 0.0.

    Lightweight build: MiDaS-small v2.1 loaded from the vendored model
    code (checkpoints/MiDaS_repo) and local checkpoint — no torch.hub
    cloning at runtime. Set --model zoedepth for the full ZoeDepth-NK
    variant (see scripts/download_models.sh).
    """
    import torch

    if model == "zoedepth":
        from PIL import Image
        print("  [depth] loading ZoeDepth (ZoeD_NK) via torch.hub")
        zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
        zoe.eval()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            out = zoe.infer_pil(Image.fromarray(rgb))
        dep = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
        dep = C.normalize01(dep.astype(np.float32))
        del zoe
        gc.collect()
        return dep

    repo = C.CHECKPOINT_DIR / "MiDaS_repo"
    ckpt = C.CHECKPOINT_DIR / "midas_v21_small_256.pt"
    if not repo.exists() or not ckpt.exists():
        raise FileNotFoundError(
            "MiDaS-small assets missing under checkpoints/ — run "
            "scripts/download_models.sh first.")
    sys.path.insert(0, str(repo))
    from midas.midas_net_custom import MidasNet_small
    from midas.transforms import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose

    print(f"  [depth] MiDaS-small v2.1 from local checkpoint "
          f"({ckpt.stat().st_size >> 20} MB)")
    net = MidasNet_small(str(ckpt), features=64, backbone="efficientnet_lite3",
                         exportable=True, non_negative=True,
                         blocks={"expand": True})
    net.eval()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tf = Compose([
        lambda im: {"image": im},
        Resize(256, 256, resize_target=None, keep_aspect_ratio=True,
               ensure_multiple_of=32, resize_method="upper_bound",
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
        lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
    ])
    with torch.no_grad():
        prediction = net(tf(rgb))
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=bgr.shape[:2],
            mode="bicubic", align_corners=False).squeeze().cpu().numpy()
    del net
    gc.collect()
    # MiDaS outputs a *disparity-like* map: larger value = closer.
    return C.normalize01(prediction.astype(np.float32))


def fuse_depth(neural: np.ndarray, prior: np.ndarray | None,
               alpha: float = 0.55) -> np.ndarray:
    """Blend neural depth with the semantic prior, then smooth jointly."""
    fused = neural if prior is None else alpha * neural + (1.0 - alpha) * prior
    # Gentle contrast shaping: keep mid-tones, compress extremes.
    fused = np.clip(fused, 0.0, 1.0) ** 0.92
    return fused.astype(np.float32)


def smooth_depth(depth: np.ndarray, guide: np.ndarray,
                 d: int = 9, sigma_color: float = 40.0) -> np.ndarray:
    """Edge-aware smoothing so layer boundaries stay crisp while interiors
    become flat parallax planes."""
    if not hasattr(cv2, "ximgproc"):
        return depth
    dep8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    out = cv2.ximgproc.jointBilateralFilter(guide, dep8, d, sigma_color, 9)
    return out.astype(np.float32) / 255.0


def run_depth(image: str | Path = C.DEFAULT_IMAGE, model: str = "midas") -> dict:
    """Entry point: painting -> fused depth map + visualizations."""
    C.ensure_dirs()
    bgr = C.load_bgr(image)
    print(f"[stage 2] depth estimation ({model}): {image}")
    neural = run_midas(bgr, model)
    C.report_ram()
    prior = load_layer_prior()
    if prior is not None:
        print("  [depth] fusing with SAM layer prior "
              f"(alpha=0.55, layers={len(C.LAYER_NAMES)})")
    fused = fuse_depth(neural, prior)
    guide = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                             (0, 0), 2.0)
    fused = smooth_depth(fused, guide)

    # --- outputs -------------------------------------------------------------
    np.save(C.DEPTH_DIR / "depth_raw.npy", fused.astype(np.float32))
    depth_u16 = (np.clip(fused, 0, 1) * 65535.0 + 0.5).astype(np.uint16)
    # Encode in the R+G channels of a PNG (16-bit grayscale is fine too).
    cv2.imwrite(str(C.DEPTH_DIR / "depth.png"), depth_u16)
    colored = cv2.applyColorMap(cv2.convertScaleAbs(
        (fused * 255).astype(np.uint8), alpha=1.0), cv2.COLORMAP_TURBO)
    C.save_png(C.DEPTH_DIR / "depth_colored.png", colored)

    # --- manifest stats --------------------------------------------------------
    stats = {}
    for name in C.LAYER_NAMES:
        p = C.LAYERS_DIR / f"{name}.png"
        if not p.exists():
            continue
        rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3:
            continue
        m = rgba[..., 3] > 127
        if m.sum() < 10:
            continue
        stats[name] = {
            "mean": round(float(fused[m].mean()), 4),
            "std": round(float(fused[m].std()), 4),
            "min": round(float(fused[m].min()), 4),
            "max": round(float(fused[m].max()), 4),
        }
    manifest_path = C.OUTPUT_DIR / "elements.json"
    if manifest_path.exists():
        manifest = C.read_json(manifest_path)
        manifest["depth"] = {
            "model": model,
            "fusion": "0.55*neural + 0.45*layer_prior" if prior is not None
                      else "neural_only",
            "layers": stats,
        }
        C.write_json(manifest_path, manifest)
    print(f"[stage 2] done: depth.png (16-bit), depth_colored.png, "
          f"depth_raw.npy -> {C.DEPTH_DIR}")
    return {"depth": fused, "stats": stats}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=str(C.DEFAULT_IMAGE))
    ap.add_argument("--model", default="midas", choices=["midas", "zoedepth"])
    args = ap.parse_args()
    run_depth(args.image, args.model)
