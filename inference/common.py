"""Shared utilities for the Van Gogh neural rendering pipeline.

Provides canonical paths, image I/O helpers and small numeric utilities so
every stage of the pipeline (SAM / depth / RAFT / export) speaks the same
language on disk:

    data/input/starry_night.png   -> pipeline input (the painting)
    output/layers/*.png           -> semantic layer cut-outs (RGBA)
    output/depth/*                -> fused depth maps
    output/flow/*                 -> RAFT dense flow fields
    output/elements.json          -> detected element manifest
    renderer/web/assets/          -> exported WebGL bundle
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
LAYERS_DIR = OUTPUT_DIR / "layers"
DEPTH_DIR = OUTPUT_DIR / "depth"
FLOW_DIR = OUTPUT_DIR / "flow"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
WEB_ASSETS_DIR = PROJECT_ROOT / "renderer" / "web" / "assets"

DEFAULT_IMAGE = DATA_DIR / "starry_night.png"

# Canonical layer names required by the project specification.
LAYER_NAMES = ["sky", "stars", "moon", "tree", "village"]

# Relative depth ordering of the semantic layers (0 = far, 1 = near).
# This encodes art-historical knowledge of the composition: the night sky
# and its whirlpools are the distant background, Venus and the eleven stars
# hang in the middle distance, the waning moon glows slightly closer, the
# sleeping village with its low mountain ridge sits near the horizon, and
# the flame-like cypress is the unmistakable foreground sentinel.
LAYER_DEPTH_PRIOR = {
    "sky": 0.08,
    "stars": 0.30,
    "moon": 0.42,
    "village": 0.74,
    "tree": 1.00,
}


def ensure_dirs() -> None:
    """Create every output directory used by the pipeline."""
    for d in (LAYERS_DIR, DEPTH_DIR, FLOW_DIR, CHECKPOINT_DIR, WEB_ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_bgr(path: str | Path) -> np.ndarray:
    """Load an image as BGR uint8 (OpenCV convention)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_png(path: str | Path, img: np.ndarray) -> None:
    """Save an image (any channel count) as PNG, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def to_unit(u8: np.ndarray) -> np.ndarray:
    """uint8 [0..255] -> float32 [0..1]."""
    return u8.astype(np.float32) / 255.0


def from_unit(f: np.ndarray) -> np.ndarray:
    """float32 [0..1] -> uint8 [0..255] with clipping."""
    return (np.clip(f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def normalize01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an arbitrary float array to [0, 1]."""
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def write_json(path: str | Path, obj) -> None:
    """Pretty-print a JSON object, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def largest_component(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return (mask > 0).astype(np.uint8) * 255
    areas = stats[1:, cv2.CC_STAT_AREA].copy()
    if min_area > 0:
        areas[areas < min_area] = 0
    best = 1 + int(np.argmax(areas))
    return (labels == best).astype(np.uint8) * 255


def feather_mask(mask_u8: np.ndarray, ksize: int = 7, blur: float = 2.0) -> np.ndarray:
    """Feather a binary mask so layer cut-outs blend without hard seams."""
    m = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE,
                         np.ones((ksize, ksize), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), blur)
    return m


def report_ram() -> None:
    """Print current RAM usage (the sandbox is memory constrained)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    print(f"    [mem] available: "
                          f"{int(line.split()[1]) / 1024 / 1024:.2f} GB")
                    return
    except OSError:
        pass
