"""Stage 3 — RAFT dense optical flow of the animated scene.

A painting is static, so RAFT has nothing to measure by itself. This stage
therefore follows the paper-level recipe for making a still artwork move:

  1. **Motion synthesis** — render "frame B": the painting warped by the
     animation's designed motion field (depth-parallax camera drift, the
     twin whirlpool rotation, star-halo spin, moon breathing, cypress
     sway). Frame A is the untouched painting.
  2. **RAFT estimation** — run RAFT-small (torchvision) on the pair
     (A, B). The estimated dense flow is the *measured* motion of the
     composed animation, in pixels, at 960x760, upsampled to full canvas.

The WebGL renderer later uses this RAFT flow texture as a dense advection
field that streams the brushwork along the measured motion, on top of the
crisp analytic element animations (rotation, sway, flicker).

Outputs:
    output/flow/frame_pair.png   A | B side-by-side (what RAFT saw)
    output/flow/flow.npy         float32 HxWx2, pixels (full resolution)
    output/flow/flow_visual.png  Middlebury colormap visualization
    output/flow/flow_arrows.png  vector-field overlay on the painting
    output/flow/flow_texture.png 8-bit RG-encoded texture for WebGL
    output/elements.json         gains "flow" stats + texture scale

Usage:
    python -m inference.optical_flow [--image PATH]
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

# RAFT working resolution (divisible by 8, close to the canvas ratio).
RAFT_SIZE = (960, 760)


# ---------------------------------------------------------------------------
# 1. Motion synthesis
# ---------------------------------------------------------------------------

def load_depth() -> np.ndarray | None:
    p = C.DEPTH_DIR / "depth_raw.npy"
    if not p.exists():
        return None
    return np.load(str(p)).astype(np.float32)


def load_manifest() -> dict:
    p = C.OUTPUT_DIR / "elements.json"
    return C.read_json(p) if p.exists() else {}


def load_layer_mask(name: str) -> np.ndarray:
    p = C.LAYERS_DIR / f"{name}.png"
    if not p.exists():
        return None
    rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return (rgba[..., 3] > 127).astype(np.float32) if rgba is not None else None


def rot2(x: np.ndarray, y: np.ndarray, cx: float, cy: float, ang: float):
    """Rotate coordinate grids around (cx, cy) by ``ang`` radians."""
    dx, dy = x - cx, y - cy
    c, s = np.cos(ang), np.sin(ang)
    return dx * c - dy * s + cx, dx * s + dy * c + cy


def synthesize_motion(h: int, w: int, manifest: dict,
                      depth: np.ndarray | None) -> np.ndarray:
    """The designed animation displacement field, in pixels: where each
    pixel at position p moves after ``dt`` (~0.5 s of animation)."""
    x = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    disp = np.zeros((h, w, 2), np.float32)
    dep = depth if depth is not None else np.full((h, w), 0.5, np.float32)

    # --- (a) depth parallax: camera drifts 6 px right & 3 px down ---------
    cam = np.array([6.0, 3.0], np.float32)
    disp += (dep[..., None] - 0.5) * cam[None, None, :]

    # --- (b) twin whirlpools rotate ---------------------------------------
    sky = load_layer_mask("sky")
    for v in manifest.get("vortices", [])[:2]:
        cx, cy = v["x"] * w, v["y"] * h
        r = v["r"] * w
        ang = np.deg2rad(3.0) * v.get("dir", 1.0)  # 3 degrees of turn
        rx, ry = rot2(x, y, cx, cy, ang)
        d = np.stack([rx - x, ry - y], axis=-1)
        dist = np.hypot(x - cx, y - cy)
        fall = (1.0 - np.clip(dist / r, 0, 1)) ** 1.5
        if sky is not None:
            fall = fall * (0.35 + 0.65 * sky)
        disp += d * fall[..., None]

    # --- (c) star halos spin -----------------------------------------------
    for s in manifest.get("stars", []):
        cx, cy = s["x"] * w, s["y"] * h
        r = max(s["r"] * w, 4.0)
        ang = np.deg2rad(2.2) * (1.0 if (int(s["x"] * 997) % 2) else -1.0)
        rx, ry = rot2(x, y, cx, cy, ang)
        d = np.stack([rx - x, ry - y], axis=-1)
        dist = np.hypot(x - cx, y - cy)
        fall = (1.0 - np.clip(dist / (r * 2.2), 0, 1)) ** 2
        disp += d * fall[..., None] * 0.8

    # --- (d) moon breathes (slow radial expansion) --------------------------
    mo = manifest.get("moon", {})
    if mo:
        cx, cy = mo["x"] * w, mo["y"] * h
        rad = mo.get("halo", 0.13) * w
        dx, dy = x - cx, y - cy
        dist = np.hypot(dx, dy) + 1e-3
        fall = np.exp(-(dist / rad) ** 2)
        disp += np.stack([dx / dist, dy / dist], axis=-1) * (1.2 * fall)[..., None]

    # --- (e) cypress sways (amplitude ramp: treetop -> roots) ---------------
    tree = load_layer_mask("tree")
    if tree is not None:
        cyp = manifest.get("cypress", {})
        top = cyp.get("top", 0.05)
        bottom = cyp.get("bottom", 0.90)
        ramp = np.clip((bottom - y / h) / max(bottom - top, 0.1), 0, 1) ** 1.25
        sway = 4.0 * np.sin(0.9 * x / w * np.pi) * ramp * tree
        disp[..., 0] += sway

    return disp


def warp_forward(bgr: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """Render frame B = painting displaced by ``disp`` (inverse-mapped so
    no holes appear: B(p) = A(p - disp(p)))."""
    h, w = bgr.shape[:2]
    x = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    map_x = x - disp[..., 0]
    map_y = y - disp[..., 1]
    return cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# 2. RAFT estimation
# ---------------------------------------------------------------------------

def run_raft_model(bgr_a: np.ndarray, bgr_b: np.ndarray) -> np.ndarray:
    """RAFT-small (torchvision) on the (A, B) pair -> flow HxWx2, pixels."""
    import torch
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    h, w = RAFT_SIZE
    a = cv2.resize(bgr_a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(bgr_b, (w, h), interpolation=cv2.INTER_AREA)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def prep(img):
        t = torch.from_numpy(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)
        return (t - mean) / std

    print("  [RAFT] loading raft_small (C_T_V2 weights, auto-download)")
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights).eval()
    print("  [RAFT] estimating dense flow at "
          f"{w}x{h} (CPU, ~12 iterations)")
    with torch.no_grad():
        flows = model(prep(a), prep(b))
    flow = flows[-1][0].permute(1, 2, 0).numpy()  # HxWx2
    del model, flows
    gc.collect()

    # Upsample to full canvas resolution.
    flow = cv2.resize(flow, (bgr_a.shape[1], bgr_a.shape[0]),
                      interpolation=cv2.INTER_LINEAR)
    return flow.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Visualisation / export
# ---------------------------------------------------------------------------

def flow_to_color(flow: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """Middlebury color wheel encoding: hue = direction, sat = 1,
    value = magnitude."""
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    if max_mag is None:
        max_mag = float(np.percentile(mag, 99)) + 1e-3
    hsv[..., 0] = ((ang * 180.0 / np.pi + 360.0) / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_arrows(bgr: np.ndarray, flow: np.ndarray, step: int = 28) -> np.ndarray:
    out = bgr.copy()
    h, w = flow.shape[:2]
    for yv in range(step // 2, h, step):
        for xv in range(step // 2, w, step):
            u, v = flow[yv, xv]
            m = float(np.hypot(u, v))
            if m < 0.6:
                continue
            scale = 3.0
            pt2 = (int(xv + u * scale), int(yv + v * scale))
            col = (0, 255, 255) if m > 4 else (180, 220, 255)
            cv2.arrowedLine(out, (xv, yv), pt2, col, 1, cv2.LINE_AA, tipLength=0.25)
    return out


def run_raft(image: str | Path = C.DEFAULT_IMAGE) -> dict:
    """Entry point: painting -> synthesized frame B -> RAFT flow bundle."""
    C.ensure_dirs()
    bgr = C.load_bgr(image)
    h, w = bgr.shape[:2]
    print(f"[stage 3] RAFT optical flow: {image} ({w}x{h})")
    manifest = load_manifest()
    depth = load_depth()

    disp = synthesize_motion(h, w, manifest, depth)
    frame_b = warp_forward(bgr, disp)
    pair = np.hstack([bgr, frame_b])
    C.save_png(C.FLOW_DIR / "frame_pair.png", pair)
    print(f"  [flow] synthesized frame B (motion pair -> "
          f"{C.FLOW_DIR / 'frame_pair.png'})")

    flow = run_raft_model(bgr, frame_b)

    mag = np.linalg.norm(flow, axis=-1)
    stats = {
        "meanMag": round(float(mag.mean()), 3),
        "p99Mag": round(float(np.percentile(mag, 99)), 3),
        "maxMag": round(float(mag.max()), 3),
    }
    scale = max(stats["p99Mag"], 1.0)

    np.save(C.FLOW_DIR / "flow.npy", flow)
    C.save_png(C.FLOW_DIR / "flow_visual.png", flow_to_color(flow, scale))
    C.save_png(C.FLOW_DIR / "flow_arrows.png", draw_arrows(bgr, flow))

    # 8-bit RG texture for WebGL: u,v in [-scale, scale] -> [0, 255].
    u8 = np.zeros((h, w, 3), np.uint8)
    u8[..., 0] = np.clip((flow[..., 0] / scale + 1.0) * 127.5, 0, 255)
    u8[..., 1] = np.clip((flow[..., 1] / scale + 1.0) * 127.5, 0, 255)
    C.save_png(C.FLOW_DIR / "flow_texture.png", u8)

    if manifest:
        manifest["flow"] = {**stats, "textureScale": round(scale, 3),
                            "resolution": [w, h]}
        C.write_json(C.OUTPUT_DIR / "elements.json", manifest)
    print(f"[stage 3] done: flow.npy, flow_visual.png, flow_arrows.png, "
          f"flow_texture.png (scale={scale:.2f}px) -> {C.FLOW_DIR}")
    return {"flow": flow, "stats": stats}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=str(C.DEFAULT_IMAGE))
    args = ap.parse_args()
    run_raft(args.image)
