# -*- coding: utf-8 -*-
"""
================================================================
 The Starry Night — Enhanced Animation Generator
 (python/starry_night_enhanced.py)
================================================================
A rebuilt & upgraded version of
CathyKernel/Animated-version-of-Van-Gogh-Starry-Night.

Key improvements over the original script:
  1. [Faithful colors] The original enhance_colors_dynamically
     (LAB saturation/contrast oscillation) has been removed — it
     pulled the palette far away from the painting. Colors now stay
     100% true to the original: animation only changes brightness
     and position, never hue.
  2. [Element-level animation] Automatic element detection
     (stars / moon / cypress / village lights), then per-element
     animation:
       - Stars: twinkle at their real positions (incl. Venus) with
         independent phases + breathing halos
       - Moon: brightness breathing + halo pulsing
       - Cypress: crown sways in the wind (wide at the top, steady
         at the roots)
       - Village: window lights flicker like candlelight
  3. [Star particles] The original scattered particles randomly
     across the whole frame (they landed on the cypress/ground);
     particles now emit from the real star positions.
  4. [Better swirls] The flow field is centered on the painting's
     twin whirlpools and only affects the sky region — the village
     and cypress are no longer warped.
  5. [Performance] The oil-painting filter was rewritten from pure
     Python double loops (minutes per frame at 1024px) to a
     vectorized implementation (tens of milliseconds per frame).
  6. [Glow] Bloom-style glow whose color is sampled from the
     painting's own bright areas, instead of a fixed yellow overlay.

Usage:
  python starry_night_enhanced.py                     # enhanced_starry_night.mp4
  python starry_night_enhanced.py --duration 12 --fps 30 --size 1024
  python starry_night_enhanced.py --no-oil --no-vignette
Dependencies: pip install opencv-python numpy requests pillow
================================================================
"""
import argparse
import os
import random
import sys
from io import BytesIO

import cv2
import numpy as np
import requests

try:
    from PIL import Image
except ImportError:
    Image = None

# ---------------------------------------------------------------- config
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
]

# Fallback star positions used when detection fails (curated on the
# 1280px Wikimedia scan and double-checked with a vision model)
FALLBACK_STARS = [
    (0.352, 0.512, 0.060), (0.107, 0.050, 0.055), (0.368, 0.049, 0.042),
    (0.228, 0.036, 0.030), (0.476, 0.016, 0.030), (0.614, 0.092, 0.050),
    (0.815, 0.278, 0.045), (0.131, 0.479, 0.048), (0.046, 0.452, 0.035),
    (0.237, 0.147, 0.035), (0.694, 0.206, 0.040),
]
FALLBACK_MOON = (0.926, 0.137, 0.135)


# ================================================================ basics (same as original)
def load_image(url_list):
    """Download the painting (multiple mirrors; or pass --image for a local file)."""
    for url in url_list:
        try:
            print(f"Downloading painting: {url}")
            resp = requests.get(url, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                "Referer": "https://commons.wikimedia.org/",
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            })
            if resp.status_code != 200:
                continue
            if Image is None:
                arr = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
                return arr
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"  Failed: {e}")
    raise RuntimeError("Could not download the painting — pass a local file with --image")


def apply_flow(img, flow):
    """Resample the image by a displacement field (as in the original)."""
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    flow_x = np.clip(x + flow[:, :, 0], 0, w - 1)
    flow_y = np.clip(y + flow[:, :, 1], 0, h - 1)
    return cv2.remap(img, flow_x.astype(np.float32), flow_y.astype(np.float32), cv2.INTER_CUBIC)


# ================================================================ new: element detection
def detect_elements(img):
    """
    Locate stars / moon / cypress / window lights.
    Returns dict(stars=[(x,y,r,phase,speed,amp)...], moon=(x,y,halo_r),
              cypress_mask=HxW uint8 sway-amplitude map, windows=[(x,y,r,phase,speed)...])
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hc = hsv[..., 0].astype(np.int32)
    Sc = hsv[..., 1].astype(np.int32)
    Vc = hsv[..., 2].astype(np.int32)
    yy, xx = np.ogrid[:h, :w]
    rng = random.Random(42)

    # ---- cypress: tall dark olive-green blob on the left
    cyp_zone = (xx < 0.30 * w) & (yy > 0.04 * h) & (yy < 0.90 * h)
    cyp_dark = (Vc < 115) & (Hc >= 18) & (Hc <= 85) & cyp_zone
    m = (cyp_dark.astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    cypress = np.zeros((h, w), np.uint8)
    if n > 1:
        best = 1 + int(np.argmax(stats[1:, 4] * 0.1 + stats[1:, 3] * 2.0))
        cypress = (labels == best).astype(np.uint8) * 255
        cypress = cv2.morphologyEx(cypress, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cypress = cv2.GaussianBlur(cypress, (0, 0), 2.0)
    ys_ = np.nonzero(cypress > 60)[0]
    cyp_top = ys_.min() / h if len(ys_) else 0.06
    cyp_bottom = ys_.max() / h if len(ys_) else 0.85
    # sway amplitude: 1 at the treetop -> 0 at the roots
    ramp = np.clip((cyp_bottom - yy / h) / max(cyp_bottom - cyp_top, 0.1), 0, 1) ** 1.25
    sway_map = ((cypress / 255.0) * ramp * 255).astype(np.uint8)

    cyp_excl = cv2.dilate(cypress, np.ones((25, 25), np.uint8)) > 0

    # ---- stars: bright warm blobs in the sky (cypress & moon excluded)
    sky = (yy < 0.56 * h) & (~cyp_excl)
    bright_warm = (Vc > 135) & (Sc > 25) & (Hc >= 10) & (Hc <= 52)
    bright_white = (Vc > 208) & (Sc < 80)
    bright = (bright_warm | bright_white) & sky
    bm = cv2.morphologyEx((bright.astype(np.uint8)) * 255, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    n2, l2, st2, ce2 = cv2.connectedComponentsWithStats(bm, 8)

    moon = None
    moon_zone = (xx > 0.52 * w) & (yy < 0.40 * h)
    moon_label, best_area = 0, 0
    for i in range(1, n2):
        cx, cy = ce2[i]
        if st2[i, 4] > best_area and moon_zone[int(cy), int(cx)]:
            best_area, moon_label = st2[i, 4], i
    if moon_label:
        sel = (l2 == moon_label) & (Vc > 225)
        if sel.sum() > 30:
            mcx, mcy = np.nonzero(sel)[1].mean(), np.nonzero(sel)[0].mean()
        else:
            mcx, mcy = ce2[moon_label]
        moon = (float(mcx / w), float(mcy / h), max(0.10, best_area ** 0.5 / w))
    if moon is None or abs(w / h - 1.2623) < 0.02:
        # standard composition: use the calibrated crescent position
        moon = FALLBACK_MOON

    stars = []
    for i in range(1, n2):
        if i == moon_label:
            continue
        x, y, bw, bh, area = st2[i]
        if area < 20 or area > 25000:
            continue
        cx, cy = ce2[i]
        if np.hypot(cx / w - moon[0], cy / h - moon[1]) < 0.30:
            continue
        stars.append([float(cx / w), float(cy / h), float(max(bw, bh) / 2 / w)])
    # merge nearby fragments
    merged = []
    for s in sorted(stars, key=lambda b: -b[2]):
        if not any(np.hypot(s[0] - t[0], s[1] - t[1]) < 0.045 for t in merged):
            merged.append(s)
    stars = merged[:14]

    # Standard composition (aspect ~1.26): use the curated star list —
    # verified twice with a vision model on the 1280px original; pure
    # threshold detection misses the star below the moon and picks up
    # stray halo fragments at the left edge.
    if abs(w / h - 1.2623) < 0.02:
        stars = [list(s) for s in FALLBACK_STARS]
    elif len(stars) < 6:
        stars = [list(s) for s in FALLBACK_STARS]
    # independent phase/speed per star
    stars = [(*s, rng.uniform(0, 6.28), rng.uniform(1.1, 2.7), 0.34) for s in stars]

    # ---- window lights: small warm blobs in the village band
    village = (yy > 0.78 * h) & (yy < 0.93 * h) & (xx > 0.30 * w) & (xx < 0.97 * w)
    win_warm = (Vc > 100) & (Sc > 45) & (Hc >= 8) & (Hc <= 55) & village
    wm = cv2.morphologyEx((win_warm.astype(np.uint8)) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n3, l3, st3, ce3 = cv2.connectedComponentsWithStats(wm, 8)
    windows = []
    for i in range(1, n3):
        x, y, bw, bh, area = st3[i]
        if 15 <= area <= 220:
            cx, cy = ce3[i]
            windows.append((float(cx / w), float(cy / h),
                            float(max(bw, bh) / w) + 0.006,
                            rng.uniform(0, 6.28), rng.uniform(2.2, 5.5)))
    print(f"Elements: {len(stars)} stars, moon@({moon[0]:.2f},{moon[1]:.2f}), "
          f"{len(windows)} windows, cypress y[{cyp_top:.2f},{cyp_bottom:.2f}]")
    return {"stars": stars, "moon": moon, "sway_map": sway_map, "windows": windows}


# ================================================================ new: element animation
def animate_stars(img, elements, t):
    """Star twinkle + halo breathing (brightness only, hue untouched)."""
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    for x, y, r, phase, speed, amp in elements["stars"]:
        cx, cy, rad = int(x * w), int(y * h), max(int(r * w), 5)
        x0, x1 = max(0, cx - rad * 3), min(w, cx + rad * 3)
        y0, y1 = max(0, cy - rad * 3), min(h, cy + rad * 3)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad
        tw = np.sin(t * speed + phase)
        core = np.exp(-d * d * 1.1)
        halo = np.exp(-d * 1.35)
        gain = 1.0 + amp * tw * (core * 0.85 + halo * 0.30)
        out[y0:y1, x0:x1] *= gain[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def animate_moon(img, elements, t):
    """Moon breathing (~14 s period)."""
    mx, my, mr = elements["moon"]
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    cx, cy, rad = int(mx * w), int(my * h), int(mr * w)
    x0, x1 = max(0, cx - rad * 2), min(w, cx + rad * 2)
    y0, y1 = max(0, cy - rad * 2), min(h, cy + rad * 2)
    if x1 <= x0 or y1 <= y0:
        return img
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad
    tw = np.sin(t * 0.45)
    core = np.exp(-d * d * 9.0)
    halo = np.exp(-d * 1.6)
    gain = 1.0 + 0.12 * tw * (core + halo * 0.4)
    out[y0:y1, x0:x1] *= gain[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def flicker_windows(img, elements, t):
    """Village window lights: candlelight-style two-frequency flicker."""
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    for x, y, r, phase, speed in elements["windows"]:
        cx, cy, rad = int(x * w), int(y * h), max(int(r * w * 3), 4)
        x0, x1 = max(0, cx - rad), min(w, cx + rad)
        y0, y1 = max(0, cy - rad), min(h, cy + rad)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r * w, 1e-5)
        fl = 0.62 * np.sin(t * speed + phase) + 0.38 * np.sin(t * speed * 2.83 + phase * 2.5)
        core = np.exp(-d * d * 1.6)
        gain = 1.0 + 0.55 * fl * core
        out[y0:y1, x0:x1] *= gain[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def sway_cypress(img, sway_map, t, intensity=1.0):
    """Cypress sway: low-frequency swing + high-frequency leaf tremor (widest at the top)."""
    h, w = img.shape[:2]
    yv = np.arange(h, dtype=np.float32) / h
    # displacement field
    dx = (sway_map / 255.0) * (
        8.5 * intensity * np.sin(t * 1.15 + yv * 4.2)[:, None]
        + 3.2 * intensity * np.sin(t * 3.03 + yv * 12.0)[:, None]
    )
    dy = (sway_map / 255.0) * 2.2 * intensity * np.cos(t * 1.0 + yv * 6.5)[:, None]
    flow = np.dstack((dx, dy)).astype(np.float32)
    return apply_flow(img, flow)


# ================================================================ rewritten: star particles
def create_starry_particles(img, elements, t, particles, density=0.05):
    """Particles emit from the REAL star positions (the original scattered
    them randomly across the whole frame, landing on the ground/cypress)."""
    h, w = img.shape[:2]
    stars = elements["stars"]
    # spawn new particles at random star positions each frame
    for _ in range(max(1, int(density * len(stars) * 2))):
        if random.random() < 0.6 and stars:
            sx, sy, sr = random.choice(stars)[:3]
            particles.append({
                "x": sx + random.uniform(-0.01, 0.01),
                "y": sy + random.uniform(-0.01, 0.01),
                "vx": random.uniform(-0.004, 0.004),
                "vy": random.uniform(-0.008, -0.002),
                "life": 1.0, "decay": random.uniform(0.008, 0.02),
                "size": random.randint(1, 2),
            })
    layer = np.zeros_like(img)
    alive = []
    for p in particles:
        p["x"] += p["vx"]; p["y"] += p["vy"]; p["life"] -= p["decay"]
        if p["life"] <= 0:
            continue
        alive.append(p)
        px, py = int(p["x"] * w), int(p["y"] * h)
        b = 255 * p["life"]
        cv2.circle(layer, (px, py), p["size"], (b, b * 0.92, b * 0.75), -1, lineType=cv2.LINE_AA)
    particles[:] = alive
    return cv2.addWeighted(img, 1.0, layer, 0.55, 0)


# ================================================================ improved: bloom-style glow
def add_glow_effect(img, strength=0.35):
    """Glow color is sampled from the painting's own bright areas
    (the original overlaid a fixed yellow, which shifted the colors)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (0, 0), 8)
    mask_f = (mask.astype(np.float32) / 255.0)[..., None]
    glow = cv2.GaussianBlur(img, (0, 0), 9).astype(np.float32)
    out = img.astype(np.float32) * (1 - strength * mask_f * 0.6) + glow * (strength * mask_f * 0.6)
    return np.clip(out, 0, 255).astype(np.uint8)


# ================================================================ improved: vectorized oil-painting filter
def fast_oil_painting(img, radius=4, levels=10):
    """Vectorized oil-painting effect (the original used pure Python double
    loops, taking minutes per 1024px frame)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    step = 256 // levels
    quant = (gray // step).astype(np.int32)
    ksize = radius * 2 + 1
    # per quantization level: indicator map + 3-channel sums via boxFilter
    counts = np.zeros((levels + 1, h, w), np.float32)
    sums = np.zeros((levels + 1, h, w, 3), np.float32)
    img_f = img.astype(np.float32)
    for lv in range(levels + 1):
        ind = (quant == lv).astype(np.float32)
        counts[lv] = cv2.boxFilter(ind, -1, (ksize, ksize), normalize=False)
        for c in range(3):
            sums[lv, :, :, c] = cv2.boxFilter(ind * img_f[:, :, c], -1, (ksize, ksize), normalize=False)
    dominant = np.argmax(counts, axis=0)
    cnt = np.maximum(counts[dominant, np.arange(h)[:, None], np.arange(w)[None, :]], 1.0)
    out = np.empty((h, w, 3), np.uint8)
    for c in range(3):
        s = sums[dominant, np.arange(h)[:, None], np.arange(w)[None, :], c]
        out[:, :, c] = np.clip(s / cnt, 0, 255).astype(np.uint8)
    return out


# ================================================================ kept: intro reveal (original smooth_unfold)
def smooth_unfold_effect(img, progress):
    h, w = img.shape[:2]
    unfolded = img.copy()
    eased = np.sin(progress * np.pi / 2)
    center_y = int(h * eased)
    if center_y < h:
        x = np.linspace(0, w - 1, w)
        y = np.linspace(center_y, h - 1, h - center_y)
        xx, yy = np.meshgrid(x, y)
        curve = 30 * (1 - eased) * (0.5 + 0.5 * np.sin(progress * np.pi * 4))
        zz = curve * np.sin(xx / w * np.pi * 2)
        remapped = cv2.remap(
            img[center_y:], xx.astype(np.float32), (yy + zz - center_y).astype(np.float32),
            interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        unfolded[center_y:] = remapped
    return unfolded


# ================================================================ improved: sky whirlpool flow field
def create_dynamic_swirl_flow_field(shape, t, cypress_excl_mask=None, max_strength=0.09):
    """
    Improvements:
      - whirlpool centers match the painting's twin swirls
        (0.42, 0.25) / (0.63, 0.19) instead of the frame center
      - the flow only affects the sky (fades below y=0.62), so the
        village is never warped
      - gentler strength (the original used 0.15 on the whole frame,
        stretching the houses out of shape)
    """
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx / w
    v = yy / h
    sky_w = np.clip((0.62 - v) / 0.20, 0, 1) ** 1.3
    if cypress_excl_mask is not None:
        sky_w *= (1.0 - cypress_excl_mask.astype(np.float32) / 255.0)

    fx = np.zeros((h, w), np.float32)
    fy = np.zeros((h, w), np.float32)
    # pseudo curl-noise advection (brushwork slowly crawls)
    n1 = cv2.GaussianBlur(np.sin(u * 18 + t * 0.7) * np.cos(v * 13 - t * 0.4), (0, 0), 9)
    n2 = cv2.GaussianBlur(np.cos(u * 15 - t * 0.5) * np.sin(v * 16 + t * 0.6), (0, 0), 9)
    fx += n1 * 2.2
    fy += n2 * 2.2
    # twin whirlpools
    for cx, cy, rad, dirn, spd in [(0.42, 0.25, 0.30, 1.0, 0.55), (0.63, 0.19, 0.20, -1.0, 0.75)]:
        dx, dy = u - cx, v - cy
        d = np.sqrt(dx * dx + dy * dy)
        infl = np.clip(1 - d / rad, 0, 1) ** 1.5
        ang = dirn * 0.16 * infl * np.sin(t * spd + d * 17.0)
        rx = dx * np.cos(ang) - dy * np.sin(ang)
        ry = dx * np.sin(ang) + dy * np.cos(ang)
        fx += (rx - dx) * w * 0.32 * infl
        fy += (ry - dy) * h * 0.32 * infl
    # cloud-band horizontal shear
    fx += 2.8 * np.sin(t * 0.22 + v * 7.5)

    flow = np.dstack((fx * sky_w * max_strength * 10, fy * sky_w * max_strength * 10))
    return flow.astype(np.float32)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="The Starry Night — enhanced animation generator")
    ap.add_argument("--duration", type=float, default=12, help="duration in seconds (default 12)")
    ap.add_argument("--fps", type=int, default=30, help="frame rate (default 30)")
    ap.add_argument("--size", type=int, default=1024, help="frame width in pixels (default 1024)")
    ap.add_argument("--output", default="enhanced_starry_night.mp4", help="output file")
    ap.add_argument("--image", default=None, help="local painting path (skips the download)")
    ap.add_argument("--no-oil", action="store_true", help="disable the oil-painting filter")
    ap.add_argument("--no-vignette", action="store_true", help="disable the vignette")
    ap.add_argument("--no-particles", action="store_true", help="disable star particles")
    args = ap.parse_args()

    print("=" * 56)
    print(" Van Gogh's Starry Night — enhanced animation")
    print(" (element-level motion, colors true to the original)")
    print("=" * 56)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"Cannot read local image: {args.image}")
    else:
        img = load_image(IMAGE_URLS)

    h, w = img.shape[:2]
    new_w = args.size
    img = cv2.resize(img, (new_w, int(h * new_w / w)))
    h, w = img.shape[:2]
    print(f"Working resolution: {w}x{h}")

    elements = detect_elements(img)
    particles = []

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, float(args.fps), (w, h))
    total = int(args.duration * args.fps)
    t_all = args.duration

    print(f"Rendering {total} frames -> {args.output}")
    for frame in range(total):
        t = frame / args.fps
        progress = frame / total

        # 1. intro reveal (first 35% of the timeline, from the original smooth_unfold)
        cur = img if progress > 0.35 else smooth_unfold_effect(img, progress / 0.35)

        # 2. cypress sway (element animation)
        cur = sway_cypress(cur, elements["sway_map"], t)

        # 3. sky whirlpool flow (sky only, gentle strength)
        flow = create_dynamic_swirl_flow_field(cur.shape, t, elements["sway_map"])
        cur = apply_flow(cur, flow)

        # 4. star twinkle / moon breathing / window flicker (element animation)
        cur = animate_stars(cur, elements, t)
        cur = animate_moon(cur, elements, t)
        cur = flicker_windows(cur, elements, t)

        # 5. star particles (emitted from real star positions)
        if not args.no_particles:
            cur = create_starry_particles(cur, elements, t, particles)

        # 6. bloom-style glow (color from the painting's bright areas)
        cur = add_glow_effect(cur)

        # 7. progressive oil-paint texture (optional; capped at 0.45, far
        #    below the original's full-strength overlay)
        if not args.no_oil and progress > 0.4:
            strength = min(0.45, (progress - 0.4) / 0.6 * 0.45)
            oil = fast_oil_painting(cur, radius=3, levels=10)
            cur = cv2.addWeighted(cur, 1 - strength, oil, strength, 0)

        # 8. very light vignette (optional)
        if not args.no_vignette:
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            xx, yy = np.meshgrid(x, y)
            vg = np.clip(1 - 0.15 * np.sqrt(xx ** 2 + yy ** 2) / np.sqrt(2), 0.85, 1)
            cur = np.clip(cur.astype(np.float32) * vg[..., None], 0, 255).astype(np.uint8)

        out.write(cur)
        if frame % args.fps == 0:
            print(f"  {t:5.1f}s / {t_all:.0f}s")

    out.release()
    print(f"Done! Output: {args.output} ({os.path.getsize(args.output) // 1024} KB)")


if __name__ == "__main__":
    main()
