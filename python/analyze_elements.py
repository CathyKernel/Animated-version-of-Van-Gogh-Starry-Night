# -*- coding: utf-8 -*-
"""
The Starry Night — element detection script (portable version)
Locates the stars / moon / cypress / village window lights and writes:
  elements.json       element coordinates (UV coords, y axis pointing down)
  masks.png           cypress sway-amplitude mask (grayscale: bright treetop -> dark roots)
  debug_detection.jpg visualization for verification

Usage:
  python3 analyze_elements.py [painting path] [output dir]
  # defaults: painting = ../assets/starry-night.jpg, output = ../assets/

Dependencies: opencv-python numpy
"""
import json
import os
import sys

import cv2
import numpy as np

# Curated star positions (double-checked with a vision model on the
# standard 1280x1014 composition)
# (x, y, r, kind)  kind: star = real star / swirl = bright swirl core
CURATED_STARS = [
    (0.352, 0.512, 0.060, "star"),   # Venus, the morning star — the brightest
    (0.107, 0.050, 0.055, "star"),   # large star, upper-left corner
    (0.368, 0.049, 0.042, "star"),   # star at top center
    (0.228, 0.036, 0.030, "star"),   # second-brightest star, upper left
    (0.476, 0.016, 0.030, "star"),   # star on the top edge
    (0.614, 0.092, 0.050, "star"),   # round star to the moon's upper left
    (0.815, 0.278, 0.045, "star"),   # star below-left of the moon
    (0.131, 0.479, 0.048, "star"),   # bright star left of the cypress
    (0.046, 0.452, 0.035, "star"),   # star at the far left edge
    (0.237, 0.147, 0.035, "star"),   # star above-right of the cypress
    (0.694, 0.206, 0.040, "star"),   # star at the right whirlpool's center
    (0.323, 0.328, 0.035, "swirl"),  # bright core of the main whirlpool eye
    (0.473, 0.202, 0.045, "swirl"),  # bright arc between the twin whirlpools
]
CURATED_MOON = (0.926, 0.137, 0.055, 0.135)  # crescent core + halo radius


def analyze(img, use_curated=True):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hc = hsv[..., 0].astype(np.int32)
    Sc = hsv[..., 1].astype(np.int32)
    Vc = hsv[..., 2].astype(np.int32)
    yy, xx = np.ogrid[:h, :w]

    # ---------- cypress: tall dark olive-green blob on the left ----------
    cyp_zone = (xx < 0.30 * w) & (yy > 0.04 * h) & (yy < 0.90 * h)
    cyp_dark = (Vc < 115) & (Hc >= 18) & (Hc <= 85) & cyp_zone
    m = cv2.morphologyEx((cyp_dark.astype(np.uint8)) * 255, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    cypress = np.zeros((h, w), np.uint8)
    if n > 1:
        best = 1 + int(np.argmax(stats[1:, 4] * 0.1 + stats[1:, 3] * 2.0))
        cypress = (labels == best).astype(np.uint8) * 255
        cypress = cv2.morphologyEx(cypress, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cypress = cv2.GaussianBlur(cypress, (0, 0), 2.0)
    ys_ = np.nonzero(cypress > 60)[0]
    top = ys_.min() / h if len(ys_) else 0.06
    bottom = ys_.max() / h if len(ys_) else 0.85
    ramp = np.clip((bottom - yy / h) / max(bottom - top, 0.1), 0, 1) ** 1.25
    sway = ((cypress / 255.0) * ramp * 255).astype(np.uint8)
    sway = cv2.GaussianBlur(sway, (0, 0), 1.5)

    # ---------- stars ----------
    cyp_excl = cv2.dilate(cypress, np.ones((25, 25), np.uint8)) > 0
    sky = (yy < 0.56 * h) & (~cyp_excl)
    bright = (((Vc > 135) & (Sc > 25) & (Hc >= 10) & (Hc <= 52))
              | ((Vc > 208) & (Sc < 80))) & sky
    bm = cv2.morphologyEx((bright.astype(np.uint8)) * 255, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    n2, l2, st2, ce2 = cv2.connectedComponentsWithStats(bm, 8)

    # moon: centroid of the V>225 pixels of the largest bright blob, upper right
    moon_zone = (xx > 0.52 * w) & (yy < 0.40 * h)
    moon_label, best_area = 0, 0
    for i in range(1, n2):
        cx, cy = ce2[i]
        if st2[i, 4] > best_area and moon_zone[int(cy), int(cx)]:
            best_area, moon_label = st2[i, 4], i
    moon = list(CURATED_MOON)
    if moon_label and not use_curated:
        sel = (l2 == moon_label) & (Vc > 225)
        if sel.sum() > 30:
            mcx, mcy = np.nonzero(sel)[1].mean(), np.nonzero(sel)[0].mean()
            moon = [float(mcx / w), float(mcy / h),
                    float(st2[moon_label, 2] / 2 / w), float(best_area ** 0.5 / w)]

    if use_curated:
        stars = [{"x": x, "y": y, "r": r, "kind": k, "amp": 0.34 if k == "star" else 0.16}
                 for x, y, r, k in CURATED_STARS]
    else:
        auto = []
        for i in range(1, n2):
            if i == moon_label:
                continue
            x, y, bw, bh, area = st2[i]
            if 20 <= area <= 25000:
                cx, cy = ce2[i]
                if np.hypot(cx / w - moon[0], cy / h - moon[1]) >= 0.30:
                    auto.append({"x": round(float(cx / w), 4), "y": round(float(cy / h), 4),
                                 "r": round(float(max(bw, bh) / 2 / w), 4),
                                 "kind": "star", "amp": 0.34})
        stars = auto[:13]

    # ---------- window lights ----------
    village = (yy > 0.78 * h) & (yy < 0.93 * h) & (xx > 0.30 * w) & (xx < 0.97 * w)
    win = (Vc > 100) & (Sc > 45) & (Hc >= 8) & (Hc <= 55) & village
    wm = cv2.morphologyEx((win.astype(np.uint8)) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n3, l3, st3, ce3 = cv2.connectedComponentsWithStats(wm, 8)
    windows = []
    for i in range(1, n3):
        x, y, bw, bh, area = st3[i]
        if 15 <= area <= 220:
            cx, cy = ce3[i]
            windows.append({"x": round(float(cx / w), 4), "y": round(float(cy / h), 4),
                            "r": round(float(max(bw, bh) / w) + 0.006, 4), "area": int(area)})

    return {
        "sway_map": sway,
        "elements": {
            "imageSize": [w, h],
            "vortices": [
                {"x": 0.42, "y": 0.25, "r": 0.30, "speed": 0.55, "dir": 1.0},
                {"x": 0.63, "y": 0.19, "r": 0.20, "speed": 0.75, "dir": -1.0},
            ],
            "stars": stars,
            "moon": {"x": moon[0], "y": moon[1], "r": moon[2], "halo": moon[3],
                     "speed": 0.45, "phase": 0.0},
            "windows": sorted(windows, key=lambda b: -b["area"])[:20],
            "cypress": {"top": round(top, 4), "bottom": round(bottom, 4)},
        },
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "assets", "starry-night.jpg")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "assets")
    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"Cannot read image: {img_path}")
    res = analyze(img)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "elements.json"), "w", encoding="utf-8") as f:
        json.dump(res["elements"], f, ensure_ascii=False, indent=2)
    cv2.imwrite(os.path.join(out_dir, "masks.png"), res["sway_map"])
    # verification visualization
    dbg = img.copy()
    h, w = img.shape[:2]
    for s in res["elements"]["stars"]:
        c = (0, 255, 255) if s["kind"] == "star" else (0, 180, 255)
        cv2.circle(dbg, (int(s["x"] * w), int(s["y"] * h)), int(s["r"] * w), c, 2)
    mo = res["elements"]["moon"]
    cv2.circle(dbg, (int(mo["x"] * w), int(mo["y"] * h)), int(mo["halo"] * w), (255, 200, 0), 3)
    for wn in res["elements"]["windows"]:
        cv2.circle(dbg, (int(wn["x"] * w), int(wn["y"] * h)), 6, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "debug_detection.jpg"), dbg)
    print(f"Done: {len(res['elements']['stars'])} stars, {len(res['elements']['windows'])} windows")
    print(f"Output directory: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
