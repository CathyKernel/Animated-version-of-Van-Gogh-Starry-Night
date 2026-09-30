"""Stage 1 — SAM semantic decomposition of The Starry Night.

Runs Meta AI's Segment Anything Model (ViT-B) over the painting, then maps
its instance masks onto the five semantic layers required by the project
specification:

    output/layers/sky.png      opaque, occlusion-inpainted background
    output/layers/stars.png    eleven stars + Venus (RGBA cut-out)
    output/layers/moon.png     the waning moon and its radiant halo (RGBA)
    output/layers/tree.png     the flame-like cypress (RGBA cut-out)
    output/layers/village.png  the village + rolling hills (RGBA cut-out)

SAM supplies high-quality object boundaries; classical color/geometry
analysis supplies the semantic labels (which mask is the cypress, which is
the moon, ...). This hybrid is the standard way to turn class-agnostic
segmentation into semantics, and it keeps the pipeline honest about what
the network actually predicts.

The stage also writes ``output/elements.json`` — the element manifest the
WebGL renderer consumes (star positions, vortex centers, moon parameters,
window lights, cypress extent, layer depth ordering).

Usage (each stage runs standalone to isolate model memory):
    python -m inference.sam_segment [--image PATH] [--device cpu]
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import common as C  # noqa: E402


# ---------------------------------------------------------------------------
# Classical analysis (semantic labelling)
# ---------------------------------------------------------------------------

def hsv_cvt(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0].astype(np.int32),
            hsv[..., 1].astype(np.int32),
            hsv[..., 2].astype(np.int32))


def detect_moon(bgr: np.ndarray) -> dict:
    """The moon = largest bright warm blob in the upper-right corner.

    The crescent core (centroid of V>225 pixels) anchors the position;
    the halo radius is estimated from the bright core extent — the painted
    halo is roughly 1.6x the bright halo, which matches the canonical
    composition (halo ~0.135 of canvas width).
    """
    h, w = bgr.shape[:2]
    Hc, Sc, Vc = hsv_cvt(bgr)
    yy, xx = np.ogrid[:h, :w]
    zone = (xx > 0.80 * w) & (yy < 0.32 * h)
    bright = (Vc > 135) & (Sc > 25) & (Hc >= 10) & (Hc <= 52) & zone
    bright |= (Vc > 208) & (Sc < 80) & zone
    m = cv2.morphologyEx(bright.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                         np.ones((9, 9), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    best, best_area = 0, 0
    for i in range(1, n):
        cx, cy = cents[i]
        if zone[int(cy), int(cx)] and stats[i, 4] > best_area:
            best_area, best = stats[i, 4], i
    if best == 0:
        # Fallback: the canonical composition position.
        return {"x": 0.926, "y": 0.137, "r": 0.055, "halo": 0.135,
                "mask": np.zeros((h, w), np.uint8), "fallback": True}
    comp = (labels == best)
    core = comp & (Vc > 225)
    sel = core if core.sum() > 30 else comp
    ys_, xs_ = np.nonzero(sel)
    mcx, mcy = float(xs_.mean() / w), float(ys_.mean() / h)
    core_r = float(np.hypot(xs_ - xs_.mean(), ys_ - ys_.mean()).max() / w)
    halo = float(np.clip(core_r * 1.6, 0.10, 0.16))
    # Full halo disc mask (painted halo is wider than the bright core).
    disc = np.zeros((h, w), np.uint8)
    cv2.circle(disc, (int(mcx * w), int(mcy * h)),
               int(halo * 1.15 * w), 255, -1)
    halo_mask = ((disc > 0) & (Vc > 85)).astype(np.uint8) * 255
    return {"x": round(mcx, 4), "y": round(mcy, 4),
            "r": round(core_r, 4), "halo": round(halo, 4),
            "mask": halo_mask, "fallback": False}


def detect_stars(bgr: np.ndarray, cypress: np.ndarray, moon: dict) -> list[dict]:
    """Eleven stars + Venus + the two bright swirl cores.

    The hard part of this painting: the whirlpool arms, Venus's halo and
    every star halo form ONE connected warm-bright mass, so plain
    connected components either swallow stars into a mega-blob or split
    them into confetti. The working recipe: find the local maxima of the
    halo-scale brightness map (>= 55 px apart) — every luminous element
    of this painting owns exactly one such dome — then anchor a seed on
    each maximum and flood the warm mask within an 80 px disc around it.
    The disc cap is what keeps neighbouring halos from bleeding into
    each other; the measured blob's centroid (not the raw peak) decides
    whether it belongs to the moon's halo or is a star of its own.
    """
    h, w = bgr.shape[:2]
    Hc, Sc, Vc = hsv_cvt(bgr)
    yy, xx = np.ogrid[:h, :w]
    # A slim dilation: Venus sits just off the cypress's right edge and
    # must survive (its centroid, not its halo, has to clear the tree).
    cyp_excl = cv2.dilate(cypress, np.ones((5, 5), np.uint8)) > 0
    sky = (yy < 0.56 * h) & (~cyp_excl)
    warm = (((Vc > 135) & (Sc > 25) & (Hc >= 10) & (Hc <= 52))
            | ((Vc > 208) & (Sc < 80))) & sky
    score = np.where(warm, Vc, 0).astype(np.float32)

    # Halo-scale smoothing, then local-max seeds with greedy NMS.
    # Replicate-padding before the maximum filter: OpenCV's dilate pads
    # with +inf by default, which silently kills every peak within
    # half a kernel of the canvas edge (the top-edge star lives there).
    sm = cv2.GaussianBlur(score, (0, 0), 7.0)
    pad = 28
    sm_p = cv2.copyMakeBorder(sm, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    k55 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
    dil_p = cv2.dilate(sm_p, k55)
    dil = dil_p[pad:pad + h, pad:pad + w]
    is_max = (dil <= sm + 1e-4) & (sm > 55)
    ys_, xs_ = np.nonzero(is_max)
    cand = []
    for py, px in zip(ys_, xs_):
        if px / w > 0.95 or px / w < 0.02:
            continue  # canvas-edge JPEG / block artifacts
        if px / w > 0.90 and py / h > 0.30:
            continue  # right-edge sky flow streaks
        cand.append((float(sm[py, px]), int(px), int(py)))
    cand.sort(reverse=True)
    seeds: list[tuple[int, int]] = []
    for val, px, py in cand:
        if len(seeds) >= 30:
            break
        if any((px - qx) ** 2 + (py - qy) ** 2 <= 55 ** 2 for qx, qy in seeds):
            continue
        seeds.append((px, py))

    # Flood each seed's own halo inside an 80 px disc.
    warm_closed = cv2.morphologyEx(warm.astype(np.uint8) * 255,
                                   cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    disc_r = 80
    stars, star_mask = [], np.zeros((h, w), np.uint8)
    for px, py in seeds:
        y0, y1 = max(0, py - disc_r), min(h, py + disc_r + 1)
        x0, x1 = max(0, px - disc_r), min(w, px + disc_r + 1)
        gy, gx = np.ogrid[y0:y1, x0:x1]
        disc = ((gy - py) ** 2 + (gx - px) ** 2) <= disc_r ** 2
        local = (warm_closed[y0:y1, x0:x1] > 0) & disc
        if local.sum() < 150:
            continue
        n, labels, stats, cents = cv2.connectedComponentsWithStats(
            local.astype(np.uint8), 8)
        seed_label = labels[py - y0, px - x0]
        if seed_label == 0:
            continue
        x, y, bw, bh, area = stats[seed_label]
        cx, cy = cents[seed_label]
        gx_, gy_ = (x0 + cx) / w, (y0 + cy) / h
        # The moon's halo rays seed domes too — their blobs stay inside
        # the halo; a real star's blob centroid clears it.
        if np.hypot(gx_ - moon["x"], gy_ - moon["y"]) < moon["halo"] * 0.8:
            continue
        r = float(max(bw, bh) / 2 / w)
        fill = float(area / (bw * bh))  # circularity proxy
        mean_v = float(Vc[y0:y1, x0:x1][labels == seed_label].mean())
        if r < 0.008 or r > 0.075 or fill < 0.40 or mean_v < 138:
            continue  # thin swirl-arm fragments are not stars
        stars.append({"x": round(gx_, 4), "y": round(gy_, 4),
                      "r": round(r, 4), "kind": "star", "amp": 0.34,
                      "bright": round(mean_v, 1)})
        star_mask[y0:y1, x0:x1] |= (labels == seed_label).astype(np.uint8) * 255

    stars.sort(key=lambda s: -s["bright"])
    return stars[:16], star_mask


def detect_swirl_cores(bgr: np.ndarray, cypress: np.ndarray,
                       moon: dict, stars: list[dict]) -> list[dict]:
    """The twin whirlpools: k-means (k=2) over bright sky pixels between
    the stars gives the two rotation centers of the great sky swirls."""
    h, w = bgr.shape[:2]
    _, _, Vc = hsv_cvt(bgr)
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")
    yy, xx = yy / h, xx / w
    cyp_excl = cv2.dilate(cypress, np.ones((15, 15), np.uint8)) > 0
    # Upper central sky band where the two whirlpool arms live.
    zone = (yy < 0.38) & (xx > 0.25) & (xx < 0.72) & (~cyp_excl)
    # Remove star and moon neighbourhoods so swirl arms dominate the cloud.
    for s in stars:
        zone &= ~(((xx - s["x"]) ** 2 + (yy - s["y"]) ** 2)
                  < (s["r"] * 1.6) ** 2)
    zone &= ~(((xx - moon["x"]) ** 2 + (yy - moon["y"]) ** 2)
              < (moon["halo"] * 1.4) ** 2)
    # Brightness-weighted pixel cloud: bright swirl arms count more.
    weight = np.clip(Vc.astype(np.float32) - 118.0, 0, 90) / 90.0
    sel = (weight > 0) & zone
    reps = np.maximum(weight[sel].astype(np.int32), 1)  # integer repetition
    pts_f = np.column_stack([xx[sel], yy[sel]])
    pts = np.repeat(pts_f, reps, axis=0).astype(np.float32)
    if len(pts) < 200:
        return [_fallback_vortex(0.42, 0.25, 0.30, 1.0),
                _fallback_vortex(0.63, 0.19, 0.20, -1.0)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    _, labels, centers = cv2.kmeans(pts, 2, None, crit, 6, cv2.KMEANS_PP_CENTERS)
    vorts = []
    for k in range(2):
        pk = pts[labels.ravel() == k]
        spread = float(max(np.std(pk[:, 0]), np.std(pk[:, 1])))
        vorts.append({"x": round(float(centers[k, 0]), 4),
                      "y": round(float(centers[k, 1]), 4),
                      "r": round(min(max(spread * 2.6, 0.16), 0.34), 4),
                      "speed": 0.55 if k == 0 else 0.75,
                      "dir": 1.0 if centers[k, 0] < 0.52 else -1.0})
    # Order left to right for stable animation directions.
    vorts.sort(key=lambda v: v["x"])
    vorts[0]["dir"], vorts[1]["dir"] = 1.0, -1.0
    vorts[0]["speed"], vorts[1]["speed"] = 0.55, 0.75
    return vorts


def _fallback_vortex(x, y, r, d):
    return {"x": x, "y": y, "r": r, "speed": 0.55, "dir": d}


def detect_cypress(bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The cypress = tall dark olive-green flame on the left of the canvas."""
    h, w = bgr.shape[:2]
    Hc, _, Vc = hsv_cvt(bgr)
    yy, xx = np.ogrid[:h, :w]
    zone = (xx < 0.32 * w) & (yy > 0.02 * h) & (yy < 0.94 * h)
    dark = (Vc < 115) & (Hc >= 18) & (Hc <= 85) & zone
    m = cv2.morphologyEx(dark.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                         np.ones((11, 11), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cyp = C.largest_component(m)
    cyp = cv2.morphologyEx(cyp, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cyp = cv2.GaussianBlur(cyp, (0, 0), 2.0)
    ys_ = np.nonzero(cyp > 60)[0]
    top = float(ys_.min() / h) if len(ys_) else 0.05
    bottom = float(ys_.max() / h) if len(ys_) else 0.88
    return (cyp > 127).astype(np.uint8) * 255, round(top, 4), round(bottom, 4)


def detect_windows(bgr: np.ndarray) -> list[dict]:
    """Village window lights: small warm blobs in the village band."""
    h, w = bgr.shape[:2]
    Hc, Sc, Vc = hsv_cvt(bgr)
    yy, xx = np.ogrid[:h, :w]
    band = (yy > 0.78 * h) & (yy < 0.93 * h) & (xx > 0.30 * w) & (xx < 0.97 * w)
    win = (Vc > 100) & (Sc > 45) & (Hc >= 8) & (Hc <= 55) & band
    wm = cv2.morphologyEx(win.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                          np.ones((2, 2), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(wm, 8)
    windows = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if 15 <= area <= 220:
            cx, cy = cents[i]
            windows.append({"x": round(float(cx / w), 4),
                            "y": round(float(cy / h), 4),
                            "r": round(float(max(bw, bh) / w) + 0.006, 4),
                            "area": int(area)})
    return sorted(windows, key=lambda b: -b["area"])[:20]


# ---------------------------------------------------------------------------
# SAM inference + semantic mapping
# ---------------------------------------------------------------------------

def diffusion_inpaint(img: np.ndarray, mask_u8: np.ndarray,
                      iters: int = 220) -> np.ndarray:
    """Laplace/diffusion fill for large holes (the revealed-behind-the-
    cypress sky). Smoother and far more plausible on big regions than
    Telea/NS inpainting, which leaves smears on holes hundreds of px wide.
    """
    out = img.astype(np.float32).copy()
    known = mask_u8 == 0
    if known.all() or (~known).sum() == 0:
        return img
    out[~known] = out[known].mean(axis=0)
    for _ in range(iters):
        blur = cv2.GaussianBlur(out, (0, 0), 4.0)
        out[~known] = blur[~known]
    return np.clip(out, 0, 255).astype(np.uint8)


def nearest_fill(img: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Voronoi texture extension: every hole pixel inherits the color of
    its nearest known pixel (distance-transform labels). Boundary strokes
    grow inward as directional streaks — real painted texture instead of
    the flat blur a Laplace fill produces on holes this large."""
    known = (mask_u8 == 0).astype(np.uint8)
    hole = known == 0
    if not hole.any() or known.all():
        return img
    _, labels = cv2.distanceTransformWithLabels(known, cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(known)
    ids = labels[ys, xs]
    src = np.zeros((int(labels.max()) + 1, 2), np.int32)
    src[ids] = np.stack([ys, xs], axis=1)
    out = img.copy()
    sel = labels[hole].astype(np.int64)
    out[hole] = img[src[sel, 0], src[sel, 1]]
    # Feather the transition band so the streak roots blend softly.
    band = cv2.dilate((hole.astype(np.uint8)) * 255,
                      np.ones((5, 5), np.uint8)) > 0
    soft = cv2.GaussianBlur(out, (0, 0), 2.0)
    out[band & hole] = soft[band & hole]
    return out


def directional_fill(img: np.ndarray, hole: np.ndarray, avoid: np.ndarray,
                      axis: str = "x") -> np.ndarray:
    """Painterly sky quilting: translate-and-stitch fill for big holes.

    Every hole row is filled from one clean sky row whose strokes are
    CONTINUOUSLY mapped onto the hole (a translation, not a clump of
    nearest pixels), so real brushwork structure survives; the source
    row drifts only a few rows between consecutive hole rows, keeping
    vertical coherence. Pixels whose translated source lands on another
    occluder fall back to the nearest valid sky pixel of the row. A
    gentle vertical blur inside the stitched area welds the rows into
    one continuous sky.

    axis is accepted for API compatibility (single strategy).
    """
    del axis
    out = img.copy()
    hole_b = hole > 0
    avoid_b = avoid > 0
    h, w = hole_b.shape
    # Sky-like pixels: blue channel clearly dominant (night sky tones).
    sky_like = (img[..., 0].astype(np.int32) >
                img[..., 2].astype(np.int32) + 12)
    rng = np.random.default_rng(7)

    rows_with_hole = np.nonzero(hole_b.any(axis=1))[0]
    if len(rows_with_hole) == 0:
        return out
    # The cypress hole needs sky shifted in from the right; the village
    # band maps the sky straight above it. Try both translations.
    base_offsets = [0, int(0.36 * w)]

    sy = float(np.clip(rows_with_hole[0], 0.03 * h, 0.55 * h))
    for y in rows_with_hole:
        sy = float(np.clip(sy + rng.uniform(-3.0, 3.0), 0.03 * h, 0.55 * h))
        syi = int(sy)
        xs = np.nonzero(hole_b[y])[0]
        row_valid = (~avoid_b[syi]) & sky_like[syi]
        valid = np.nonzero(row_valid)[0]
        tries = 0
        while len(valid) < 16 and tries < 8:
            sy = float(np.clip(sy + rng.uniform(-30, 30), 0.02 * h, 0.58 * h))
            syi = int(sy)
            row_valid = (~avoid_b[syi]) & sky_like[syi]
            valid = np.nonzero(row_valid)[0]
            tries += 1
        if len(valid) < 16:
            continue  # no sky in this row context -> Voronoi fallback

        # pick the translation with the best valid coverage
        best_T, best_cov = 0, -1.0
        for T0 in base_offsets:
            T = T0 + int(rng.uniform(-8, 8))
            sx = np.clip(xs + T, 0, w - 1)
            cov = float(row_valid[sx].mean())
            if cov > best_cov:
                best_cov, best_T = cov, T
        sx = np.clip(xs + best_T, 0, w - 1)
        ok = row_valid[sx]
        if ok.any():
            out[y, xs[ok]] = img[syi, sx[ok]]
        bad = ~ok
        if bad.any():
            xs2 = xs[bad]
            idx = np.searchsorted(valid, xs2)
            right = valid[np.clip(idx, 0, len(valid) - 1)]
            left = valid[np.clip(idx - 1, 0, len(valid) - 1)]
            pick = np.where((right - xs2) <= (xs2 - left), right, left)
            out[y, xs2] = img[syi, pick]

    # Weld the stitched rows: a gentle vertical-only blur inside the hole
    # removes seam lines between differently-translated rows while the
    # horizontal stroke texture (the painterly part) stays untouched.
    seam = cv2.GaussianBlur(out, (1, 3), 0)   # vertical kernel only
    out[hole_b] = seam[hole_b]
    return out


def texture_fill(bgr: np.ndarray, mask_u8: np.ndarray,
                 tall_hole: np.ndarray | None = None,
                 wide_hole: np.ndarray | None = None) -> np.ndarray:
    """Fill the occluded foreground with plausible sky brushwork.

    Strategy: painterly row quilting over the whole occluder set (real
    sky rows recomposed into the holes), then Voronoi nearest-stroke
    extension for any pixels the quilter could not source. Fast
    Signal Regression replaces both when ximgproc provides inpaint.
    (``tall_hole`` / ``wide_hole`` are accepted for API compatibility
    and folded into the same quilting pass.)
    """
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "inpaint"):
            out = cv2.ximgproc.inpaint(bgr, mask_u8, 60,
                                       cv2.ximgproc.INPAINT_FSR_FAST)
            if out is not None:
                return out
    except (AttributeError, cv2.error):
        pass
    out = directional_fill(bgr, mask_u8, mask_u8)
    remaining = mask_u8.copy()
    filled = (out != bgr).any(axis=2)
    remaining[filled] = 0
    if (remaining > 0).any():
        out = nearest_fill(out, remaining)
    return out


def run_sam_masks(bgr: np.ndarray, device: str = "cpu") -> list[dict]:
    """Run SAM automatic mask generation (with a disk cache so layer
    heuristics can be iterated without re-running the encoder)."""
    cache = C.OUTPUT_DIR / "sam_masks.npy"
    if cache.exists():
        try:
            arrs = np.load(str(cache), allow_pickle=True)
            print(f"  [SAM] loaded {len(arrs)} cached masks from {cache.name}")
            return list(arrs)
        except Exception:
            pass

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    ckpt = C.CHECKPOINT_DIR / "sam_vit_b_01ec64.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"SAM checkpoint missing: {ckpt}\n"
            "Run scripts/download_models.sh first.")
    print(f"  [SAM] loading ViT-B checkpoint ({ckpt.stat().st_size >> 20} MB)")
    sam = sam_model_registry["vit_b"](checkpoint=str(ckpt))
    sam.to(device=device)
    print("  [SAM] running automatic mask generation (CPU, be patient)")
    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=24,
        points_per_batch=64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        crop_n_layers=0,
        min_mask_region_area=64,
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = gen.generate(rgb)
    del sam, gen
    gc.collect()
    print(f"  [SAM] {len(results)} instance masks generated")
    slim = [{"segmentation": r["segmentation"], "area": int(r["area"])}
            for r in results]
    try:
        np.save(str(cache), np.array(slim, dtype=object), allow_pickle=True)
    except Exception:
        pass
    return slim


def sam_union_in_zone(masks: list[dict], zone_mask: np.ndarray,
                      overlap_thresh: float = 0.55) -> np.ndarray:
    """Union of SAM masks whose pixel mass inside ``zone_mask`` exceeds
    ``overlap_thresh`` — used to snap semantic layers onto SAM boundaries."""
    h, w = zone_mask.shape
    zone_area = float((zone_mask > 0).sum())
    out = np.zeros((h, w), np.uint8)
    for m in masks:
        seg = m["segmentation"]
        inter = float((seg & (zone_mask > 0)).sum())
        if zone_area > 0 and inter / zone_area >= overlap_thresh:
            out |= seg.astype(np.uint8) * 255
    return out


def build_layers(bgr: np.ndarray, sam_masks: list[dict]) -> dict:
    """Map SAM instances + classical cues onto the five semantic layers."""
    h, w = bgr.shape[:2]
    yy, xx = np.ogrid[:h, :w]

    # --- classical anchors -------------------------------------------------
    moon = detect_moon(bgr)
    cyp_raw, cyp_top, cyp_bottom = detect_cypress(bgr)
    stars, star_mask = detect_stars(bgr, cyp_raw, moon)
    vortices = detect_swirl_cores(bgr, cyp_raw, moon, stars)
    windows = detect_windows(bgr)

    # Peaks sitting on a whirlpool eye are swirl cores, not stars: they
    # animate with a gentler amplitude and join the great rotation.
    for s in stars:
        for v in vortices:
            if np.hypot(s["x"] - v["x"], s["y"] - v["y"]) < v["r"] * 0.6:
                s["kind"] = "swirl"
                s["amp"] = 0.16

    # --- SAM-snapped layer masks -------------------------------------------
    # tree: SAM masks living inside the cypress region (union with the
    # classical mask keeps the flame's wispy tip that SAM sometimes cuts).
    tree_zone = cv2.dilate(cyp_raw, np.ones((31, 31), np.uint8))
    sam_tree = sam_union_in_zone(sam_masks, tree_zone, 0.35)
    tree = cv2.bitwise_or(cv2.bitwise_and(sam_tree, tree_zone), cyp_raw)
    tree = C.largest_component(tree)
    tree = cv2.morphologyEx(tree, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # village: SAM masks whose centroid falls below the horizon band.
    village_zone = ((yy > 0.60 * h) & (xx > 0.26 * w)).astype(np.uint8) * 255
    sam_village = sam_union_in_zone(sam_masks, village_zone, 0.30)
    if (sam_village > 0).sum() < 0.05 * h * w:  # SAM too sparse -> band
        sam_village = village_zone
    # Village = everything below the horizon minus the cypress foot.
    village = cv2.bitwise_and(sam_village, village_zone)
    village = cv2.bitwise_and(village, cv2.bitwise_not(tree))
    village = cv2.morphologyEx(village, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # moon: SAM mask best matching the halo disc (falls back to halo mask).
    sam_moon = sam_union_in_zone(sam_masks, moon["mask"], 0.25)
    moon_mask = cv2.bitwise_or(
        cv2.bitwise_and(sam_moon, cv2.dilate(moon["mask"], np.ones((15, 15), np.uint8))),
        moon["mask"])
    moon_mask = C.feather_mask(moon_mask, 7, 1.5)

    # stars: classical blobs are sharper than SAM for glowing halos; keep
    # SAM only where it confirms a blob (>60% covered) to avoid greediness.
    stars_mask = star_mask.copy()
    for m in sam_masks:
        seg = m["segmentation"]
        if (seg & (star_mask > 0)).sum() > 0.60 * max(seg.sum(), 1):
            stars_mask |= seg.astype(np.uint8) * 255
    stars_mask = cv2.GaussianBlur(stars_mask, (0, 0), 1.0)

    # sky: the residual background — opaque, with occluded areas inpainted.
    fg = cv2.bitwise_or(cv2.bitwise_or(tree, village),
                        cv2.bitwise_or(moon_mask, stars_mask))
    sky_mask = cv2.bitwise_not(cv2.dilate(fg, np.ones((5, 5), np.uint8)))

    layers = {
        "sky": {"mask": sky_mask, "cutout": None},
        "stars": {"mask": stars_mask, "cutout": None},
        "moon": {"mask": moon_mask, "cutout": None},
        "tree": {"mask": tree, "cutout": None},
        "village": {"mask": village, "cutout": None},
    }

    # --- RGBA cut-outs + inpainted background --------------------------------
    # The sky layer is written by its own branch below (opaque,
    # occlusion-filled); the other four are alpha cut-outs.
    for name in ["stars", "moon", "tree", "village"]:
        layer = layers[name]
        alpha = C.feather_mask(layer["mask"], 5, 1.2)
        rgba = np.dstack([bgr, alpha])
        layer["cutout"] = rgba
        C.save_png(C.LAYERS_DIR / f"{name}.png", rgba)

    # Sky layer: fill what EVERY foreground element hides so parallax
    # never shows ghost copies — the tall cypress hole mirrors the sky
    # on its right, the wide village band mirrors the sky above, and
    # the compact moon/star holes get Voronoi stroke extension.
    occluders = cv2.bitwise_or(cv2.bitwise_or(tree, village),
                               cv2.bitwise_or(moon_mask, stars_mask))
    occluders = cv2.dilate(occluders, np.ones((13, 13), np.uint8))
    sky_rgb = texture_fill(
        bgr, occluders,
        tall_hole=cv2.dilate(tree, np.ones((13, 13), np.uint8)),
        wide_hole=cv2.dilate(village, np.ones((13, 13), np.uint8)))
    sky_alpha = np.full((h, w), 255, np.uint8)
    C.save_png(C.LAYERS_DIR / "sky.png", np.dstack([sky_rgb, sky_alpha]))

    # --- manifest ------------------------------------------------------------
    manifest = {
        "imageSize": [w, h],
        "layers": {},
        "vortices": vortices,
        "stars": stars,
        "moon": {k: moon[k] for k in ("x", "y", "r", "halo")},
        "windows": windows,
        "cypress": {"top": cyp_top, "bottom": cyp_bottom},
    }
    for name, layer in layers.items():
        m = (layer["mask"] > 127).astype(np.uint8)
        area = float(m.sum() / (h * w))
        entry = {"area": round(area, 4), "depth": C.LAYER_DEPTH_PRIOR[name]}
        if m.sum() > 20:
            ys, xs = np.nonzero(m)
            entry["bbox"] = [int(xs.min()), int(ys.min()),
                             int(xs.max()), int(ys.max())]
            entry["centroid"] = [round(float(xs.mean() / w), 4),
                                 round(float(ys.mean() / h), 4)]
        manifest["layers"][name] = entry
    C.write_json(C.OUTPUT_DIR / "elements.json", manifest)
    return manifest


def make_montage(bgr: np.ndarray) -> None:
    """QA montage: the five layers over a checkerboard (revealing each
    layer's true alpha coverage) plus the inpainted sky background."""
    def with_label(img: np.ndarray, text: str) -> np.ndarray:
        hh, ww = img.shape[:2]
        label = np.full((34, ww, 3), 24, np.uint8)
        cv2.putText(label, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([label, img])

    tiles = []
    for name in C.LAYER_NAMES:
        p = C.LAYERS_DIR / f"{name}.png"
        if not p.exists():
            continue
        rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        h, w = rgba.shape[:2]
        # 16px checkerboard so alpha coverage is visible at a glance.
        ys, xs = np.ogrid[:h, :w]
        board = ((((ys // 16) + (xs // 16)) % 2) * 60 + 96).astype(np.uint8)
        checker = cv2.merge([board, board, board])
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        comp = (rgba[..., :3].astype(np.float32) * a
                + checker.astype(np.float32) * (1 - a)).astype(np.uint8)
        tiles.append(with_label(comp, name.upper()))
    if tiles:
        tiles.append(with_label(bgr.copy(), "ORIGINAL"))
        rows = [np.hstack(tiles[:3]), np.hstack(tiles[3:])]
        montage = np.vstack(rows)
        h, w = montage.shape[:2]
        scale = 1800 / w
        montage = cv2.resize(montage, (1800, int(h * scale)))
        C.save_png(C.LAYERS_DIR / "montage.png", montage)
        print(f"  [QA ] montage -> {C.LAYERS_DIR / 'montage.png'}")


def run_sam(image: str | Path = C.DEFAULT_IMAGE, device: str = "cpu") -> dict:
    """Entry point: painting -> five semantic layers + element manifest."""
    C.ensure_dirs()
    bgr = C.load_bgr(image)
    print(f"[stage 1] SAM semantic decomposition: {image} "
          f"({bgr.shape[1]}x{bgr.shape[0]})")
    sam_masks = run_sam_masks(bgr, device)
    manifest = build_layers(bgr, sam_masks)
    make_montage(bgr)
    n_stars, n_win = len(manifest["stars"]), len(manifest["windows"])
    print(f"[stage 1] done: {len(C.LAYER_NAMES)} layers, "
          f"{n_stars} star-like blobs, {n_win} window lights")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=str(C.DEFAULT_IMAGE))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-sam", action="store_true",
                    help="skip SAM (classical layers only) for debugging")
    args = ap.parse_args()
    if args.no_sam:
        C.ensure_dirs()
        bgr = C.load_bgr(args.image)
        build_layers(bgr, [])
        make_montage(bgr)
    else:
        run_sam(args.image, args.device)
