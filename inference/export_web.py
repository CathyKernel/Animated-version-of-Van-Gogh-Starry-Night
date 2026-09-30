"""Stage 4 — Export the neural pipeline outputs as a WebGL asset bundle.

Asset philosophy ("preserves original brush strokes and colors"): every
semantic layer shares ONE painting texture (JPEG); what makes a layer is
only its alpha mask (grayscale PNG). The renderer therefore physically
cannot alter Van Gogh's palette — it can only displace, mask and modulate
brightness. This also keeps the whole bundle around ~1 MB instead of
~30 MB of RGBA PNGs.

Writes:
    renderer/web/assets/painting.jpg      shared RGB texture of the painting
    renderer/web/assets/mask_<layer>.png  per-layer alpha masks (stars,
                                           moon, tree, village)
    renderer/web/assets/depth.png         8-bit fused depth (near = white)
    renderer/web/assets/flow.png          RAFT flow, RG-encoded (+scale in
                                           the manifest)
    renderer/web/assets/manifest.json     element manifest + asset registry
    renderer/web/assets/scene-data.js     everything above base64-embedded
                                           (single-file offline / file:// mode)
    renderer/web/assets/shaders.js        the two GLSL sources embedded

Usage:
    python -m inference.export_web [--no-embed]
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import common as C  # noqa: E402

MASK_LAYERS = ["stars", "moon", "tree", "village"]  # sky = opaque base
JPEG_QUALITY = 90


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def export_assets(embed: bool = True) -> dict:
    """Entry point: pipeline outputs -> renderer/web/assets bundle."""
    C.ensure_dirs()
    img = Image.open(C.DEFAULT_IMAGE).convert("RGB")
    w, h = img.size
    print(f"[stage 4] exporting WebGL assets ({w}x{h})")

    assets_dir = C.WEB_ASSETS_DIR
    assets_dir.mkdir(parents=True, exist_ok=True)

    registry = {"imageSize": [w, h], "layers": {}, "assets": {}}

    # --- shared painting texture ------------------------------------------
    painting_path = assets_dir / "painting.jpg"
    img.save(str(painting_path), "JPEG", quality=JPEG_QUALITY, optimize=True)
    registry["assets"]["painting"] = painting_path.name

    # --- per-layer alpha masks ----------------------------------------------
    for name in MASK_LAYERS:
        src = C.LAYERS_DIR / f"{name}.png"
        if not src.exists():
            raise FileNotFoundError(f"layer missing: {src} (run stage 1 first)")
        rgba = np.array(Image.open(src))
        alpha = rgba[..., 3]
        # Light blur keeps the mask smooth at grazing parallax angles.
        a_img = Image.fromarray(alpha, mode="L")
        mask_path = assets_dir / f"mask_{name}.png"
        a_img.save(str(mask_path), "PNG", optimize=True)
        registry["assets"][f"mask_{name}"] = mask_path.name
        frac = float((alpha > 127).mean())
        registry["layers"][name] = {
            "coverage": round(frac, 4),
            "depth": C.LAYER_DEPTH_PRIOR[name],
        }
    registry["layers"]["sky"] = {"coverage": 1.0,
                                 "depth": C.LAYER_DEPTH_PRIOR["sky"]}

    # --- depth texture (8-bit) -----------------------------------------------
    dep_src = C.DEPTH_DIR / "depth_raw.npy"
    if dep_src.exists():
        dep = np.load(str(dep_src))
        d8 = (np.clip(dep, 0, 1) * 255).astype(np.uint8)
        depth_path = assets_dir / "depth.png"
        Image.fromarray(d8, mode="L").save(str(depth_path), "PNG", optimize=True)
        registry["assets"]["depth"] = depth_path.name

    # --- RAFT flow texture (RG) ------------------------------------------------
    flow_src = C.FLOW_DIR / "flow_texture.png"
    if flow_src.exists():
        flow_path = assets_dir / "flow.png"
        Image.open(flow_src).save(str(flow_path), "PNG", optimize=True)
        registry["assets"]["flow"] = flow_path.name

    # --- manifest ------------------------------------------------------------
    manifest_path = C.OUTPUT_DIR / "elements.json"
    if manifest_path.exists():
        elements = C.read_json(manifest_path)
        registry["elements"] = elements
        registry["flowScale"] = elements.get("flow", {}).get("textureScale", 4.0)

    (assets_dir / "manifest.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8")

    # --- shaders bundle ----------------------------------------------------
    shader_dir = C.PROJECT_ROOT / "renderer" / "shaders"
    shaders = {}
    for key, fname in (("flow", "optical_flow.frag"),
                       ("parallax", "depth_parallax.frag")):
        p = shader_dir / fname
        if p.exists():
            shaders[key] = p.read_text(encoding="utf-8")
    if shaders:
        js = "window.__SHADERS__ = " + json.dumps(shaders) + ";\n"
        (assets_dir / "shaders.js").write_text(js, encoding="utf-8")
        print(f"  [web ] shaders.js: {list(shaders)}")

    # --- single-file offline bundle -----------------------------------------
    if embed:
        payload = {
            "imageSize": [w, h],
            "layers": registry["layers"],
            "flowScale": registry.get("flowScale", 4.0),
            "elements": registry.get("elements", {}),
            "files": {},
        }
        for key in list(registry["assets"].values()):
            p = assets_dir / key
            payload["files"][key] = b64(p)
        if shaders:
            payload["shaders"] = shaders
        js = ("window.__STARRY__ = "
              + json.dumps(payload, separators=(",", ":")) + ";\n")
        (assets_dir / "scene-data.js").write_text(js, encoding="utf-8")
        size_mb = (assets_dir / "scene-data.js").stat().st_size / 1048576
        print(f"  [web ] scene-data.js embedded bundle: {size_mb:.2f} MB")

    total = sum((assets_dir / k).stat().st_size
                for k in registry["assets"].values())
    print(f"[stage 4] done: {len(registry['assets'])} assets "
          f"({total / 1024:.0f} KB) -> {assets_dir}")
    return registry


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-embed", action="store_true",
                    help="skip the base64 single-file bundle")
    args = ap.parse_args()
    export_assets(embed=not args.no_embed)
