"""Van Gogh neural rendering pipeline — orchestrator.

Runs the four stages in order, each in its own subprocess so the
memory of a 375 MB SAM encoder never overlaps the depth / flow models
(the demo targets machines with as little as 4 GB RAM):

    1. sam_segment    SAM ViT-B   -> 5 semantic layers + element manifest
    2. depth_estimate MiDaS-small -> fused depth (neural + SAM prior)
    3. optical_flow   RAFT-small  -> dense flow of the designed motion
    4. export_web                 -> WebGL asset bundle in renderer/web

Usage:
    python -m inference.run_pipeline                 # full pipeline
    python -m inference.run_pipeline --stage flow    # a single stage
    python -m inference.run_pipeline --inprocess     # debug, one process
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import common as C  # noqa: E402

STAGES = [
    ("sam_segment", "SAM semantic decomposition"),
    ("depth_estimate", "MiDaS depth + semantic prior fusion"),
    ("optical_flow", "RAFT dense flow of designed motion"),
    ("export_web", "WebGL asset export"),
]


def run_stage(name: str, arg: str | None = None) -> None:
    t0 = time.time()
    print(f"\n=== stage: {name} — {dict(STAGES).get(name, '')} ===")
    cmd = [sys.executable, "-u", "-m", f"inference.{name}"]
    if name == "depth_estimate" and arg:
        cmd += ["--model", arg]
    elif arg:
        cmd += ["--image", arg]
    proc = subprocess.run(cmd, cwd=str(C.PROJECT_ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"stage '{name}' failed with code {proc.returncode}")
    print(f"=== stage: {name} done in {time.time() - t0:.1f}s ===")


def run_pipeline(image: str | None = None, depth_model: str = "midas") -> None:
    print("Van Gogh neural rendering pipeline")
    print(f"  input : {image or C.DEFAULT_IMAGE}")
    print(f"  models: SAM vit_b + {depth_model} + RAFT-small (lightweight)")
    total = time.time()
    for name, _ in STAGES:
        if name == "depth_estimate":
            run_stage(name, depth_model)
        else:
            run_stage(name, image)
    print(f"\n[pipeline] all stages complete in {time.time() - total:.1f}s")
    print(f"[pipeline] open renderer/web/index.html "
          f"(or serve it) to view the animated painting")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=None,
                    help="input painting (default data/input/starry_night.png)")
    ap.add_argument("--stage", choices=[s[0] for s in STAGES],
                    help="run a single stage only")
    ap.add_argument("--depth-model", default="midas",
                    choices=["midas", "zoedepth"])
    ap.add_argument("--inprocess", action="store_true",
                    help="run stages in-process (debugging; more RAM)")
    args = ap.parse_args()

    if args.stage:
        run_stage(args.stage, args.depth_model if args.stage == "depth_estimate"
                  else args.image)
    elif args.inprocess:
        from inference.sam_segment import run_sam
        from inference.depth_estimate import run_depth
        from inference.optical_flow import run_raft
        from inference.export_web import export_assets
        img = args.image or str(C.DEFAULT_IMAGE)
        run_sam(img)
        run_depth(img, args.depth_model)
        run_raft(img)
        export_assets()
    else:
        run_pipeline(args.image, args.depth_model)
