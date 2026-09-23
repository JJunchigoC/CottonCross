"""Run every model on every real plate once and cache what the fiber counter needs.

    python scripts/count_features.py                   # all models, all seeds
    python scripts/count_features.py --archs cfxnet --seeds 42
    python scripts/count_features.py --dataset bench      # synthetic counting benchmark

Output: results/counting/features[_bench]/<model>_s<seed>.json, one record per image.  Counting
itself (`scripts/count_fibers.py`) then runs on these caches in seconds, so the counting
rule can be studied without touching the GPU again.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cottoncross.common import load_json, read_image, save_json  # noqa: E402
from cottoncross.count import plate_features  # noqa: E402
from cottoncross.predict import classical, load_model, predict  # noqa: E402

ARCHS = ["unet", "attunet", "unetpp", "resunet", "dscnet", "cfxnet"]
SEEDS = [42, 43, 44]
# Staple used only for the interference cut in the crossing filter on the real plates
# (docs/development/crossing_count_rule.md); counting itself selects S per model.
REAL_STAPLE_PRIOR = 1400.0


def _rows(dataset):
    if dataset == "real":
        return [dict(id=r["id"], path=ROOT / "data/real" / r["roi"], split=r["split"],
                     group=r["group"], staple=REAL_STAPLE_PRIOR)
                for r in load_json(ROOT / "data/real/manifest.json")]
    root = ROOT / "data/count_bench"
    staple = float(load_json(root / "provenance.json")["staple_pixels"])
    return [dict(id=r["id"], path=root / r["image"], split=r["split"], group="synthetic",
                 fibers=r["fibers"], crossings=r["crossings"], staple=staple)
            for r in load_json(root / "manifest.json")]


def run_one(name, checkpoint, method, rows, out):
    if out.exists():
        print(f"[skip] {out.name}", flush=True)
        return
    model, device = load_model(checkpoint) if checkpoint else (None, None)
    thresholds = None
    if model is not None and getattr(model, "crossing_threshold", None) is not None:
        thresholds = dict(joint=model.crossing_threshold)
    plates, t0 = {}, time.time()
    for r in rows:
        gray = read_image(r["path"], True)
        bundle = predict(gray, model, device) if model is not None else classical(gray)[0]
        feats = plate_features(bundle, gray, method=method, thresholds=thresholds,
                               staple=r["staple"])
        feats.update(split=r["split"], group=r["group"])
        if "fibers" in r:
            feats["true_count"] = r["fibers"]
            feats["true_crossings"] = r["crossings"]
        plates[r["id"]] = feats
    save_json(out, dict(model=name, checkpoint=str(checkpoint) if checkpoint else None,
                        linker=method, thresholds=thresholds, plates=plates))
    print(f"[done] {out.name}  {len(rows)} plates  {time.time() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="*", default=ARCHS)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--no-classical", action="store_true")
    p.add_argument("--dataset", default="real", choices=["real", "bench"])
    p.add_argument("--runs", default="runs/main", help="folder holding <arch>_s<seed>/best.pt")
    p.add_argument("--weights", default="best.pt", help="checkpoint file inside each run")
    p.add_argument("--out")
    args = p.parse_args()
    rows = _rows(args.dataset)
    out = ROOT / (args.out or f"results/counting/features{'' if args.dataset == 'real' else '_bench'}")
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_classical:
        run_one("classical", None, "angle_joint", rows, out / "classical_s0.json")
    for seed in args.seeds:
        for arch in args.archs:
            ck = ROOT / args.runs / f"{arch}_s{seed}" / args.weights
            if not ck.exists():
                print(f"[warn] missing {ck}", flush=True)
                continue
            run_one(arch, ck, "cfx", rows, out / f"{arch}_s{seed}.json")


if __name__ == "__main__":
    main()
