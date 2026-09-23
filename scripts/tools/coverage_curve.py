"""Evidence coverage at several ridge-strength levels, not just the brightest cores.

Most of the label-free real-plate statistics (substrate false alarm, equivariance error,
crossing repeatability, fragmentation) improve when a detector simply finds less.  The
counterweight is meant to be evidence coverage - but at the original 0.05% quantile it
only probes the very brightest ridge cores, where every method saturates near 1.0, so it
does not actually penalise a timid detector.

This script sweeps the quantile from the brightest cores down to faint evidence and
writes the whole curve, so "found less" becomes visible instead of being rewarded.

    python scripts/tools/coverage_curve.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cottoncross.background import ridge_response  # noqa: E402
from cottoncross.common import load_json, read_image, save_json  # noqa: E402
from cottoncross.predict import classical, load_model, predict  # noqa: E402

# Fraction of pixels above the threshold.  The plates are ~0.15% fiber, so 0.0005 is the
# brightest cores only and 0.01 reaches well into faint and ambiguous evidence.
FRACTIONS = [0.0005, 0.001, 0.002, 0.005, 0.01]


def curve_for(checkpoint, real, split, limit=None):
    root = Path(real)
    rows = [r for r in load_json(root / "manifest.json") if r["split"] == split]
    if limit:
        rows = rows[:limit]
    model, device = (load_model(checkpoint) if checkpoint else (None, None))
    per_fraction = {f: [] for f in FRACTIONS}
    for r in rows:
        gray = read_image(root / r["roi"], True)
        bundle = predict(gray, model, device) if model is not None else classical(gray)[0]
        mask = bundle["mask"] > 0.5
        response = ridge_response(gray)
        for f in FRACTIONS:
            level = float(np.quantile(response, 1.0 - f))
            evidence = response > level
            per_fraction[f].append(float(mask[evidence].mean()) if evidence.any()
                                   else float("nan"))
    return {str(f): dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v, ddof=1)))
            for f, v in per_fraction.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", default="data/real")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="results/diagnostics/coverage_curve.json")
    p.add_argument("--limit", type=int)
    a = p.parse_args()

    jobs = [("classical", None)]
    for run in sorted((ROOT / "runs/main").glob("*_s42")):
        jobs.append((run.name.replace("_s42", ""), run / "best.pt"))
    for name in ("raw_input", "no_adaptation"):
        run = ROOT / "runs/ablation" / name
        if (run / "best.pt").exists():
            jobs.append((f"ablation:{name}", run / "best.pt"))

    out = dict(fractions=FRACTIONS, split=a.split, methods={},
               reading="fraction of pixels above each ridge-strength level that the "
                       "detector also calls fiber. The plates are about 0.15% fiber, so "
                       "0.0005 is the brightest cores and 0.01 reaches faint evidence. A "
                       "timid detector holds up at 0.0005 and falls away at 0.01.")
    for name, checkpoint in jobs:
        out["methods"][name] = curve_for(checkpoint, a.real, a.split, a.limit)
        row = "  ".join(f"{f}:{out['methods'][name][str(f)]['mean']:.3f}"
                        for f in FRACTIONS)
        print(f"{name:20s} {row}", flush=True)
    save_json(ROOT / a.out, out)
    print(f"written -> {ROOT / a.out}")


if __name__ == "__main__":
    main()
