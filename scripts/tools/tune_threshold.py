"""Pick each model's crossing-heatmap threshold on the VALIDATION split.

F1 at a fixed 0.5 threshold rewards whichever model happens to be calibrated there, which
is not the property we are comparing.  This script sweeps the threshold on `val` for every
trained model and stores the argmax, so the test table can report F1 at a threshold that
never saw the test split.  Average precision is reported alongside and needs no threshold
at all.

Only the network forward pass and peak extraction are needed here - no linking - so a
model takes seconds.

    python scripts/tools/tune_threshold.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cottoncross.common import load_json, read_image, save_json  # noqa: E402
from cottoncross.linking import heatmap_points  # noqa: E402
from cottoncross.metrics import point_counts, point_scores  # noqa: E402
from cottoncross.predict import load_model, predict  # noqa: E402

THRESHOLDS = np.linspace(0.05, 0.95, 19)


def sweep(checkpoint, data, split, limit, tolerance=6):
    root = Path(data)
    rows = [r for r in load_json(root / "manifest.json") if r["split"] == split]
    if limit:
        rows = rows[:limit]
    model, device = load_model(checkpoint)
    cached = []
    for r in rows:
        gray = read_image(root / r["image"], True)
        bundle = predict(gray, model, device, tile=min(384, r["size"]),
                         stride=min(288, r["size"]))
        pts = heatmap_points(bundle["joint"], threshold=float(THRESHOLDS[0]))
        scores = [float(bundle["joint"][int(round(p[1])), int(round(p[0]))]) for p in pts]
        with np.load(root / r["target"]) as z:
            truth = z["points"].reshape(-1, 2).copy()
        cached.append((pts, scores, truth))

    curve = []
    for thr in THRESHOLDS:
        counts = dict(tp=0, fp=0, fn=0)
        for pts, scores, truth in cached:
            keep = [p for p, s in zip(pts, scores) if s >= thr]
            c = point_counts(keep, truth, tolerance)
            for k in counts:
                counts[k] += c[k]
        s = point_scores(counts)
        curve.append(dict(threshold=float(thr), precision=s["precision"],
                          recall=s["recall"], f1=s["f1"]))
    best = max(curve, key=lambda d: d["f1"])
    return best, curve, len(cached)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/main")
    p.add_argument("--data", default="data/synth")
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    for run in sorted((ROOT / a.runs).glob("*_s*")):
        target = run / "crossing_threshold.json"
        checkpoint = run / "best.pt"
        if not checkpoint.exists():
            continue
        if target.exists() and not a.force:
            print(f"[skip] {run.name}")
            continue
        began = time.time()
        best, curve, n = sweep(checkpoint, a.data, a.split, a.limit)
        save_json(target, dict(
            threshold=best["threshold"], validation=best, curve=curve,
            split=a.split, images=n, tolerance_px=6,
            note="selected on the validation split only; the test split played no part"))
        print(f"{run.name:16s} threshold={best['threshold']:.2f} "
              f"val F1={best['f1']:.4f} ({time.time() - began:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
