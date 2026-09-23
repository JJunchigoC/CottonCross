"""Select the linker hyperparameters on the synthetic VALIDATION split.

The embedding weight, the overlap weight and the rejection margin are free parameters of
the crossing linker.  Choosing them on the test split would be cheating, so the search
runs on `val` and writes `configs/linker.json`, which `linking.analyze` then loads.
The test split is touched only afterwards, with the chosen values frozen.

    python scripts/tools/tune_linker.py --checkpoint runs/main/cfxnet_s42/best.pt
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cottoncross.common import load_json, save_json, read_image  # noqa: E402
from cottoncross.linking import COSTS, analyze  # noqa: E402
from cottoncross.metrics import (instance_scores, pair_accuracy, point_counts,  # noqa: E402
                                 point_scores)
from cottoncross.predict import binarise, load_model, predict  # noqa: E402

GRID = dict(
    embedding=[0.0, 0.6, 1.2, 2.0],
    overlap=[0.0, 0.35, 0.7],
    margin_threshold=[0.0, 0.06, 0.12],
)


def cache_predictions(checkpoint, data, split, limit):
    """Predict once; the search then only re-runs the (cheap) linker."""
    root = Path(data)
    rows = [r for r in load_json(root / "manifest.json") if r["split"] == split]
    if limit:
        rows = rows[:limit]
    model, device = load_model(checkpoint)
    cached = []
    for i, r in enumerate(rows):
        gray = read_image(root / r["image"], True)
        bundle = predict(gray, model, device, tile=min(384, r["size"]),
                         stride=min(288, r["size"]))
        with np.load(root / r["target"]) as z:
            truth = dict(instances=z["instances"].copy(),
                         points=z["points"].reshape(-1, 2).copy(),
                         instance_id=z["instance_id"].copy())
        cached.append((gray, bundle, binarise(bundle), truth))
        if (i + 1) % 32 == 0:
            print(f"  cached {i + 1}/{len(rows)}", flush=True)
    return cached


def score(cached, costs, margin):
    counts = dict(tp=0, fp=0, fn=0)
    crossing = dict(tp=0, fp=0, fn=0)
    correct = committed = 0
    abstention = []
    for gray, bundle, binary, truth in cached:
        _, graph = analyze(binary.astype(np.uint8), gray, bundle, method="cfx",
                           costs=costs, margin_threshold=margin)
        s = instance_scores(graph["fibers"], truth["instances"])
        for k in counts:
            counts[k] += s[k]
        pts = [j["xy"] for j in graph["joints"] if j["degree"] >= 4]
        c = point_counts(pts, truth["points"], 6)
        for k in crossing:
            crossing[k] += c[k]
        p = pair_accuracy(graph, truth["instance_id"])
        correct += p["correct_pairs"]
        committed += p["committed_pairs"]
        abstention.append(p["abstention_rate"])
    return dict(instance_f1=point_scores(counts)["f1"],
                graph_crossing_f1=point_scores(crossing)["f1"],
                pair_accuracy=correct / committed if committed else float("nan"),
                committed_pairs=committed,
                abstention_rate=float(np.mean(abstention)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/main/cfxnet_s42/best.pt")
    p.add_argument("--data", default="data/synth")
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--out", default="configs/linker.json")
    a = p.parse_args()

    began = time.time()
    print(f"caching predictions on the {a.split} split ...", flush=True)
    cached = cache_predictions(a.checkpoint, a.data, a.split, a.limit)

    trials = []
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        setting = dict(zip(keys, values))
        costs = dict(COSTS)
        costs["embedding"] = setting["embedding"]
        costs["overlap"] = setting["overlap"]
        result = score(cached, costs, setting["margin_threshold"])
        trials.append(dict(**setting, **result))
        print(f"  emb={setting['embedding']:.2f} ovl={setting['overlap']:.2f} "
              f"margin={setting['margin_threshold']:.2f} -> "
              f"instF1={result['instance_f1']:.4f} "
              f"crossF1={result['graph_crossing_f1']:.4f} "
              f"pairAcc={result['pair_accuracy']:.4f} "
              f"abstain={result['abstention_rate']:.3f}", flush=True)

    # Selection criterion, fixed before the search: trajectory F1 on the validation
    # split rounded to three decimals, ties broken by pairing accuracy.  Abstention is
    # deliberately NOT optimised here - it is an operating point, reported separately.
    best = max(trials, key=lambda t: (round(t["instance_f1"], 3), t["pair_accuracy"]))
    costs = dict(COSTS)
    costs["embedding"] = best["embedding"]
    costs["overlap"] = best["overlap"]
    payload = dict(
        costs=costs, margin_threshold=best["margin_threshold"],
        angle_limit=40.0, unmatched=0.9,
        selected_on=dict(split=a.split, images=len(cached), checkpoint=a.checkpoint),
        criterion="highest trajectory F1 on the validation split (rounded to three "
                  "decimals), ties broken by pairing accuracy. Abstention is an "
                  "operating point, not something the search optimises. The test split "
                  "played no part in this choice.",
        validation=best, grid=GRID, trials=trials,
        seconds=round(time.time() - began, 1))
    save_json(ROOT / a.out, payload)
    print(f"\nselected: embedding={best['embedding']}, overlap={best['overlap']}, "
          f"margin={best['margin_threshold']}  (val instF1 {best['instance_f1']:.4f}, "
          f"pair accuracy {best['pair_accuracy']:.4f})")
    print(f"written -> {ROOT / a.out}")


if __name__ == "__main__":
    main()
