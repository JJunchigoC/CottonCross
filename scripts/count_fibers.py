"""Per-image fiber counting for every model, scored against a reference count.

    python scripts/count_fibers.py                 # both benchmarks
    python scripts/count_fibers.py --only bench

Two benchmarks, one protocol:

bench  synthetic plates with an EXACT count (data/count_bench); parameters are selected
       on its val split, scores are on its test split.
real   the 75 real plates against data/real/fiber_counts_reference.csv; parameters are
       selected on the train+val plates, scores are on the held-out test plates.  The
       reference is a visual count (see the CSV's `source` column); only plates whose
       reference confidence is high or medium are scored, bundles are excluded.

Two protocols, the same for every model so no model can win by a better-tuned counter:

calibrated (primary)  only the staple length S in pixels is selected (minimum MAE on the
                      selection split); the debris cut-off stays at the physical prior
                      d = 0.10 of a staple (3.5 mm) declared in cottoncross/count.py.  S is
                      a camera calibration - the one number a new setup has to provide.
tuned (secondary)     S and d both selected on the selection split.  With a few dozen
                      selection images the chosen d proved unstable across seeds
                      (0.02-0.20 on the real plates), which is why it is not the primary
                      protocol; it is reported so the effect of that choice is visible.

Inputs are the caches written by `count_features.py`.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cottoncross.common import load_json, save_json  # noqa: E402
from cottoncross.count import DEBRIS_FRACTION, chain_lengths, count_plate  # noqa: E402
from cottoncross.metrics import bootstrap_ci, paired_test  # noqa: E402

# "length" is the primary (calibrated) protocol; "length_tuned" the secondary one.
ESTIMATORS = ("length", "length_tuned", "chains", "trajectories", "components")
STAPLES = np.arange(700, 2001, 25)
DEBRIS = (0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
MODELS = ["classical", "unet", "attunet", "unetpp", "resunet", "dscnet", "cfxnet"]
OURS = "cfxnet"


def reference(kind):
    if kind == "bench":
        return None
    rows = csv.DictReader(open(ROOT / "data/real/fiber_counts_reference.csv", encoding="utf-8"))
    return {r["id"]: int(r["count"]) for r in rows
            if r["count"] and r["confidence"] in ("high", "medium")}


def truth_of(plates, ref):
    if ref is None:
        return {i: int(f["true_count"]) for i, f in plates.items()}
    return {i: ref[i] for i in plates if i in ref}


def predictions(plates, chains, ids, estimator, staple, debris):
    key = ("length" if estimator.startswith("length")
           else "chains_count" if estimator == "chains" else estimator)
    return np.array([count_plate(plates[i], staple, debris_fraction=debris,
                                 chains=chains[i])[key] for i in ids])


def select(plates, chains, truth, ids, estimator):
    """Grid search on the selection split; ties broken toward exact accuracy, then d=0.1."""
    t = np.array([truth[i] for i in ids])
    best = None
    debris = DEBRIS if estimator == "length_tuned" else (DEBRIS_FRACTION,)
    staples = STAPLES if estimator != "components" else (1300,)
    for s in staples:
        for d in debris:
            p = predictions(plates, chains, ids, estimator, float(s), d)
            key = (float(np.abs(p - t).mean()), -float((p == t).mean()), abs(d - 0.1))
            if best is None or key < best[0]:
                best = (key, float(s), d)
    return dict(staple=best[1], debris_fraction=best[2], selection_mae=best[0][0],
                selection_accuracy=-best[0][1], selection_images=len(ids))


def scores(pred, true):
    err = pred - true
    return dict(mae=float(np.abs(err).mean()), accuracy=float((err == 0).mean()),
                within_one=float((np.abs(err) <= 1).mean()), bias=float(err.mean()),
                rmse=float(np.sqrt((err ** 2).mean())), n=int(len(err)))


def evaluate(kind, feature_dir, out):
    ref = reference(kind)
    select_splits = ("val",) if kind == "bench" else ("train", "val")
    rows, summary = [], {}
    files = sorted(Path(feature_dir).glob("*.json"))
    for path in files:
        data = load_json(path)
        name, seed = path.stem.rsplit("_s", 1)
        plates = data["plates"]
        truth = truth_of(plates, ref)
        chains = {i: chain_lengths(f) for i, f in plates.items()}
        sel_ids = sorted(i for i in truth if plates[i]["split"] in select_splits)
        test_ids = sorted(i for i in truth if plates[i]["split"] == "test")
        entry = {}
        for est in ESTIMATORS:
            chosen = select(plates, chains, truth, sel_ids, est)
            pred = predictions(plates, chains, test_ids, est, chosen["staple"],
                               chosen["debris_fraction"])
            true = np.array([truth[i] for i in test_ids])
            entry[est] = dict(selected=chosen, test=scores(pred, true))
            for i, p_, t_ in zip(test_ids, pred, true):
                rows.append(dict(model=name, seed=int(seed), estimator=est, id=i,
                                 true=int(t_), pred=int(p_), error=int(p_ - t_)))
        summary.setdefault(name, {})[seed] = entry
        print(f"[{kind}] {path.stem:14s} length mae={entry['length']['test']['mae']:.3f} "
              f"acc={entry['length']['test']['accuracy']:.3f}  "
              f"tuned mae={entry['length_tuned']['test']['mae']:.3f}  "
              f"traj mae={entry['trajectories']['test']['mae']:.3f}  "
              f"comp mae={entry['components']['test']['mae']:.3f}", flush=True)

    aggregate = {}
    for name, seeds in summary.items():
        aggregate[name] = {}
        for est in ESTIMATORS:
            per_seed = [seeds[s][est]["test"] for s in sorted(seeds)]
            agg = {}
            for k in ("mae", "accuracy", "within_one", "bias", "rmse"):
                v = np.array([p[k] for p in per_seed])
                agg[k] = float(v.mean())
                agg[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            per_image = {}
            for r in rows:
                if r["model"] == name and r["estimator"] == est:
                    per_image.setdefault(r["id"], []).append(abs(r["error"]))
            ci = bootstrap_ci([np.mean(v) for v in per_image.values()])
            agg.update(mae_ci=[ci["low"], ci["high"]], seeds=len(per_seed),
                       images=per_seed[0]["n"])
            aggregate[name][est] = agg

    # Paired significance: ours vs each model, per-image absolute error averaged over seeds.
    tests = {}
    for est in ESTIMATORS:
        per_image = {}
        for r in rows:
            if r["estimator"] == est:
                per_image.setdefault(r["model"], {}).setdefault(r["id"], []).append(abs(r["error"]))
        if OURS not in per_image:
            continue
        ours = per_image[OURS]
        for name, other in per_image.items():
            if name == OURS:
                continue
            ids = sorted(set(ours) & set(other))
            a = np.array([np.mean(ours[i]) for i in ids])
            b = np.array([np.mean(other[i]) for i in ids])
            tests.setdefault(est, {})[name] = paired_test(a, b)

    # Error broken down by the true count (where the difficulty of crossings grows).
    by_count = {}
    for r in rows:
        if r["estimator"] != "length":
            continue
        by_count.setdefault(r["model"], {}).setdefault(r["true"], []).append(abs(r["error"]))
    by_count = {m: {str(k): dict(mae=float(np.mean(v)), n=len(v))
                    for k, v in sorted(d.items())} for m, d in by_count.items()}

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = dict(benchmark=kind, protocol=dict(
        selection_splits=list(select_splits), test_split="test", staple_grid=[
            int(STAPLES[0]), int(STAPLES[-1]), 25], debris_grid=list(DEBRIS),
        rule="per model and estimator: minimum selection MAE, ties -> accuracy -> d near 0.1",
        calibrated=f"length: only S selected, d fixed at {DEBRIS_FRACTION}",
        tuned="length_tuned: S and d selected",
        reference=("exact synthetic count" if kind == "bench"
                   else "data/real/fiber_counts_reference.csv, confidence high/medium")),
        per_seed=summary, aggregate=aggregate, significance=tests, by_true_count=by_count)
    save_json(out / "summary.json", result)
    order = [m for m in MODELS if m in aggregate]
    print(f"\n[{kind}] test, mean over seeds (length estimator)")
    for m in order:
        a = aggregate[m]["length"]
        print(f"  {m:10s} MAE {a['mae']:.3f}±{a['mae_std']:.3f}  exact {a['accuracy']:.3f}  "
              f"±1 {a['within_one']:.3f}  bias {a['bias']:+.3f}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["bench", "real"])
    p.add_argument("--features", help="custom feature folder (with --only)")
    p.add_argument("--out", help="custom output folder (with --only)")
    args = p.parse_args()
    jobs = [("bench", ROOT / "results/counting/features_bench", ROOT / "results/counting/bench"),
            ("real", ROOT / "results/counting/features", ROOT / "results/counting/real")]
    if args.features:
        jobs = [(args.only, Path(args.features), Path(args.out))]
    for kind, feats, out in jobs:
        if args.only and kind != args.only:
            continue
        if not feats.exists():
            print(f"[skip] {feats} missing")
            continue
        evaluate(kind, feats, out)


if __name__ == "__main__":
    main()
