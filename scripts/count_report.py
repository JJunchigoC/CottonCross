"""Counting tables, the every-model x every-plate count sheet, and the app configuration.

    python scripts/count_report.py

Reads results/counting/{bench,real}/summary.json (written by count_fibers.py) and the
feature caches, and writes

results/counting/real/all_plates.csv     fiber count of EVERY real plate by EVERY model
                                          (seed-median, plus each seed), next to the
                                          reference count and its confidence
results/tables/table_counting_methods.csv counting rule x benchmark, per backbone
results/tables/table_counting.csv         backbone comparison under the primary rule
configs/counting.json                     staple length and checkpoint the app uses
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cottoncross.common import load_json, save_json  # noqa: E402
from cottoncross.count import chain_lengths, count_plate  # noqa: E402

MODELS = ["classical", "unet", "attunet", "unetpp", "resunet", "dscnet", "cfxnet"]
OURS = "cfxnet"
ESTIMATOR_NAMES = {
    "components": "connected components (blob count)",
    "trajectories": "linked trajectories >= 0.5 staple",
    "chains": "chain-level staple rule",
    "length": "pooled staple-length rule (primary)",
    "length_tuned": "pooled staple-length rule, d also tuned",
}


def all_plates(summary):
    rows = {r["id"]: dict(r) for r in csv.DictReader(
        open(ROOT / "data/real/fiber_counts_reference.csv", encoding="utf-8"))}
    header = ["id", "split", "group", "reference", "confidence"]
    table = {i: dict(id=i, split=r["split"], group=r["group"], reference=r["count"],
                     confidence=r["confidence"]) for i, r in rows.items()}
    for path in sorted((ROOT / "results/counting/features").glob("*.json")):
        name, seed = path.stem.rsplit("_s", 1)
        chosen = summary["per_seed"][name][seed]["length"]["selected"]
        plates = load_json(path)["plates"]
        for i, f in plates.items():
            n = count_plate(f, chosen["staple"], debris_fraction=chosen["debris_fraction"],
                            chains=chain_lengths(f))["length"]
            table[i][f"{name}_s{seed}"] = n
    columns = []
    for m in MODELS:
        seeds = sorted({k for row in table.values() for k in row if k.startswith(m + "_s")})
        for i, row in table.items():
            values = [row[k] for k in seeds if k in row]
            row[m] = int(np.median(values)) if values else ""
        columns += [m] + seeds
    out = ROOT / "results/counting/real/all_plates.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header + columns, extrasaction="ignore")
        w.writeheader()
        for i in sorted(table):
            w.writerow(table[i])
    print(f"  all plates x all models -> {out}")
    return out


def tables(bench, real):
    out = ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for est, label in ESTIMATOR_NAMES.items():
        for m in MODELS:
            b = bench["aggregate"].get(m, {}).get(est)
            r = real["aggregate"].get(m, {}).get(est)
            if not b and not r:
                continue
            rows.append(dict(
                rule=label, estimator=est, model=m,
                bench_mae=b and round(b["mae"], 4), bench_mae_std=b and round(b["mae_std"], 4),
                bench_exact=b and round(b["accuracy"], 4),
                bench_within1=b and round(b["within_one"], 4),
                real_mae=r and round(r["mae"], 4), real_mae_std=r and round(r["mae_std"], 4),
                real_exact=r and round(r["accuracy"], 4),
                real_within1=r and round(r["within_one"], 4)))
    with open(out / "table_counting_methods.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    primary = [r for r in rows if r["estimator"] == "length"]
    with open(out / "table_counting.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(primary[0]))
        w.writeheader()
        w.writerows(primary)
    print(f"  counting tables -> {out}")


def app_config(real):
    """Staple and checkpoint for the application.

    The paper's model (CFX-Net) is the default; among its seeds the one with the lowest
    counting error on the SELECTION plates (train+val) is used, with that seed's selected
    staple.  The test plates play no part in this choice.
    """
    seeds = real["per_seed"][OURS]
    best = min(seeds, key=lambda s: (seeds[s]["length"]["selected"]["selection_mae"], s))
    chosen = seeds[best]["length"]["selected"]
    cfg = dict(
        checkpoint=f"runs/main/{OURS}_s{best}/best.pt",
        staple_px=chosen["staple"], debris_fraction=chosen["debris_fraction"],
        selected_on="real train+val plates with a high/medium reference count",
        selection_mae=chosen["selection_mae"], seed=int(best),
        note="staple_px is a camera calibration: re-select it for a different microscope "
             "or magnification (scripts/count_fibers.py on a few counted images), or "
             "pass --staple to the app.")
    save_json(ROOT / "configs/counting.json", cfg)
    print(f"  app config -> configs/counting.json ({cfg['checkpoint']}, "
          f"staple {cfg['staple_px']:.0f} px)")


def main():
    bench = load_json(ROOT / "results/counting/bench/summary.json")
    real = load_json(ROOT / "results/counting/real/summary.json")
    all_plates(real)
    tables(bench, real)
    app_config(real)


if __name__ == "__main__":
    main()
