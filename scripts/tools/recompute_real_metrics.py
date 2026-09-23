"""Rebuild the real-plate metrics from the saved per-plate JSON files.

The length statistics were redefined after the inference run (the earlier fragmentation
and dispersion definitions rewarded a detector that shatters every fiber).  Every number
they need - each trajectory's length, the skeleton size, the substrate and equivariance
statistics - is already in `results/real/main/<method>/<plate>.json`, so the corrected metrics
are recomputed from those files instead of running inference again.

    python scripts/tools/recompute_real_metrics.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cottoncross.common import load_json, save_json  # noqa: E402
from cottoncross.evaluate_real import _write_summary_csv, length_statistics  # noqa: E402
from cottoncross.metrics import bootstrap_ci  # noqa: E402

CARRIED = ("sfar", "coverage", "equivariance_mse", "crossing_repeatability",
           "crossing_repeatability_counts", "crossings", "graph_crossings",
           "unresolved_joints", "skeleton_pixels", "mask_fraction")


def rebuild(folder, expected_length):
    old = load_json(folder / "metrics.json")
    records = []
    for previous in old["per_plate"]:
        plate = folder / f"{previous['id']}.json"
        if not plate.exists():
            raise SystemExit(f"missing {plate}; rerun evaluate_real for {folder.name}")
        data = load_json(plate)
        lengths = [f["length_pixels"] for f in data["fibers"]]
        record = {k: previous[k] for k in CARRIED if k in previous}
        record.update(id=previous["id"], page=previous["page"], split=previous["split"])
        record.update(length_statistics(lengths, previous["skeleton_pixels"],
                                        expected_length))
        records.append(record)
    summary = dict(old)
    summary["per_plate"] = records
    for field in ("sfar", "coverage", "equivariance_mse", "crossing_repeatability",
                  "fragmentation", "tracks", "substantial_tracks",
                  "length_recovery_error", "length_recovery_error_p90"):
        summary[field] = bootstrap_ci([r[field] for r in records])
    summary["length_cv"] = bootstrap_ci([r["length"]["cv"] for r in records])
    summary["length_p90"] = bootstrap_ci([r["length"]["p90"] for r in records])
    summary["length_median"] = bootstrap_ci([r["length"]["median"] for r in records])
    summary["metric_revision"] = (
        "length statistics recomputed from the saved per-plate trajectories: "
        "fragmentation now counts all tracks (the earlier version counted only long "
        "ones and therefore rewarded shattering), dispersion is measured on substantial "
        "tracks only, and length_recovery_error was added so a detector whose longest "
        "trajectory is a fraction of a staple cannot win on a tight spread alone")
    save_json(folder / "metrics.json", summary)
    _write_summary_csv(folder / "summary.csv", records)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/real/main")
    p.add_argument("--calibration", default="data/calibration.json")
    a = p.parse_args()
    expected = float(load_json(ROOT / a.calibration)["staple_pixels"])
    for folder in sorted((ROOT / a.results).iterdir()):
        if not (folder / "metrics.json").exists():
            continue
        summary = rebuild(folder, expected)
        print(f"{folder.name:12s} frag {summary['fragmentation']['mean']:7.3f}  "
              f"tracks {summary['tracks']['mean']:7.1f}  "
              f"len_cv {summary['length_cv']['mean']:.3f}  "
              f"len_err {summary['length_recovery_error']['mean']:.3f}")


if __name__ == "__main__":
    main()
