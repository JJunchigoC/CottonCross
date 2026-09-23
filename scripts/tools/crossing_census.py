"""How many crossings, and how many fibers take part in them, on each real plate.

This reads the per-plate JSON that `evaluate_real` already wrote, so it adds no new
inference.  It answers the practical question "what did the system actually find?" while
keeping the distinction the rest of the project insists on:

* a *crossing candidate* is where the algorithm believes two fibers overlap in the
  projection.  It is not verified against a human, and a 2D overlap is not proof that
  the two fibers touch;
* a *trajectory* is a reconstructed path, not a physical fiber.  One real fiber can be
  broken into several trajectories, and two fibers chained through a crossing can be
  merged into one.  Short debris is excluded by the `substantial` flag.

A trajectory is counted as participating in a crossing when its polyline passes within
`tolerance` pixels of that crossing's centre.

    python scripts/tools/crossing_census.py --method cfxnet
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cottoncross.common import load_json, save_json  # noqa: E402


def passes_through(points, centre, tolerance):
    """Does this polyline come within `tolerance` px of the crossing centre?"""
    pts = np.asarray(points, float)
    if len(pts) < 2:
        return bool(len(pts) and np.linalg.norm(pts[0] - centre) <= tolerance)
    seg = pts[1:] - pts[:-1]
    length = np.einsum("ij,ij->i", seg, seg)
    length[length == 0] = 1e-9
    t = np.clip(np.einsum("ij,ij->i", centre - pts[:-1], seg) / length, 0.0, 1.0)
    closest = pts[:-1] + t[:, None] * seg
    return bool(np.min(np.linalg.norm(closest - centre, axis=1)) <= tolerance)


def census_plate(data, tolerance=12.0, min_degree=4):
    crossings = [j for j in data["joints"] if j["degree"] >= min_degree]
    resolved = [j for j in crossings if not j["ambiguous"] and j["accepted_pairs"]]
    substantial = [f for f in data["fibers"] if f["substantial"]]
    involved = set()
    per_crossing = []
    for j in crossings:
        centre = np.asarray(j["xy_original"], float)
        members = [f["id"] for f in substantial
                   if passes_through(f["points_xy_original"], centre, tolerance)]
        involved.update(members)
        per_crossing.append(dict(joint=j["id"], degree=j["degree"],
                                 ambiguous=j["ambiguous"],
                                 substantial_fibers=len(members)))
    return dict(
        id=data["id"], page=data["page"], split=data["split"],
        crossing_candidates=len(crossings),
        resolved_crossings=len(resolved),
        abstained_crossings=len(crossings) - len(resolved),
        heatmap_crossings=len(data.get("crossings_heatmap_xy_original", [])),
        branch_candidates=sum(1 for j in data["joints"] if j["degree"] == 3),
        substantial_fibers=len(substantial),
        fibers_in_crossings=len(involved),
        all_trajectories=len(data["fibers"]),
        per_crossing=per_crossing)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="cfxnet")
    p.add_argument("--results", default="results/real/main")
    p.add_argument("--tolerance", type=float, default=12.0)
    p.add_argument("--out", default="results/diagnostics/crossing_census.json")
    a = p.parse_args()

    folder = ROOT / a.results / a.method
    plates = [census_plate(load_json(f), a.tolerance)
              for f in sorted(folder.glob("p*.json"))]
    if not plates:
        raise SystemExit(f"no per-plate JSON under {folder}")

    def total(key):
        return int(sum(r[key] for r in plates))

    summary = dict(
        method=a.method, plates=len(plates), tolerance_px=a.tolerance,
        totals={k: total(k) for k in
                ("crossing_candidates", "resolved_crossings", "abstained_crossings",
                 "heatmap_crossings", "branch_candidates", "substantial_fibers",
                 "fibers_in_crossings", "all_trajectories")},
        per_plate_mean={k: round(float(np.mean([r[k] for r in plates])), 2) for k in
                        ("crossing_candidates", "resolved_crossings",
                         "heatmap_crossings", "substantial_fibers",
                         "fibers_in_crossings")},
        caveats=[
            "a crossing candidate is an algorithm output, not a human-verified crossing;"
            " these plates carry no ground truth",
            "a 2D overlap is not proof of physical contact, and gives no depth order",
            "a trajectory is not a fiber: one fiber can break into several trajectories"
            " and two fibers can be merged into one through a crossing",
            "'substantial' means a trajectory at least 0.25 x the calibrated 35 mm"
            " staple; shorter debris is excluded",
        ],
        per_plate=plates)
    save_json(ROOT / a.out, summary)

    header = ["plate", "page", "crossing_candidates", "resolved", "abstained",
              "heatmap_crossings", "substantial_fibers", "fibers_in_crossings",
              "all_trajectories"]
    with (ROOT / a.results / a.method / "crossing_census.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in plates:
            w.writerow([r["id"], r["page"], r["crossing_candidates"],
                        r["resolved_crossings"], r["abstained_crossings"],
                        r["heatmap_crossings"], r["substantial_fibers"],
                        r["fibers_in_crossings"], r["all_trajectories"]])

    print(f"{'plate':12s} {'cross':>6s} {'resolved':>9s} {'heatmap':>8s} "
          f"{'fibers':>7s} {'in-cross':>9s}")
    for r in plates:
        print(f"{r['id']:12s} {r['crossing_candidates']:6d} {r['resolved_crossings']:9d} "
              f"{r['heatmap_crossings']:8d} {r['substantial_fibers']:7d} "
              f"{r['fibers_in_crossings']:9d}")
    print(f"\ntotals over {len(plates)} plates: {summary['totals']}")
    print(f"per-plate mean: {summary['per_plate_mean']}")


if __name__ == "__main__":
    main()
