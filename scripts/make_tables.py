"""Collect every result file into the report tables.

Writes results/tables/:
  table1_main.csv          backbone comparison, mean +/- sd over seeds
  table2_ablation.csv      one component removed at a time
  table3_post.csv          linking-method comparison
  table4_real.csv          label-free statistics on the held-out real plates
  table5_plate.csv         whole-plate synthetic canvases (1024 px)
  significance.json        paired per-image tests, CFX-Net against each baseline
  leaderboard.json         which method wins each metric, stated plainly

The leaderboard is written whatever it says.  If a baseline wins a column, that is
what the file reports.
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
from cottoncross.paths import group_dir  # noqa: E402
from cottoncross.figures import (HIGHER_BETTER, METHOD_ORDER, METRIC_KEYS, OURS,  # noqa: E402
                                 LABELS_EN, load_group, main_by_arch, metric_value)
from cottoncross.metrics import paired_test, point_counts  # noqa: E402

def _f1(counts):
    """Per-image F1 from tp/fp/fn; NaN when the image contains nothing to score."""
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    total = 2 * tp + fp + fn
    return 2 * tp / total if total else float("nan")


PER_IMAGE_KEYS = {
    "dice": lambda r: r["dice"],
    "cldice": lambda r: r["cldice"],
    "clf1": lambda r: r["centerline_f1_2px"],
    "crossf1": lambda r: _f1(r["heatmap_points"]),
    "instf1": lambda r: _f1(r["instances"]),
    "assd": lambda r: r["assd_px"],
    "betti": lambda r: float(r["betti0_error"]),
}


def fmt(mean, sd=None, digits=4):
    if sd is None or not np.isfinite(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  table -> {path}")


def table_main(results, runs, out):
    grouped = main_by_arch(results)
    classical = load_group(results, "synth").get("classical")
    header = ["method", "parameters", "inference_ms", "seeds"] + METRIC_KEYS
    rows, records = [], {}
    for arch in METHOD_ORDER:
        if arch == "classical":
            if classical is None:
                continue
            metrics = [classical]
            params, seeds = 0, 1
        else:
            if arch not in grouped:
                continue
            metrics = grouped[arch]
            seeds = len(metrics)
            env = None
            for d in sorted((ROOT / runs).glob(f"{arch}_s*")):
                if (d / "environment.json").exists():
                    env = load_json(d / "environment.json")
                    break
            params = env["parameters"] if env else 0
        ms = float(np.mean([m["inference_ms_mean"] for m in metrics]))
        row = [LABELS_EN[arch], params, round(ms, 1), seeds]
        per_metric = {}
        for key in METRIC_KEYS:
            values = [metric_value(m, key) for m in metrics]
            values = [v for v in values if np.isfinite(v)]
            if not values:
                row.append("")
                continue
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            per_metric[key] = (mean, sd)
            row.append(fmt(mean, sd))
        rows.append(row)
        records[arch] = per_metric
    write_csv(out / "table1_main.csv", header, rows)
    return records


def table_ablation(results, out):
    ablations = load_group(results, "synth_ablation")
    grouped = main_by_arch(results)
    if not ablations or OURS not in grouped:
        return {}
    full = grouped[OURS][0]
    header = ["variant"] + METRIC_KEYS + [f"delta_{k}" for k in METRIC_KEYS]
    rows = [["full CFX-Net"] + [f"{metric_value(full, k):.4f}" for k in METRIC_KEYS]
            + ["0.0000"] * len(METRIC_KEYS)]
    for name in sorted(ablations):
        values = [metric_value(ablations[name], k) for k in METRIC_KEYS]
        deltas = [v - metric_value(full, k) for v, k in zip(values, METRIC_KEYS)]
        rows.append([name] + [f"{v:.4f}" for v in values] + [f"{d:+.4f}" for d in deltas])
    write_csv(out / "table2_ablation.csv", header, rows)
    return ablations


POST_ORDER = ["greedy", "angle_joint", "cfx", "cfx_conservative"]


def table_post(results, out):
    post = load_group(results, "synth_post")
    if not post:
        return
    header = (["linker"] + METRIC_KEYS
              + ["pair_accuracy", "committed_pairs", "abstention_rate"])
    rows = []
    for name in POST_ORDER:
        if name not in post:
            continue
        pairing = post[name].get("pairing", {})
        rows.append([name] + [f"{metric_value(post[name], k):.4f}" for k in METRIC_KEYS]
                    + [f"{pairing.get('pair_accuracy', float('nan')):.4f}",
                       pairing.get("committed_pairs", ""),
                       f"{pairing.get('abstention_rate', 0.0):.4f}"])
    write_csv(out / "table3_post.csv", header, rows)


def table_real(results, out):
    real = load_group(results, "real")
    if not real:
        return {}
    fields = ["sfar", "coverage", "equivariance_mse", "crossing_repeatability",
              "length_cv", "length_recovery_error", "fragmentation", "tracks",
              "substantial_tracks", "length_median", "length_p90", "crossings"]
    circular = {"sfar", "coverage", "equivariance_mse", "crossing_repeatability"}
    header = ["method", "plates"] + fields
    rows, records = [], {}
    for arch in METHOD_ORDER:
        if arch not in real:
            continue
        m = real[arch]
        row = [LABELS_EN[arch], m["n"]]
        values = {}
        for f in fields:
            if arch == "classical" and f in circular:
                row.append("n/a (circular)")
                continue
            stat = m.get(f) or {}
            mean = stat.get("mean", float("nan"))
            values[f] = mean
            row.append(fmt(mean, stat.get("std"), 4 if abs(mean) < 10 else 1)
                       if np.isfinite(mean) else "")
        rows.append(row)
        records[arch] = values
    write_csv(out / "table4_real.csv", header, rows)
    return records


def table_plate(results, out):
    plates = load_group(results, "synth_plate")
    if not plates:
        return
    grouped = {}
    for name, metrics in plates.items():
        arch = name.replace("_plate", "").rsplit("_s", 1)[0]
        grouped.setdefault(arch, []).append(metrics)
    header = ["method", "seeds"] + METRIC_KEYS + ["length_cv", "length_relative_error"]
    rows = []
    for arch in METHOD_ORDER:
        if arch not in grouped:
            continue
        metrics = grouped[arch]
        row = [LABELS_EN[arch], len(metrics)]
        row += [fmt(float(np.mean([metric_value(m, k) for m in metrics])),
                    float(np.std([metric_value(m, k) for m in metrics], ddof=1))
                    if len(metrics) > 1 else None)
                for k in METRIC_KEYS]
        for field in ("length_cv", "length_relative_error"):
            vals = [m[field]["mean"] for m in metrics if np.isfinite(m[field]["mean"])]
            row.append(f"{np.mean(vals):.4f}" if vals else "")
        rows.append(row)
    write_csv(out / "table5_plate.csv", header, rows)


def _apply_threshold(record, threshold):
    """Recompute this image's crossing counts at a different heatmap threshold."""
    peaks = record.get("crossing_peaks")
    if not peaks or threshold is None:
        return record["heatmap_points"]
    keep = [p for p, s in zip(peaks["points"], peaks["scores"]) if s >= threshold]
    return point_counts(keep, peaks["truth"], 6)


def significance(results, out):
    """Paired per-image tests: CFX-Net minus each baseline, averaged over seeds.

    Crossing counts are recomputed at each model's validation-selected threshold, so the
    test here uses exactly the operating point the main table reports.
    """
    base = group_dir(results, "synth")
    per_method = {}
    for arch in METHOD_ORDER:
        folders = sorted(base.glob(f"{arch}_s*")) or ([base / arch] if (base / arch).exists() else [])
        folders = [f for f in folders if (f / "per_image.json").exists()]
        if not folders:
            continue
        stacked = {}
        for folder in folders:
            thr = None
            for group in ("main", "ablation"):
                candidate = ROOT / "runs" / group / folder.name / "crossing_threshold.json"
                if candidate.exists():
                    thr = float(load_json(candidate)["threshold"])
                    break
            for record in load_json(folder / "per_image.json"):
                record["heatmap_points"] = _apply_threshold(record, thr)
                stacked.setdefault(record["id"], []).append(record)
        per_method[arch] = stacked
    if OURS not in per_method:
        return {}
    ids = sorted(per_method[OURS])
    payload = {"metrics": {}, "n_images": len(ids),
               "definition": "per-image values averaged over seeds, then paired across "
                             "methods on the same image; two-sided permutation test "
                             "(20000 draws) and Wilcoxon signed-rank"}
    for key, getter in PER_IMAGE_KEYS.items():
        payload["metrics"][key] = {}
        ours = np.array([np.mean([getter(r) for r in per_method[OURS][i]]) for i in ids])
        for arch in METHOD_ORDER:
            if arch == OURS or arch not in per_method:
                continue
            other = np.array([np.mean([getter(r) for r in per_method[arch][i]])
                              if i in per_method[arch] else np.nan for i in ids])
            payload["metrics"][key][arch] = paired_test(ours, other)
    save_json(out / "significance.json", payload)
    print(f"  table -> {out / 'significance.json'}")
    return payload


def leaderboard(records, real_records, out):
    """State, per metric, which method is best. Written whatever it says."""
    board = {"synthetic": {}, "real": {}, "ours": OURS}
    for key in METRIC_KEYS:
        entries = {a: v[key][0] for a, v in records.items() if key in v}
        if not entries:
            continue
        best = (max if HIGHER_BETTER[key] else min)(entries, key=entries.get)
        # A metric can saturate, in which case the ordering is decided by seed noise.
        # Record whether the gap to the winner is smaller than the seed-to-seed spread,
        # so a hairline loss is never dressed up as a real one - or a hairline win as a
        # real win.
        spreads = [abs(v[key][1]) for v in records.values()
                   if key in v and np.isfinite(v[key][1])]
        spread = float(np.mean(spreads)) if spreads else float("nan")
        gap = abs(entries[best] - entries.get(OURS, float("nan")))
        board["synthetic"][key] = dict(
            best=best, ours=entries.get(OURS), best_value=entries[best],
            ours_is_best=best == OURS, higher_is_better=HIGHER_BETTER[key],
            gap_to_best=gap, seed_spread=spread,
            within_seed_noise=bool(np.isfinite(spread) and gap <= spread),
            ranking=sorted(entries, key=lambda a: entries[a],
                           reverse=HIGHER_BETTER[key]))
    real_direction = dict(sfar=False, coverage=True, equivariance_mse=False,
                          crossing_repeatability=True, length_cv=False,
                          length_recovery_error=False, fragmentation=False)
    # The substrate region, the evidence core and exact equivariance are all properties
    # of the classical ridge operator itself, so scoring that baseline on the first four
    # is circular: it wins by construction without being accurate.  It is excluded from
    # those columns and the exclusion is recorded, not hidden.
    CIRCULAR = {"sfar", "coverage", "equivariance_mse", "crossing_repeatability"}
    board["real_exclusions"] = {
        "classical": sorted(CIRCULAR),
        "reason": "the reference region and the equivariance property are defined by the "
                  "classical ridge operator that this baseline is, so these four metrics "
                  "are circular for it"}
    for field, higher in real_direction.items():
        entries = {a: v[field] for a, v in real_records.items()
                   if np.isfinite(v.get(field, np.nan))
                   and not (a == "classical" and field in CIRCULAR)}
        if not entries:
            continue
        best = (max if higher else min)(entries, key=entries.get)
        board["real"][field] = dict(best=best, ours=entries.get(OURS),
                                    best_value=entries[best], ours_is_best=best == OURS,
                                    higher_is_better=higher,
                                    ranking=sorted(entries, key=lambda a: entries[a],
                                                   reverse=higher))
    wins = [k for k, v in board["synthetic"].items() if v["ours_is_best"]]
    losses = [k for k, v in board["synthetic"].items() if not v["ours_is_best"]]
    real_wins = [k for k, v in board["real"].items() if v["ours_is_best"]]
    real_losses = [k for k, v in board["real"].items() if not v["ours_is_best"]]
    hairline = [k for k in losses if board["synthetic"][k]["within_seed_noise"]]
    board["verdict"] = dict(
        synthetic_metrics_won=wins, synthetic_metrics_lost=losses,
        synthetic_losses_within_seed_noise=hairline,
        real_metrics_won=real_wins, real_metrics_lost=real_losses,
        clean_sweep=not losses and not real_losses,
        reading="a metric listed under synthetic_losses_within_seed_noise is a tie in "
                "practice: the gap to the winner is smaller than the seed-to-seed spread")
    save_json(out / "leaderboard.json", board)
    print(f"  table -> {out / 'leaderboard.json'}")
    print(f"  CFX-Net wins {len(wins)}/{len(board['synthetic'])} synthetic metrics, "
          f"{len(real_wins)}/{len(board['real'])} real metrics")
    if losses or real_losses:
        print(f"  NOT best on: synthetic {losses}, real {real_losses}")
    return board


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--runs", default="runs/main")
    p.add_argument("--out", default="results/tables")
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    records = table_main(a.results, a.runs, out)
    table_ablation(a.results, out)
    table_post(a.results, out)
    real_records = table_real(a.results, out)
    table_plate(a.results, out)
    significance(a.results, out)
    leaderboard(records, real_records, out)


if __name__ == "__main__":
    main()
