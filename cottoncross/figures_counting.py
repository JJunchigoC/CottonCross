"""Counting figures (fig19-fig25): per-image fiber counts for every model.

Same rules as `figures.py` - colour follows the model, one value axis per panel, dot
plots instead of truncated bars, a diverging map with a neutral midpoint for signed
errors.  Inputs are results/counting/* written by scripts/count_fibers.py and
scripts/count_report.py.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .common import load_json
from .count import chain_lengths, count_plate
from .figures import (GRID, INK, INK_2, INK_MUTED, METHOD_ORDER, OURS, PALETTE, SURFACE,
                      _save, diverging_cmap, label)

RULES = [("components", "connected components"), ("trajectories", "linked trajectories"),
         ("chains", "chain-level staple rule"), ("length", "pooled staple rule")]
BENCHES = [("bench", "Counting benchmark (exact count, 160 test plates)"),
           ("real", "Real plates (visual reference, 18 test plates)")]


def _summary(results, kind):
    path = Path(results) / "counting" / kind / "summary.json"
    return load_json(path) if path.exists() else None


def _per_image(results, kind):
    path = Path(results) / "counting" / kind / "per_image.csv"
    if not path.exists():
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def fig_counting_rules(out, results="results"):
    """How much of the counting accuracy comes from the counting rule itself."""
    sums = {k: _summary(results, k) for k, _ in BENCHES}
    if not all(sums.values()):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6), sharey=True)
    for ax, (kind, title) in zip(axes, BENCHES):
        agg = sums[kind]["aggregate"]
        models = [m for m in METHOD_ORDER if m in agg]
        offsets = np.linspace(-0.27, 0.27, len(models))
        for r, (rule, _) in enumerate(RULES):
            for off, m in zip(offsets, models):
                v = agg[m].get(rule, {}).get("mae")
                if v is None:
                    continue
                ax.scatter(max(v, 0.02), r + off, s=64 if m == OURS else 40,
                           c=PALETTE[m], edgecolor=INK if m == OURS else SURFACE,
                           linewidth=1.4 if m == OURS else 1.0, zorder=3,
                           label=label(m) if r == 0 else None)
        ax.set_xscale("log")
        ax.set_yticks(range(len(RULES)))
        ax.set_yticklabels([name for _, name in RULES])
        ax.set_xlabel("count MAE (fibers per image, log scale, lower is better)")
        ax.set_title(title, color=INK, loc="left")
        ax.grid(axis="y", visible=False)
        for r in range(len(RULES) - 1):
            ax.axhline(r + 0.5, color=GRID, linewidth=0.8)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("The counting rule matters more than the backbone: blob counting is off by "
                 "several fibers per image, the staple rules by a fraction of one",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    return _save(fig, out, "fig19_counting_rules")


def fig_counting_models(out, results="results"):
    """Backbone comparison under the two staple rules, with seed spread."""
    sums = {k: _summary(results, k) for k, _ in BENCHES}
    if not all(sums.values()):
        return None
    panels = [(kind, rule, key) for kind, _ in BENCHES
              for rule, key in (("length", "mae"), ("chains", "mae"), ("chains", "accuracy"))]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.2))
    names = dict(RULES)
    for ax, (kind, rule, key) in zip(axes.ravel(), panels):
        agg = sums[kind]["aggregate"]
        models = [m for m in METHOD_ORDER if m in agg]
        means = [agg[m][rule][key] for m in models]
        errs = [agg[m][rule].get(key + "_std", 0.0) for m in models]
        y = np.arange(len(models))[::-1]
        best = min(means) if key == "mae" else max(means)
        ax.axvline(best, color=INK_MUTED, linestyle=(0, (3, 3)), linewidth=1)
        ax.errorbar(means, y, xerr=errs, fmt="none", ecolor=INK_MUTED, elinewidth=1.2,
                    capsize=3, zorder=2)
        ax.scatter(means, y, s=78, c=[PALETTE[m] for m in models], edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        for yi, v in zip(y, means):
            ax.text(v, yi + 0.26, f"{v:.3f}", ha="center", va="bottom", color=INK_2,
                    fontsize=7.4, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
        ax.set_yticks(y)
        ax.set_yticklabels([label(m) for m in models], fontsize=7.8)
        ax.set_ylim(-0.8, len(models) - 0.2)
        ax.grid(axis="y", visible=False)
        what = "count MAE (lower is better)" if key == "mae" else "exact-count accuracy"
        ax.set_title(f"{'Benchmark' if kind == 'bench' else 'Real plates'}: {names[rule]}",
                     color=INK, loc="left", fontsize=9.5)
        ax.set_xlabel(what, fontsize=8)
    fig.suptitle("Per-image fiber count by backbone: mean over three seeds, whiskers are one "
                 "standard deviation, dashed line marks the best model",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, out, "fig20_counting_models")


def fig_count_confusion(out, results="results", rule="chains"):
    rows = [r for r in _per_image(results, "bench") if r["estimator"] == rule]
    if not rows:
        return None
    models = [m for m in METHOD_ORDER if any(r["model"] == m for r in rows)]
    truths = sorted({int(r["true"]) for r in rows})
    preds = list(range(0, max(max(truths) + 2, max(int(r["pred"]) for r in rows) + 1)))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE, "#9ec5f4", "#2a78d6", "#0d3a73"])
    fig, axes = plt.subplots(1, len(models), figsize=(2.55 * len(models) + 0.8, 3.4),
                             sharey=True)
    for ax, m in zip(axes, models):
        mat = np.zeros((len(truths), len(preds)))
        for r in rows:
            if r["model"] == m:
                p = min(int(r["pred"]), preds[-1])
                mat[truths.index(int(r["true"])), p] += 1
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(truths)):
            for j in range(len(preds)):
                if mat[i, j] >= 0.05:
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if mat[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(preds)))
        ax.set_xticklabels([str(p) for p in preds], fontsize=7)
        ax.set_yticks(range(len(truths)))
        ax.set_yticklabels([str(t) for t in truths], fontsize=7)
        ax.grid(False)
        ax.set_title(label(m), color=INK, fontsize=8.5, loc="left")
        ax.set_xlabel("predicted count", fontsize=7.5)
        diag = np.mean([mat[i, t] for i, t in enumerate(truths) if t < len(preds)])
        ax.text(0.98, 0.02, f"diag {diag:.2f}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7, color=INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
    axes[0].set_ylabel("true count")
    fig.suptitle("Counting benchmark, chain-level staple rule: row-normalised confusion of "
                 "true vs predicted fiber count (three seeds pooled)",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return _save(fig, out, "fig21_count_confusion")


def fig_real_plate_counts(out, results="results"):
    """Every model's count on every real plate, coloured by its error."""
    path = Path(results) / "counting" / "real" / "all_plates.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    models = [m for m in METHOD_ORDER if m in rows[0]]
    ids = [r["id"] for r in rows]
    ref = [int(r["reference"]) if r["reference"] else None for r in rows]
    scored = [r["confidence"] in ("high", "medium") for r in rows]
    counts = np.array([[float(r[m]) if r[m] != "" else np.nan for r in rows] for m in models])
    err = np.array([[c - t if (t is not None and s) else np.nan
                     for c, t, s in zip(row, ref, scored)] for row in counts])
    fig, ax = plt.subplots(figsize=(22, 0.52 * (len(models) + 1) + 2.4))
    shown = np.vstack([np.full((1, len(ids)), np.nan), err])
    im = ax.imshow(np.clip(shown, -3, 3), cmap=diverging_cmap(), vmin=-3, vmax=3,
                   aspect="auto")
    ax.set_facecolor("#e9e8e4")
    for j, (t, s) in enumerate(zip(ref, scored)):
        txt = "B" if t is None else str(t)
        ax.text(j, 0, txt, ha="center", va="center", fontsize=6.5,
                color=INK if s else INK_MUTED, fontweight="bold" if s else "normal")
    for i in range(len(models)):
        for j in range(len(ids)):
            if np.isfinite(counts[i, j]):
                e = err[i, j]
                ax.text(j, i + 1, f"{int(counts[i, j])}", ha="center", va="center",
                        fontsize=6,
                        color="#ffffff" if np.isfinite(e) and abs(e) >= 2 else INK)
    splits = [r["split"] for r in rows]
    for j in range(1, len(ids)):
        if splits[j] != splits[j - 1]:
            ax.axvline(j - 0.5, color=INK, linewidth=1.4)
    ax.set_yticks(range(len(models) + 1))
    ax.set_yticklabels(["reference"] + [label(m) for m in models], fontsize=8)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([i.split("_")[0] for i in ids], rotation=90, fontsize=6)
    ax.grid(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
    bar.set_label("count - reference (grey = no confident reference)", fontsize=8)
    ax.set_title("Fiber count of every real plate by every model (seed median).  Black lines "
                 "separate train | val | test; B = bundle, not countable; bold reference = "
                 "scored (high/medium confidence)", color=INK, loc="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "fig22_real_plate_counts")


def fig_count_by_n(out, results="results"):
    rows = _per_image(results, "bench")
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    for ax, rule in zip(axes, ("length", "chains")):
        sub = [r for r in rows if r["estimator"] == rule]
        truths = sorted({int(r["true"]) for r in sub})
        for m in METHOD_ORDER:
            vals = [np.mean([abs(int(r["error"])) for r in sub
                             if r["model"] == m and int(r["true"]) == t]) for t in truths]
            if not any(np.isfinite(vals)):
                continue
            ax.plot(truths, vals, marker="o", color=PALETTE[m], label=label(m),
                    linewidth=2.6 if m == OURS else 1.6, zorder=3 if m == OURS else 2)
        ax.set_xticks(truths)
        ax.set_xlabel("true number of fibers in the image")
        ax.set_title(dict(RULES)[rule], color=INK, loc="left")
    axes[0].set_ylabel("count MAE")
    axes[1].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Counting benchmark: error grows with the number of fibers, i.e. with the "
                 "number of crossings to resolve", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return _save(fig, out, "fig23_count_by_n")


def fig_staple_evidence(out, results="results"):
    """The equal-staple prior on the real plates: recovered length per reference fiber."""
    summary = _summary(results, "real")
    feats = Path(results) / "counting" / "features"
    if summary is None or not feats.exists():
        return None
    ref = {r["id"]: int(r["count"]) for r in csv.DictReader(
        open("data/real/fiber_counts_reference.csv", encoding="utf-8"))
        if r["count"] and r["confidence"] in ("high", "medium")}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.3))
    ax = axes[0]
    for m in METHOD_ORDER:
        path = feats / f"{m}_s42.json" if m != "classical" else feats / "classical_s0.json"
        if not path.exists():
            continue
        plates = load_json(path)["plates"]
        ratio = []
        for i, n in ref.items():
            c = chain_lengths(plates[i])
            ratio.append(c[c >= 140].sum() / n)
        ax.hist(ratio, bins=np.linspace(600, 3200, 40), histtype="step", linewidth=2.2
                if m == OURS else 1.3, color=PALETTE[m], label=label(m))
    chosen = summary["per_seed"][OURS]
    staples = [v["length"]["selected"]["staple"] for v in chosen.values()]
    ax.axvspan(min(staples), max(staples), color=PALETTE[OURS], alpha=0.12, linewidth=0)
    ax.set_xlabel("recovered fiber length per reference fiber (px), seed 42")
    ax.set_ylabel("plates")
    ax.set_title("Recovered length clusters at one staple", color=INK, loc="left")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    grid = np.arange(900, 2001, 25)
    for m in METHOD_ORDER:
        path = feats / f"{m}_s42.json" if m != "classical" else feats / "classical_s0.json"
        if not path.exists():
            continue
        plates = load_json(path)["plates"]
        ids = [i for i in ref if plates[i]["split"] in ("train", "val")]
        chains = {i: chain_lengths(plates[i]) for i in ids}
        mae = [np.mean([abs(count_plate(plates[i], s, chains=chains[i])["length"] - ref[i])
                        for i in ids]) for s in grid]
        ax.plot(grid, mae, color=PALETTE[m], linewidth=2.4 if m == OURS else 1.3,
                label=label(m))
    ax.set_xlabel("staple length S (px) - the one calibration number")
    ax.set_ylabel("count MAE, train+val plates")
    ax.set_title("Selecting S on the selection plates (35 mm = S px)", color=INK, loc="left")
    fig.suptitle("Equal-staple prior on the real plates: every fiber is 35 mm, so length "
                 "divided by one staple counts fibers", color=INK, fontsize=11, x=0.02,
                 ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return _save(fig, out, "fig24_staple_evidence")


def fig_counting_examples(out, results="results", plates=("p062_i01", "p069_i01",
                                                             "p058_i01", "p073_i01")):
    """What the application shows the user, on held-out real test plates."""
    cfg_path = Path("configs/counting.json")
    if not cfg_path.exists():
        return None
    from .app import count_image, draw
    from .common import read_image
    from .predict import load_model
    cfg = load_json(cfg_path)
    model, device = load_model(cfg["checkpoint"])
    ref = {r["id"]: r for r in csv.DictReader(
        open("data/real/fiber_counts_reference.csv", encoding="utf-8"))}
    fig, axes = plt.subplots(1, len(plates), figsize=(4.6 * len(plates), 5.6))
    for ax, pid in zip(axes, plates):
        gray = read_image(Path("data/real/roi") / f"{pid}.png", True)
        record, parts = count_image(gray, model, device, cfg["staple_px"],
                                    cfg["debris_fraction"])
        image = draw(gray, record, parts)
        ax.imshow(image[:, :, ::-1])
        ax.axis("off")
        r = ref.get(pid, {})
        ax.set_title(f"{pid.split('_')[0]}: counted {record['fibers']}, reference "
                     f"{r.get('count', '?')} ({r.get('confidence', '?')})",
                     color=INK, fontsize=9, loc="left")
    fig.suptitle("Application output on held-out real plates (CFX-Net): each traced chain in "
                 "its own colour (a fiber may span several), removed interference in grey, "
                 "crossing candidates circled",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, out, "fig25_counting_examples")


COUNTING_FIGURES = {
    "counting_rules": fig_counting_rules,
    "counting_models": fig_counting_models,
    "count_confusion": fig_count_confusion,
    "real_plate_counts": fig_real_plate_counts,
    "count_by_n": fig_count_by_n,
    "staple_evidence": fig_staple_evidence,
    "counting_examples": fig_counting_examples,
}
