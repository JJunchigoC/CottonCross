"""Where every result lives.  One map, so no script hard-codes a results folder.

results/
    synthetic/main       held-out synthetic test, every model and seed
    synthetic/plate      plate-sized synthetic canvases
    synthetic/ablation   ablation runs on the synthetic test
    synthetic/linker     linker comparison (greedy / angle_joint / cfx)
    real/main            label-free evaluation on the real test plates, per model
    real/ablation        the same for the ablation variants
    real/review          offline review page
    counting/            per-image fiber counts (real plates and counting benchmark)
    tables/              every table in the report
    diagnostics/         background, domain-gap, coverage, census and run logs
"""
from __future__ import annotations

from pathlib import Path

GROUPS = {
    "synth": "synthetic/main",
    "synth_plate": "synthetic/plate",
    "synth_ablation": "synthetic/ablation",
    "synth_post": "synthetic/linker",
    "real": "real/main",
    "real_ablation": "real/ablation",
    "review": "real/review",
    "counting": "counting",
    "tables": "tables",
    "diagnostics": "diagnostics",
}

DIAGNOSTICS = ("background_report.json", "domain_gap.json", "coverage_curve.json",
               "crossing_census.json", "crossing_threshold_census.json",
               "experiment_log.json")


def group_dir(results, group):
    """Folder of a logical result group (e.g. 'synth' -> results/synthetic/main)."""
    return Path(results) / GROUPS.get(group, group)


def diagnostic(results, name):
    """Path of one diagnostics file, e.g. diagnostic('results', 'domain_gap.json')."""
    return Path(results) / GROUPS["diagnostics"] / name
