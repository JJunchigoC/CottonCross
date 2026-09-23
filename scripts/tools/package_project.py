"""Write the delivery manifest and per-file checksums; optionally build the archive.

    python scripts/tools/package_project.py             manifest + checksums
    python scripts/tools/package_project.py --archive   also build the ZIP next to the project

Run it only after the experiment matrix and the report have finished, so the manifest
describes what is actually on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cottoncross.common import load_json, save_json  # noqa: E402
from cottoncross.paths import group_dir  # noqa: E402

SKIP_DIRS = {"__pycache__", ".venv", ".git", ".idea", ".vscode", "logs"}
UNCOMPRESSED = {".png", ".jpg", ".jpeg", ".npz", ".pt", ".zip"}


def walk():
    for directory, folders, names in os.walk(ROOT):
        folders[:] = sorted(d for d in folders if d not in SKIP_DIRS)
        for name in sorted(names):
            if name.endswith(".pyc") or name == "FILES.sha256":
                continue
            yield Path(directory) / name


def digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def optional(path, key=None):
    p = ROOT / path
    if not p.exists():
        return None
    data = load_json(p)
    return data if key is None else data.get(key)


def manifest():
    runs = {}
    for group in ("main", "ablation"):
        base = ROOT / "runs" / group
        if not base.exists():
            continue
        runs[group] = {d.name: load_json(d / "summary.json")
                       for d in sorted(base.iterdir())
                       if (d / "summary.json").exists()}
    results = {}
    for group in ("synth", "synth_plate", "synth_ablation", "synth_post", "real"):
        base = group_dir(ROOT / "results", group)
        if not base.exists():
            continue
        results[group] = sorted(d.name for d in base.iterdir()
                                if (d / "metrics.json").exists())
    board = optional("results/tables/leaderboard.json")
    info = dict(
        project="CottonCross",
        subtitle="label-first, annotation-free cotton fiber crossing analysis",
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        status="executed research prototype; real-image accuracy NOT evaluated "
               "(no human ground truth exists for these plates)",
        real_accuracy=None,
        real_accuracy_reason="the 75 plates carry no human annotation; only label-free "
                             "consistency statistics are reported for them",
        calibration=optional("data/calibration.json"),
        synthetic=optional("data/synth/provenance.json"),
        real_dataset=optional("data/real/summary.json"),
        crossing_windows={k: v for k, v in (optional("data/real/crossing_windows.json") or {}).items()
                          if k != "windows"},
        real_augmentation_examples=optional("data/real_aug/manifest.json", "count"),
        background_report={k: v for k, v in (optional("results/diagnostics/background_report.json") or {}).items()
                           if k != "per_plate"},
        experiment_log=optional("results/diagnostics/experiment_log.json"),
        runs=runs,
        results=results,
        leaderboard=(board or {}).get("verdict"),
        entrypoints=dict(readme="README.md", results="docs/实验结果.md",
                         design="docs/研究设计与参考文献.md",
                         figures="docs/figures/", tables="results/tables/",
                         review="results/real/review/index.html",
                         weights="runs/main/cfxnet_s42/best.pt"),
    )
    save_json(ROOT / "docs/delivery/DELIVERY.json", info)
    print(f"manifest -> {ROOT / 'docs/delivery/DELIVERY.json'}")
    return info


def checksums():
    items = sorted(walk(), key=lambda p: p.relative_to(ROOT).as_posix())
    lines = [f"{digest(p)}  {p.relative_to(ROOT).as_posix()}" for p in items]
    (ROOT / "docs/delivery/FILES.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"checksums -> {ROOT / 'docs/delivery/FILES.sha256'} ({len(lines)} files)")
    return items


def archive(items):
    target = ROOT.parent / f"{ROOT.name}_完整项目.zip"
    items = list(items) + [ROOT / "docs/delivery/FILES.sha256"]
    with zipfile.ZipFile(target, "w", allowZip64=True) as z:
        for p in items:
            stored = p.suffix.lower() in UNCOMPRESSED
            z.write(p, arcname=f"{ROOT.name}/" + p.relative_to(ROOT).as_posix(),
                    compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,
                    compresslevel=None if stored else 6)
    with zipfile.ZipFile(target) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"archive CRC failure at {bad}")
        count = len(z.namelist())
    (ROOT.parent / f"{target.name}.sha256").write_text(
        f"{digest(target)}  {target.name}\n", encoding="utf-8")
    print(f"archive -> {target} ({count} entries, {target.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive", action="store_true")
    a = p.parse_args()
    manifest()
    items = checksums()
    if a.archive:
        archive(items)
