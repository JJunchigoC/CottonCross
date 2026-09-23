"""Apply the pre-registered CFX-Net v1 / v2 rule and act on it.

The rule was written to `docs/development/variant_selection_rule.md` BEFORE the deciding
number existed: the winner is the variant with the higher mean crossing F1 on the
synthetic VALIDATION split, because that is the quantity the OCA revision was aimed at.
The training-time validation score is reported either way but does not decide.

    python scripts/tools/select_variant.py            # report the comparison, change nothing
    python scripts/tools/select_variant.py --apply    # also swap the runs and configs
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cottoncross.common import load_json, save_json  # noqa: E402

SEEDS = [42, 43, 44]
RULE = ROOT / "docs/development/variant_selection_rule.md"


def collect(runs, names):
    scores, crossings = [], []
    for name in names:
        run = Path(runs) / name
        if (run / "summary.json").exists():
            scores.append(load_json(run / "summary.json")["best_val_score"])
        if (run / "crossing_threshold.json").exists():
            crossings.append(load_json(run / "crossing_threshold.json")["validation"]["f1"])
    return scores, crossings


def compare():
    archive = load_json(ROOT / "docs/development/cfxnet_v1_validation.json")["runs"]
    v1_scores = [archive[f"cfxnet_s{s}"]["summary"]["best_val_score"] for s in SEEDS
                 if f"cfxnet_s{s}" in archive]
    v1_cross = [archive[f"cfxnet_s{s}"]["val_crossing"]["f1"] for s in SEEDS
                if f"cfxnet_s{s}" in archive and "val_crossing" in archive[f"cfxnet_s{s}"]]
    v2_scores, v2_cross = collect(ROOT / "runs/main", [f"cfxnet_s{s}" for s in SEEDS])
    return dict(
        v1=dict(val_score=v1_scores, val_crossing_f1=v1_cross),
        v2=dict(val_score=v2_scores, val_crossing_f1=v2_cross))


def apply_v1():
    """Put the archived v1 runs back in place and pin the config to the v1 module."""
    for seed in SEEDS:
        source = ROOT / "runs/legacy" / f"cfxnet_v1_s{seed}"
        target = ROOT / "runs/main" / f"cfxnet_s{seed}"
        if not source.exists():
            raise SystemExit(f"cannot restore v1: {source} is missing")
        if target.exists():
            shutil.move(str(target), str(ROOT / "runs/legacy" / f"cfxnet_v2_s{seed}"))
        shutil.move(str(source), str(target))
        cfg = load_json(target / "config.json")
        cfg.setdefault("model", {})["oca_version"] = "v1"
        save_json(target / "config.json", cfg)
    linker = ROOT / "docs/development/linker_v1.json"
    if linker.exists():
        shutil.copy(linker, ROOT / "configs/linker.json")
    print("restored CFX-Net v1 into runs/main and pinned oca_version=v1")


def write_conclusion(data, winner, applied):
    if not RULE.exists():
        return
    text = RULE.read_text(encoding="utf-8")
    head = text.split("## 结论", 1)[0]

    def row(values):
        return (f"{np.mean(values):.4f}" if values else "—")

    body = [
        "## 结论", "",
        f"| 版本 | 验证分数（均值） | **验证交叉 F1（均值，决定指标）** |",
        "|---|---:|---:|",
        f"| v1 | {row(data['v1']['val_score'])} | **{row(data['v1']['val_crossing_f1'])}** |",
        f"| v2 | {row(data['v2']['val_score'])} | **{row(data['v2']['val_crossing_f1'])}** |",
        "",
        f"按事先写定的准则，**采用 {winner}**。" + ("（已执行文件替换。）" if applied else ""),
        "",
    ]
    if winner == "v1":
        body += ["第二版的改动（通道最大值 + 按图标准化 + 1×1 卷积注入 + 门控残差）在验证集上"
                 "**没有改善交叉检测**，因此按准则放弃。这是一次失败的尝试，保留在代码里作为"
                 "`oca_version=\"v2\"` 选项，并在此如实记录——写论文时应当把它写成消融的一部分，"
                 "而不是隐去。", ""]
    else:
        body += ["第二版在决定指标上更好，因此采用。若其训练期验证分数低于第一版，这一点在上表"
                 "中可见，必须一并报告，不能只讲有利的一面。", ""]
    RULE.write_text(head + "\n".join(body), encoding="utf-8")
    print(f"conclusion written to {RULE}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    data = compare()
    for name in ("v1", "v2"):
        print(f"{name}: val_score={data[name]['val_score']} "
              f"val_crossing_f1={data[name]['val_crossing_f1']}")
    if not data["v2"]["val_crossing_f1"]:
        raise SystemExit("v2 crossing thresholds not available yet; run tune_threshold.py")
    v1 = float(np.mean(data["v1"]["val_crossing_f1"]))
    v2 = float(np.mean(data["v2"]["val_crossing_f1"]))
    winner = "v2" if v2 > v1 else "v1"
    print(f"\nmean validation crossing F1:  v1 {v1:.4f}   v2 {v2:.4f}  ->  keep {winner}")
    if a.apply and winner == "v1":
        apply_v1()
    elif a.apply:
        for seed in SEEDS:
            cfg_path = ROOT / "runs/main" / f"cfxnet_s{seed}" / "config.json"
            if cfg_path.exists():
                cfg = load_json(cfg_path)
                cfg.setdefault("model", {})["oca_version"] = "v2"
                save_json(cfg_path, cfg)
        print("kept CFX-Net v2 and pinned oca_version=v2")
    write_conclusion(data, winner, a.apply)
    return winner


if __name__ == "__main__":
    main()
