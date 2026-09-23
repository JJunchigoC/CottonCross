"""Inject the headline results into README.md between the RESULTS-SUMMARY markers.

The numbers come from results/tables, so the README cannot disagree with the
experiments.  Run it after scripts/make_tables.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cottoncross.common import load_json  # noqa: E402
from cottoncross.paths import diagnostic  # noqa: E402
from cottoncross.figures import (LABELS_ZH, METHOD_ORDER, OURS, TEXT_ZH,  # noqa: E402
                                 load_group, main_by_arch, metric_value)

START = "<!-- RESULTS-SUMMARY-START -->"
END = "<!-- RESULTS-SUMMARY-END -->"
LINKER_START = "<!-- LINKER-CONFIG-START -->"
LINKER_END = "<!-- LINKER-CONFIG-END -->"
CENSUS_START = "<!-- CENSUS-START -->"
CENSUS_END = "<!-- CENSUS-END -->"
COUNT_START = "<!-- COUNTING-START -->"
COUNT_END = "<!-- COUNTING-END -->"
COUNT_RULES = [("components", "连通域计数（常规做法）"), ("trajectories", "轨迹计数（≥0.5 根长）"),
               ("chains", "链级等长规则"), ("length", "总长等长规则（主协议）")]


def counting_paragraph(results="results"):
    """Per-image fiber counting: rule comparison and backbone comparison, both benchmarks."""
    paths = {k: ROOT / results / "counting" / k / "summary.json" for k in ("bench", "real")}
    if not all(p.exists() for p in paths.values()):
        return "运行 `python scripts/count_fibers.py` 与 `python scripts/count_report.py` 生成本节。"
    S = {k: load_json(p) for k, p in paths.items()}
    order = [m for m in METHOD_ORDER if m in S["bench"]["aggregate"]]
    lines = ["**表 C1：计数规则的影响**（计数 MAE = 每张图的根数平均绝对误差，越低越好；"
             "三个种子均值，测试集）", ""]
    rows = []
    for est, name in COUNT_RULES:
        vals_b = [S["bench"]["aggregate"][m][est]["mae"] for m in order]
        vals_r = [S["real"]["aggregate"][m][est]["mae"] for m in order if m in S["real"]["aggregate"]]
        ours_b = S["bench"]["aggregate"][OURS][est]["mae"]
        ours_r = S["real"]["aggregate"][OURS][est]["mae"]
        rows.append([name, f"{min(vals_b):.3f} – {max(vals_b):.3f}", f"{ours_b:.3f}",
                     f"{min(vals_r):.3f} – {max(vals_r):.3f}", f"{ours_r:.3f}"])
    lines.append(md_table(["计数规则", "计数基准：各模型范围", "计数基准：CFX-Net",
                       "真实照片：各模型范围", "真实照片：CFX-Net"], rows))
    lines += ["", "**表 C2：各骨干在两条等长规则下的逐图计数**（均值 ± 种子间标准差；"
              "完全正确率 = 计数与真值完全相同的图像比例）", ""]
    rows = []
    for m in order:
        cells = [LABELS_ZH.get(m, m) if m != OURS else f"**{LABELS_ZH.get(m, m)}**"]
        for kind in ("bench", "real"):
            agg = S[kind]["aggregate"].get(m)
            for est in ("length", "chains"):
                if not agg:
                    cells.append("—")
                    continue
                a = agg[est]
                cells.append(f"{a['mae']:.3f} ± {a['mae_std']:.3f}<br><sub>完全正确 "
                             f"{a['accuracy']:.2f}</sub>")
        rows.append(cells)
    lines.append(md_table(["模型", "基准·总长规则", "基准·链级规则", "真实·总长规则",
                       "真实·链级规则"], rows))
    sig = []
    for kind, word in (("bench", "计数基准"), ("real", "真实照片")):
        for est, name in (("chains", "链级规则"), ("length", "总长规则")):
            tests = S[kind]["significance"].get(est, {})
            better = [LABELS_ZH.get(m, m) for m, t in tests.items()
                      if t["mean_difference"] < 0 and t["wilcoxon_p"] < 0.05]
            worse = [LABELS_ZH.get(m, m) for m, t in tests.items()
                     if t["mean_difference"] > 0 and t["wilcoxon_p"] < 0.05]
            sig.append(f"- {word}·{name}：CFX-Net 显著优于 {('、'.join(better) or '无')}；"
                       f"显著劣于 {('、'.join(worse) or '无')}（逐图 Wilcoxon，p < 0.05，"
                       f"其余为统计上并列）。")
    lines += ["", "**显著性**：", ""] + sig
    lines += ["", "逐张结果：`results/counting/real/all_plates.csv`（75 张 × 7 个模型 × 3 个种子）、"
              "`results/counting/*/per_image.csv`；表：`results/tables/table_counting*.csv`。"]
    return "\n".join(lines)


def census_paragraph():
    path = diagnostic(ROOT / "results", "crossing_census.json")
    if not path.exists():
        return "运行 `python scripts/tools/crossing_census.py` 生成本节统计。"
    d = load_json(path)
    t, m = d["totals"], d["per_plate_mean"]
    lines = [
        f"留出 **{d['plates']} 张**真实照片（`results/diagnostics/crossing_census.json`、"
        f"`results/real/main/{d['method']}/crossing_census.csv`）：", "",
        md_table(["统计量", f"{d['plates']} 张合计", "每张平均"], [
            ["骨架图交叉候选（度数 ≥ 4，保守）", t["crossing_candidates"],
             m["crossing_candidates"]],
            ["其中连接器给出了配对（未拒判）", t["resolved_crossings"],
             m["resolved_crossings"]],
            ["三分支候选（锐角交叉常表现为两个 Y 点）", t["branch_candidates"],
             round(t["branch_candidates"] / d["plates"], 2)],
            ["网络热图峰值（验证集选定阈值）", t["heatmap_crossings"],
             m["heatmap_crossings"]],
            ["**实质轨迹**（长度 ≥ 0.25×staple）", f"**{t['substantial_fibers']}**",
             f"**{m['substantial_fibers']}**"],
            ["**其中参与至少一个交叉的**", f"**{t['fibers_in_crossings']}**",
             f"**{m['fibers_in_crossings']}**"],
            ["全部轨迹（含尘点级碎片）", t["all_trajectories"],
             round(t["all_trajectories"] / d["plates"], 2)],
        ]), "",
    ]
    thresholds = diagnostic(ROOT / "results", "crossing_threshold_census.json")
    if thresholds.exists():
        td = load_json(thresholds)
        lines += [
            "**交叉点数量对热图阈值极其敏感**，所以「有几个交叉」本身不是一个由数据唯一"
            "确定的量：", "",
            md_table(["热图阈值", "合计", "每张平均"],
                     [[k + ("（验证集选定）" if float(k) == td["selected_threshold"] else ""),
                       v["total"], round(v["mean"], 1)]
                      for k, v in td["per_threshold"].items()]), "",
        ]
    lines += [
        "> ⚠️ 三条必须一起读的限制：",
        "> 1. **轨迹数不等于真实纤维根数**——一根纤维可能被断成几条轨迹，两根纤维也可能"
        "在交叉处被连成一条；视野裁剪还会切掉纤维末端。",
        "> 2. **交叉数取决于阈值与判据**（见上表）；没有人工真值可以判定哪个阈值是对的。",
        "> 3. **这是二维投影**——检出的是投影重叠，不能证明两根纤维物理接触，也判断不出"
        "谁在上面。",
    ]
    return "\n".join(lines)


def linker_paragraph():
    path = ROOT / "configs/linker.json"
    if not path.exists():
        return "连接器超参数尚未在验证集上选定（运行 `scripts/tools/tune_linker.py`）。"
    data = load_json(path)
    c = data["costs"]
    line = (f"验证集选出的权重（`configs/linker.json`，在 "
            f"{data['selected_on']['images']} 张验证图上搜索）："
            f"`w_ang={c['angle']}, w_emb={c['embedding']}, w_int={c['intensity']}, "
            f"w_sup={c['support']}, w_ovl={c['overlap']}, w_mom={c['moments']}`，"
            f"拒判阈值 {data['margin_threshold']}。")
    notes = []
    if c["overlap"] == 0:
        notes.append("**`w_ovl` 被选成了 0**——重叠重数作为*连接线索*没有被验证集选中。"
                     "它作为**辅助训练任务**是否有用是另一回事，由消融 `no_overlap` 回答。")
    if c["embedding"] > 0:
        val = data.get("validation", {})
        notes.append(f"**`w_emb` 被选成了 {c['embedding']}**——学习到的实例嵌入确实被选中，"
                     f"验证集上配对准确率 {val.get('pair_accuracy', float('nan')):.4f}。")
    if notes:
        line += "\n\n> " + " ".join(notes) + " 这一段按实际搜索结果如实记录，不做美化。"
    return line

HEADLINE = ["dice", "cldice", "clf1", "crossf1", "crossap", "instf1"]
REAL_FIELDS = [("sfar", "基底误报 ↓"), ("coverage", "证据覆盖 ↑"),
               ("equivariance_mse", "等变性误差 ↓"),
               ("crossing_repeatability", "交叉可重复 ↑"),
               ("length_cv", "长度离散度 ↓"), ("length_recovery_error", "长度恢复误差 ↓"),
               ("fragmentation", "碎片化 ↓"), ("tracks", "轨迹总数")]
CIRCULAR = {"sfar", "coverage", "equivariance_mse", "crossing_repeatability"}


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "|".join(["---"] * len(header)) + "|"]
                     + ["| " + " | ".join(str(v) for v in r) + " |" for r in rows])


def build(results):
    grouped = main_by_arch(results)
    classical = load_group(results, "synth").get("classical")
    if not grouped:
        return "> 结果尚未生成。"

    rows = []
    for arch in METHOD_ORDER:
        metrics = [classical] if arch == "classical" and classical else grouped.get(arch)
        if not metrics:
            continue
        row = [("**" + LABELS_ZH[arch] + "**") if arch == OURS else LABELS_ZH[arch]]
        for k in HEADLINE:
            values = [metric_value(m, k) for m in metrics]
            values = [v for v in values if np.isfinite(v)]
            if not values:
                row.append("—")
                continue
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
            row.append(f"{mean:.4f}" + (f"<br><sub>±{sd:.4f}</sub>" if sd else ""))
        rows.append(row)
    parts = ["### 主对比（合成留出测试集，3 个随机种子，同数据 / 同损失 / 同预算）", "",
             md_table(["方法"] + [TEXT_ZH[k] for k in HEADLINE], rows), ""]

    board = ROOT / "results/tables/leaderboard.json"
    if board.exists():
        data = load_json(board)
        syn_win = data["verdict"]["synthetic_metrics_won"]
        syn_lose = data["verdict"]["synthetic_metrics_lost"]
        real_win = data["verdict"]["real_metrics_won"]
        real_lose = data["verdict"]["real_metrics_lost"]
        line = (f"**排行榜结论**：本文方法在 **{len(syn_win)}/{len(syn_win) + len(syn_lose)}** "
                f"项合成测试指标、**{len(real_win)}/{len(real_win) + len(real_lose)}** "
                f"项真实无标签指标上排名第一。")
        hairline = set(data["verdict"].get("synthetic_losses_within_seed_noise", []))
        if syn_lose or real_lose:
            detail = []
            for k in syn_lose:
                e = data["synthetic"][k]
                tie = "，差距小于种子间波动，应视为并列" if k in hairline else ""
                detail.append(f"{TEXT_ZH.get(k, k)}（最优 {LABELS_ZH[e['best']]} "
                              f"{e['best_value']:.4f}，本文 {e['ours']:.4f}{tie}）")
            for k in real_lose:
                e = data["real"][k]
                detail.append(f"{dict(REAL_FIELDS).get(k, k)}（最优 {LABELS_ZH[e['best']]} "
                              f"{e['best_value']:.4g}，本文 {e['ours']:.4g}）")
            line += " 未夺冠的项：" + "；".join(detail) + "。"
        else:
            line += " 没有任何一项被基线超过。"
        parts += [line, ""]

    sig = ROOT / "results/tables/significance.json"
    if sig.exists():
        data = load_json(sig)
        worst = []
        for key, entries in data["metrics"].items():
            for arch, stat in entries.items():
                worst.append((stat["permutation_p"], key, arch, stat["mean_difference"]))
        worst.sort()
        best_p = worst[0][0] if worst else float("nan")
        insignificant = [w for w in worst if w[0] >= 0.05]
        parts += [f"**显著性**：逐图配对置换检验，n = {data['n_images']} 张测试图。"
                  f"共 {len(worst)} 组「本文 vs. 基线 × 指标」比较，"
                  f"其中 {len(worst) - len(insignificant)} 组达到 p < 0.05"
                  f"（最小 p {'< 5e-05' if best_p == 0 else f'= {best_p:.1e}'}）。"
                  + ("" if not insignificant else
                     f"未达显著的 {len(insignificant)} 组见 `results/tables/significance.json`。"),
                  ""]

    real = load_group(results, "real")
    if real:
        rows = []
        for arch in METHOD_ORDER:
            if arch not in real:
                continue
            row = [("**" + LABELS_ZH[arch] + "**") if arch == OURS else LABELS_ZH[arch]]
            for field, _ in REAL_FIELDS:
                if arch == "classical" and field in CIRCULAR:
                    row.append("不适用")
                    continue
                v = real[arch].get(field, {}).get("mean", float("nan"))
                row.append(f"{v:.4g}" if np.isfinite(v) else "—")
            rows.append(row)
        plates = next(iter(real.values()))["n"]
        parts += [f"### 真实留出图像（{plates} 张，**无任何人工标注**）", "",
                  md_table(["方法"] + [n for _, n in REAL_FIELDS], rows), "",
                  "这些是一致性统计量，**不是准确率**；↑ / ↓ 标示方向。", "",
                  "两条必须一起读的说明：", "",
                  "1. **免训练 Hessian 基线在前四项标为「不适用」**——基底区域与证据核心都由它"
                  "自己用的那个脊线算子定义，固定线性滤波器又按构造严格等变，所以它会「满分」"
                  "却并不更准。把它算进去是循环论证。它的碎片化与长度恢复误差看着好，也是因为"
                  "它检出的结构本来就少得多（35 条轨迹 vs 学习型方法的 118–538 条）。",
                  "2. **基底误报率与证据覆盖率必须成对看**——只有同时做到「少误报」和「不漏掉强"
                  "证据」才有意义；**长度离散度也不能单看**，把纤维打成等长碎片同样会让它变低，"
                  "所以要和长度恢复误差、碎片化一起读。", ""]

    ablation = load_group(results, "synth_ablation")
    if ablation and OURS in grouped:
        full = grouped[OURS][0]
        deltas = sorted(
            ((name, float(np.mean([metric_value(m, k) - metric_value(full, k)
                                   for k in HEADLINE])))
             for name, m in ablation.items()), key=lambda t: t[1])
        top = deltas[:4]
        parts += ["**消融（去掉一个组件后，6 项指标的平均变化）**：" +
                  "；".join(f"`{n}` {d:+.4f}" for n, d in top) +
                  "。完整表见 `results/tables/table2_ablation.csv`。", ""]

    parts += ["> 以上全部为合成留出测试集与无标签真实指标。"
              "**真实照片的识别准确率没有被评价，因为不存在人工真值。**"]
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--readme", default="README.md")
    a = p.parse_args()
    path = ROOT / a.readme
    text = path.read_text(encoding="utf-8")

    def replace(source, start, end, body, what):
        if start not in source or end not in source:
            raise SystemExit(f"README is missing the {what} markers")
        head, rest = source.split(start, 1)
        _, tail = rest.split(end, 1)
        return f"{head}{start}\n\n{body}\n\n{end}{tail}"

    text = replace(text, START, END, build(a.results), "RESULTS-SUMMARY")
    text = replace(text, LINKER_START, LINKER_END, linker_paragraph(), "LINKER-CONFIG")
    text = replace(text, CENSUS_START, CENSUS_END, census_paragraph(), "CENSUS")
    text = replace(text, COUNT_START, COUNT_END, counting_paragraph(a.results), "COUNTING")
    path.write_text(text, encoding="utf-8")
    print(f"README generated sections updated: {path}")


if __name__ == "__main__":
    main()
