"""Write docs/实验结果.md straight from the result files.

Every number in the report is read from results/, so the document cannot drift away
from the experiments.  Run it after scripts/make_tables.py.
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
from cottoncross.figures import (HIGHER_BETTER, LABELS_ZH, METHOD_ORDER,  # noqa: E402
                                 METRIC_KEYS, OURS, TEXT_ZH, load_group, main_by_arch,
                                 metric_value)

REAL_FIELDS = [
    ("sfar", "基底误报率 SFAR↓", False),
    ("coverage", "证据覆盖率 COV↑", True),
    ("equivariance_mse", "等变性误差 EQE↓", False),
    ("crossing_repeatability", "交叉可重复性 CREP↑", True),
    ("length_cv", "长度离散度 LCV↓", False),
    ("length_recovery_error", "长度恢复误差 LERR↓", False),
    ("fragmentation", "碎片化 FRAG↓", False),
]
REAL_EXTRA = [("tracks", "轨迹总数"), ("substantial_tracks", "实质轨迹"),
              ("length_median", "长度中位数（px）")]
CIRCULAR = {"sfar", "coverage", "equivariance_mse", "crossing_repeatability"}

ABLATION_ZH = {
    "no_topology": "去掉 soft-clDice 拓扑损失",
    "no_orientation": "去掉方向矩监督",
    "no_overlap": "去掉重叠重数分类头",
    "no_embedding": "去掉实例嵌入（连接退回纯几何联合匹配）",
    "no_deep": "去掉深监督",
    "no_oca": "去掉方向门控交叉注意力 OCA",
    "no_dcm": "去掉空洞上下文模块 DCM",
    "no_adaptation": "去掉无标注真实域自适应",
    "raw_input": "去掉背景归一化（改用普通局部对比度输入）",
    "no_equal_length": "去掉 35 mm 等长先验（纤维长度随机）",
    "no_complete_crossing": "去掉完整交叉保证（不做拒绝重采样）",
    "no_real_background": "去掉真实背景载体（纯合成背景）",
}


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def best_marker(value, values, higher):
    if not np.isfinite(value) or not values:
        return ""
    best = max(values) if higher else min(values)
    return " **←最优**" if abs(value - best) < 1e-12 else ""


def section_main(results, runs):
    grouped = main_by_arch(results)
    classical = load_group(results, "synth").get("classical")
    if not grouped:
        return "主对比结果尚未生成。\n"
    keys = ["dice", "cldice", "clf1", "crossf1", "crossap", "instf1", "pairacc",
            "assd", "betti"]
    columns = {}
    for arch in METHOD_ORDER:
        metrics = ([classical] if arch == "classical" and classical else
                   grouped.get(arch))
        if not metrics:
            continue
        columns[arch] = metrics
    rows = []
    for arch, metrics in columns.items():
        env = None
        for d in sorted((ROOT / runs).glob(f"{arch}_s*")):
            if (d / "environment.json").exists():
                env = load_json(d / "environment.json")
                break
        params = f"{env['parameters'] / 1e6:.2f}M" if env else "—"
        row = [LABELS_ZH[arch], params, len(metrics)]
        for k in keys:
            values = [metric_value(m, k) for m in metrics]
            values = [v for v in values if np.isfinite(v)]
            if not values:
                row.append("—")
                continue
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
            row.append(f"{mean:.4f}" + (f" ± {sd:.4f}" if sd is not None else ""))
        rows.append(row)
    header = ["方法", "参数量", "种子数"] + [TEXT_ZH[k] for k in keys]
    text = table(header, rows)

    board = ROOT / "results/tables/leaderboard.json"
    if board.exists():
        data = load_json(board)
        wins = data["verdict"]["synthetic_metrics_won"]
        losses = data["verdict"]["synthetic_metrics_lost"]
        hairline = set(data["verdict"].get("synthetic_losses_within_seed_noise", []))
        text += (f"\n\n本文方法在 **{len(wins)}/{len(wins) + len(losses)}** 项合成测试指标上排名第一。")
        if losses:
            text += "未夺冠的指标：\n\n"
            rows = []
            for k in losses:
                e = data["synthetic"][k]
                rows.append([TEXT_ZH[k], LABELS_ZH[e["best"]], f"{e['best_value']:.4f}",
                             f"{e['ours']:.4f}", f"{e['gap_to_best']:.4f}",
                             f"{e['seed_spread']:.4f}",
                             "是（视为并列）" if k in hairline else "否"])
            text += table(["指标", "最优方法", "最优值", "本文", "差距", "种子间波动",
                           "差距是否小于波动"], rows)
            if hairline:
                text += ("\n\n最后一列为「是」的项，差距小于各方法在三个种子之间的自身波动，"
                         "**不能据此说谁更好**，写论文时应表述为并列。")
        else:
            text += "没有任何一项被基线超过。"
    return text


def section_significance(results):
    path = ROOT / "results/tables/significance.json"
    if not path.exists():
        return "显著性检验尚未生成。\n"
    data = load_json(path)
    rows = []
    for key, entries in data["metrics"].items():
        for arch, stat in entries.items():
            p = stat["permutation_p"]
            mark = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            rows.append([TEXT_ZH.get(key, key), LABELS_ZH[arch], f"{stat['mean_difference']:+.4f}",
                         f"{p:.2e}" if p > 0 else "<5e-05", f"{stat['wilcoxon_p']:.2e}",
                         f"{stat['cohens_d']:+.2f}" if np.isfinite(stat["cohens_d"]) else "—",
                         mark])
    return (f"逐图配对检验，样本量 n = {data['n_images']} 张合成测试图，"
            "差值为 **本文方法减去基线**。\n\n"
            + table(["指标", "对照方法", "平均差值", "置换检验 p", "Wilcoxon p",
                     "Cohen's d", "显著性"], rows))


def section_ablation(results):
    ablations = load_group(results, "synth_ablation")
    grouped = main_by_arch(results)
    if not ablations or OURS not in grouped:
        return "消融实验尚未生成。\n"
    full = grouped[OURS][0]
    keys = ["dice", "cldice", "clf1", "crossf1", "instf1"]
    rows = [["完整 CFX-Net（本文）"] + [f"{metric_value(full, k):.4f}" for k in keys]]
    order = sorted(ablations, key=lambda n: np.mean(
        [metric_value(ablations[n], k) - metric_value(full, k) for k in keys]))
    for name in order:
        values = [metric_value(ablations[name], k) for k in keys]
        cells = []
        for v, k in zip(values, keys):
            delta = v - metric_value(full, k)
            cells.append(f"{v:.4f} ({delta:+.4f})")
        rows.append([ABLATION_ZH.get(name, name)] + cells)
    return ("括号内是相对完整模型的变化；负值说明该组件确实有贡献。"
            "每个消融只改一处，其余完全相同。\n\n"
            + table(["变体"] + [TEXT_ZH[k] for k in keys], rows))


def section_post(results):
    post = load_group(results, "synth_post")
    if not post:
        return "后处理对比尚未生成。\n"
    names = {"greedy": "角度贪心（参考论文做法）", "angle_joint": "几何联合匹配",
             "cfx": "嵌入引导联合匹配（本文）",
             "cfx_conservative": "本文 + 拒判（保守操作点）"}
    keys = ["instf1", "crossf1", "clf1"]
    rows = []
    for m in ("greedy", "angle_joint", "cfx", "cfx_conservative"):
        if m not in post:
            continue
        pairing = post[m].get("pairing", {})
        rows.append([names[m]] + [f"{metric_value(post[m], k):.4f}" for k in keys]
                    + [f"{pairing.get('pair_accuracy', float('nan')):.4f}",
                       str(pairing.get("committed_pairs", "—")),
                       f"{pairing.get('abstention_rate', 0.0) * 100:.1f}%"])
    linker = (load_json(ROOT / "configs/linker.json")
              if (ROOT / "configs/linker.json").exists() else None)
    note = ""
    if linker:
        c = linker["costs"]
        note = (f"\n\n连接器超参数在**验证集**上选定（{linker['selected_on']['images']} 张）："
                f"嵌入权重 {c['embedding']}、重叠权重 {c['overlap']}、"
                f"拒判阈值 {linker['margin_threshold']}。准则是轨迹 F1（保留三位小数），"
                f"并列时看配对准确率；拒判率是一个操作点，不参与搜索目标。"
                f"**测试集在这一步完全没有参与。**")
    return ("同一个网络输出、同一组阈值，只更换交叉点连接算法。\n\n"
            "**配对准确率**是指：在连接器**确实做出**的配对中，有多大比例把同一根真值纤维的两个端口连在一起。"
            "它与**拒判率**成对阅读——拒判会牺牲一点轨迹 F1，换取剩下那些配对更可靠。\n\n"
            + table(["连接算法"] + [TEXT_ZH[k] for k in keys]
                    + ["配对准确率", "已提交配对数", "拒判率"], rows) + note)


def section_variant_comparison(runs):
    """CFX-Net before and after the OCA conditioning fix, on VALIDATION only."""
    archive = ROOT / "docs/development/cfxnet_v1_validation.json"
    if not archive.exists():
        return ""
    old = load_json(archive)["runs"]
    rows = []
    for name in sorted(old):
        run = ROOT / runs / name
        new_summary = load_json(run / "summary.json") if (run / "summary.json").exists() else None
        new_cross = (load_json(run / "crossing_threshold.json")
                     if (run / "crossing_threshold.json").exists() else None)
        rows.append([
            name,
            f"{old[name]['summary']['best_val_score']:.4f}",
            f"{new_summary['best_val_score']:.4f}" if new_summary else "—",
            f"{old[name].get('val_crossing', {}).get('f1', float('nan')):.4f}",
            f"{new_cross['validation']['f1']:.4f}" if new_cross else "—",
            f"{old[name]['prior_gain']:+.4f}",
        ])
    v2 = ROOT / "runs/legacy"
    for row in rows:
        name = row[0]
        alt = v2 / name.replace("cfxnet_", "cfxnet_v2_")
        if (alt / "summary.json").exists():
            row[2] = f"{load_json(alt / 'summary.json')['best_val_score']:.4f}"
        if (alt / "crossing_threshold.json").exists():
            row[4] = f"{load_json(alt / 'crossing_threshold.json')['validation']['f1']:.4f}"
    rule = ROOT / "docs/development/variant_selection_rule.md"
    verdict = ""
    if rule.exists() and "采用 v1" in rule.read_text(encoding="utf-8"):
        verdict = ("\n\n**结论：第二版被放弃。** 按事先写定的准则（决定指标 = 验证集交叉点 F1），"
                   "第二版在该指标上更差，因此恢复第一版。这条负面结果说明：诊断没错"
                   "（先验通路确实没被使用），但把它修好并不能提升交叉检测——OCA 的作用主要"
                   "通过特征门控而非显式 logit 先验实现。第二版保留为 `oca_version=\"v2\"` 选项。")
    return ("\n\n## 2b. 开发记录：CFX-Net 的两个版本（只比验证集）\n\n"
            "第一版把方向描述子 `top2` 乘一个标量增益加到交叉热图 logit 上。训练后检查发现"
            "**该增益学成了约 0**（见最后一列），也就是这条先验通路没有被用上。第二版改为"
            "通道最大值 + 按图标准化 + 零初始化 1×1 卷积注入，并把注意力融合改成门控残差。\n\n"
            "**取舍准则在第二版的数字产生之前就写定**（`docs/development/variant_selection_rule.md`），"
            "决定指标是验证集交叉点 F1；测试集全程未参与。\n\n"
            + table(["运行", "v1 验证分数", "v2 验证分数", "v1 验证交叉 F1",
                     "v2 验证交叉 F1", "v1 学到的先验增益"], rows) + verdict)


def section_real_ablation(results):
    """Which design choices only pay off on the real plates?"""
    variants = load_group(results, "real_ablation")
    full = load_group(results, "real").get(OURS)
    if not variants or not full:
        return ""
    fields = [("sfar", "基底误报 SFAR↓"), ("coverage", "证据覆盖 COV↑"),
              ("equivariance_mse", "等变性误差 EQE↓"),
              ("crossing_repeatability", "交叉可重复 CREP↑"),
              ("fragmentation", "碎片化 FRAG↓"), ("tracks", "轨迹总数")]
    rows = [["完整 CFX-Net（本文）"] + [f"{full[f]['mean']:.5g}" for f, _ in fields]]
    for name in sorted(variants):
        m = variants[name]
        cells = []
        for f, _ in fields:
            value = m[f]["mean"]
            ratio = value / full[f]["mean"] if full[f]["mean"] else float("nan")
            cells.append(f"{value:.5g}" + (f" ({ratio:.1f}×)" if np.isfinite(ratio)
                                           and (ratio > 1.5 or ratio < 0.67) else ""))
        rows.append([ABLATION_ZH.get(name, name)] + cells)
    return ("\n\n## 6b. 只有在真实图像上才看得出来的设计选择\n\n"
            "合成测试集的背景载体来自同一个分布，所以在那上面看不出背景归一化的价值——"
            "消融表里 `raw_input` 甚至略好。但把同一批模型放到**真实照片**上，差别非常大。"
            "括号里是相对完整模型的倍数。\n\n"
            + table(["变体"] + [n for _, n in fields], rows)
            + "\n\n**这正是「合成集上看不出、真实数据上才暴露」的典型例子**，写论文时应当明确指出："
              "只在模拟分布内评价会系统性低估背景处理这类组件的作用。")


def section_census(results):
    """The plain answer: what did the system actually find on the real plates?"""
    path = diagnostic(ROOT / results, "crossing_census.json")
    thresholds = diagnostic(ROOT / results, "crossing_threshold_census.json")
    if not path.exists():
        return ""
    d = load_json(path)
    t = d["totals"]
    m = d["per_plate_mean"]
    rows = [
        ["骨架图交叉候选（度数 ≥ 4，保守）", t["crossing_candidates"],
         m["crossing_candidates"]],
        ["其中连接器给出了配对（未拒判）", t["resolved_crossings"], m["resolved_crossings"]],
        ["三分支候选（锐角交叉常表现为两个 Y 点）", t["branch_candidates"],
         round(t["branch_candidates"] / d["plates"], 2)],
        ["网络热图峰值（验证集选定阈值）", t["heatmap_crossings"], m["heatmap_crossings"]],
        ["实质轨迹（长度 ≥ 0.25×staple）", t["substantial_fibers"], m["substantial_fibers"]],
        ["其中参与至少一个交叉的轨迹", t["fibers_in_crossings"], m["fibers_in_crossings"]],
        ["全部轨迹（含尘点级碎片）", t["all_trajectories"],
         round(t["all_trajectories"] / d["plates"], 2)],
    ]
    text = (f"\n\n## 6d. 在真实照片上到底找到了什么\n\n"
            f"留出 {d['plates']} 张真实图，方法为 {LABELS_ZH.get(d['method'], d['method'])}。"
            f"轨迹是否「参与交叉」按几何判定：轨迹折线距交叉中心 "
            f"{d['tolerance_px']:.0f} px 以内。\n\n"
            + table(["统计量", f"{d['plates']} 张合计", "每张平均"], rows))
    if thresholds.exists():
        td = load_json(thresholds)
        rows = [[f"{k}" + ("（验证集选定）" if float(k) == td["selected_threshold"] else ""),
                 v["total"], round(v["mean"], 1)]
                for k, v in td["per_threshold"].items()]
        text += ("\n\n**交叉点数量对热图阈值极其敏感**，所以「有几个交叉」本身不是一个"
                 "由数据唯一确定的量：\n\n"
                 + table(["热图阈值", "合计", "每张平均"], rows))
    text += ("\n\n> ⚠️ 三条必须一起读的限制：\n"
             "> 1. **轨迹数不等于真实纤维根数**——一根纤维可能被断成几条轨迹，两根纤维也可能"
             "在交叉处被连成一条；视野裁剪还会切掉纤维末端。\n"
             "> 2. **交叉数取决于阈值与判据**，上表已经列出这种敏感性；"
             "没有人工真值可以判定哪个阈值是对的。\n"
             "> 3. **这是二维投影**——检出的是投影重叠，不能证明两根纤维物理接触，"
             "也判断不出谁在上面。")
    return text


def section_coverage(results):
    """The counterweight metric, measured properly."""
    path = diagnostic(ROOT / results, "coverage_curve.json")
    if not path.exists():
        return ""
    data = load_json(path)
    fractions = data["fractions"]
    rows = []
    for name, curve in data["methods"].items():
        arch = name.split(":")[-1]
        display = LABELS_ZH.get(arch, arch)
        if name.startswith("ablation:"):
            display = "消融：" + ABLATION_ZH.get(arch, arch)
        rows.append([display] + [f"{curve[str(f)]['mean']:.3f}" for f in fractions])
    return ("\n\n## 6c. 覆盖率的制衡作用在单一阈值下会失效\n\n"
            "除覆盖率外，真实图像上的每一个无标签统计量（基底误报、等变性、可重复性、碎片化）"
            "在模型**检得越少**时都越好看。覆盖率本应是制衡，但原来用的是最强的 0.05% 脊线核心——"
            "那里所有方法都饱和在 0.99 以上，制衡不起作用。把阈值放宽到微弱证据区才能看出差别：\n\n"
            + table(["方法"] + [f"前 {f * 100:g}% 证据" for f in fractions], rows)
            + "\n\n**这条结论限定了本文自己的结果**：CFX-Net 最低的基底误报率，部分是用「对微弱"
              "证据也最保守」换来的——它是所有学习型方法里最保守的一个，而不是无争议地最准的"
              "一个。免训练基线在这一列看着最高，但证据本身就由它定义，同样是循环。没有人工"
              "标注，无法判断被跳过的弱结构是真纤维还是噪声。")


def section_real(results):
    real = load_group(results, "real")
    if not real:
        return "真实图像评价尚未生成。\n"
    rows = []
    per_field = {}
    for field, _, higher in REAL_FIELDS:
        per_field[field] = [real[a][field]["mean"] for a in METHOD_ORDER
                            if a in real and np.isfinite(real[a][field]["mean"])
                            and not (a == "classical" and field in CIRCULAR)]
    for arch in METHOD_ORDER:
        if arch not in real:
            continue
        row = [LABELS_ZH[arch]]
        for field, _, higher in REAL_FIELDS:
            if arch == "classical" and field in CIRCULAR:
                row.append("不适用（循环）")
                continue
            stat = real[arch][field]
            v = stat["mean"]
            if not np.isfinite(v):
                row.append("—")
                continue
            lo, hi = stat.get("low"), stat.get("high")
            cell = f"{v:.4g}"
            if lo is not None and np.isfinite(lo):
                cell += f" [{lo:.4g}, {hi:.4g}]"
            row.append(cell + best_marker(v, per_field[field], higher))
        for field, _ in REAL_EXTRA:
            v = real[arch].get(field, {}).get("mean", float("nan"))
            row.append(f"{v:.1f}" if np.isfinite(v) else "—")
        rows.append(row)
    header = (["方法"] + [name for _, name, _ in REAL_FIELDS]
              + [name for _, name in REAL_EXTRA])
    plates = next(iter(real.values()))["n"]
    calib = load_json(ROOT / "data/calibration.json") if (ROOT / "data/calibration.json").exists() else {}
    note = ""
    if calib:
        note = (f"\n\n最后一列可以和标定值对照：staple 标定为 **{calib['staple_pixels']:.0f} px**，"
                "而该标定是用 **train 划分 + 免训练检测器**做的，这里是 **test 划分 + 学习到的模型**，"
                "所以两者相符是一次跨划分、跨方法的一致性检验（不是循环论证，但也不是独立真值）。"
                "「实质轨迹」指长度不小于 0.25×staple 的轨迹，用来排除尘点级碎片。")
    return (f"留出真实图像 {plates} 张，全部 **没有人工标注**。方括号是 95% bootstrap 区间。"
            "这些是一致性统计量，不是准确率。\n\n" + table(header, rows) + note)


def section_counting(results):
    """Same generated block as the README, so the two documents cannot disagree."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from fill_readme import counting_paragraph

    return counting_paragraph(results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--runs", default="runs/main")
    p.add_argument("--out", default="docs/实验结果.md")
    a = p.parse_args()

    calib = load_json(ROOT / "data/calibration.json") if (ROOT / "data/calibration.json").exists() else {}
    prov = load_json(ROOT / "data/synth/provenance.json") if (ROOT / "data/synth/provenance.json").exists() else {}
    bg_path = diagnostic(ROOT / "results", "background_report.json")
    bg = load_json(bg_path) if bg_path.exists() else {}
    log_path = diagnostic(ROOT / "results", "experiment_log.json")
    log = load_json(log_path) if log_path.exists() else {}
    env = {}
    for d in sorted((ROOT / a.runs).glob("*_s*")):
        if (d / "environment.json").exists():
            env = load_json(d / "environment.json")
            break

    parts = ["# 实验结果（自动生成，请勿手工编辑）",
             "",
             "本文件由 `python scripts/make_report.py` 从 `results/` 下的原始结果文件生成，"
             "因此文中每一个数字都可以在结果文件里逐项核对。",
             ""]
    parts += ["## 0. 运行环境与实验预算", ""]
    if env:
        parts.append(table(["项目", "值"], [
            ["GPU", env.get("gpu", "—")],
            ["PyTorch", env.get("torch", "—")],
            ["数值精度", env.get("amp", "—")],
            ["输入表示", env.get("input_mode", "—")],
            ["标签策略", env.get("label_policy", "—")],
        ]))
    if log:
        parts += ["", f"各阶段耗时（秒）：`{log.get('stages')}`，"
                      f"训练步数 {log.get('steps')}，无标注自适应从第 {log.get('warmup')} 步开始，"
                      f"batch size {log.get('batch')}，随机种子 {log.get('seeds')}。"]
    parts += ["", "## 1. 数据与先验", ""]
    if calib:
        parts.append(f"- **尺度标定**：由 35 mm 等长先验反推，staple = "
                     f"{calib['staple_pixels']:.1f} px，即 {calib['pixels_per_mm']:.2f} px/mm。"
                     f"该值是 **下界**（{calib['method']}）。")
    if prov:
        g = prov["complete_crossing_guarantee"]
        parts.append(f"- **完整交叉保证**：{g['satisfied']}/{g['of']} 张合格画布满足"
                     f"「交点距边界 ≥ margin 且四条分支在画布内各 ≥ {prov['arm_pixels']:.0f} px」。")
        parts.append(f"- **等长先验**：每根纤维弧长严格等于 {prov['staple_pixels']:.1f} px；"
                     f"背景载体只取自 **{prov['background_carrier_split']}** 划分。")
        parts.append(f"- **语料规模**：训练/验证/测试 = {prov['counts']}，"
                     f"另有 {prov['plate_count']} 张 {prov['plate_canvas']}px 整版画布。")
    gap_path = diagnostic(ROOT / "results", "domain_gap.json")
    gap = load_json(gap_path) if gap_path.exists() else {}
    if gap:
        parts.append(f"- **归一化后的域差**：在网络实际看到的表示里，合成纤维前景亮度约为真实的 "
                     f"{gap['foreground_ratio']:.2f} 倍，背景噪声水平约为 "
                     f"{gap['background_ratio']:.2f} 倍（1.00 表示两域一致）。"
                     f"{gap['caveat']}。")
    if bg:
        parts.append(f"- **背景处理效果**（{bg['n']} 张真实图）：大尺度明暗起伏占动态范围的比例由 "
                     f"{bg['illumination_span_raw']:.3f} 降到 {bg['illumination_span_snr']:.3f}；"
                     f"颗粒区与光滑区的高频对比度之比由 {bg['grain_ratio_raw']:.2f} 降到 "
                     f"{bg['grain_ratio_snr']:.2f}（1.00 表示两者已无法区分）。")
    parts += ["", "## 2. 主对比：相同数据、相同损失、相同预算，只换骨干网络", "",
              section_main(a.results, a.runs),
              section_variant_comparison(a.runs),
              "", "## 3. 统计显著性", "", section_significance(a.results),
              "", "## 4. 组件消融", "", section_ablation(a.results),
              "", "## 5. 交叉点连接算法对比", "", section_post(a.results),
              "", "## 6. 真实图像上的无标签评价", "", section_real(a.results),
              section_real_ablation(a.results),
              section_coverage(a.results),
              section_census(a.results),
              "", "## 7. 逐图纤维计数（每个模型、每张照片）", "", section_counting(a.results),
              "", "## 8. 读数注意事项", "",
              "- 第 2–5 节全部在 **留出的合成测试集** 上测量，衡量的是模型在所建模分布上的表现，"
              "不是真实照片的准确率。",
              "- 第 6 节在真实照片上测量，但这些照片 **没有人工真值**，所以报告的是一致性统计量"
              "（误报、覆盖、等变性、可重复性、长度离散度、碎片化），不能写成「准确率」。",
              "- 交叉点用容差 6 px 的一对一匹配统计，检测器无法靠在同一真值附近堆点来抬高召回。",
              "- 轨迹 F1 是本项目定义的诊断指标（一对一匹配、中心线 F1 ≥ 0.5、容差 2 px），"
              "不是公开榜单指标。",
              "- 第 7 节的计数基准有精确真值；真实照片的参考根数是目视估计（未经人工核验），"
              "且可评分的测试照片只有 18 张。",
              "- 三个随机种子只能支撑「差异稳定」的说法，不足以宣称 SOTA。",
              ""]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
