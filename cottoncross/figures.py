"""Every figure in the report, generated from the saved result files.

Colour rules follow the project's data-viz standard: one fixed hue per method for the
whole document (colour follows the entity, never its rank), a single value axis per
panel, recessive grid and axes, text in ink rather than series colour, and a legend
whenever more than one series is on screen.  The palette is validated by
`scripts/tools/validate_palette.py` - run it after any change.

    python -m cottoncross.figures --all
    python -m cottoncross.figures --only main_comparison --lang zh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

from .paths import diagnostic, group_dir
from .common import load_json  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3df"

PALETTE = {
    "classical": "#4a3aa7",   # violet
    "unet": "#008300",        # green
    "attunet": "#e87ba4",     # magenta
    "unetpp": "#eda100",      # yellow
    "resunet": "#1baf7a",     # aqua
    "dscnet": "#eb6834",      # orange
    "cfxnet": "#2a78d6",      # blue - ours
}
METHOD_ORDER = ["classical", "unet", "attunet", "unetpp", "resunet", "dscnet", "cfxnet"]
OURS = "cfxnet"

LABELS_EN = {
    "classical": "Hessian ridge", "unet": "U-Net", "attunet": "Attention U-Net",
    "unetpp": "U-Net++", "resunet": "Residual U-Net", "dscnet": "DSCNet",
    "cfxnet": "CFX-Net (ours)",
}
LABELS_ZH = {
    "classical": "Hessian 脊线", "unet": "U-Net", "attunet": "Attention U-Net",
    "unetpp": "U-Net++", "resunet": "残差 U-Net", "dscnet": "DSCNet",
    "cfxnet": "CFX-Net（本文）",
}

TEXT_EN = dict(
    dice="Dice", cldice="clDice", clf1="Centerline F1", crossf1="Crossing F1",
    crossap="Crossing AP", instf1="Trajectory F1", pairacc="Pairing accuracy",
    assd="ASSD (px, lower better)",
    betti="Betti-0 error (lower better)", overlap="Overlap Dice",
    step="training step", val="validation score", loss="training loss",
    recall="recall", precision="precision", params="parameters (M)",
    lengthpx="trajectory length (px)", count="count",
    sfar="substrate false alarm", coverage="evidence coverage",
    eqe="equivariance error", crep="crossing repeatability",
    lcv="length dispersion (CV)", frag="fragmentation",
)
TEXT_ZH = dict(
    dice="Dice", cldice="clDice", clf1="中心线 F1", crossf1="交叉点 F1",
    crossap="交叉点 AP", instf1="轨迹 F1", pairacc="配对准确率",
    assd="ASSD（像素，越低越好）",
    betti="Betti-0 误差（越低越好）", overlap="重叠区 Dice",
    step="训练步数", val="验证分数", loss="训练损失",
    recall="召回率", precision="精确率", params="参数量（百万）",
    lengthpx="轨迹长度（像素）", count="数量",
    sfar="基底误报率", coverage="证据覆盖率",
    eqe="等变性误差", crep="交叉点可重复性",
    lcv="长度离散度（CV）", frag="碎片化指数",
)

# Longest phrases first: replacement is applied in order over the already-formatted
# string, so a specific sentence must not be broken up by a shorter fragment first.
TRANSLATIONS = [
    ("The counting rule matters more than the backbone: blob counting is off by several "
     "fibers per image, the staple rules by a fraction of one",
     "计数规则比骨干网络更重要：连通域计数每张图差好几根，等长规则只差零点几根"),
    ("Counting benchmark (exact count, 160 test plates)", "计数基准（精确真值，160 张测试图）"),
    ("Real plates (visual reference, 18 test plates)", "真实照片（目视参考计数，18 张测试图）"),
    ("count MAE (fibers per image, log scale, lower is better)",
     "计数 MAE（每张图的根数误差，对数坐标，越低越好）"),
    ("Per-image fiber count by backbone: mean over three seeds, whiskers are one standard "
     "deviation, dashed line marks the best model",
     "各骨干网络的逐图纤维计数：三个种子的均值，须线为一个标准差，虚线标出最优模型"),
    ("Counting benchmark, chain-level staple rule: row-normalised confusion of true vs "
     "predicted fiber count (three seeds pooled)",
     "计数基准（链级等长规则）：真实根数 vs 预测根数的行归一化混淆矩阵（三个种子合并）"),
    ("Fiber count of every real plate by every model (seed median).  Black lines separate "
     "train | val | test; B = bundle, not countable; bold reference = scored "
     "(high/medium confidence)",
     "每个模型对每张真实照片的纤维计数（种子中位数）。黑线分隔 训练 | 验证 | 测试；"
     "B = 纤维束，无法计数；加粗参考值 = 参与评分（高/中置信度）"),
    ("count - reference (grey = no confident reference)", "计数 − 参考（灰色 = 无可信参考）"),
    ("Counting benchmark: error grows with the number of fibers, i.e. with the number of "
     "crossings to resolve", "计数基准：误差随纤维根数增加，即随需要解开的交叉增多而增长"),
    ("true number of fibers in the image", "图中真实纤维根数"),
    ("Equal-staple prior on the real plates: every fiber is 35 mm, so length divided by one "
     "staple counts fibers", "真实照片上的等长先验：每根纤维都是 35 mm，总长度除以单根长度即为根数"),
    ("recovered fiber length per reference fiber (px), seed 42",
     "每根参考纤维对应的恢复长度（像素），种子 42"),
    ("Recovered length clusters at one staple", "恢复长度聚集在一根纤维的长度处"),
    ("staple length S (px) - the one calibration number", "单根纤维长度 S（像素）——唯一的标定量"),
    ("count MAE, train+val plates", "计数 MAE（训练+验证照片）"),
    ("Selecting S on the selection plates (35 mm = S px)", "在选择集照片上选定 S（35 mm = S 像素）"),
    ("Application output on held-out real plates (CFX-Net): each traced chain in its own "
     "colour (a fiber may span several), removed interference in grey, crossing candidates "
     "circled",
     "应用程序在留出真实照片上的输出（CFX-Net）：每条追踪链一种颜色（一根纤维可能由几条链组成），"
     "被剔除的干扰为灰色，候选交叉点画圈"),
    ("connected components", "连通域计数"), ("linked trajectories", "连接后的轨迹计数"),
    ("chain-level staple rule", "链级等长规则"), ("pooled staple rule", "总长等长规则"),
    ("count MAE (lower is better)", "计数 MAE（越低越好）"),
    ("exact-count accuracy", "计数完全正确率"), ("predicted count", "预测根数"),
    ("true count", "真实根数"), ("count MAE", "计数 MAE"), ("Benchmark: ", "计数基准："),
    ("Real plates: ", "真实照片："), ("counted ", "计数 "), (", reference ", "，参考 "),
    ("reference", "参考"), ("plates", "照片数"), ("diag ", "对角 "),
    ("Background normalisation removes the two nuisances that cause false fibers",
     "背景归一化消除了两类会造成假纤维的干扰"),
    ("Label-first samples: geometry drawn first, image rendered from it "
     "(blue = verified complete crossing, amber = clipped)",
     "标签先行样例：先画几何、再由几何渲染图像（蓝圈 = 已验证的完整交叉，橙圈 = 被裁断）"),
    ("Real-plate augmentation is crossing-centred: every crop carries a whole crossing "
     "(circle = detected junction)",
     "真实图像的增强以交叉为中心：每个裁剪块都含一个完整交叉（圆圈 = 检出的关节点）"),
    ("Held-out synthetic test set: mean over seeds, whiskers are one standard deviation, "
     "dashed line marks the best method",
     "合成留出测试集：多种子均值，误差棒为一倍标准差，虚线标出最优方法"),
    ("Per-metric ranking among the six learned backbones, rescaled per axis\n"
     "(the training-free baseline is far below this range and is omitted)",
     "六个学习型骨干在各指标上的排名（逐轴归一化）\n（免训练基线远低于该范围，未画出）"),
    ("The 'empty' column is the deliberate negative case: no fibers, so a correct "
     "detector scores 1.0 trivially",
     "「empty」列是刻意设置的负例：没有纤维，正确的检测器自然得 1.0"),
    ("Crossing resolution: identical predictions, four linkers. Abstention trades "
     "trajectory F1 for pairing accuracy.",
     "交叉解析：同一组预测，四种连接算法。拒判用轨迹 F1 换配对准确率。"),
    ("Real plates, held-out split: label-free consistency statistics "
     "(95% bootstrap interval)",
     "真实留出图像：无标签一致性统计量（95% bootstrap 区间）"),
    ("Equal-staple prior as a label-free check on the real plates",
     "把等长先验用作真实图像上的无标签检验"),
    ("Paired per-image difference (CFX-Net minus baseline) and permutation test\n"
     "(*** p<0.001, ** p<0.01, * p<0.05; numbers are raw differences)",
     "逐图配对差值（本文减基线）与置换检验\n（*** p<0.001，** p<0.01，* p<0.05；数字为原始差值）"),
    ("input   |   ground-truth centerlines and crossings   |   CFX-Net trajectories, "
     "junctions and heatmap peaks",
     "输入   |   真值中心线与交叉点   |   CFX-Net 轨迹、关节点与热图峰值"),
    ("Component and data ablation: one change at a time, seed 42",
     "组件与数据消融：每次只改一处，种子 42"),
    ("effect of removing the component (negative = it helps)",
     "去掉该组件的影响（负值 = 该组件有贡献）"),
    ("synthetic validation score (mean over 3 seeds, band = range)",
     "合成验证分数（3 个种子均值，带宽为范围）"),
    ("training loss, seed 42 (40-step moving average)", "训练损失，种子 42（40 步滑动平均）"),
    ("crossing detection, precision-recall over heatmap threshold",
     "交叉点检测：随热图阈值变化的精确率-召回曲线"),
    ("F1 against operating threshold", "F1 随工作阈值的变化"),
    ("(a) length dispersion per plate (lower is better)", "(a) 每张图的长度离散度（越低越好）"),
    ("(b) longest trajectories against the staple prior", "(b) 最长轨迹与等长先验的对照"),
    ("(a) criterion on one canvas", "(a) 单张画布上的判据"),
    ("(b) four of eight sectors must be occupied", "(b) 八个扇区中至少四个被占据"),
    ("(a) Dice by geometry case", "(a) 按几何策略的 Dice"),
    ("(b) Crossing F1 by geometry case (fixed 0.5 heatmap threshold)",
     "(b) 按几何策略的交叉点 F1（固定 0.5 热图阈值）"),
    ("large-scale shading (lower = flatter)", "大尺度明暗（越低越平）"),
    ("grain contrast, textured / smooth half (1.0 = flat)",
     "颗粒对比：纹理半幅 / 光滑半幅（1.0 = 已无差别）"),
    ("zero-complete canvases are the deliberate\n'empty' and 'near miss' negatives",
     "完整交叉为零的画布是刻意设置的\n「empty」与「near miss」负例"),
    ("unlabelled real\nadaptation starts", "无标注真实\n域自适应开始"),
    ("blue = CFX-Net better, red = baseline better", "蓝 = 本文更好，红 = 基线更好"),
    ("marker area is inference time per one 384 px tile", "标记面积为单块 384 px 的推理耗时"),
    ("cumulative fraction of plates", "累积图像比例"),
    ("crossings per training canvas", "每张训练画布的交叉数"),
    ("90th percentile ", "第 90 百分位 "),
    ("calibrated 35 mm staple", "标定的 35 mm 等长"),
    ("interior margin", "内部边距"),
    ("tinted area in (e) is excluded from\nthe false-alarm measurement",
     "(e) 中着色区域不计入误报率测量"),
    ("all crossings", "全部交叉"),
    ("Why one coverage number is not enough: every label-free statistic except coverage "
     "rewards finding less",
     "为什么单一覆盖率数字不够：除覆盖率外，所有无标签统计量都奖励「少检出」"),
    ("(a) evidence coverage from the brightest cores to faint evidence",
     "(a) 从最强脊线核心到微弱证据的覆盖率"),
    ("(b) the operating point: fewer false alarms costs faint fibers",
     "(b) 工作点：更少的误报是用漏掉弱纤维换来的"),
    ("fraction of pixels above the ridge level (0.15% of a plate is fiber)",
     "高于该脊线强度的像素比例（整幅约 0.15% 是纤维）"),
    ("fraction also called fiber", "同时被判为纤维的比例"),
    ("substrate false alarm (lower is better)", "基底误报率（越低越好）"),
    ("coverage of faint evidence at ", "微弱证据覆盖率（阈值 "),
    (" (higher is better)", "，越高越好）"),
    ("crossings present", "存在的交叉"),
    ("shading", "明暗"),
    ("grain", "颗粒"),
    (" to ", " → "),
    ("n = ", "n = "),
    ("verified complete", "已验证完整"),
    ("guarantee holds on", "保证成立于"),
    ("eligible canvases", "张合格画布"),
    ("heatmap threshold", "热图阈值"),
    ("final third, zoomed", "最后三分之一，放大"),
    ("higher is better", "越高越好"),
    ("lower is better", "越低越好"),
    ("pairing accuracy", "配对准确率"),
    ("abstention rate", "拒判率"),
    ("Cost against ", "代价对比："),
    ("(c) ", "(c) "),
    ("raw detail", "原图细节"),
    ("SNR channel", "SNR 通道"),
    ("flattened", "平坦化"),
    ("substrate region", "基底区域"),
    ("local SNR", "局部 SNR"),
    ("coherence", "相干性"),
    ("raw plate", "原图"),
    ("plates", "张图"),
    ("raw", "原图"),
    ("flip", "翻转"),
    ("deg", "°"),
    ("ms", "ms"),
]

_STATE = dict(lang="en", labels=LABELS_EN, text=TEXT_EN)


def tr(value):
    """Translate an already-formatted label when the Chinese figures are being built."""
    if _STATE["lang"] != "zh" or not isinstance(value, str):
        return value
    for english, chinese in TRANSLATIONS:
        if english in value:
            value = value.replace(english, chinese)
    return value


_PATCHED = False


def _install_translation():
    """Route every title, axis label and legend entry through `tr` once.

    Wrapping the matplotlib call sites here, rather than sprinkling `tr(...)` through
    forty figure functions, means a figure added later is translated automatically and
    an English build is bit-identical to before (tr is the identity when lang != zh).
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure

    def wrap_first(cls, name):
        original = getattr(cls, name)

        def patched(self, *args, _original=original, **kwargs):
            if args:
                args = (tr(args[0]),) + args[1:]
            if "label" in kwargs:
                kwargs["label"] = tr(kwargs["label"])
            return _original(self, *args, **kwargs)

        setattr(cls, name, patched)

    for name in ("set_title", "set_xlabel", "set_ylabel", "annotate"):
        wrap_first(Axes, name)
    wrap_first(Figure, "suptitle")
    wrap_first(Colorbar, "set_label")

    def wrap_label_kwarg(cls, name):
        original = getattr(cls, name)

        def patched(self, *args, _original=original, **kwargs):
            if "label" in kwargs:
                kwargs["label"] = tr(kwargs["label"])
            return _original(self, *args, **kwargs)

        setattr(cls, name, patched)

    for name in ("plot", "scatter", "hist", "errorbar", "fill"):
        wrap_label_kwarg(Axes, name)

    def wrap_bar(name):
        original = getattr(Axes, name)

        def patched(self, *args, _original=original, **kwargs):
            # A categorical bar chart passes its tick labels as the first argument.
            # `bar` and `barh` name their other positionals differently, so nothing but
            # the first is touched.
            if args and isinstance(args[0], (list, tuple)) and args[0] \
                    and isinstance(args[0][0], str):
                args = ([tr(v) for v in args[0]],) + args[1:]
            if "label" in kwargs:
                kwargs["label"] = tr(kwargs["label"])
            return _original(self, *args, **kwargs)

        setattr(Axes, name, patched)

    for name in ("bar", "barh"):
        wrap_bar(name)

    original_text = Axes.text

    def patched_text(self, x, y, s, *args, **kwargs):
        return original_text(self, x, y, tr(s), *args, **kwargs)

    Axes.text = patched_text

    for name in ("set_xticklabels", "set_yticklabels"):
        original = getattr(Axes, name)

        def patched(self, labels, *args, _original=original, **kwargs):
            return _original(self, [tr(v) for v in labels], *args, **kwargs)

        setattr(Axes, name, patched)


def set_language(lang):
    _STATE["lang"] = lang
    _STATE["labels"] = LABELS_ZH if lang == "zh" else LABELS_EN
    _STATE["text"] = TEXT_ZH if lang == "zh" else TEXT_EN
    _install_translation()
    if lang == "zh":
        for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
            if Path(candidate).exists():
                font_manager.fontManager.addfont(candidate)
                plt.rcParams["font.family"] = font_manager.FontProperties(
                    fname=candidate).get_name()
                break
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def label(key):
    return _STATE["labels"].get(key, key)


def text(key):
    return _STATE["text"].get(key, key)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 200,
        "lines.linewidth": 2.0, "savefig.bbox": "tight",
    })


def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if _STATE["lang"] == "en" else "_zh"
    path = out / f"{name}{suffix}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path}")
    return path


# ------------------------------------------------------------------ result loading
def crossing_threshold(metrics, runs="runs"):
    """The threshold this model picked on the validation split, if one was selected."""
    tag = metrics.get("tag") or ""
    for group in ("main", "ablation"):
        candidate = Path(runs) / group / tag.replace("_plate", "") / "crossing_threshold.json"
        if candidate.exists():
            return float(load_json(candidate)["threshold"])
    return None


def _has_heatmap(metrics):
    """The training-free baseline emits no crossing heatmap at all.

    Reporting 0 for it would suggest it tried and failed; the honest entry is "not
    applicable", which is what a NaN becomes in the tables and figures.
    """
    return metrics.get("architecture") != "classical"


def _crossf1_at_val_threshold(metrics):
    """Test F1 at the threshold this model selected on the validation split.

    A fixed 0.5 cut would reward whichever model happens to be calibrated there, which is
    not what the comparison is about.  The threshold comes from `tune_threshold.py`;
    the fixed cut is used only when that has not been run.
    """
    if not _has_heatmap(metrics):
        return float("nan")
    thr = crossing_threshold(metrics)
    curve = metrics.get("heatmap_pr_curve")
    if thr is None or not curve:
        return metrics["heatmap_points"]["f1"]
    nearest = min(curve, key=lambda d: abs(d["threshold"] - thr))
    return nearest["f1"]


METRIC_SPEC = [
    ("dice", lambda m: m["pixel"]["dice"]["mean"], True),
    ("cldice", lambda m: m["pixel"]["cldice"]["mean"], True),
    ("clf1", lambda m: m["pixel"]["centerline_f1_2px"]["mean"], True),
    ("crossf1", _crossf1_at_val_threshold, True),
    ("crossap", lambda m: m["heatmap_ap"] if _has_heatmap(m) else float("nan"), True),
    ("instf1", lambda m: m["instances"]["f1"], True),
    ("pairacc", lambda m: m["pairing"]["pair_accuracy"], True),
    ("assd", lambda m: m["pixel"]["assd_px"]["mean"], False),
    ("betti", lambda m: m["pixel"]["betti0_error"]["mean"], False),
]
METRIC_KEYS = [k for k, _, _ in METRIC_SPEC]
HIGHER_BETTER = {k: hi for k, _, hi in METRIC_SPEC}


def metric_value(metrics, key):
    for name, fn, _ in METRIC_SPEC:
        if name == key:
            try:
                return float(fn(metrics))
            except (KeyError, TypeError):
                return float("nan")
    raise KeyError(key)


def load_group(root, group):
    out = {}
    base = group_dir(root, group)
    if not base.exists():
        return out
    for d in sorted(base.iterdir()):
        f = d / "metrics.json"
        if f.exists():
            out[d.name] = load_json(f)
    return out


def main_by_arch(results="results"):
    """{arch: [metrics per seed]} from results/synthetic/main/<arch>_s<seed>/."""
    raw = load_group(results, "synth")
    grouped = {}
    for name, metrics in raw.items():
        arch = name.rsplit("_s", 1)[0] if "_s" in name else name
        grouped.setdefault(arch, []).append(metrics)
    return grouped


# --------------------------------------------------------------------- data figures
def fig_background(out, real="data/real", results="results"):
    from .background import normalise
    from .common import read_image
    from .predict import substrate_mask

    gray = read_image(Path(real) / "roi/p001_i01.png", True)
    ch = normalise(gray)
    sub = substrate_mask(gray)
    report = load_json(diagnostic(results, "background_report.json"))

    tinted = np.dstack([gray] * 3).astype(float) / 255.0
    tinted[~sub] = 0.45 * tinted[~sub] + 0.55 * np.array([0.92, 0.41, 0.20])
    fig = plt.figure(figsize=(13.4, 6.4))
    grid = fig.add_gridspec(2, 15, height_ratios=[3, 1.05], hspace=0.32, wspace=1.6)
    panels = [(gray, "(a) raw plate", "gray"),
              (ch[0], "(b) flattened", "gray"),
              (ch[1], "(c) local SNR", "gray"),
              (ch[2], "(d) coherence", "magma"),
              (np.clip(tinted, 0, 1), "(e) substrate region", None)]
    for i, (image, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(grid[0, 3 * i:3 * i + 3])
        if cmap is None:
            ax.imshow(image)
        else:
            lo, hi = np.percentile(image, [1, 99.5])
            ax.imshow(image, cmap=cmap, vmin=lo, vmax=max(hi, lo + 1e-6))
        ax.set_title(title, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)

    ax = fig.add_subplot(grid[1, 0:6])
    names = ["raw", "flattened", "SNR channel"]
    values = [report["illumination_span_raw"], report["illumination_span_flat"],
              report["illumination_span_snr"]]
    bars = ax.barh(names, values, color=PALETTE[OURS], height=0.58,
                   edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, values):
        ax.text(v + 0.035, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center",
                color=INK_2, fontsize=8)
    ax.set_xlim(0, 1.2)
    ax.set_title("large-scale shading (lower = flatter)", color=INK, loc="left", fontsize=9)
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(grid[1, 6:12])
    names = ["raw detail", "SNR channel"]
    values = [report["grain_ratio_raw"], report["grain_ratio_snr"]]
    bars = ax.barh(names, values, color=PALETTE["dscnet"], height=0.58,
                   edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, values):
        ax.text(v + 0.07, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center",
                color=INK_2, fontsize=8)
    ax.axvline(1.0, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.set_xlim(0, 3.0)
    ax.set_title("grain contrast, textured / smooth half (1.0 = flat)",
                 color=INK, loc="left", fontsize=9)
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(grid[1, 12:15])
    ax.axis("off")
    ax.text(0, 1.02, f"n = {report['n']} plates", color=INK, fontsize=9, va="top")
    ax.text(0, 0.72, f"shading {report['illumination_span_raw']:.2f} to "
                     f"{report['illumination_span_snr']:.2f}", color=INK_2, fontsize=8.5,
            va="top")
    ax.text(0, 0.46, f"grain {report['grain_ratio_raw']:.2f} to "
                     f"{report['grain_ratio_snr']:.2f}", color=INK_2, fontsize=8.5, va="top")
    ax.text(0, 0.20, "tinted area in (e) is excluded from\nthe false-alarm measurement",
            color=INK_MUTED, fontsize=8, va="top")

    fig.suptitle("Background normalisation removes the two nuisances that cause false fibers",
                 color=INK, fontsize=11, x=0.02, ha="left")
    return _save(fig, out, "fig01_background")


def fig_synthetic(out, synth="data/synth", n=12):
    from .common import read_image

    root = Path(synth)
    rows = [r for r in load_json(root / "manifest.json") if r["split"] == "train"]
    wanted = ["cross", "acute", "orthogonal", "parallel", "near_miss", "multi"]
    picks = []
    for case in wanted:
        picks += [r for r in rows if r["case"] == case][:2]
    picks = picks[:n]

    fig, axes = plt.subplots(2, len(picks) // 2, figsize=(2.05 * len(picks) // 2, 5.2))
    for ax, r in zip(axes.ravel(), picks):
        image = read_image(root / r["image"], True)
        with np.load(root / r["target"]) as z:
            center = z["center"]
            points = z["points"].reshape(-1, 2)
            complete = z["points_complete"].reshape(-1).astype(bool)
        rgb = np.dstack([image] * 3).astype(float) / 255.0
        rgb[center > 120] = [0.24, 0.90, 0.35]
        ax.imshow(rgb)
        for p, c in zip(points, complete):
            ax.add_patch(Circle(p, 13, fill=False, linewidth=1.4,
                                edgecolor=PALETTE[OURS] if c else "#eda100"))
        ax.set_title(r["case"], color=INK, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Label-first samples: geometry drawn first, image rendered from it "
                 "(blue = verified complete crossing, amber = clipped)",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.2)
    return _save(fig, out, "fig02_synthetic")


def fig_complete_crossing(out, synth="data/synth"):
    from .common import read_image

    root = Path(synth)
    prov = load_json(root / "provenance.json")
    rows = [r for r in load_json(root / "manifest.json") if r["split"] == "train"]
    pick = next(r for r in rows if r["complete_crossings"] > 0 and r["case"] == "cross")
    meta = load_json(root / pick["meta"])
    image = read_image(root / pick["image"], True)
    record = next(c for c in meta["crossings"] if c["complete"])
    cx, cy = record["xy"]
    arm = meta["arm_pixels"]
    margin = meta["margin"]
    size = meta["size"]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    ax = axes[0]
    ax.imshow(image, cmap="gray")
    ax.add_patch(Rectangle((margin, margin), size - 2 * margin, size - 2 * margin,
                           fill=False, edgecolor=PALETTE["dscnet"], linewidth=1.4,
                           linestyle=(0, (5, 3))))
    ax.add_patch(Circle((cx, cy), arm, fill=False, edgecolor=PALETTE[OURS], linewidth=1.6))
    for curve in meta["curves_xy"]:
        c = np.asarray(curve)
        ax.plot(c[:, 0], c[:, 1], color="#1baf7a", linewidth=1.1, alpha=0.9)
    ax.plot([cx], [cy], marker="o", markersize=7, color=PALETTE[OURS])
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_title("(a) criterion on one canvas", color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.text(margin + 6, margin + 18, f"interior margin {margin}px", color=PALETTE["dscnet"],
            fontsize=8)
    ax.annotate(f"arm >= {arm:.0f}px", xy=(cx, cy - arm), xytext=(0, -14),
                textcoords="offset points", ha="center", color=PALETTE[OURS], fontsize=8)

    ax = axes[1]
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    for k in range(8):
        a0, a1 = k * np.pi / 4, (k + 1) * np.pi / 4
        t = np.linspace(a0, a1, 24)
        ax.fill(np.concatenate([[0], np.cos(t)]), np.concatenate([[0], np.sin(t)]),
                color=GRID if k % 2 else "#f2f1ed", linewidth=0)
    for angle in (0.35, 0.35 + np.pi, 2.1, 2.1 + np.pi):
        ax.plot([0, np.cos(angle)], [0, np.sin(angle)], color="#1baf7a", linewidth=2.4)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=PALETTE[OURS], linewidth=1.6))
    ax.plot([0], [0], marker="o", markersize=8, color=PALETTE[OURS])
    ax.set_title("(b) four of eight sectors must be occupied", color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

    ax = axes[2]
    cap = 8
    complete = np.clip([r["complete_crossings"] for r in rows], 0, cap)
    total = np.clip([r["crossings"] for r in rows], 0, cap)
    centres = np.arange(cap + 1)
    tc = np.bincount(total, minlength=cap + 1)[:cap + 1]
    cc = np.bincount(complete, minlength=cap + 1)[:cap + 1]
    ax.bar(centres - 0.21, tc, width=0.40, color=INK_MUTED, label="crossings present",
           edgecolor=SURFACE, linewidth=1.6)
    ax.bar(centres + 0.21, cc, width=0.40, color=PALETTE[OURS], label="verified complete",
           edgecolor=SURFACE, linewidth=1.6)
    ax.set_xticks(centres)
    ax.set_xticklabels([str(v) for v in centres[:-1]] + [f"{cap}+"])
    ax.set_xlabel("crossings per training canvas")
    ax.set_ylabel(text("count"))
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    guarantee = prov["complete_crossing_guarantee"]
    ax.set_title(f"(c) guarantee holds on {guarantee['satisfied']}/{guarantee['of']} "
                 "eligible canvases", color=INK, fontsize=9.5)
    ax.annotate("zero-complete canvases are the deliberate\n'empty' and 'near miss' "
                "negatives", xy=(0.36, 0.72), xycoords="axes fraction", fontsize=8,
                color=INK_MUTED)
    fig.tight_layout()
    return _save(fig, out, "fig03_complete_crossing")


def fig_real_augmentation(out, aug="data/real_aug", n=12):
    from .common import read_image

    root = Path(aug)
    manifest = load_json(root / "manifest.json")
    items = manifest["items"]
    picks = [items[i] for i in np.linspace(0, len(items) - 1, n).astype(int)]
    cols = n // 2
    fig, axes = plt.subplots(2, cols, figsize=(1.85 * cols, 4.9))
    for ax, item in zip(axes.ravel(), picks):
        image = read_image(root / item["file"], True)
        ax.imshow(image, cmap="gray")
        jx, jy = item["junction_xy"]
        rot, flip = item["rot"], item["flip"]
        size = image.shape[0]
        x, y = jx, jy
        for _ in range(rot // 90):
            x, y = y, size - 1 - x
        if flip:
            x = size - 1 - x
        ax.add_patch(Circle((x, y), 34, fill=False, edgecolor=PALETTE[OURS], linewidth=1.5))
        ax.set_title(f"{item['plate']}  {rot}deg{' flip' if flip else ''}",
                     color=INK_2, fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Real-plate augmentation is crossing-centred: every crop carries a whole "
                 "crossing (circle = detected junction)", color=INK, fontsize=11,
                 x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.2)
    return _save(fig, out, "fig04_real_augmentation")


# ------------------------------------------------------------------ result figures
def fig_training(out, runs="runs/main"):
    base = Path(runs)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    zoom = []
    for arch in METHOD_ORDER:
        if arch == "classical":
            continue
        curves, losses = [], []
        for d in sorted(base.glob(f"{arch}_s*")):
            log = d / "training_log.json"
            if not log.exists():
                continue
            rows = load_json(log)
            pts = [(r["step"], r["val_score"]) for r in rows if "val_score" in r]
            if pts:
                curves.append(np.array(pts))
            losses.append(np.array([(r["step"], r["loss"]) for r in rows]))
        if not curves:
            continue
        steps = curves[0][:, 0]
        values = np.stack([c[:, 1] for c in curves if len(c) == len(steps)])
        mean = values.mean(0)
        axes[0].plot(steps, mean, color=PALETTE[arch], label=label(arch))
        if len(values) > 1:
            axes[0].fill_between(steps, values.min(0), values.max(0),
                                 color=PALETTE[arch], alpha=0.13, linewidth=0)
        zoom.append((arch, steps, mean, values))
        if losses:
            smooth = np.convolve(losses[0][:, 1], np.ones(40) / 40, mode="valid")
            axes[1].plot(losses[0][40 - 1:, 0], smooth, color=PALETTE[arch],
                         label=label(arch))
    warm = None
    for d in sorted(base.glob("*_s*")):
        if (d / "config.json").exists():
            warm = load_json(d / "config.json")["warmup"]
            break
    for ax in axes:
        if warm:
            ax.axvline(warm, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1)
    lo, hi = axes[0].get_ylim()
    axes[0].annotate("unlabelled real\nadaptation starts",
                     xy=(warm or 0, lo + 0.18 * (hi - lo)), xytext=(7, 0),
                     textcoords="offset points", color=INK_MUTED, fontsize=7.5,
                     va="center")
    axes[0].set_xlabel(text("step"))
    axes[0].set_ylabel(text("val"))
    axes[0].set_title("synthetic validation score (mean over 3 seeds, band = range)",
                      color=INK, loc="left")
    axes[0].legend(ncol=2, loc="lower left", fontsize=7.5)
    # The curves converge to within a few thousandths, so an inset carries the detail
    # that the full-range axis cannot show.
    if zoom:
        inset = axes[0].inset_axes([0.44, 0.30, 0.53, 0.42])
        tail = max(1, len(zoom[0][1]) // 3)
        for arch, steps_a, mean_a, values_a in zoom:
            inset.plot(steps_a[-tail:], mean_a[-tail:], color=PALETTE[arch], linewidth=1.6)
            if len(values_a) > 1:
                inset.fill_between(steps_a[-tail:], values_a.min(0)[-tail:],
                                   values_a.max(0)[-tail:], color=PALETTE[arch],
                                   alpha=0.13, linewidth=0)
        inset.tick_params(labelsize=6.5)
        inset.set_title("final third, zoomed", fontsize=7, color=INK_MUTED, loc="left")
        inset.grid(color=GRID, linewidth=0.6)
        for s in inset.spines.values():
            s.set_color(GRID)
    axes[1].set_xlabel(text("step"))
    axes[1].set_ylabel(text("loss"))
    axes[1].set_yscale("log")
    axes[1].set_title("training loss, seed 42 (40-step moving average)", color=INK,
                      loc="left")
    axes[1].legend(ncol=2, loc="upper right", fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "fig05_training")


def fig_main_comparison(out, results="results", keys=("dice", "cldice", "clf1", "crossf1",
                                                "crossap", "instf1", "pairacc", "assd")):
    grouped = main_by_arch(results)
    classical = load_group(results, "synth").get("classical")
    # Dot plot, not bars.  These metrics cluster near the top of their range, so a bar
    # chart would need a truncated axis - and a truncated bar exaggerates a difference of
    # a few thousandths.  A dot encodes position only, so a non-zero axis is honest.
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.4))
    for ax, key in zip(axes.ravel(), keys):
        names, means, errs, colours, best = [], [], [], [], None
        for arch in METHOD_ORDER:
            if arch == "classical":
                if classical is None:
                    continue
                values = [metric_value(classical, key)]
            else:
                if arch not in grouped:
                    continue
                values = [metric_value(m, key) for m in grouped[arch]]
            values = [v for v in values if np.isfinite(v)]
            if not values:
                continue
            names.append(label(arch))
            means.append(float(np.mean(values)))
            errs.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            colours.append(PALETTE[arch])
        if not names:
            continue
        y = np.arange(len(names))[::-1]
        best = (max(means) if HIGHER_BETTER[key] else min(means))
        ax.axvline(best, color=INK_MUTED, linestyle=(0, (3, 3)), linewidth=1)
        ax.errorbar(means, y, xerr=errs, fmt="none", ecolor=INK_MUTED, elinewidth=1.2,
                    capsize=3, zorder=2)
        ax.scatter(means, y, s=78, c=colours, edgecolor=SURFACE, linewidth=2, zorder=3)
        for yi, m, e in zip(y, means, errs):
            # The dashed best-method line runs through the winner's own label, so the
            # text gets an opaque backing to stay legible.
            ax.text(m, yi + 0.26, f"{m:.4f}", ha="center", va="bottom", color=INK_2,
                    fontsize=7.4, zorder=4,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.8)
        ax.set_ylim(-0.8, len(names) - 0.2)
        ax.set_title(text(key), color=INK, loc="left")
        ax.grid(axis="y", visible=False)
        lo = min(m - e for m, e in zip(means, errs))
        hi = max(m + e for m, e in zip(means, errs))
        pad = (hi - lo) * 0.28 + 1e-6
        ax.set_xlim(lo - pad, hi + pad)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Held-out synthetic test set: mean over seeds, whiskers are one standard "
                 "deviation, dashed line marks the best method",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, out, "fig06_main_comparison")


def fig_pr_curves(out, results="results"):
    grouped = main_by_arch(results)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for arch in METHOD_ORDER:
        if arch not in grouped:
            continue
        curves = [m.get("heatmap_pr_curve") for m in grouped[arch]]
        curves = [c for c in curves if c]
        if not curves:
            continue
        recall = np.mean([[p["recall"] for p in c] for c in curves], axis=0)
        precision = np.mean([[p["precision"] for p in c] for c in curves], axis=0)
        f1 = np.mean([[p["f1"] for p in c] for c in curves], axis=0)
        thr = [p["threshold"] for p in curves[0]]
        ap = np.mean([m["heatmap_ap"] for m in grouped[arch]])
        axes[0].plot(recall, precision, color=PALETTE[arch],
                     label=f"{label(arch)}  AP={ap:.3f}", marker="o", markersize=3.5,
                     markerfacecolor=SURFACE, markeredgewidth=1.2)
        axes[1].plot(thr, f1, color=PALETTE[arch], label=label(arch))
    axes[0].set_xlabel(text("recall"))
    axes[0].set_ylabel(text("precision"))
    axes[0].set_title("crossing detection, precision-recall over heatmap threshold",
                      color=INK, loc="left")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].set_xlabel("heatmap threshold")
    axes[1].set_ylabel("F1")
    axes[1].set_title("F1 against operating threshold", color=INK, loc="left")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, out, "fig07_pr_curves")


def fig_radar(out, results="results"):
    grouped = main_by_arch(results)
    keys = ["dice", "cldice", "clf1", "crossf1", "crossap", "instf1", "pairacc"]
    # The training-free baseline is far below every learned method; including it would
    # compress the learned methods onto the rim and hide the comparison this chart is
    # for.  It stays in the main table instead.
    series = {arch: [float(np.mean([metric_value(m, k) for m in grouped[arch]]))
                     for k in keys]
              for arch in METHOD_ORDER if arch != "classical" and arch in grouped}
    if len(series) < 2:
        return None
    matrix = np.array([series[a] for a in series])
    lo, hi = matrix.min(0), matrix.max(0)
    normed = (matrix - lo) / np.maximum(hi - lo, 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False)
    closed = np.concatenate([angles, angles[:1]])
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, polar=True)
    for (arch, _), row in zip(series.items(), normed):
        values = np.concatenate([row, row[:1]])
        ax.plot(closed, values, color=PALETTE[arch], label=label(arch),
                linewidth=2.4 if arch == OURS else 1.6,
                alpha=1.0 if arch == OURS else 0.85)
        if arch == OURS:
            ax.fill(closed, values, color=PALETTE[arch], alpha=0.12, linewidth=0)
    ax.set_xticks(angles)
    ax.set_xticklabels([text(k) for k in keys], fontsize=8.5, color=INK_2)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels([], fontsize=8)
    ax.set_rlabel_position(97)
    ax.text(np.deg2rad(97), 1.0, "best of the six", fontsize=7.5, color=INK_MUTED,
            ha="left", va="bottom")
    ax.text(np.deg2rad(97), 0.0, "worst of the six", fontsize=7.5, color=INK_MUTED,
            ha="left", va="bottom")
    ax.set_ylim(-0.12, 1.12)
    ax.grid(color=GRID)
    ax.spines["polar"].set_color(GRID)
    ax.set_title("Per-metric ranking among the six learned backbones, rescaled per axis\n"
                 "(the training-free baseline is far below this range and is omitted)",
                 color=INK, pad=24, fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=8)
    return _save(fig, out, "fig08_radar")


def fig_by_case(out, results="results"):
    grouped = main_by_arch(results)
    classical = load_group(results, "synth").get("classical")
    entries = {}
    for arch in METHOD_ORDER:
        if arch == "classical" and classical is not None:
            entries[arch] = classical.get("by_case", {})
        elif arch in grouped:
            cases = grouped[arch][0].get("by_case", {})
            merged = {}
            for c in cases:
                merged[c] = dict(
                    dice=float(np.mean([m["by_case"][c]["dice"] for m in grouped[arch]])),
                    crossing_f1=float(np.mean([m["by_case"][c]["crossing_f1"]
                                               for m in grouped[arch]])))
            entries[arch] = merged
    cases = sorted({c for v in entries.values() for c in v})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    titles = ("(a) Dice by geometry case",
              "(b) Crossing F1 by geometry case (fixed 0.5 heatmap threshold)")
    for ax, metric, title in zip(axes, ("dice", "crossing_f1"), titles):
        matrix = np.array([[entries[a].get(c, {}).get(metric, np.nan) for c in cases]
                           for a in entries])
        im = ax.imshow(matrix, cmap="Blues", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels(cases, rotation=28, ha="right", fontsize=8)
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([label(a) for a in entries], fontsize=8)
        ax.grid(False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    mid = (np.nanmin(matrix) + np.nanmax(matrix)) / 2
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=7.2,
                            color="#ffffff" if matrix[i, j] > mid else INK)
        ax.set_title(title, color=INK, loc="left", fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("The 'empty' column is the deliberate negative case: no fibers, so a "
                 "correct detector scores 1.0 trivially", color=INK_MUTED, fontsize=8.5,
                 x=0.01, ha="left", y=0.02)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _save(fig, out, "fig09_by_case")


def diverging_cmap():
    """Two hues with a neutral midpoint, as the project's colour standard specifies."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "cfx_diverging", ["#2a78d6", "#9ec5f4", "#f0efec", "#f3a9a8", "#e34948"])


def fig_ablation(out, results="results"):
    ablations = load_group(results, "synth_ablation")
    grouped = main_by_arch(results)
    if not ablations or OURS not in grouped:
        return None
    full = grouped[OURS][0]
    keys = ["dice", "cldice", "clf1", "crossf1", "crossap", "instf1", "pairacc"]
    names = list(ablations)
    # Sign convention: negative always means "removing this made the model worse", i.e.
    # the component earns its place.  Lower-is-better metrics are flipped accordingly.
    effect = np.array([[(metric_value(ablations[n], k) - metric_value(full, k))
                        * (1 if HIGHER_BETTER[k] else -1) for k in keys] for n in names])
    order = np.argsort(np.nanmean(effect, axis=1))
    names = [names[i] for i in order]
    effect = effect[order]

    span = float(np.nanmax(np.abs(effect))) or 1.0
    fig, ax = plt.subplots(figsize=(1.35 * len(keys) + 4.6, 0.45 * len(names) + 2.6))
    im = ax.imshow(effect, cmap=diverging_cmap(), vmin=-span, vmax=span, aspect="auto")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([text(k) for k in keys], rotation=26, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=8.5)
    ax.grid(False)
    for i in range(effect.shape[0]):
        for j in range(effect.shape[1]):
            if np.isfinite(effect[i, j]):
                ax.text(j, i, f"{effect[i, j]:+.3f}", ha="center", va="center", fontsize=7.2,
                        color="#ffffff" if abs(effect[i, j]) > span * 0.62 else INK)
    bar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("effect of removing the component (negative = it helps)", fontsize=8)
    ax.set_title("Component and data ablation: one change at a time, seed 42",
                 color=INK, loc="left")
    fig.tight_layout()
    return _save(fig, out, "fig10_ablation")


def fig_post(out, results="results"):
    post = load_group(results, "synth_post")
    if not post:
        return None
    names = ["greedy", "angle_joint", "cfx", "cfx_conservative"]
    order = [m for m in names if m in post]
    labels = {"greedy": "angle greedy", "angle_joint": "joint matching",
              "cfx": "embedding-guided\n(ours)", "cfx_conservative": "ours\n+ abstention"}
    colours = {"greedy": PALETTE["unet"], "angle_joint": PALETTE["dscnet"],
               "cfx": PALETTE[OURS], "cfx_conservative": PALETTE["resunet"]}
    panels = [("instf1", lambda m: metric_value(post[m], "instf1"), text("instf1")),
              ("crossf1", lambda m: metric_value(post[m], "crossf1"), text("crossf1")),
              ("pair", lambda m: post[m].get("pairing", {}).get("pair_accuracy", np.nan),
               "pairing accuracy"),
              ("abstain", lambda m: post[m].get("pairing", {}).get("abstention_rate", 0.0),
               "abstention rate")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.1))
    for ax, (key, getter, title) in zip(axes, panels):
        values = [getter(m) for m in order]
        bars = ax.bar(range(len(order)), values, width=0.62,
                      color=[colours[m] for m in order], edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
                    color=INK_2, fontsize=8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([labels[m] for m in order], rotation=20, ha="right", fontsize=7.6)
        ax.set_title(title, color=INK, loc="left", fontsize=9.5)
        ax.grid(axis="x", visible=False)
        finite = [v for v in values if np.isfinite(v)]
        if finite:
            lo, hi = min(finite), max(finite)
            span = max(hi - lo, 1e-6)
            ax.set_ylim(max(0, lo - span * 0.6), hi + span * 0.4)
    fig.suptitle("Crossing resolution: identical predictions, four linkers. "
                 "Abstention trades trajectory F1 for pairing accuracy.",
                 color=INK, fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return _save(fig, out, "fig11_post")


def fig_real_labelfree(out, results="results"):
    real = load_group(results, "real")
    if not real:
        return None
    specs = [("sfar", "sfar", False), ("coverage", "coverage", True),
             ("equivariance_mse", "eqe", False), ("crossing_repeatability", "crep", True),
             ("length_cv", "lcv", False), ("fragmentation", "frag", False)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for ax, (field, key, higher) in zip(axes.ravel(), specs):
        names, means, errs, colours = [], [], [], []
        for arch in METHOD_ORDER:
            if arch not in real:
                continue
            stat = real[arch].get(field)
            if not stat or not np.isfinite(stat.get("mean", np.nan)):
                continue
            names.append(label(arch))
            means.append(stat["mean"])
            errs.append([max(stat["mean"] - stat.get("low", stat["mean"]), 0),
                         max(stat.get("high", stat["mean"]) - stat["mean"], 0)])
            colours.append(PALETTE[arch])
        if not names:
            continue
        x = np.arange(len(names))
        errs = np.array(errs).T
        ax.bar(x, means, yerr=errs, width=0.66, color=colours,
               edgecolor=SURFACE, linewidth=2,
               error_kw=dict(ecolor=INK_MUTED, elinewidth=1.1, capsize=3))
        top = max(m + e for m, e in zip(means, errs[1])) if names else 1.0
        for xi, m, e in zip(x, means, errs[1]):
            ax.text(xi, m + e + top * 0.03, f"{m:.4g}", ha="center", va="bottom",
                    color=INK_2, fontsize=7.4)
        ax.set_ylim(0, top * 1.25)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=28, ha="right", fontsize=7.6)
        arrow = "higher is better" if higher else "lower is better"
        ax.set_title(f"{text(key)}  ({arrow})", color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Real plates, held-out split: label-free consistency statistics "
                 "(95% bootstrap interval)", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, out, "fig12_real_labelfree")


def fig_length(out, results="results", calibration="data/calibration.json"):
    real = load_group(results, "real")
    if not real:
        return None
    staple = load_json(calibration)["staple_pixels"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    for arch in METHOD_ORDER:
        if arch not in real:
            continue
        plates = real[arch].get("per_plate", [])
        cvs = [p["length"]["cv"] for p in plates if np.isfinite(p["length"].get("cv", np.nan))]
        p90 = [p["length"].get("p90", np.nan) for p in plates]
        p90 = [v for v in p90 if np.isfinite(v)]
        if cvs:
            axes[0].plot(sorted(cvs), np.linspace(0, 1, len(cvs)), color=PALETTE[arch],
                         label=label(arch))
        if p90:
            axes[1].plot(sorted(p90), np.linspace(0, 1, len(p90)), color=PALETTE[arch],
                         label=label(arch))
    axes[0].set_xlabel(text("lcv"))
    axes[0].set_ylabel("cumulative fraction of plates")
    axes[0].set_title("(a) length dispersion per plate (lower is better)",
                      color=INK, loc="left", fontsize=9.5)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].axvline(staple, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    axes[1].annotate(f"calibrated 35 mm staple = {staple:.0f} px",
                     xy=(staple, 0.42), xytext=(-8, 0), textcoords="offset points",
                     color=INK_MUTED, fontsize=8, va="center", ha="right", rotation=90)
    axes[1].set_xlabel(f"90th percentile {text('lengthpx')}")
    axes[1].set_ylabel("cumulative fraction of plates")
    axes[1].set_title("(b) longest trajectories against the staple prior",
                      color=INK, loc="left", fontsize=9.5)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Equal-staple prior as a label-free check on the real plates",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, out, "fig13_length")


def fig_efficiency(out, results="results", runs="runs/main", key="crossap"):
    grouped = main_by_arch(results)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    points = []
    for arch in METHOD_ORDER:
        if arch not in grouped:
            continue
        env = None
        for d in sorted(Path(runs).glob(f"{arch}_s*")):
            if (d / "environment.json").exists():
                env = load_json(d / "environment.json")
                break
        if env is None:
            continue
        values = [metric_value(m, key) for m in grouped[arch]]
        points.append((arch, env["parameters"] / 1e6, float(np.mean(values)),
                       float(np.mean([m["inference_ms_mean"] for m in grouped[arch]]))))
    if not points:
        return None
    # Every point is direct-labelled: this is a scatter, where the palette's all-pairs
    # guarantee does not hold, so identity must not rest on colour alone.
    order = sorted(range(len(points)), key=lambda i: points[i][2])
    offsets = {}
    for rank, i in enumerate(order):
        offsets[i] = (11, 6) if rank % 2 == 0 else (11, -16)
    for i, (arch, params, value, ms) in enumerate(points):
        ax.scatter([params], [value], s=70 + ms * 2.0, color=PALETTE[arch],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{label(arch)}\n{ms:.0f} ms", (params, value),
                    textcoords="offset points", xytext=offsets[i], fontsize=8,
                    color=INK_2, zorder=4)
    xs = [p[1] for p in points]
    ax.set_xlim(0, max(xs) * 1.35)
    ax.set_xlabel(text("params"))
    ax.set_ylabel(text(key))
    ax.set_title(f"Cost against {text(key)}; marker area is inference time per one "
                 "384 px tile", color=INK, loc="left", fontsize=9.5)
    fig.tight_layout()
    return _save(fig, out, "fig14_efficiency")


def fig_significance(out, results="results"):
    tests = Path(results) / "tables" / "significance.json"
    if not tests.exists():
        return None
    data = load_json(tests)
    keys = list(data["metrics"])
    others = [a for a in METHOD_ORDER if a != OURS and a in data["metrics"][keys[0]]]
    matrix = np.array([[data["metrics"][k][a]["mean_difference"] for a in others]
                       for k in keys])
    pvals = np.array([[data["metrics"][k][a]["permutation_p"] for a in others]
                      for k in keys])
    # Colour by benefit, not by raw sign: on a lower-is-better metric a negative
    # difference is good news.  The printed number stays the raw difference.
    benefit = matrix * np.array([[1.0 if HIGHER_BETTER.get(k, True) else -1.0]
                                 for k in keys])
    span = float(np.nanmax(np.abs(benefit))) or 1.0
    fig, ax = plt.subplots(figsize=(1.35 * len(others) + 4.2, 0.62 * len(keys) + 2.4))
    im = ax.imshow(benefit, cmap=diverging_cmap().reversed(), vmin=-span, vmax=span)
    ax.set_xticks(range(len(others)))
    ax.set_xticklabels([label(a) for a in others], rotation=26, ha="right", fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([text(k) for k in keys], fontsize=8.5)
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            star = "***" if pvals[i, j] < 0.001 else "**" if pvals[i, j] < 0.01 else \
                "*" if pvals[i, j] < 0.05 else "n.s."
            ax.text(j, i, f"{matrix[i, j]:+.3f}\n{star}", ha="center", va="center",
                    fontsize=7.4,
                    color="#ffffff" if abs(matrix[i, j]) > span * 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                 label="CFX-Net minus baseline, per image")
    ax.set_title("Paired per-image difference and permutation test "
                 "(*** p<0.001, ** p<0.01, * p<0.05)", color=INK, loc="left", fontsize=9.5)
    fig.tight_layout()
    return _save(fig, out, "fig15_significance")


def fig_qualitative(out, results="results", n=4):
    from .common import read_image

    fig = None
    for arch in (OURS,):
        folder = group_dir(results, "synth") / f"{arch}_s42"
        samples = sorted(folder.glob("sample_*.jpg"))[:n]
        if not samples:
            return None
        fig, axes = plt.subplots(len(samples), 1, figsize=(11.5, 3.0 * len(samples)))
        axes = np.atleast_1d(axes)
        for ax, path in zip(axes, samples):
            ax.imshow(read_image(path)[:, :, ::-1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0].set_title("input   |   ground-truth centerlines and crossings   |   "
                          "CFX-Net trajectories, junctions and heatmap peaks",
                          color=INK, loc="left")
    fig.tight_layout()
    return _save(fig, out, "fig16_qualitative")


def fig_real_qualitative(out, results="results", plate=None, archs=("classical", "unet", OURS)):
    from .common import read_image

    tiles = []
    for arch in archs:
        folder = group_dir(results, "real") / arch
        files = sorted(folder.glob("*_overlay.jpg"))
        if not files:
            continue
        pick = next((f for f in files if plate and plate in f.name), files[0])
        tiles.append((arch, read_image(pick)[:, :, ::-1], pick.stem.replace("_overlay", "")))
    if not tiles:
        return None
    fig, axes = plt.subplots(1, len(tiles), figsize=(4.4 * len(tiles), 5.4))
    axes = np.atleast_1d(axes)
    for ax, (arch, image, name) in zip(axes, tiles):
        ax.imshow(image)
        ax.set_title(f"{label(arch)}", color=PALETTE[arch])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"Held-out real plate {tiles[0][2]}: reconstructed trajectories "
                 "(no ground truth exists for this image)", color=INK, fontsize=11,
                 x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _save(fig, out, "fig17_real_qualitative")


def fig_coverage_tradeoff(out, results="results"):
    """What the single-threshold coverage number hides: the sensitivity trade-off."""
    path = diagnostic(results, "coverage_curve.json")
    real = load_group(results, "real")
    if not path.exists():
        return None
    data = load_json(path)
    fractions = data["fractions"]
    faint = str(fractions[-1])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for name, curve in data["methods"].items():
        arch = name.split(":")[-1]
        colour = PALETTE.get(arch, INK_MUTED)
        style_kw = dict(linestyle=(0, (4, 2))) if name.startswith("ablation:") else {}
        axes[0].plot(fractions, [curve[str(f)]["mean"] for f in fractions],
                     color=colour, marker="o", markersize=4, markerfacecolor=SURFACE,
                     markeredgewidth=1.2,
                     label=label(arch) if arch in PALETTE else arch, **style_kw)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("fraction of pixels above the ridge level (0.15% of a plate is fiber)")
    axes[0].set_ylabel("fraction also called fiber")
    axes[0].set_title("(a) evidence coverage from the brightest cores to faint evidence",
                      color=INK, loc="left", fontsize=9.5)
    axes[0].legend(fontsize=7.5, ncol=2)

    points = [(name.split(":")[-1], real[name.split(":")[-1]]["sfar"]["mean"],
               curve[faint]["mean"])
              for name, curve in data["methods"].items()
              if not name.startswith("ablation:") and name.split(":")[-1] in real]
    order = sorted(range(len(points)), key=lambda i: points[i][1])
    for rank, i in enumerate(order):
        arch, x, y = points[i]
        axes[1].scatter([x], [y], s=90, color=PALETTE.get(arch, INK_MUTED),
                        edgecolor=SURFACE, linewidth=2, zorder=3)
        axes[1].annotate(label(arch), (x, y), textcoords="offset points",
                         xytext=(11, 5) if rank % 2 == 0 else (11, -13),
                         fontsize=8, color=INK_2)
    axes[1].margins(x=0.22, y=0.14)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("substrate false alarm (lower is better)")
    axes[1].set_ylabel(f"coverage of faint evidence at {faint} (higher is better)")
    axes[1].set_title("(b) the operating point: fewer false alarms costs faint fibers",
                      color=INK, loc="left", fontsize=9.5)
    fig.suptitle("Why one coverage number is not enough: every label-free statistic "
                 "except coverage rewards finding less",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, out, "fig18_coverage_tradeoff")


def fig_pipeline(out):
    """Schematic of the label-first pipeline; drawn, not photographed."""
    boxes = [
        (0.02, "sample geometry\n(35 mm worm-like chains)", PALETTE[OURS]),
        (0.20, "exact labels\nfor free", PALETTE[OURS]),
        (0.38, "render image\nfrom label", PALETTE["dscnet"]),
        (0.56, "background\nnormalisation", PALETTE["resunet"]),
        (0.74, "CFX-Net\nmulti-task", PALETTE["unetpp"]),
    ]
    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for x, caption, colour in boxes:
        ax.add_patch(Rectangle((x, 0.52), 0.155, 0.26, facecolor=SURFACE,
                               edgecolor=colour, linewidth=2.2))
        ax.text(x + 0.078, 0.65, caption, ha="center", va="center", fontsize=9, color=INK)
        if x < 0.74:
            ax.add_patch(FancyArrowPatch((x + 0.157, 0.65), (x + 0.178, 0.65),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color=INK_MUTED, linewidth=1.4))
    ax.add_patch(Rectangle((0.56, 0.12), 0.155, 0.24, facecolor=SURFACE,
                           edgecolor=INK_MUTED, linewidth=2.0, linestyle=(0, (4, 3))))
    ax.text(0.638, 0.24, "real plates\nNO labels", ha="center", va="center", fontsize=9,
            color=INK_2)
    ax.add_patch(FancyArrowPatch((0.715, 0.24), (0.80, 0.50), arrowstyle="-|>",
                                 mutation_scale=13, color=INK_MUTED, linewidth=1.4,
                                 connectionstyle="arc3,rad=0.2"))
    ax.text(0.745, 0.36, "teacher-student\nconsistency", fontsize=8, color=INK_MUTED)
    ax.add_patch(Rectangle((0.92, 0.52), 0.06, 0.26, facecolor=SURFACE,
                           edgecolor=PALETTE["attunet"], linewidth=2.2))
    ax.text(0.95, 0.65, "linking\n+\ntracks", ha="center", va="center", fontsize=8.5,
            color=INK)
    ax.add_patch(FancyArrowPatch((0.897, 0.65), (0.918, 0.65), arrowstyle="-|>",
                                 mutation_scale=13, color=INK_MUTED, linewidth=1.4))
    ax.text(0.02, 0.92, "Label-first pipeline: annotation cost is zero because the label "
                        "is drawn before the image exists", fontsize=11, color=INK)
    ax.text(0.02, 0.05, "Human annotation is never used, at any stage.", fontsize=9,
            color=INK_MUTED)
    return _save(fig, out, "fig00_pipeline")


ALL = {
    "pipeline": fig_pipeline,
    "background": fig_background,
    "synthetic": fig_synthetic,
    "complete_crossing": fig_complete_crossing,
    "real_augmentation": fig_real_augmentation,
    "training": fig_training,
    "main_comparison": fig_main_comparison,
    "pr_curves": fig_pr_curves,
    "radar": fig_radar,
    "by_case": fig_by_case,
    "ablation": fig_ablation,
    "post": fig_post,
    "real_labelfree": fig_real_labelfree,
    "length": fig_length,
    "efficiency": fig_efficiency,
    "significance": fig_significance,
    "qualitative": fig_qualitative,
    "real_qualitative": fig_real_qualitative,
    "coverage_tradeoff": fig_coverage_tradeoff,
}

# Counting figures live in their own module; registered here so --all builds them too.
# (Skipped in the __main__ copy, which hands over to the package module below.)
if __name__ != "__main__":
    from .figures_counting import COUNTING_FIGURES  # noqa: E402

    ALL.update(COUNTING_FIGURES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/figures")
    p.add_argument("--only", nargs="*")
    p.add_argument("--lang", default="en", choices=["en", "zh"])
    p.add_argument("--results", default="results")
    p.add_argument("--runs", default="runs/main")
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    set_language(a.lang)
    style()
    names = a.only or list(ALL)
    made = []
    import inspect

    for name in names:
        fn = ALL.get(name)
        if fn is None:
            print(f"  [warn] no figure named {name}")
            continue
        try:
            accepted = inspect.signature(fn).parameters
            extra = {}
            if "results" in accepted:
                extra["results"] = a.results
            if "runs" in accepted:
                extra["runs"] = a.runs
            path = fn(a.out, **extra)
            if path:
                made.append(str(path))
        except FileNotFoundError as exc:
            print(f"  [skip] {name}: missing input ({exc})")
        except Exception as exc:  # a missing result must not stop the rest
            print(f"  [fail] {name}: {type(exc).__name__}: {exc}")
    print(json.dumps(made, indent=2))


if __name__ == "__main__":
    # Run through the package module so the counting figures (which import it) share the
    # same language state as the rest.
    from cottoncross import figures as _package

    _package.main()
