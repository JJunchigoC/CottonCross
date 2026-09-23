"""Evaluation metrics.

Grouped by what they answer:

* occupancy      - does the model find fiber pixels?      Dice, IoU, precision, recall
* topology       - does it find CONNECTED fibers?         clDice, ASSD, Betti-0 error
* crossings      - does it find the intersections?        one-to-one P/R/F1, AP
* trajectories   - does it follow a fiber THROUGH one?    instance F1, length statistics
* substrate      - does it reject the background?         false-alarm rate

Point matching is a threshold-gated maximum-cardinality one-to-one assignment, so a
detector cannot inflate recall by predicting many points near one ground-truth crossing.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .topology import zhang_suen


# ------------------------------------------------------------------------- occupancy
def overlap_scores(pred, truth):
    p = np.asarray(pred).astype(bool)
    t = np.asarray(truth).astype(bool)
    i = int((p & t).sum())
    ps, ts = int(p.sum()), int(t.sum())
    union = ps + ts - i
    return dict(dice=(2 * i) / (ps + ts) if ps + ts else 1.0,
                iou=i / union if union else 1.0,
                precision=i / ps if ps else float(ts == 0),
                recall=i / ts if ts else float(ps == 0))


# -------------------------------------------------------------------------- topology
def cldice(pred, truth):
    p, t = np.asarray(pred) > 0, np.asarray(truth) > 0
    sp, st = zhang_suen(p), zhang_suen(t)
    if not sp.any() and not st.any():
        return 1.0
    precision = float((sp * t).sum()) / max(int(sp.sum()), 1)
    recall = float((st * p).sum()) / max(int(st.sum()), 1)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def centerline_f1(pred, truth, tolerance=2):
    p, t = np.asarray(pred) > 0, np.asarray(truth) > 0
    if not p.any() or not t.any():
        return float(not p.any() and not t.any())
    precision = float((ndi.distance_transform_edt(~t)[p] <= tolerance).mean())
    recall = float((ndi.distance_transform_edt(~p)[t] <= tolerance).mean())
    return 2 * precision * recall / max(precision + recall, 1e-8)


def assd(pred, truth):
    """Average symmetric surface distance between two centerline sets, in pixels."""
    p, t = np.asarray(pred) > 0, np.asarray(truth) > 0
    if not p.any() or not t.any():
        return float("nan")
    dp = ndi.distance_transform_edt(~p)[t]
    dt = ndi.distance_transform_edt(~t)[p]
    return float((dp.sum() + dt.sum()) / (dp.size + dt.size))


def betti0_error(pred, truth):
    """Absolute difference in the number of connected components (Betti-0)."""
    lp, np_ = ndi.label(np.asarray(pred) > 0, structure=np.ones((3, 3)))
    lt, nt = ndi.label(np.asarray(truth) > 0, structure=np.ones((3, 3)))
    return abs(int(np_) - int(nt)), int(np_), int(nt)


# ------------------------------------------------------------------------- crossings
def point_counts(pred, truth, tolerance=6):
    p = np.asarray(pred, dtype=float).reshape(-1, 2)
    t = np.asarray(truth, dtype=float).reshape(-1, 2)
    n, m = len(p), len(t)
    if not n or not m:
        return dict(tp=0, fp=n, fn=m, distance_sum=0.0)
    distance = cdist(p, t)
    # Dummy rows/columns turn this into maximum-cardinality matching under a gate.
    cost = np.ones((n + m, n + m), float)
    cost[n:, m:] = 0.0
    cost[:n, :m] = np.where(distance <= tolerance,
                            distance / (tolerance + 1) / max(n, m), 1e6)
    a, b = linear_sum_assignment(cost)
    tp, dsum = 0, 0.0
    for i, j in zip(a, b):
        if i < n and j < m and distance[i, j] <= tolerance:
            tp += 1
            dsum += float(distance[i, j])
    return dict(tp=int(tp), fp=n - int(tp), fn=m - int(tp), distance_sum=dsum)


def point_scores(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return dict(tp=tp, fp=fp, fn=fn,
                precision=tp / (tp + fp) if tp + fp else float(fn == 0),
                recall=tp / (tp + fn) if tp + fn else float(fp == 0),
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
                mean_distance=counts.get("distance_sum", 0.0) / tp if tp else float("nan"))


def heatmap_average_precision(records, tolerance=6, thresholds=None):
    """AP over crossing-heatmap thresholds; records hold (score, matched) point lists."""
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19)
    curve = []
    for thr in thresholds:
        counts = dict(tp=0, fp=0, fn=0)
        for rec in records:
            keep = [p for p, s in zip(rec["points"], rec["scores"]) if s >= thr]
            c = point_counts(keep, rec["truth"], tolerance)
            for k in counts:
                counts[k] += c[k]
        s = point_scores(counts)
        curve.append(dict(threshold=float(thr), precision=s["precision"],
                          recall=s["recall"], f1=s["f1"]))
    order = sorted(curve, key=lambda d: d["recall"])
    ap, prev_r = 0.0, 0.0
    for point in order:
        ap += (point["recall"] - prev_r) * point["precision"]
        prev_r = point["recall"]
    return float(ap), curve


# ---------------------------------------------------------------------- trajectories
def rasterise(fiber, shape):
    import cv2

    mask = np.zeros(shape, np.uint8)
    pts = np.rint(np.asarray(fiber["points_xy"], dtype=float)).astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(mask, [pts], bool(fiber.get("closed", False)), 1, 1)
    elif len(pts) == 1:
        mask[np.clip(pts[0, 1], 0, shape[0] - 1), np.clip(pts[0, 0], 0, shape[1] - 1)] = 1
    return mask > 0


def instance_scores(fibers, instances, tolerance=2, minimum_f1=0.5):
    """One-to-one trajectory matching on tolerance-aware centerline F1.

    A diagnostic 2D trajectory score.  Fragments cost precision, merges cost recall.
    """
    truth = [m > 0 for m in instances if m.sum() >= 8]
    n, m = len(fibers), len(truth)
    if not n or not m:
        return dict(tp=0, fp=n, fn=m, matched_f1_sum=0.0, mean_matched_f1=0.0,
                    predicted=n, true=m)
    shape = truth[0].shape
    pred = [rasterise(f, shape) for f in fibers]
    dist_t = [ndi.distance_transform_edt(~t) for t in truth]
    dist_p = [ndi.distance_transform_edt(~p) if p.any() else None for p in pred]
    scores = np.zeros((n, m))
    for i, p in enumerate(pred):
        if not p.any():
            continue
        for j, t in enumerate(truth):
            precision = float((dist_t[j][p] <= tolerance).mean())
            recall = float((dist_p[i][t] <= tolerance).mean())
            scores[i, j] = 2 * precision * recall / max(precision + recall, 1e-8)
    a, b = linear_sum_assignment(-scores)
    matched = scores[a, b]
    tp = int((matched >= minimum_f1).sum())
    return dict(tp=tp, fp=n - tp, fn=m - tp, matched_f1_sum=float(matched.sum()),
                mean_matched_f1=float(matched[matched >= minimum_f1].mean()) if tp else 0.0,
                predicted=n, true=m)


def pair_accuracy(graph, instance_id, tolerance=2):
    """Are the pairs the linker committed to actually the same fiber?

    Each port is assigned the ground-truth instance that occupies most of the skeleton
    pixels it was sampled on (looked up in a small neighbourhood, since the predicted
    skeleton does not sit exactly on the true centerline).  A committed pair is correct
    when both of its ports carry the same instance.

    Reported together with the abstention rate, this is what shows whether the rejection
    rule buys anything: abstaining should raise accuracy on what remains.
    """
    height, width = instance_id.shape
    lookup = np.zeros_like(instance_id)
    # Dilate the label map so a one-pixel offset does not read background.
    for dy in range(-tolerance, tolerance + 1):
        for dx in range(-tolerance, tolerance + 1):
            shifted = np.roll(np.roll(instance_id, dy, axis=0), dx, axis=1)
            lookup = np.where(lookup > 0, lookup, shifted)

    def port_instance(port):
        pts = np.asarray(port.get("instance_pixels") or [port["point_yx"]], dtype=int)
        if not len(pts):
            return 0
        ys = np.clip(pts[:, 0], 0, height - 1)
        xs = np.clip(pts[:, 1], 0, width - 1)
        values = lookup[ys, xs]
        values = values[values > 0]
        if not len(values):
            return 0
        counts = np.bincount(values)
        return int(np.argmax(counts))

    correct = total = 0
    joints = graph["joints"]
    for joint in joints:
        ports = joint["ports"]
        for a, b in joint["accepted_pairs"]:
            ia, ib = port_instance(ports[a]), port_instance(ports[b])
            if ia == 0 or ib == 0:
                continue
            total += 1
            correct += int(ia == ib)
    abstained = sum(bool(j["solution"]["ambiguous"]) for j in joints)
    return dict(correct_pairs=correct, committed_pairs=total,
                pair_accuracy=correct / total if total else float("nan"),
                abstention_rate=abstained / len(joints) if joints else 0.0,
                joints=len(joints))


def length_consistency(fibers, expected=None, min_length=40.0):
    """Dispersion of reconstructed trajectory lengths under the equal-staple prior.

    All fibers in this specimen set have the same true length, so a better reconstruction
    produces a tighter length distribution.  The statistic needs no annotation.
    """
    lengths = np.array([f["length_pixels"] for f in fibers
                        if f["length_pixels"] >= min_length], float)
    if len(lengths) < 2:
        return dict(n=int(len(lengths)), cv=float("nan"), median=float(lengths.mean()) if len(lengths) else 0.0,
                    relative_error=float("nan"), long_fraction=0.0)
    median = float(np.median(lengths))
    mad = float(np.median(np.abs(lengths - median)))
    out = dict(n=int(len(lengths)), median=median,
               cv=float(1.4826 * mad / max(median, 1e-6)),
               p90=float(np.percentile(lengths, 90)))
    if expected:
        out["relative_error"] = float(abs(np.percentile(lengths, 90) - expected) / expected)
        out["long_fraction"] = float((lengths >= 0.6 * expected).mean())
    return out


# ------------------------------------------------------------------------- substrate
def substrate_false_alarm(pred_mask, substrate_mask):
    """Fraction of detector-independent substrate pixels claimed as fiber."""
    s = np.asarray(substrate_mask, bool)
    if not s.any():
        return float("nan")
    return float(np.asarray(pred_mask, bool)[s].mean())


# ------------------------------------------------------------------------ aggregation
def bootstrap_ci(values, statistic=np.mean, n=2000, alpha=0.05, seed=0):
    values = np.asarray([v for v in values if np.isfinite(v)], float)
    if len(values) < 2:
        return dict(mean=float(values.mean()) if len(values) else float("nan"),
                    low=float("nan"), high=float("nan"), n=int(len(values)))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), (n, len(values)))
    samples = statistic(values[draws], axis=1)
    return dict(mean=float(statistic(values)),
                low=float(np.percentile(samples, 100 * alpha / 2)),
                high=float(np.percentile(samples, 100 * (1 - alpha / 2))),
                std=float(values.std(ddof=1)), n=int(len(values)))


def paired_test(a, b, n=20000, seed=0):
    """Paired permutation test plus Wilcoxon signed-rank on per-image differences."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 3:
        return dict(n=int(len(a)), mean_difference=float("nan"), permutation_p=float("nan"),
                    wilcoxon_p=float("nan"))
    d = a - b
    observed = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], (n, len(d)))
    null = (signs * d).mean(axis=1)
    perm_p = float((np.abs(null) >= abs(observed) - 1e-12).mean())
    try:
        from scipy.stats import wilcoxon

        wil = float(wilcoxon(a, b, zero_method="zsplit").pvalue) if np.any(d != 0) else 1.0
    except Exception:
        wil = float("nan")
    return dict(n=int(len(a)), mean_difference=observed, permutation_p=perm_p,
                wilcoxon_p=wil,
                cohens_d=float(observed / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"))
