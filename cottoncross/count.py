"""Per-image fiber counting under the equal-staple prior.

Every fiber in a sample is the same 35 mm staple, which gives a counting rule that
does not care how the fibers cross:

    N = round( total length of genuine fiber centerline / staple length )

Crossings break a fiber into several skeleton pieces and make trajectory counting
fragile, but they do not change how much centerline there is.  What the rule does need is

1. a centerline that is complete (no fiber lost to low contrast), and
2. a centerline that is clean - every short speck, dust grain or scratch that survives
   segmentation adds length that is not fiber.

Both depend on the segmentation model, which is exactly what the counting benchmark
compares.  The pipeline here is shared by every model, so differences in count accuracy
come from the model, not from the counting rule.

Stages
------
`plate_features`  model output -> linked trajectories with length, ends, end directions
                  and mean centerline confidence (GPU work happens once, results cached)
`close_gaps`      re-joins trajectory ends separated by a short collinear gap, the
                  typical break a faint stretch of fiber leaves in the mask
`count_plate`     debris filter (a piece far shorter than a staple that is not joined to
                  anything longer cannot be a fiber) and the three estimators below
`calibrate_staple` label-free staple length in pixels from complete trajectories

Estimators reported for every model
-----------------------------------
components    connected skeleton pieces above a size floor - what a naive blob counter
              reports; crossings merge fibers and breaks split them
trajectories  linked trajectories longer than half a staple - resolves crossings but is
              hurt by every unresolved crossing and every break
length        the equal-staple rule above after gap closing and debris removal
chains        chain-level form of the same rule: every gap-closed chain longer than
              CHAIN_FRACTION of a staple is counted as round(its length / staple) fibers
              on its own (two fibers merged through a crossing give 2), and only the
              leftover fragments are pooled by length.  A fiber traced at 75% of its
              length still counts as one, where the pooled rule lets such losses add up
              (three fibers at 75% pool to 2.25 staples = 2); what it needs in exchange is
              that crossings are resolved, i.e. that a chain is one fiber, not a jumble.

Rounding is half-up (floor(x + 0.5)), not Python's round-half-to-even.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .predict import postprocess

# Fixed from the physical prior, not tuned on any split that is scored.  A staple is
# 35 mm; a piece shorter than DEBRIS_FRACTION of that (about 3.5 mm) that stays isolated
# after gap closing is treated as interference.
DEBRIS_FRACTION = 0.10
TRAJECTORY_FRACTION = 0.50
CHAIN_FRACTION = 0.60
GAP_PIXELS = 45
GAP_ANGLE_DEG = 40.0
COMPONENT_MIN_PIXELS = 60
BORDER_PIXELS = 12


def _direction(points, head, span=12):
    """Unit vector pointing OUT of the trajectory at one end."""
    n = len(points)
    if n < 2:
        return [0.0, 0.0]
    k = min(span, n - 1)
    if head:
        v = points[0] - points[k]
    else:
        v = points[-1] - points[-1 - k]
    norm = float(np.linalg.norm(v))
    return (v / norm).tolist() if norm > 1e-6 else [0.0, 0.0]


def _near_border(p, shape, margin=BORDER_PIXELS):
    h, w = shape
    x, y = p
    return bool(x < margin or y < margin or x > w - 1 - margin or y > h - 1 - margin)


def crossings_on_fibers(fibers, points, shape, staple, debris_fraction=DEBRIS_FRACTION,
                        tolerance=8.0):
    """Keep only crossing candidates that lie on a counted fiber (not on interference)."""
    if not fibers or not points:
        return []
    chains = close_gaps(fibers)
    canvas = np.zeros(shape, np.uint8)
    for chain in chains:
        if sum(fibers[i]["length"] for i in chain) < debris_fraction * staple:
            continue
        for i in chain:
            pts = np.rint(fibers[i]["points"]).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, 1, 1, cv2.LINE_8)
    if not canvas.any():
        return []
    distance = ndi.distance_transform_edt(canvas == 0)
    h, w = shape
    keep = []
    for x, y in points:
        yi, xi = int(np.clip(round(y), 0, h - 1)), int(np.clip(round(x), 0, w - 1))
        if distance[yi, xi] <= tolerance:
            keep.append([float(x), float(y)])
    return keep


def plate_features(bundle, gray, method="cfx", thresholds=None, staple=None):
    """Everything the estimators need, small enough to cache as JSON.

    With `staple`, crossing candidates are also filtered to those lying on a counted fiber
    (docs/development/crossing_count_rule.md) and stored with their positions.
    """
    binary, skeleton, graph = postprocess(bundle, gray, method=method, thresholds=thresholds)
    center = bundle["center"]
    h, w = gray.shape
    fibers = []
    for f in graph["fibers"]:
        pts = np.asarray(f["points_xy"], float)
        if len(pts) == 0:
            continue
        ix = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
        fibers.append(dict(
            length=float(f["length_pixels"]), closed=bool(f["closed"]),
            head=pts[0].tolist(), tail=pts[-1].tolist(),
            head_dir=_direction(pts, True), tail_dir=_direction(pts, False),
            head_border=_near_border(pts[0], (h, w)), tail_border=_near_border(pts[-1], (h, w)),
            confidence=float(center[iy, ix].mean())))
    labels, n = ndi.label(skeleton > 0, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())[1:] if n else np.zeros(0, int)
    out = dict(shape=[h, w], fibers=fibers, component_pixels=sizes.astype(int).tolist(),
               skeleton_pixels=int((skeleton > 0).sum()),
               crossings=int(graph["crossing_candidates"]),
               heatmap_crossings=len(graph.get("heatmap_crossings_xy") or []))
    if staple:
        geometry = [dict(f, points=np.asarray(g["points_xy"], float))
                    for f, g in zip(fibers, [g for g in graph["fibers"] if len(g["points_xy"])])]
        junctions = [j["xy"] for j in graph["joints"] if j["degree"] >= 4]
        out["crossing_points_heatmap"] = crossings_on_fibers(
            geometry, graph.get("heatmap_crossings_xy") or [], (h, w), staple)
        out["crossing_points_skeleton"] = crossings_on_fibers(geometry, junctions, (h, w), staple)
        out["crossing_staple"] = staple
    return out


def close_gaps(fibers, max_gap=GAP_PIXELS, max_angle=GAP_ANGLE_DEG):
    """Join collinear trajectory ends across short gaps; returns chains of fiber indices.

    Two ends are joined when they are closer than `max_gap`, each end points toward the
    other within `max_angle`, and neither is claimed by a better (shorter, straighter)
    candidate - a greedy one-to-one matching on the gap cost.
    """
    ends = [(i, np.asarray(f[k], float), np.asarray(f[k + "_dir"], float))
            for i, f in enumerate(fibers) if not f["closed"] for k in ("head", "tail")]
    cos_limit = math.cos(math.radians(max_angle))
    candidates = []
    if len(ends) >= 2:
        tree = cKDTree(np.array([e[1] for e in ends]))
        for a, b in tree.query_pairs(max_gap):
            ia, pa, da = ends[a]
            ib, pb, db = ends[b]
            if ia == ib:
                continue
            gap = pb - pa
            d = float(np.linalg.norm(gap))
            if d < 1e-6:
                candidates.append((0.0, a, b))
                continue
            u = gap / d
            ca, cb = float(da @ u), float(-(db @ u))
            if ca < cos_limit or cb < cos_limit:
                continue
            candidates.append((d * (3.0 - ca - cb), a, b))
    parent = list(range(len(fibers)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used = set()
    for _, a, b in sorted(candidates):
        if a in used or b in used:
            continue
        ra, rb = find(ends[a][0]), find(ends[b][0])
        if ra == rb:
            continue
        used.update((a, b))
        parent[ra] = rb
    chains = {}
    for i in range(len(fibers)):
        chains.setdefault(find(i), []).append(i)
    return list(chains.values())


def chain_lengths(features, gap=GAP_PIXELS, angle=GAP_ANGLE_DEG):
    """Summed trajectory length of every gap-closed chain (independent of the staple)."""
    fibers = features["fibers"]
    if not fibers:
        return np.zeros(0)
    lengths = np.array([f["length"] for f in fibers], float)
    return np.array([lengths[c].sum() for c in close_gaps(fibers, gap, angle)], float)


def half_up(x):
    return int(math.floor(x + 0.5))


def count_plate(features, staple, debris_fraction=DEBRIS_FRACTION,
                trajectory_fraction=TRAJECTORY_FRACTION, gap=GAP_PIXELS,
                angle=GAP_ANGLE_DEG, component_min=COMPONENT_MIN_PIXELS, chains=None,
                chain_fraction=CHAIN_FRACTION):
    """Per-image counts; pass `chains=chain_lengths(features)` to reuse gap closing."""
    fibers = features["fibers"]
    comps = np.asarray(features["component_pixels"], float)
    lengths = np.array([f["length"] for f in fibers], float)
    chain_len = chain_lengths(features, gap, angle) if chains is None else chains
    kept = chain_len[chain_len >= debris_fraction * staple]
    total = float(kept.sum())
    by_length = max(1, half_up(total / staple)) if total > 0 else 0
    whole = kept[kept >= chain_fraction * staple]
    rest = float(kept[kept < chain_fraction * staple].sum())
    by_chain = int(sum(max(1, half_up(l / staple)) for l in whole)) + half_up(rest / staple)
    if total > 0:
        by_chain = max(1, by_chain)
    return dict(
        chains_count=by_chain,
        components=int((comps >= component_min).sum()),
        trajectories=int((lengths >= trajectory_fraction * staple).sum()),
        length=by_length,
        total_length=total,
        raw_length=float(lengths.sum()),
        debris_removed=int((chain_len < debris_fraction * staple).sum()),
        debris_length=float(chain_len[chain_len < debris_fraction * staple].sum()),
        chains=int(len(kept)))


def complete_lengths(features, gap=GAP_PIXELS, angle=GAP_ANGLE_DEG):
    """Lengths of gap-closed chains whose two ends are free and away from the border.

    A chain that ends at the image border is truncated; a chain whose end sits on a
    crossing may have been cut there.  What is left are the fibers the model traced from
    one free tip to the other, the only direct measurement of a staple in pixels.
    """
    fibers = features["fibers"]
    if not fibers:
        return []
    out = []
    for chain in close_gaps(fibers, gap, angle):
        if any(fibers[i]["closed"] for i in chain):
            continue
        if any(fibers[i]["head_border"] or fibers[i]["tail_border"] for i in chain):
            continue
        out.append(float(sum(fibers[i]["length"] for i in chain)))
    return out


def calibrate_staple(feature_list, floor=300.0, quantile=0.75):
    """Label-free staple length: an upper quantile of complete chain lengths.

    Chains through an unresolved crossing come out short, never long, and fragments are
    removed by the floor, so the upper part of the complete-chain distribution is where
    whole fibers sit.  Returned with the sample it was computed from.
    """
    pool = [v for feats in feature_list for v in complete_lengths(feats) if v >= floor]
    if not pool:
        return dict(staple=float("nan"), n=0)
    pool = np.asarray(pool)
    return dict(staple=float(np.quantile(pool, quantile)), n=int(len(pool)),
                median=float(np.median(pool)), p90=float(np.quantile(pool, 0.9)),
                quantile=quantile, floor=floor)
