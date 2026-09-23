"""Scale calibration from the equal-length prior, and crossing-centred real sampling.

Two things the rest of the pipeline needs and that can both be obtained WITHOUT any
human annotation:

1. `calibrate_scale` - the magnification of the plates is unknown, but every fiber in
   this specimen set is cut to the same nominal staple length (35 mm).  Isolated fibers
   (skeleton components with exactly two endpoints and no interior junction) therefore
   all have the same traced pixel length L.  We estimate L as a robust upper mode of the
   isolated-trace length distribution and report px/mm = L / 35.  The estimate is a
   calibration, not a measurement of any single fiber.

2. `crossing_windows` - unsupervised crossing candidates on real plates, used to crop
   real training/augmentation tiles that are GUARANTEED to contain a complete crossing
   (junction interior to the window, all four arms supported inside it).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from .background import ridge_response
from .common import load_json, read_image, save_json
from .topology import branch_count, prune_spurs, zhang_suen

NOMINAL_STAPLE_MM = 35.0


def fiber_mask(gray: np.ndarray, quantile: float = 0.995, min_size: int = 40) -> np.ndarray:
    """Unsupervised fiber mask via hysteresis on the background-normalised ridge map."""
    r = ridge_response(gray)
    high = max(float(np.quantile(r, quantile)), 1e-5)
    strong = r > high
    weak = r > high * 0.30
    mask = ndi.binary_propagation(strong, mask=weak)
    lab, _ = ndi.label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def _trace_length(coords: np.ndarray) -> float:
    """Euclidean traced length of a non-branching skeleton component."""
    pixels = {tuple(p) for p in coords.tolist()}
    neighbours = {}
    for y, x in pixels:
        ns = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y + dy, x + dx)
                if (not dy and not dx) or q not in pixels:
                    continue
                if dy and dx and ((y + dy, x) in pixels or (y, x + dx) in pixels):
                    continue
                ns.append(q)
        neighbours[(y, x)] = ns
    if any(len(v) > 2 for v in neighbours.values()):
        return 0.0
    ends = [p for p, ns in neighbours.items() if len(ns) == 1]
    if len(ends) != 2:
        return 0.0
    path = [ends[0]]
    seen = {ends[0]}
    while True:
        nxt = [p for p in neighbours[path[-1]] if p not in seen]
        if not nxt:
            break
        path.append(nxt[0])
        seen.add(nxt[0])
    if len(path) != len(pixels):
        return 0.0
    arr = np.array(path, dtype=float)
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def isolated_trace_lengths(gray: np.ndarray) -> list[float]:
    sk = prune_spurs(zhang_suen(fiber_mask(gray)), 8)
    lab, n = ndi.label(sk, structure=np.ones((3, 3)))
    lengths = []
    for i, box in enumerate(ndi.find_objects(lab), 1):
        if box is None:
            continue
        coords = np.argwhere(lab[box] == i) + np.array([box[0].start, box[1].start])
        if len(coords) < 30:
            continue
        lengths.append(_trace_length(coords))
    return [v for v in lengths if v > 0]


def traced_lengths(gray: np.ndarray) -> list[float]:
    """Trajectory lengths after crossings are resolved, from the classical mask."""
    from .topology import analyze

    _, graph = analyze(fiber_mask(gray).astype(np.uint8), gray, method="optimal")
    return [float(f["length_pixels"]) for f in graph["fibers"] if f["length_pixels"] > 0]


def calibrate_scale(real_root, split="train", limit=None):
    """Bootstrap pixels-per-millimetre from the equal-length prior.

    Fragmentation can only SHORTEN a traced fiber, never lengthen it, so the upper tail
    of the trace-length distribution is the part that carries the scale.  We take the
    90th percentile of the per-plate maxima of trajectories reconstructed from the
    classical (training-free) mask.  Because the classical mask still breaks fibers, the
    result is an explicit LOWER BOUND on the true staple length in pixels; the learned
    pipeline re-estimates it in `eval_real.refine_calibration`.
    """
    root = Path(real_root)
    rows = [r for r in load_json(root / "manifest.json") if split == "all" or r["split"] == split]
    if limit:
        rows = rows[:limit]
    per_plate_max, pooled, fragments = [], [], []
    for r in rows:
        gray = read_image(root / r["roi"], True)
        lengths = traced_lengths(gray)
        fragments.extend(isolated_trace_lengths(gray))
        pooled.extend(lengths)
        if lengths:
            per_plate_max.append(max(lengths))
    if not per_plate_max:
        raise RuntimeError("calibration found no traceable fibers")
    staple_px = float(np.percentile(per_plate_max, 90))
    return dict(
        staple_pixels=staple_px,
        pixels_per_mm=staple_px / NOMINAL_STAPLE_MM,
        nominal_staple_mm=NOMINAL_STAPLE_MM,
        plates_used=len(per_plate_max),
        traces=len(pooled),
        fragment_p95=float(np.percentile(fragments, 95)) if fragments else 0.0,
        plate_max_p50=float(np.percentile(per_plate_max, 50)),
        plate_max_p90=staple_px,
        pooled_p95=float(np.percentile(pooled, 95)) if pooled else 0.0,
        split=split,
        bound="lower",
        method=("90th percentile of per-plate maximum reconstructed trajectory length "
                "from the training-free mask, interpreted through the 35 mm equal-staple "
                "prior; an unknown-magnification calibration and a lower bound, not a "
                "per-fiber length measurement"),
    )


def crossing_windows(gray: np.ndarray, window: int, arm: int = 48, margin_ratio: float = 0.28,
                     max_windows: int = 64):
    """Windows that provably contain a COMPLETE crossing.

    A candidate is kept only when (a) the junction sits at least `margin` pixels away
    from every window border and (b) each of the >=4 skeleton arms leaving the junction
    stays inside the window for at least `arm` pixels.  This is what makes every exported
    real augmentation tile contain a whole crossing rather than a clipped one.
    """
    mask = fiber_mask(gray)
    sk = prune_spurs(zhang_suen(mask), 8)
    counts = branch_count(sk)
    junction = counts >= 3
    if not junction.any():
        return []
    lab, n = ndi.label(ndi.binary_dilation(junction, iterations=3), structure=np.ones((3, 3)))
    h, w = gray.shape
    margin = int(window * margin_ratio)
    half = window // 2
    out = []
    for i, box in enumerate(ndi.find_objects(lab), 1):
        if box is None:
            continue
        pts = np.argwhere((lab[box] == i) & junction[box]) + np.array([box[0].start, box[1].start])
        if not len(pts):
            continue
        cy, cx = pts.mean(axis=0)
        degree = int(counts[tuple(pts[np.argmax(counts[pts[:, 0], pts[:, 1]])])])
        cy = int(np.clip(round(cy), half, h - half - 1)) if h > window else h // 2
        cx = int(np.clip(round(cx), half, w - half - 1)) if w > window else w // 2
        y0, x0 = cy - half, cx - half
        if y0 < 0 or x0 < 0 or y0 + window > h or x0 + window > w:
            continue
        jy, jx = pts.mean(axis=0) - np.array([y0, x0])
        if not (margin <= jy <= window - margin and margin <= jx <= window - margin):
            continue
        tile_sk = sk[y0:y0 + window, x0:x0 + window]
        # Arm support: skeleton pixels inside an annulus around the junction, split into
        # 8 angular sectors.  A complete crossing lights up at least 4 opposite sectors.
        yy, xx = np.ogrid[:window, :window]
        rr = np.hypot(yy - jy, xx - jx)
        ring = tile_sk & (rr > arm * 0.5) & (rr <= arm)
        if ring.sum() < 8:
            continue
        ay, ax = np.nonzero(ring)
        sector = ((np.arctan2(ay - jy, ax - jx) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        occupied = np.bincount(sector, minlength=8) >= 2
        if occupied.sum() < 4:
            continue
        out.append(dict(xyxy=[x0, y0, x0 + window, y0 + window],
                        junction_xy=[float(jx), float(jy)],
                        junction_plate_xy=[float(x0 + jx), float(y0 + jy)], degree=degree,
                        arm_sectors=int(occupied.sum()),
                        fiber_fraction=float(mask[y0:y0 + window, x0:x0 + window].mean())))
    out.sort(key=lambda d: (-d["arm_sectors"], -d["fiber_fraction"]))
    # Neighbouring skeleton pixels of one physical crossing produce near-duplicate
    # windows; keep the best per crossing so the sampler sees distinct sites.
    kept = []
    for cand in out:
        p = np.array(cand["junction_plate_xy"])
        if all(np.linalg.norm(p - np.array(k["junction_plate_xy"])) > window * 0.25 for k in kept):
            kept.append(cand)
        if len(kept) >= max_windows:
            break
    return kept


def main(real="data/real", out="data/calibration.json", limit=None):
    info = calibrate_scale(real, "train", limit)
    save_json(out, info)
    print(info)
    return info


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--real", default="data/real")
    p.add_argument("--out", default="data/calibration.json")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    main(a.real, a.out, a.limit)
