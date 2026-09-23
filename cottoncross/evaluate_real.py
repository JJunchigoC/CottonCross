"""Label-free evaluation on the real plates.

No human annotation exists for these micrographs, so accuracy against a human reference
cannot be reported and is not.  What *can* be measured without labels are properties any
correct detector must have, and that a wrong one violates:

SFAR   substrate false-alarm rate - fiber pixels claimed inside a "definitely substrate"
       region.  Lower is better.
COV    evidence coverage - fraction of the strongest classical ridge cores that the model
       also calls fiber.  Reported alongside SFAR so a model cannot win by predicting
       nothing.  Higher is better.
EQE    equivariance error - the prediction must transform with the image under flips and
       a half turn.  Lower is better.
CREP   crossing repeatability - crossings found in the transformed view, mapped back,
       matched one-to-one against those found in the original.  Higher is better.
LCV    length dispersion among substantial trajectories.  Lower is better, but only
       meaningful together with LERR.
LERR   length recovery error - |median substantial trajectory length - calibrated
       staple| / staple.  Lower is better.
FRAG   fragmentation - ALL trajectories per 1000 px of recovered centerline.  Lower is
       better.

Two warnings that belong with these numbers:

1. **The classical ridge baseline cannot be scored on SFAR, COV, EQE or CREP.**  The
   substrate region and the evidence core are both derived from the very ridge operator
   that baseline uses, and a fixed linear filter is exactly equivariant by construction.
   It therefore scores perfectly on all four without being accurate - the comparison is
   circular and those cells must be reported as not applicable, not as a win.
2. LCV alone rewards a detector that shatters every fiber into equally tiny pieces, which
   is why LERR and FRAG are reported beside it.

`--annotations` remains available for a genuine human-labelled crossing set; the script
refuses to score anything not marked complete by a human, and never substitutes model
output for ground truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .common import load_json, read_image, save_json
from .metrics import bootstrap_ci, point_counts, point_scores
from .predict import (classical, load_model, postprocess, predict, substrate_mask)
from .background import ridge_response

TRANSFORMS = ("flip_x", "flip_y", "rot180")


def _apply(image, name):
    if name == "flip_x":
        return np.ascontiguousarray(np.fliplr(image))
    if name == "flip_y":
        return np.ascontiguousarray(np.flipud(image))
    return np.ascontiguousarray(np.rot90(image, 2))


def _invert_map(points, name, shape):
    h, w = shape
    out = []
    for x, y in points:
        if name == "flip_x":
            out.append([w - 1 - x, y])
        elif name == "flip_y":
            out.append([x, h - 1 - y])
        else:
            out.append([w - 1 - x, h - 1 - y])
    return out


def evidence_core(gray, quantile=0.9995):
    response = ridge_response(gray)
    return response > max(float(np.quantile(response, quantile)), 1e-8)


def evaluate_plate(gray, model, device, method, thresholds, expected_length,
                   transforms=TRANSFORMS, tolerance=8):
    # Use this model's validation-selected crossing threshold, so the per-plate output
    # sits at the same operating point the tables report rather than a generic 0.5.
    selected = getattr(model, "crossing_threshold", None) if model is not None else None
    if selected is not None:
        thresholds = dict(thresholds or {})
        thresholds.setdefault("joint", selected)
    if model is not None:
        bundle = predict(gray, model, device)
    else:
        bundle, _ = classical(gray)
    binary, skeleton, graph = postprocess(bundle, gray, method=method, thresholds=thresholds)
    pred_mask = bundle["mask"] > 0.5

    substrate = substrate_mask(gray)
    core = evidence_core(gray)
    sfar = float(pred_mask[substrate].mean()) if substrate.any() else float("nan")
    coverage = float(pred_mask[core].mean()) if core.any() else float("nan")

    base_points = graph.get("heatmap_crossings_xy") or [j["xy"] for j in graph["joints"]
                                                        if j["degree"] >= 4]
    eqe, crep = [], dict(tp=0, fp=0, fn=0)
    for name in transforms:
        alt = _apply(gray, name)
        if model is not None:
            alt_bundle = predict(alt, model, device)
        else:
            alt_bundle, _ = classical(alt)
        back = {k: (None if v is None else _apply_channels(v, name))
                for k, v in alt_bundle.items()}
        for key in ("mask", "center", "joint"):
            if back.get(key) is not None and bundle.get(key) is not None:
                eqe.append(float(np.mean((back[key] - bundle[key]) ** 2)))
        _, _, alt_graph = postprocess(alt_bundle, alt, method=method, thresholds=thresholds)
        alt_points = (alt_graph.get("heatmap_crossings_xy")
                      or [j["xy"] for j in alt_graph["joints"] if j["degree"] >= 4])
        mapped = _invert_map(alt_points, name, gray.shape)
        c = point_counts(mapped, base_points, tolerance)
        for k in ("tp", "fp", "fn"):
            crep[k] += c[k]

    skeleton_length = float(skeleton.sum())
    floor = 0.25 * (expected_length or 0.0)
    substantial = [f for f in graph["fibers"] if f["length_pixels"] >= floor]
    return dict(
        sfar=sfar, coverage=coverage,
        equivariance_mse=float(np.mean(eqe)) if eqe else float("nan"),
        crossing_repeatability=point_scores(crep)["f1"],
        crossing_repeatability_counts=crep,
        crossings=len(base_points),
        graph_crossings=int(graph["crossing_candidates"]),
        unresolved_joints=int(graph["unresolved_joints"]),
        skeleton_pixels=skeleton_length,
        mask_fraction=float(pred_mask.mean()),
        **length_statistics([f["length_pixels"] for f in graph["fibers"]],
                            skeleton_length, expected_length),
    ), graph, skeleton, bundle


def length_statistics(lengths, skeleton_length, expected_length, floor_ratio=0.25):
    """Length-based statistics that a fragmenting detector cannot win.

    An earlier version measured dispersion over every track and counted only the long
    ones as fragments.  Both favoured a detector that shatters every fiber: uniformly
    tiny pieces have a tight length distribution, and they contribute almost no "long"
    tracks.  The corrected versions are:

    * `fragmentation` counts ALL reconstructed tracks per 1000 px of recovered
      centerline, so shattering a fiber is penalised;
    * `length_cv` is measured on the substantial tracks only, and is meaningless unless
      read together with
    * `length_recovery_error`, the relative gap between the 90th percentile of
      substantial track length and the calibrated staple - a detector whose longest
      trajectory is a tenth of a fiber fails here no matter how tight its spread is.
    """
    lengths = np.asarray(lengths, float)
    floor = floor_ratio * (expected_length or 0.0)
    substantial = lengths[lengths >= floor] if floor else lengths
    stats = dict(tracks=int(len(lengths)), substantial_tracks=int(len(substantial)),
                 substantial_floor_px=float(floor),
                 fragmentation=float(1000.0 * len(lengths) / max(skeleton_length, 1.0)))
    if len(substantial) >= 2:
        median = float(np.median(substantial))
        mad = float(np.median(np.abs(substantial - median)))
        p90 = float(np.percentile(substantial, 90))
        stats["length"] = dict(n=int(len(substantial)), median=median, p90=p90,
                               cv=float(1.4826 * mad / max(median, 1e-6)))
    else:
        p90 = float(substantial.max()) if len(substantial) else 0.0
        stats["length"] = dict(n=int(len(substantial)), median=p90, p90=p90,
                               cv=float("nan"))
    # Two recovery errors, because they answer different questions and the honest thing
    # is to show both.  The MEDIAN one is the headline: it asks whether a typical
    # reconstructed fiber has the right length.  The p90 one is an upper-tail statistic
    # and is biased upward here, since only a handful of substantial tracks exist per
    # plate and the longest of them is often two fibers chained through a crossing.
    if expected_length:
        stats["length_recovery_error"] = float(
            abs(stats["length"]["median"] - expected_length) / expected_length)
        stats["length_recovery_error_p90"] = float(
            abs(stats["length"]["p90"] - expected_length) / expected_length)
    else:
        stats["length_recovery_error"] = float("nan")
        stats["length_recovery_error_p90"] = float("nan")
    return stats


def _apply_channels(arr, name):
    if arr.ndim == 2:
        return _apply(arr, name)
    return np.stack([_apply(arr[i], name) for i in range(arr.shape[0])])


def _plate_record(row, graph, stats, expected_length):
    """Per-plate output a user can actually work with: trajectories in ORIGINAL pixels."""
    x0, y0, _, _ = row["roi_xyxy"]
    floor = 0.25 * (expected_length or 0.0)
    joints = [dict(id=j["id"], xy_roi=j["xy"],
                   xy_original=[j["xy"][0] + x0, j["xy"][1] + y0],
                   degree=j["degree"], type=j["type"],
                   ambiguous=bool(j["solution"]["ambiguous"]),
                   margin=j["solution"].get("margin"),
                   accepted_pairs=j["accepted_pairs"],
                   network_score=j.get("network_score"))
              for j in graph["joints"]]
    fibers = []
    for f in graph["fibers"]:
        pts = np.asarray(f["points_xy"], float) + [x0, y0]
        fibers.append(dict(id=f["id"], length_pixels=round(f["length_pixels"], 2),
                           closed=bool(f["closed"]),
                           substantial=bool(f["length_pixels"] >= floor),
                           points_xy_original=np.round(pts, 1).tolist()))
    return dict(
        id=row["id"], page=row["page"], split=row["split"],
        roi_xyxy=row["roi_xyxy"],
        coordinate_system="`*_original` are pixels of the image extracted from the PDF; "
                          "subtract roi_xyxy[:2] to get ROI pixels",
        crossings_heatmap_xy=graph.get("heatmap_crossings_xy", []),
        crossings_heatmap_xy_original=[[p[0] + x0, p[1] + y0]
                                       for p in graph.get("heatmap_crossings_xy", [])],
        joints=joints, fibers=fibers,
        statistics={k: v for k, v in stats.items() if k != "crossing_repeatability_counts"},
        warning="pixel units only, no physical scale bar. Trajectory count is NOT the "
                "number of fibers in the specimen, and a 2D crossing is not proof of "
                "physical contact or of which fiber lies on top.")


def run(real="data/real", checkpoint=None, out="results/real_eval", split="test",
        method="cfx", limit=None, expected_length=None, thresholds=None, tag=None,
        save_overlays=True):
    root = Path(real)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_json(root / "manifest.json")
            if split == "all" or r["split"] == split]
    if limit:
        rows = rows[:limit]
    model, device = (load_model(checkpoint) if checkpoint else (None, None))
    from .predict import overlay
    from .common import write_image

    records = []
    for i, r in enumerate(rows):
        gray = read_image(root / r["roi"], True)
        rec, graph, skeleton, bundle = evaluate_plate(
            gray, model, device, method, thresholds, expected_length)
        rec.update(id=r["id"], page=r["page"], split=r["split"])
        records.append(rec)
        if save_overlays:
            write_image(out / f"{r['id']}_overlay.jpg", overlay(gray, skeleton, graph, True))
            if bundle.get("joint") is not None:
                write_image(out / f"{r['id']}_joint.png",
                            (np.clip(bundle["joint"], 0, 1) * 255).astype(np.uint8))
            save_json(out / f"{r['id']}.json", _plate_record(r, graph, rec, expected_length))
        print(f"  real {i + 1}/{len(rows)} {r['id']}: sfar={rec['sfar']:.4f} "
              f"cov={rec['coverage']:.3f} eqe={rec['equivariance_mse']:.5f} "
              f"crep={rec['crossing_repeatability']:.3f} tracks={rec['tracks']}", flush=True)

    summary = dict(
        split=split, n=len(records), method=method, tag=tag,
        architecture=(model.config["arch"] if model is not None else "classical"),
        checkpoint=str(checkpoint) if checkpoint else None,
        sfar=bootstrap_ci([r["sfar"] for r in records]),
        coverage=bootstrap_ci([r["coverage"] for r in records]),
        equivariance_mse=bootstrap_ci([r["equivariance_mse"] for r in records]),
        crossing_repeatability=bootstrap_ci([r["crossing_repeatability"] for r in records]),
        length_cv=bootstrap_ci([r["length"]["cv"] for r in records]),
        length_p90=bootstrap_ci([r["length"].get("p90", float("nan")) for r in records]),
        length_recovery_error=bootstrap_ci([r["length_recovery_error"] for r in records]),
        length_recovery_error_p90=bootstrap_ci(
            [r["length_recovery_error_p90"] for r in records]),
        length_median=bootstrap_ci([r["length"]["median"] for r in records]),
        fragmentation=bootstrap_ci([r["fragmentation"] for r in records]),
        tracks=bootstrap_ci([r["tracks"] for r in records]),
        substantial_tracks=bootstrap_ci([r["substantial_tracks"] for r in records]),
        crossings=bootstrap_ci([r["crossings"] for r in records]),
        expected_length_px=expected_length,
        ground_truth="none; these are label-free consistency statistics, not accuracy",
        per_plate=records)
    save_json(out / "metrics.json", summary)
    _write_summary_csv(out / "summary.csv", records)
    return summary


def _write_summary_csv(path, records):
    """A plain per-plate table: what was found on each photograph."""
    import csv

    fields = ["id", "page", "split", "crossings", "graph_crossings", "unresolved_joints",
              "tracks", "substantial_tracks", "length_p90", "length_cv", "fragmentation",
              "sfar", "coverage", "equivariance_mse", "crossing_repeatability"]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for r in records:
            row = []
            for key in fields:
                if key == "length_p90":
                    row.append(round(r["length"].get("p90", float("nan")), 1))
                elif key == "length_cv":
                    row.append(round(r["length"].get("cv", float("nan")), 4))
                else:
                    value = r.get(key)
                    row.append(round(value, 5) if isinstance(value, float) else value)
            writer.writerow(row)
    print(f"  summary -> {path}")


# ----------------------------------------------------------------- optional human GT
SCHEMA = "cottoncross.crossing-ground-truth.v1"


def score_against_annotations(real, checkpoint, annotations, split="test", method="cfx",
                              tolerance=12, out=None):
    """Score crossing detection against a human-made file.

    Reads the file the review page exports.  A page that a human has not explicitly
    marked complete is refused rather than partially scored, and model output is never
    substituted for a missing annotation.
    """
    root = Path(real)
    data = load_json(annotations)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA!r}, found {data.get('schema')!r}")
    pages = {a["id"]: a for a in data["annotations"]}
    rows = [r for r in load_json(root / "manifest.json") if r["split"] == split]
    # Validate the annotation file before touching a checkpoint, so a refusal is about
    # the annotations and never gets mistaken for a missing-model error.
    annotated = [r for r in rows if r["id"] in pages]
    for r in annotated:
        if not pages[r["id"]].get("complete"):
            raise ValueError(f"page {r['id']} is not marked complete by a human annotator; "
                             "partial annotation cannot be scored")
    if not annotated:
        raise ValueError(f"no annotated pages in split {split!r}; nothing can be scored")
    model, device = load_model(checkpoint)
    counts = dict(tp=0, fp=0, fn=0, distance_sum=0.0)
    used, per_page = [], []
    for r in annotated:
        entry = pages[r["id"]]
        gray = read_image(root / r["roi"], True)
        bundle = predict(gray, model, device)
        _, _, graph = postprocess(bundle, gray, method=method)
        # Annotations live in original extracted-image pixels; predictions in ROI pixels.
        x0, y0, _, _ = r["roi_xyxy"]
        predicted = [[p[0] + x0, p[1] + y0] for p in graph["heatmap_crossings_xy"]]
        truth = entry.get("points_xy_original", [])
        c = point_counts(predicted, truth, tolerance)
        for k in counts:
            counts[k] += c[k]
        per_page.append(dict(id=r["id"], annotated=len(truth), predicted=len(predicted),
                             **{k: c[k] for k in ("tp", "fp", "fn")}))
        used.append(r["id"])
    summary = dict(point_scores(counts), pages=used, per_page=per_page,
                   tolerance_px=tolerance, split=split, source=str(annotations),
                   note="human-annotated crossing detection on real plates; the only "
                        "number in this project that is a real-image accuracy")
    if out:
        save_json(out, summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real", default="data/real")
    p.add_argument("--checkpoint")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--out", default="results/real_eval")
    p.add_argument("--split", default="test", choices=["all", "train", "val", "test"])
    p.add_argument("--method", default="cfx", choices=["greedy", "angle_joint", "cfx"])
    p.add_argument("--limit", type=int)
    p.add_argument("--expected-length", type=float)
    p.add_argument("--annotations")
    a = p.parse_args()
    if a.annotations:
        print(score_against_annotations(a.real, a.checkpoint, a.annotations, a.split, a.method))
    else:
        s = run(a.real, None if a.baseline else a.checkpoint, a.out, a.split, a.method,
                a.limit, a.expected_length)
        print({k: v for k, v in s.items() if k != "per_plate"})
