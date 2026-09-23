"""Held-out evaluation on the synthetic corpus.

Everything reported here is measured on canvases whose seeds were never used for
training or threshold selection.  It quantifies performance on the modelled
distribution; it is not a measurement of accuracy on the real plates, for which no
human ground truth exists.  Real-plate behaviour is covered by `evaluate_real`.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import load_json, read_image, save_json, write_image
from .metrics import (assd, betti0_error, bootstrap_ci, centerline_f1, cldice,
                      heatmap_average_precision, instance_scores, length_consistency,
                      overlap_scores, pair_accuracy, point_counts, point_scores)
from .predict import classical, load_model, overlay, postprocess, predict

PIXEL_KEYS = ["dice", "iou", "precision", "recall", "cldice", "centerline_f1_2px",
              "centerline_f1_3px", "assd_px", "overlap_dice"]


def _gt(root, row):
    with np.load(Path(root) / row["target"]) as z:
        instances = z["instances"]
        center = np.max(instances, axis=0) if len(instances) else np.zeros_like(z["mask"])
        return dict(mask=z["mask"] > 0, center=center > 0, instances=instances.copy(),
                    points=z["points"].reshape(-1, 2).copy(),
                    complete=z["points_complete"].reshape(-1).astype(bool).copy(),
                    overlap=z["overlap"].copy(), instance_id=z["instance_id"].copy())


def evaluate_split(checkpoint, data="data/synth", split="test", method="cfx",
                   out="results/eval", limit=None, samples=12, thresholds=None,
                   tolerance=6, expected_length=None, device="auto", tag=None,
                   margin_threshold=None):
    root = Path(data)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_json(root / "manifest.json") if r["split"] == split]
    if limit:
        rows = rows[:limit]
    model, dev = (load_model(checkpoint, device) if checkpoint else (None, None))
    records, curve_records = [], []
    inference_ms = []

    for i, r in enumerate(rows):
        gray = read_image(root / r["image"], True)
        truth = _gt(root, r)
        began = time.time()
        if model is not None:
            bundle = predict(gray, model, dev, tile=min(384, r["size"]),
                             stride=min(288, r["size"]))
        else:
            bundle, _ = classical(gray)
        inference_ms.append((time.time() - began) * 1000)
        binary, skeleton, graph = postprocess(bundle, gray, method=method,
                                              thresholds=thresholds,
                                              margin_threshold=margin_threshold)
        pred_mask = bundle["mask"] > 0.5
        b0_err, b0_pred, b0_true = betti0_error(skeleton, truth["center"])
        rec = dict(id=r["id"], case=r["case"], size=r["size"],
                   **overlap_scores(pred_mask, truth["mask"]),
                   cldice=cldice(pred_mask, truth["mask"]),
                   centerline_f1_2px=centerline_f1(skeleton, truth["center"], 2),
                   centerline_f1_3px=centerline_f1(skeleton, truth["center"], 3),
                   assd_px=assd(skeleton, truth["center"]),
                   betti0_error=b0_err, betti0_pred=b0_pred, betti0_true=b0_true)

        if bundle.get("overlap") is not None:
            pred2 = np.argmax(bundle["overlap"], axis=0) == 2
            rec["overlap_dice"] = overlap_scores(pred2, truth["overlap"] >= 2)["dice"]
        else:
            rec["overlap_dice"] = float("nan")

        heat_pts = graph.get("heatmap_crossings_xy", [])
        heat_scores = graph.get("heatmap_crossing_scores", [])
        graph_pts = [j["xy"] for j in graph["joints"] if j["degree"] >= 4]
        rec["heatmap_points"] = point_counts(heat_pts, truth["points"], tolerance)
        rec["graph_points"] = point_counts(graph_pts, truth["points"], tolerance)
        rec["complete_points"] = point_counts(heat_pts, truth["points"][truth["complete"]],
                                              tolerance)
        rec["instances"] = instance_scores(graph["fibers"], truth["instances"])
        rec["pairing"] = pair_accuracy(graph, truth["instance_id"])
        rec["length"] = length_consistency(graph["fibers"], expected_length)
        rec["true_crossings"] = int(len(truth["points"]))
        rec["complete_crossings"] = int(truth["complete"].sum())
        rec["predicted_tracks"] = len(graph["fibers"])
        rec["unresolved_joints"] = graph["unresolved_joints"]
        # Keep the raw peaks, their scores and the truth points so any threshold can be
        # applied after the fact; the table and the significance test must be able to use
        # the same validation-selected threshold without another inference pass.
        rec["crossing_peaks"] = dict(
            points=[[round(float(v), 2) for v in p] for p in heat_pts],
            scores=[round(float(s), 4) for s in heat_scores],
            truth=[[round(float(v), 2) for v in p] for p in truth["points"]])
        records.append(rec)
        curve_records.append(dict(points=heat_pts, scores=heat_scores,
                                  truth=truth["points"]))

        if i < samples:
            base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            gt = base.copy()
            gt[truth["center"]] = (40, 230, 60)
            for p in truth["points"]:
                cv2.circle(gt, (int(round(p[0])), int(round(p[1]))), 8, (40, 40, 255), 1,
                           cv2.LINE_AA)
            write_image(out / f"sample_{i:03d}.jpg",
                        np.hstack([base, gt, overlay(gray, skeleton, graph, True)]))
        if (i + 1) % 100 == 0:
            print(f"  eval {split} {i + 1}/{len(rows)}", flush=True)

    summary = _aggregate(records, curve_records, tolerance)
    summary.update(split=split, n=len(records), method=method, tag=tag,
                   checkpoint=str(checkpoint) if checkpoint else None,
                   architecture=(model.config["arch"] if model is not None else "classical"),
                   inference_ms_mean=float(np.mean(inference_ms)),
                   inference_ms_median=float(np.median(inference_ms)),
                   point_tolerance_px=tolerance,
                   linker=load_json("configs/linker.json") if Path("configs/linker.json").exists() else None,
                   margin_threshold_override=margin_threshold,
                   domain="held-out procedural synthetic canvases; NOT real-plate accuracy")
    save_json(out / "per_image.json", records)
    save_json(out / "metrics.json", summary)
    return summary


def _aggregate(records, curve_records, tolerance):
    summary = {"pixel": {}}
    for key in PIXEL_KEYS:
        summary["pixel"][key] = bootstrap_ci([r[key] for r in records])
    summary["pixel"]["betti0_error"] = bootstrap_ci([r["betti0_error"] for r in records])

    for key in ("heatmap_points", "graph_points", "complete_points"):
        counts = {k: sum(r[key][k] for r in records) for k in ("tp", "fp", "fn")}
        counts["distance_sum"] = sum(r[key]["distance_sum"] for r in records)
        summary[key] = point_scores(counts)
    # `complete_points` matches ALL predictions against only the verified-complete subset
    # of the ground truth, so a prediction that correctly found a clipped crossing is
    # counted as a false positive there.  Only its recall is meaningful: what fraction of
    # complete crossings the detector finds.  Precision and F1 are kept for completeness
    # but must not be quoted.
    summary["complete_points"]["reading"] = ("quote recall only; precision is depressed by "
                                             "correct detections of clipped crossings")

    counts = {k: sum(r["instances"][k] for r in records) for k in ("tp", "fp", "fn")}
    summary["instances"] = dict(
        point_scores(counts),
        mean_matched_f1=float(np.mean([r["instances"]["mean_matched_f1"] for r in records])),
        definition="one-to-one trajectory match at centerline F1 >= 0.5, tolerance 2 px")

    ap, curve = heatmap_average_precision(curve_records, tolerance)
    summary["heatmap_ap"] = ap
    summary["heatmap_pr_curve"] = curve

    committed = sum(r["pairing"]["committed_pairs"] for r in records)
    summary["pairing"] = dict(
        pair_accuracy=(sum(r["pairing"]["correct_pairs"] for r in records) / committed
                       if committed else float("nan")),
        committed_pairs=committed,
        abstention_rate=float(np.mean([r["pairing"]["abstention_rate"] for r in records])),
        definition="fraction of the crossing pairings the linker committed to that join "
                   "two ports of the same ground-truth fiber; read with abstention_rate")

    lengths = [r["length"]["cv"] for r in records if np.isfinite(r["length"]["cv"])]
    summary["length_cv"] = bootstrap_ci(lengths)
    rel = [r["length"].get("relative_error", float("nan")) for r in records]
    summary["length_relative_error"] = bootstrap_ci(rel)

    cases = sorted({r["case"] for r in records})
    summary["by_case"] = {
        c: dict(n=sum(r["case"] == c for r in records),
                dice=float(np.mean([r["dice"] for r in records if r["case"] == c])),
                cldice=float(np.mean([r["cldice"] for r in records if r["case"] == c])),
                crossing_f1=point_scores({
                    k: sum(r["heatmap_points"][k] for r in records if r["case"] == c)
                    for k in ("tp", "fp", "fn")})["f1"])
        for c in cases}
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--data", default="data/synth")
    p.add_argument("--split", default="test", choices=["val", "test", "plate"])
    p.add_argument("--method", default="cfx", choices=["greedy", "angle_joint", "cfx"])
    p.add_argument("--out", default="results/eval")
    p.add_argument("--limit", type=int)
    p.add_argument("--expected-length", type=float)
    a = p.parse_args()
    s = evaluate_split(None if a.baseline else a.checkpoint, a.data, a.split, a.method,
                       a.out, a.limit, expected_length=a.expected_length)
    print({k: v for k, v in s.items() if k not in ("heatmap_pr_curve", "by_case")})
