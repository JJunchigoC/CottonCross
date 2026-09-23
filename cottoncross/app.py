"""Count cotton fibers in new photographs - the application entry point.

    python -m cottoncross.app --input path/to/photos --out outputs/my_batch
    python -m cottoncross.app --input one_photo.jpg --roi none

For every image: crop the plate region (automatic, or `--roi none` for an already
cropped image), run the segmentation model, trace and link the fibers through their
crossings, drop interference that is far shorter than a staple, and count fibers with the
equal-staple rule  N = round(total fiber length / staple length).

Outputs in --out:
    counts.csv                    one row per image (count, crossings, lengths, review flag)
    <image>_count.jpg             overlay: each kept chain in its own colour, removed
                                  interference in grey, crossing candidates circled

The fiber count is the validated output.  `crossings` is the number of skeleton junctions
of degree >= 4 on counted fibers: a candidate list for review, not a validated count (it
also fires where a single fiber is tightly waved, see docs/development/crossing_count_rule.md).
    <image>.json                  trajectories in ORIGINAL image pixels

The staple length and debris fraction default to configs/counting.json, which the
counting benchmark writes for the selected model.  A different microscope or
magnification needs a different staple length: pass --staple, or count a few images by
hand and run scripts/count_fibers.py to re-select it.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from .common import load_json, read_image, save_json, write_image
from .count import close_gaps
from .predict import classical, load_model, postprocess, predict
from .prepare import automatic_roi

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEFAULT_CONFIG = "configs/counting.json"
# If total length / staple sits this far from an integer, the count is flagged for a look.
REVIEW_RESIDUAL = 0.35


def _images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_TYPES)


def count_image(gray, model, device, staple, debris_fraction, method="cfx"):
    """Count one (already cropped) grayscale image; returns the record and draw data."""
    if model is not None:
        bundle = predict(gray, model, device)
        thresholds = (dict(joint=model.crossing_threshold)
                      if getattr(model, "crossing_threshold", None) is not None else None)
    else:
        bundle, thresholds = classical(gray)[0], None
    _, _, graph = postprocess(bundle, gray, method=method, thresholds=thresholds)
    fibers = []
    for f in graph["fibers"]:
        pts = np.asarray(f["points_xy"], float)
        if len(pts) < 2:
            continue
        k = min(12, len(pts) - 1)
        head, tail = pts[0] - pts[k], pts[-1] - pts[-1 - k]
        fibers.append(dict(points=pts, length=float(f["length_pixels"]), closed=f["closed"],
                           head=pts[0], tail=pts[-1],
                           head_dir=head / max(np.linalg.norm(head), 1e-6),
                           tail_dir=tail / max(np.linalg.norm(tail), 1e-6)))
    chains = close_gaps(fibers) if fibers else []
    kept, removed = [], []
    for c in chains:
        length = sum(fibers[i]["length"] for i in c)
        (kept if length >= debris_fraction * staple else removed).append((length, c))
    total = sum(l for l, _ in kept)
    ratio = total / staple
    count = int(max(1, round(ratio))) if total > 0 else 0
    # A crossing, for the user, is where two of the COUNTED fibers meet: a skeleton
    # junction of degree >= 4 lying on a kept chain.  The raw crossing heatmap also fires
    # on tight bends and on interference, so it is not what is reported here.
    kept_pts = [fibers[i]["points"] for _, c in kept for i in c]
    kept_pts = np.concatenate(kept_pts) if kept_pts else np.zeros((0, 2))
    crossings = []
    for j in graph["joints"]:
        if j["degree"] < 4 or not len(kept_pts):
            continue
        if np.min(np.linalg.norm(kept_pts - np.asarray(j["xy"]), axis=1)) <= 10:
            crossings.append(j["xy"])
    record = dict(fibers=count, length_ratio=round(ratio, 3),
                  residual=round(ratio - count, 3),
                  review=bool(abs(ratio - count) > REVIEW_RESIDUAL or count == 0),
                  crossings=len(crossings), total_fiber_length_px=round(total, 1),
                  kept_chains=len(kept), interference_removed=len(removed),
                  interference_length_px=round(sum(l for l, _ in removed), 1))
    return record, dict(fibers=fibers, kept=kept, removed=removed, crossings=crossings)


def draw(gray, record, parts):
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image = cv2.addWeighted(image, 0.75, np.zeros_like(image), 0.25, 0)
    fibers = parts["fibers"]
    for _, chain in parts["removed"]:
        for i in chain:
            pts = np.rint(fibers[i]["points"]).astype(np.int32)
            cv2.polylines(image, [pts], False, (150, 150, 150), 1, cv2.LINE_AA)
    # Fixed, distinguishable colours for kept chains (BGR).
    palette = [(255, 144, 30), (60, 200, 255), (90, 220, 90), (200, 110, 255),
               (60, 90, 255), (255, 220, 80), (180, 180, 40), (240, 120, 180)]
    for n, (_, chain) in enumerate(sorted(parts["kept"], key=lambda t: -t[0])):
        colour = palette[n % len(palette)]
        for i in chain:
            pts = np.rint(fibers[i]["points"]).astype(np.int32)
            cv2.polylines(image, [pts], False, colour, 3, cv2.LINE_AA)
    for x, y in parts["crossings"]:
        cv2.circle(image, (int(round(x)), int(round(y))), 14, (40, 40, 255), 2, cv2.LINE_AA)
    text = f"fibers: {record['fibers']}   crossing candidates: {record['crossings']}"
    if record["review"]:
        text += "   [check]"
    scale = max(0.8, gray.shape[1] / 1100)
    cv2.putText(image, text, (20, int(48 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), int(6 * scale), cv2.LINE_AA)
    cv2.putText(image, text, (20, int(48 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), int(2 * scale), cv2.LINE_AA)
    return image


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True, help="an image file or a folder of images")
    p.add_argument("--out", default="outputs/count")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", help="model checkpoint; default from the config")
    p.add_argument("--classical", action="store_true", help="training-free baseline")
    p.add_argument("--staple", type=float, help="staple length in pixels (35 mm)")
    p.add_argument("--debris", type=float, help="debris cut-off as a fraction of a staple")
    p.add_argument("--roi", default="auto", choices=["auto", "none"])
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)

    cfg = load_json(a.config) if Path(a.config).exists() else {}
    staple = a.staple or cfg.get("staple_px")
    debris = a.debris if a.debris is not None else cfg.get("debris_fraction", 0.1)
    if not staple:
        raise SystemExit("no staple length: pass --staple or create configs/counting.json")
    checkpoint = None if a.classical else (a.checkpoint or cfg.get("checkpoint"))
    model, device = load_model(checkpoint, a.device) if checkpoint else (None, None)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in _images(a.input):
        colour = read_image(path)
        x0, y0, x1, y1 = (automatic_roi(colour) if a.roi == "auto"
                          else [0, 0, colour.shape[1], colour.shape[0]])
        gray = cv2.cvtColor(colour[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        record, parts = count_image(gray, model, device, staple, debris)
        record = dict(image=str(path), roi_xyxy=[x0, y0, x1, y1], **record)
        rows.append(record)
        write_image(out / f"{path.stem}_count.jpg", draw(gray, record, parts))
        fibers = parts["fibers"]
        save_json(out / f"{path.stem}.json", dict(
            record, staple_px=staple, debris_fraction=debris,
            model=checkpoint or "classical",
            fibers_xy_original=[[[float(x + x0), float(y + y0)] for x, y in
                                 np.concatenate([fibers[i]["points"] for i in chain])]
                                for _, chain in parts["kept"]],
            crossings_xy_original=[[float(x + x0), float(y + y0)]
                                   for x, y in parts["crossings"]]))
        flag = "  <- check" if record["review"] else ""
        print(f"{path.name}: {record['fibers']} fibers, {record['crossings']} crossing candidates "
              f"(length ratio {record['length_ratio']:.2f}){flag}", flush=True)
    if rows:
        keys = [k for k in rows[0] if k != "roi_xyxy"]
        with open(out / "counts.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n{len(rows)} images -> {out / 'counts.csv'}")
    return rows


if __name__ == "__main__":
    main()
