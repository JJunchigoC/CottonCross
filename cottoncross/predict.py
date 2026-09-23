"""Inference: tiled prediction, the training-free baseline, post-processing, overlays."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi

from .background import normalise, ridge_response
from .calibrate import fiber_mask
from pathlib import Path

from .common import load_json, starts
from .linking import analyze, heatmap_points
from .models import FiberModel, activate

DEFAULT_THRESHOLDS = dict(center=0.40, mask=0.45, joint=0.50)


def load_model(checkpoint, device="auto", weights="teacher"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    extra = dict(cfg.get("model") or {})
    # Checkpoints written before the OCA revision carry a scalar `prior_gain` instead of
    # the 1x1 prior convolution; infer the variant from the weights so an archived run
    # still loads without editing its config.
    if cfg["arch"] == "cfxnet" and "oca_version" not in extra:
        state = ck.get("teacher") or ck["model"]
        extra["oca_version"] = "v1" if "prior_gain" in state else "v2"
    model = FiberModel(cfg["arch"], cfg["base"], cfg["depth"], strip=cfg.get("strip", 11),
                       **extra).to(device)
    model.load_state_dict(ck[weights if weights in ck else "model"])
    model.eval()
    model.config = cfg
    # The crossing-heatmap threshold chosen for this model on the validation split, if
    # `tune_threshold.py` has been run.  Carrying it on the model keeps the delivered
    # per-plate outputs at the same operating point the tables report, instead of the
    # generic 0.5 default.
    selected = Path(checkpoint).parent / "crossing_threshold.json"
    model.crossing_threshold = (float(load_json(selected)["threshold"])
                                if selected.exists() else None)
    return model, device


@torch.no_grad()
def predict(gray, model, device, tile=384, stride=288, amp=True, mode=None):
    """Hann-blended tiled prediction; returns the activated bundle at full resolution."""
    mode = mode or getattr(model, "config", {}).get("input_mode", "bgn")
    h, w = gray.shape
    pad_h, pad_w = max(0, tile - h), max(0, tile - w)
    padded = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    hh, ww = padded.shape
    channels = None
    weight = np.zeros((hh, ww), np.float32)
    taper = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), 0.03).astype(np.float32)
    for y in starts(hh, tile, stride):
        for x in starts(ww, tile, stride):
            patch = padded[y:y + tile, x:x + tile]
            batch = torch.from_numpy(normalise(patch, mode)[None]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=amp and device != "cpu"):
                logits = model(batch)
            bundle = activate(logits.float())
            flat = torch.cat([bundle["mask"], bundle["center"], bundle["joint"],
                              bundle["moments"], bundle["overlap"], bundle["embed"]], 1)
            arr = flat[0].cpu().numpy()
            if channels is None:
                channels = np.zeros((arr.shape[0], hh, ww), np.float32)
            channels[:, y:y + tile, x:x + tile] += arr * taper
            weight[y:y + tile, x:x + tile] += taper
    channels = (channels / weight[None])[:, :h, :w]
    embed = channels[10:18]
    embed = embed / np.maximum(np.linalg.norm(embed, axis=0, keepdims=True), 1e-6)
    return dict(mask=channels[0], center=channels[1], joint=channels[2],
                moments=channels[3:7], overlap=channels[7:10], embed=embed)


def classical(gray, quantile=0.995):
    """Training-free multi-scale ridge detector used as the classical baseline."""
    response = ridge_response(gray)
    mask = fiber_mask(gray, quantile=quantile)
    peak = float(np.quantile(response, 0.999)) or 1.0
    return dict(mask=mask.astype(np.float32), center=np.clip(response / peak, 0, 1),
                joint=None, moments=None, overlap=None, embed=None), response


def binarise(bundle, thresholds=None, min_size=10):
    t = dict(DEFAULT_THRESHOLDS)
    t.update(thresholds or {})
    binary = (bundle["center"] > t["center"]) & (bundle["mask"] > t["mask"])
    labels, _ = ndi.label(binary, structure=np.ones((3, 3)))
    counts = np.bincount(labels.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[labels]


def postprocess(bundle, gray, method="cfx", thresholds=None, seed=42, min_size=10,
                margin_threshold=None):
    binary = binarise(bundle, thresholds, min_size)
    skeleton, graph = analyze(binary.astype(np.uint8), gray, bundle, method=method, seed=seed,
                              margin_threshold=margin_threshold)
    if bundle.get("joint") is not None:
        thr = (thresholds or DEFAULT_THRESHOLDS).get("joint", DEFAULT_THRESHOLDS["joint"])
        pts = heatmap_points(bundle["joint"], thr)
        graph["heatmap_crossings_xy"] = pts
        graph["heatmap_crossing_scores"] = [
            float(bundle["joint"][int(round(p[1])), int(round(p[0]))]) for p in pts]
    else:
        graph["heatmap_crossings_xy"] = []
        graph["heatmap_crossing_scores"] = []
    graph["thresholds"] = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    graph["binary_mask_fraction"] = float(binary.mean())
    return binary, skeleton, graph


def substrate_mask(gray, distance=30, quantile=0.995, minimum_fraction=0.05):
    """Detector-independent 'definitely substrate' region.

    Any pixel at least `distance` away from every pixel that clears a deliberately
    permissive classical ridge threshold.  The threshold passes roughly three times more
    pixels than the plates actually contain fiber, so the region excludes anything that
    could plausibly be a fiber.  It uses no learned model, so the false-alarm rate it
    defines is comparable across methods.
    """
    response = ridge_response(gray)
    permissive = response > max(float(np.quantile(response, quantile)), 1e-8)
    field = ndi.distance_transform_edt(~permissive)
    mask = field >= distance
    while mask.mean() < minimum_fraction and distance > 6:
        distance = max(6, distance // 2)
        mask = field >= distance
    return mask


def overlay(gray, skeleton, graph, draw_heatmap_points=False):
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image[skeleton > 0] = (40, 220, 60)
    for fiber in graph["fibers"]:
        rng = np.random.default_rng(fiber["id"] + 911)
        colour = tuple(int(v) for v in rng.integers(70, 255, 3))
        pts = np.rint(np.asarray(fiber["points_xy"], float)).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(image, [pts], bool(fiber["closed"]), colour, 2, cv2.LINE_AA)
    for j in graph["joints"]:
        p = tuple(int(round(v)) for v in j["xy"])
        ambiguous = bool(j["solution"]["ambiguous"])
        colour = (0, 165, 255) if ambiguous else (40, 40, 255)
        cv2.circle(image, p, 9, colour, 2, cv2.LINE_AA)
    if draw_heatmap_points:
        for p in graph.get("heatmap_crossings_xy", []):
            cv2.drawMarker(image, (int(round(p[0])), int(round(p[1]))), (255, 220, 40),
                           cv2.MARKER_TILTED_CROSS, 12, 1, cv2.LINE_AA)
    return image
