"""Synthetic fiber-counting benchmark with exact ground truth.

The real plates have no human count, so the counting claim is backed by a benchmark whose
answer is known by construction - the same label-first principle as the training data:
decide the fibers first, then render the image.

Each canvas is plate-sized (1536 px, the scale of a real ROI) and contains

* N complete fibers of exactly one staple of arc length (N = 1..5, at least one crossing
  whenever N >= 2) - the only things that count;
* the interference the real photographs show, none of which counts:
  - short fiber-like fragments (lint, broken ends; well under a staple),
  - dust motes and bright specks,
  - faded stretches along the true fibers (poor focus / low contrast), which break a
    fiber in the mask without changing how many fibers there are;
* a real substrate carrier from the training plates, as in the training corpus.

Canvases are generated from fixed seeds disjoint from the training corpus.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .common import save_json, write_image
from .synthgen import _carrier_crop, carrier_library, crossing_records, render, worm_chain

COUNT_P = {1: 0.10, 2: 0.30, 3: 0.30, 4: 0.20, 5: 0.10}


def _inside_fraction(curve, size, margin):
    ok = ((curve[:, 0] >= margin) & (curve[:, 0] < size - margin)
          & (curve[:, 1] >= margin) & (curve[:, 1] < size - margin))
    return float(ok.mean())


def fiber_layout(rng, size, n, staple, margin=40, min_inside=0.97, tries=400):
    """n staple-length fibers that stay inside the canvas, with a crossing if n >= 2."""
    for _ in range(tries):
        curves = []
        persistence = float(rng.uniform(250.0, 900.0))
        attempts = 0
        while len(curves) < n and attempts < 200:
            attempts += 1
            start = rng.uniform(0.15, 0.85, 2) * size
            c = worm_chain(rng, start, rng.uniform(0, 2 * np.pi), staple, persistence)
            if _inside_fraction(c, size, margin) >= min_inside:
                curves.append(c)
        if len(curves) < n:
            continue
        records = crossing_records(curves, size, 40.0, margin)
        if n == 1 or records:
            return curves, records
    raise RuntimeError("could not place fibers")


def add_debris(rng, image, size, staple, count):
    """Short fiber-like fragments that are interference, never fibers."""
    out = image.astype(np.float32) / 255.0
    signal = np.zeros_like(out)
    for _ in range(count):
        length = float(rng.uniform(0.015, 0.14)) * staple
        c = worm_chain(rng, rng.uniform(0, 1, 2) * size, rng.uniform(0, 2 * np.pi), length,
                       float(rng.uniform(15.0, 200.0)))
        pts = np.rint(c).astype(np.int32)
        tube = np.zeros((size, size), np.uint8)
        cv2.polylines(tube, [pts], False, 1, int(rng.integers(1, 5)), cv2.LINE_8)
        signal = np.maximum(signal, float(rng.uniform(0.10, 0.45))
                            * gaussian_filter(tube.astype(np.float32), rng.uniform(0.5, 1.4)))
    # Bright specks and blobs, larger and brighter than the training-corpus dust.
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    for _ in range(int(rng.integers(20, 60))):
        px, py = rng.integers(0, size, 2)
        r2 = rng.uniform(2.0, 30.0)
        box = slice(max(0, py - 20), py + 20), slice(max(0, px - 20), px + 20)
        signal[box] = np.maximum(signal[box], rng.uniform(0.05, 0.35) * np.exp(
            -((xx[box] - px) ** 2 + (yy[box] - py) ** 2) / r2))
    return (np.clip(out + signal, 0, 1) * 255).astype(np.uint8)


def fade_stretches(rng, image, curves, count):
    """Low-contrast stretches along true fibers: the fiber is still there, faintly."""
    background = cv2.medianBlur(image, 21)
    out = image.astype(np.float32)
    for _ in range(count):
        c = curves[int(rng.integers(len(curves)))]
        k = int(rng.integers(len(c) // 10, len(c) - len(c) // 10))
        span = int(rng.integers(4, 14))  # 8-28 px of arc at step 2
        seg = np.rint(c[max(0, k - span):k + span]).astype(np.int32)
        m = np.zeros(image.shape, np.uint8)
        cv2.polylines(m, [seg], False, 1, 9, cv2.LINE_8)
        m = gaussian_filter(m.astype(np.float32), 2.0)
        keep = float(rng.uniform(0.0, 0.35))
        out = out * (1 - m * (1 - keep)) + background * m * (1 - keep)
    return np.clip(out, 0, 255).astype(np.uint8)


def carrier_mosaic(rng, library, size):
    """Plate-sized substrate: real grain from training tiles over one smooth illumination.

    Tiles differ in mean brightness, so blending them directly leaves a patchwork.  Only
    their high-frequency grain is mosaicked; the low-frequency level is a single smooth
    field, as on a real plate.
    """
    unit = min(library[0].shape[:2])
    acc = np.zeros((size, size), np.float32)
    weight = np.zeros((size, size), np.float32)
    taper = np.outer(np.hanning(unit), np.hanning(unit)).astype(np.float32) + 0.05
    levels = []
    for y in range(0, size, unit // 2):
        for x in range(0, size, unit // 2):
            y0, x0 = min(y, size - unit), min(x, size - unit)
            crop = _carrier_crop(rng, library, unit)
            levels.append(float(np.median(crop)))
            acc[y0:y0 + unit, x0:x0 + unit] += (crop - cv2.GaussianBlur(crop, (0, 0), 24)) * taper
            weight[y0:y0 + unit, x0:x0 + unit] += taper
    grain = acc / np.maximum(weight, 1e-6)
    seed = rng.normal(0, 1, (4, 4)).astype(np.float32)
    field = cv2.resize(seed, (size, size), interpolation=cv2.INTER_CUBIC)
    level = float(np.median(levels)) + field * rng.uniform(3.0, 10.0)
    return np.clip(level + grain, 0, 255).astype(np.uint8)


def generate(real="data/real", out="data/count_bench", size=1536, staple=1300.0,
             counts=(40, 160), seed=31_000_000, carrier_probability=0.85):
    out = Path(out)
    library = carrier_library(real, 512)
    rows = []
    for split, n, offset in (("val", counts[0], 0), ("test", counts[1], 500_000)):
        (out / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            s = seed + offset + i
            rng = np.random.default_rng(s)
            n_fibers = int(rng.choice(list(COUNT_P), p=list(COUNT_P.values())))
            curves, records = fiber_layout(rng, size, n_fibers, staple)
            carrier = (carrier_mosaic(rng, library, size)
                       if rng.random() < carrier_probability else None)
            image, _ = render(rng, curves, size, carrier)
            n_debris = int(rng.integers(15, 70))
            n_fades = int(rng.integers(0, 3 * n_fibers + 1))
            image = fade_stretches(rng, image, curves, n_fades)
            image = add_debris(rng, image, size, staple, n_debris)
            stem = f"{split}/{i:04d}"
            write_image(out / f"{stem}.png", image)
            rows.append(dict(id=stem, split=split, image=f"{stem}.png", seed=s,
                             fibers=n_fibers, crossings=len(records),
                             debris_fragments=n_debris, faded_stretches=n_fades))
            save_json(out / f"{stem}.json", dict(
                seed=s, fibers=n_fibers, staple_pixels=staple,
                curves_xy=[np.round(c, 2).tolist() for c in curves],
                crossings=records, debris_fragments=n_debris, faded_stretches=n_fades))
            if (i + 1) % 20 == 0:
                print(f"count bench {split} {i + 1}/{n}", flush=True)
    save_json(out / "manifest.json", rows)
    save_json(out / "provenance.json", dict(
        purpose="per-image fiber counting with exact ground truth",
        generation_order="fibers first, image rendered from them",
        canvas=size, staple_pixels=staple, count_distribution=COUNT_P,
        interference=["short fiber-like fragments 1.5-14% of a staple",
                      "dust motes and specks", "faded stretches along true fibers"],
        background_carrier_split="train only", seed=seed, counts=list(counts)))
    return rows


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/count_bench")
    p.add_argument("--staple", type=float, default=1300.0)
    a = p.parse_args()
    generate(out=a.out, staple=a.staple)
