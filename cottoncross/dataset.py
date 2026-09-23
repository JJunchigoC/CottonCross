"""Datasets and augmentation.

Synthetic pairs carry exact labels; real plates carry none and are used only through
the unlabelled consistency branch.  Both go through `background.normalise`, so the
network sees one photometric convention.

Geometric augmentation is label-consistent: a rotation or flip also transforms the
orientation moments, because those channels describe a direction, not an intensity.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .background import normalise
from .common import load_json, read_image
from .losses import TARGET_CHANNELS


def photometric(x, rng, strong=False):
    """Intensity-only jitter, applied before background normalisation."""
    x = x.astype(np.float32) / 255.0
    x = np.clip(x * rng.uniform(0.82, 1.20) + rng.uniform(-0.05, 0.05), 0, 1)
    if strong:
        x = x ** rng.uniform(0.7, 1.45)
        x = x + rng.normal(0, rng.uniform(0, 0.022), x.shape)
        if rng.random() < 0.35:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 0.9))
        if rng.random() < 0.25:
            # A slow shading ramp, the residual the flat-field step never removes exactly.
            yy, xx = np.mgrid[:x.shape[0], :x.shape[1]].astype(np.float32)
            x = x + rng.uniform(-0.08, 0.08) * xx / x.shape[1] + rng.uniform(-0.08, 0.08) * yy / x.shape[0]
    return np.clip(x, 0, 1).astype(np.float32) * 255.0


def geometric(image, target, rng):
    """Random dihedral transform applied jointly to image and every label channel."""
    k = int(rng.integers(4))
    image = np.rot90(image, k)
    target = np.rot90(target, k, axes=(-2, -1)).copy()
    # theta -> theta - k*pi/2:  cos2/sin2 flip sign each quarter turn, 4th order is fixed.
    target[3:5] *= (-1.0) ** k
    if rng.random() < 0.5:
        image = np.fliplr(image)
        target = np.flip(target, axis=-1).copy()
        target[[4, 6]] *= -1.0          # theta -> pi - theta
    return np.ascontiguousarray(image), np.ascontiguousarray(target)


def pack_target(npz, size):
    """Stack the stored labels into the (10, H, W) float tensor the losses expect."""
    out = np.zeros((TARGET_CHANNELS, size, size), np.float32)
    out[0] = npz["mask"].astype(np.float32)
    out[1] = npz["center"].astype(np.float32) / 255.0
    out[2] = npz["joint"].astype(np.float32) / 255.0
    out[3:7] = npz["moments"].astype(np.float32) / 127.0
    out[7] = npz["orientation_valid"].astype(np.float32)
    out[8] = np.clip(npz["overlap"], 0, 2).astype(np.float32)
    out[9] = npz["instance_id"].astype(np.float32)
    return out


class SynthDataset(Dataset):
    """Synthetic canvases, optionally cropped around a verified complete crossing.

    `crop_mode="crossing"` keeps the guarantee that holds at canvas level - every
    training image contains a whole crossing - after cropping: the window is centred on
    a crossing tagged complete, jittered only as far as the arm length allows, so all
    four arms stay inside.  Canvases with no complete crossing (the deliberate `empty`
    and `near_miss` negatives) fall back to a uniform random crop.
    """

    def __init__(self, root, split="train", augment=True, seed=42, crop=None, mode="bgn",
                 crop_mode="crossing", arm_margin=1.15):
        self.root = Path(root)
        self.rows = [r for r in load_json(self.root / "manifest.json") if r["split"] == split]
        if not self.rows:
            raise ValueError(f"synthetic split {split!r} is empty at {root}")
        self.augment = augment
        self.crop = crop
        self.seed = seed
        self.mode = mode
        self.crop_mode = crop_mode
        self.arm_margin = arm_margin

    def __len__(self):
        return len(self.rows)

    def _crop_origin(self, row, rng, points):
        size, crop = row["size"], self.crop
        if self.crop_mode != "crossing" or not len(points):
            v = rng.integers(0, size - crop + 1, 2)
            return int(v[0]), int(v[1])
        cx, cy = points[int(rng.integers(len(points)))]
        # Keep the crossing at least `arm` pixels from every crop border.
        arm = min(crop // 2 - 4, int(56 * self.arm_margin))
        lo_x = int(np.clip(cx - crop + arm, 0, size - crop))
        hi_x = int(np.clip(cx - arm, 0, size - crop))
        lo_y = int(np.clip(cy - crop + arm, 0, size - crop))
        hi_y = int(np.clip(cy - arm, 0, size - crop))
        x = int(rng.integers(min(lo_x, hi_x), max(lo_x, hi_x) + 1))
        y = int(rng.integers(min(lo_y, hi_y), max(lo_y, hi_y) + 1))
        return y, x

    def __getitem__(self, i):
        r = self.rows[i]
        rng = np.random.default_rng((self.seed * 1_000_003 + r["seed"] * 7 + i) % (2 ** 63))
        image = read_image(self.root / r["image"], True)
        with np.load(self.root / r["target"]) as z:
            target = pack_target(z, r["size"])
            points = z["points"].reshape(-1, 2)
            complete = z["points_complete"].reshape(-1).astype(bool)
        if self.crop and r["size"] > self.crop:
            y, x = self._crop_origin(r, rng, points[complete])
            image = image[y:y + self.crop, x:x + self.crop]
            target = target[:, y:y + self.crop, x:x + self.crop]
        if self.augment:
            image, target = geometric(image, target, rng)
            image = photometric(image, rng, strong=True)
        return torch.from_numpy(normalise(image, self.mode)), torch.from_numpy(target)


class RealCrops:
    """Unlabelled real crops, at least half of them centred on a complete crossing.

    `crossing_windows.json` is produced by `realprep` with the training-free detector;
    it contains no human annotation, only sampling locations.
    """

    def __init__(self, root, size=384, seed=42, crossing_fraction=0.6):
        self.root = Path(root)
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.crossing_fraction = crossing_fraction
        rows = [r for r in load_json(self.root / "manifest.json") if r["split"] == "train"]
        self.plates = {r["id"]: read_image(self.root / r["roi"], True) for r in rows}
        windows = load_json(self.root / "crossing_windows.json")
        self.windows = [w for w in windows["windows"] if w["id"] in self.plates]
        if not self.windows:
            raise ValueError("no cached crossing windows for the training split")
        tiles = [r for r in load_json(self.root / "tiles.json") if r["split"] == "train"]
        scores = np.array([r["signal_score"] for r in tiles], np.float64)
        scores = np.clip(scores, 1e-4, 0.05)
        # 30% uniform keeps empty substrate in the batch as an explicit negative.
        self.tile_weights = 0.3 / len(scores) + 0.7 * scores / scores.sum()
        self.tiles = tiles

    def _crop_at(self, plate, cy, cx):
        half = self.size // 2
        h, w = plate.shape
        y = int(np.clip(cy - half, 0, max(h - self.size, 0)))
        x = int(np.clip(cx - half, 0, max(w - self.size, 0)))
        crop = plate[y:y + self.size, x:x + self.size]
        if crop.shape != (self.size, self.size):
            crop = cv2.copyMakeBorder(crop, 0, self.size - crop.shape[0], 0,
                                      self.size - crop.shape[1], cv2.BORDER_REFLECT_101)
        return crop

    def sample(self, batch):
        out = []
        for _ in range(batch):
            if self.rng.random() < self.crossing_fraction:
                w = self.windows[int(self.rng.integers(len(self.windows)))]
                x0, y0, x1, y1 = w["xyxy"]
                cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
                jitter = int(self.size * 0.12)
                cy += int(self.rng.integers(-jitter, jitter + 1))
                cx += int(self.rng.integers(-jitter, jitter + 1))
                crop = self._crop_at(self.plates[w["id"]], cy, cx)
            else:
                t = self.tiles[int(self.rng.choice(len(self.tiles), p=self.tile_weights))]
                plate = self.plates[t["source_id"]]
                x0, y0, _, _ = t["xyxy_original"]
                roi_x, roi_y = self.roi_origin(t["source_id"])
                cy = y0 - roi_y + 256
                cx = x0 - roi_x + 256
                crop = self._crop_at(plate, cy, cx)
            crop = np.rot90(crop, int(self.rng.integers(4)))
            if self.rng.random() < 0.5:
                crop = np.fliplr(crop)
            out.append(np.ascontiguousarray(crop))
        return out

    def roi_origin(self, plate_id):
        if not hasattr(self, "_origins"):
            self._origins = {r["id"]: (r["roi_xyxy"][0], r["roi_xyxy"][1])
                             for r in load_json(self.root / "manifest.json")}
        return self._origins[plate_id]


def real_views(images, rng, teacher_joint=None, use_mask=True, mode="bgn"):
    """Weak view for the teacher, strong + context-masked view for the student.

    Masking a window around the most confident crossing forces the student to recover
    the junction from the surrounding fiber context rather than from local texture
    (masked-image consistency, Hoyer et al., CVPR 2023).
    """
    views = []
    for i, x in enumerate(images):
        w = photometric(x, rng, strong=False)
        s = photometric(x, rng, strong=True)
        if use_mask and rng.random() < 0.7:
            h, wd = s.shape
            if teacher_joint is not None and teacher_joint[i].max() > 0.35:
                cy, cx = np.unravel_index(int(teacher_joint[i].argmax()), teacher_joint[i].shape)
            else:
                cy = int(rng.integers(32, h - 32))
                cx = int(rng.integers(32, wd - 32))
            half = int(rng.integers(10, 28))
            s[max(0, cy - half):cy + half, max(0, cx - half):cx + half] = float(np.median(s))
        views.append((w, s))
    # Every random draw above happened in order on the caller's generator; the transform
    # below is deterministic, so running it on a thread pool changes speed only.
    # OpenCV releases the GIL, and this step otherwise dominates the adaptation step.
    with ThreadPoolExecutor(max_workers=4) as pool:
        done = list(pool.map(lambda v: (normalise(v[0], mode), normalise(v[1], mode)), views))
    weak = np.stack([d[0] for d in done])
    strong = np.stack([d[1] for d in done])
    return torch.from_numpy(weak), torch.from_numpy(strong)
