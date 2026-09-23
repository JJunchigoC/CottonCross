"""Background normalisation (BGN) for cotton-fiber micrographs.

The raw plates contain three background nuisances that a naive detector reports as
fibers: (1) a slowly varying illumination field, (2) a strongly textured, high-variance
speckle region (visible as the bright grainy band in the lower half of every plate) and
(3) dark device bezels at the borders.

`normalise` turns a raw grayscale plate into a 3-channel representation in which those
three nuisances are explicitly factored out.  The SAME function is applied to synthetic
and to real images, so the network never has to bridge an illumination/contrast gap that
we can remove analytically.

Channels
--------
0  flattened   : illumination-corrected intensity, unit local dynamic range
1  snr         : fine detail divided by the LOCAL noise scale.  Grain regions have a
                 large local noise scale, so their response is divided down; a thin
                 fiber on a smooth substrate keeps a large response.
2  coherence   : structure-tensor anisotropy of the fine detail.  Elongated ridges give
                 values near 1, isotropic speckle gives values near 0.

`validity` returns a boolean map of pixels that may contain fiber evidence at all
(not device bezel, not saturated).  Everything outside it is forced to background.
"""
from __future__ import annotations

import cv2
import numpy as np

# Scales are expressed in native ROI pixels.  A cotton fiber here is ~3-4 px wide.
FINE_SIGMA = 1.6          # matched to the fiber cross-section
ILLUM_SIGMA = 24.0        # large enough to contain no fiber energy
NOISE_GRID = 32           # local noise scale estimated on a 32 px grid
TENSOR_SIGMA = 3.5        # structure-tensor integration scale


def _as_float(gray: np.ndarray) -> np.ndarray:
    x = gray.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return x


def illumination(x: np.ndarray) -> np.ndarray:
    """Large-scale illumination field.

    A morphological opening with a disk far wider than a fiber removes the bright thin
    structures before the low-pass, so fibers do not leak into (and get subtracted by)
    their own background estimate.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    opened = cv2.morphologyEx(x, cv2.MORPH_OPEN, k)
    return cv2.GaussianBlur(opened, (0, 0), ILLUM_SIGMA)


def noise_scale(detail: np.ndarray, grid: int = NOISE_GRID) -> np.ndarray:
    """Per-pixel noise scale from a robust local MAD, estimated on a coarse grid.

    The grainy substrate and the smooth substrate differ by roughly an order of
    magnitude here; dividing by this map is what stops the grain from being reported as
    thousands of tiny fibers.
    """
    h, w = detail.shape
    gh, gw = max(1, h // grid), max(1, w // grid)
    small = cv2.resize(np.abs(detail), (gw, gh), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 3) if min(gh, gw) >= 3 else small
    sigma = 1.4826 * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    floor = max(float(np.median(sigma)) * 0.25, 1e-3)
    return np.maximum(sigma, floor)


def coherence(detail: np.ndarray, sigma: float = TENSOR_SIGMA) -> np.ndarray:
    """Structure-tensor anisotropy in [0, 1]; high on elongated ridges only."""
    gx = cv2.Sobel(detail, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(detail, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    delta = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    trace = jxx + jyy
    return (delta / (trace + 1e-8)).astype(np.float32)


def validity(gray: np.ndarray) -> np.ndarray:
    """Pixels that can carry fiber evidence: not a dark bezel, not saturated."""
    x = _as_float(gray)
    illum = illumination(x)
    return ((illum > 0.10) & (x < 0.995)).astype(np.uint8)


def naive(gray: np.ndarray) -> np.ndarray:
    """Ablation input: plain local-contrast channels, no noise scaling, no coherence.

    This is what a pipeline looks like when the substrate is not modelled: a global
    high-pass keeps the illumination gradient at bay but leaves the grainy region with
    the same contrast as a real fiber.
    """
    x = _as_float(gray)
    background = cv2.GaussianBlur(x, (0, 0), 12.0)
    residual = x - background
    scale = max(float(np.std(residual)), 0.008)
    local = np.clip(residual / (4 * scale), -1, 1)
    fine = np.clip((x - cv2.GaussianBlur(x, (0, 0), 2.0)) / (4 * scale), -1, 1)
    return np.stack([x, local, fine]).astype(np.float32)


def normalise(gray: np.ndarray, mode: str = "bgn") -> np.ndarray:
    """Return the 3-channel background-normalised representation, float32 (3, H, W)."""
    if mode == "raw":
        return naive(gray)
    if mode != "bgn":
        raise ValueError(f"unknown input mode {mode!r}")
    x = _as_float(gray)
    illum = illumination(x)
    flat = x - illum

    detail = flat - cv2.GaussianBlur(flat, (0, 0), 4.0 * FINE_SIGMA)
    sigma = noise_scale(detail)
    snr = np.clip(detail / (4.0 * sigma), -1.0, 1.0)

    coh = coherence(detail)
    # Suppress coherence where there is no signal at all, otherwise pure noise
    # directions produce spurious anisotropy.
    coh = coh * np.clip(np.abs(snr) * 2.0, 0.0, 1.0)

    span = max(float(np.quantile(np.abs(flat), 0.995)), 1e-3)
    flattened = np.clip(flat / (2.0 * span), -1.0, 1.0)

    valid = ((illum > 0.10) & (x < 0.995)).astype(np.float32)
    out = np.stack([flattened, snr, coh]).astype(np.float32) * valid[None]
    return out


def ridge_response(gray: np.ndarray, sigmas=(1.2, 1.8, 2.6, 3.6)) -> np.ndarray:
    """Multi-scale bright-ridge (Hessian) response on the background-normalised image.

    Used by the classical baseline and by the unsupervised crossing-candidate sampler
    that drives real-image augmentation.
    """
    chans = normalise(gray)
    x = chans[1]
    best = None
    for s in sigmas:
        hxx = cv2.Sobel(cv2.GaussianBlur(x, (0, 0), s), cv2.CV_32F, 2, 0, ksize=3) * s * s
        hyy = cv2.Sobel(cv2.GaussianBlur(x, (0, 0), s), cv2.CV_32F, 0, 2, ksize=3) * s * s
        hxy = cv2.Sobel(cv2.GaussianBlur(x, (0, 0), s), cv2.CV_32F, 1, 1, ksize=3) * s * s
        delta = np.sqrt((hxx - hyy) ** 2 + 4.0 * hxy * hxy)
        lo = (hxx + hyy - delta) / 2.0
        hi = (hxx + hyy + delta) / 2.0
        strength = np.maximum(-lo, 0.0) * np.exp(-0.5 * (hi / (np.abs(lo) + 1e-6)) ** 2)
        best = strength if best is None else np.maximum(best, strength)
    return (best * chans[2]).astype(np.float32)
