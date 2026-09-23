"""Staple-prior pseudo-label refinement (SPR) for the unlabelled real adaptation.

Every fiber in a sample is a 35 mm staple, about 1300-1500 px on these plates, while an
adaptation crop is 256 px.  A staple therefore can never lie entirely inside a crop.  So
a structure the teacher calls "fiber" that

* lies entirely inside the crop (does not reach the crop border),
* is short - its skeleton is below `max_length` px, the counting debris cut-off of
  0.1 staple, and
* keeps a clearance from every structure that could still be part of a staple (anything
  long or anything reaching the border), so a fiber broken by a faint stretch is not
  touched,

is interference by the staple prior itself: grain on the substrate, lint, dust.  SPR hands
those pixels to the student as confident background.  No human label is involved - the
only input is the physical fact the whole method is built on.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

EIGHT = np.ones((3, 3), bool)


def staple_negatives(prob, threshold=0.5, max_length=140, border=6, clearance=24, grow=2):
    """Boolean map of pixels that the staple prior rules out as fiber.

    `prob` is the teacher's fiber-occupancy probability, shape (B, H, W) or (H, W).
    """
    squeeze = prob.ndim == 2
    prob = prob[None] if squeeze else prob
    out = np.zeros(prob.shape, bool)
    for b in range(prob.shape[0]):
        binary = prob[b] > threshold
        labels, n = ndi.label(binary, structure=EIGHT)
        if n == 0:
            continue
        index = np.arange(1, n + 1)
        lengths = ndi.sum(skeletonize(binary), labels, index)
        h, w = binary.shape
        reaches = np.zeros(n, bool)
        for k, sl in enumerate(ndi.find_objects(labels)):
            ys, xs = sl
            reaches[k] = (ys.start < border or xs.start < border
                          or ys.stop > h - border or xs.stop > w - border)
        keep = reaches | (lengths > max_length)
        candidates = ~keep
        if not candidates.any():
            continue
        if keep.any() and clearance > 0:
            protected = np.isin(labels, index[keep])
            distance = ndi.distance_transform_edt(~protected)
            gap = ndi.minimum(distance, labels, index)
            candidates &= gap > clearance
        if not candidates.any():
            continue
        negative = np.isin(labels, index[candidates])
        if grow:
            negative = ndi.binary_dilation(negative, EIGHT, iterations=grow)
            if keep.any():
                negative &= ~np.isin(labels, index[keep])
        out[b] = negative
    return out[0] if squeeze else out
