"""Tests for the equal-staple fiber counter and its benchmark."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cottoncross.count import chain_lengths, close_gaps, count_plate, half_up  # noqa: E402
from cottoncross.prior import staple_negatives  # noqa: E402


def fiber(points, closed=False):
    pts = np.asarray(points, float)
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    k = min(12, len(pts) - 1)
    head, tail = pts[0] - pts[k], pts[-1] - pts[-1 - k]
    return dict(length=length, closed=closed, head=pts[0].tolist(), tail=pts[-1].tolist(),
                head_dir=(head / np.linalg.norm(head)).tolist(),
                tail_dir=(tail / np.linalg.norm(tail)).tolist(),
                head_border=False, tail_border=False, confidence=1.0)


def line(x0, x1, y, step=2.0):
    xs = np.arange(x0, x1 + 1e-9, step)
    return np.stack([xs, np.full_like(xs, y)], 1)


def features(fibers):
    return dict(fibers=fibers, component_pixels=[int(f["length"]) for f in fibers])


class TestCounter(unittest.TestCase):
    def test_half_up_is_not_bankers_rounding(self):
        self.assertEqual(half_up(2.5), 3)
        self.assertEqual(half_up(0.5), 1)
        self.assertEqual(half_up(1.49), 1)

    def test_collinear_break_is_closed(self):
        # One fiber broken by a 20 px faint stretch: two pieces, one chain.
        a, b = fiber(line(0, 600, 100)), fiber(line(620, 1300, 100))
        chains = close_gaps([a, b])
        self.assertEqual(len(chains), 1)

    def test_perpendicular_ends_are_not_joined(self):
        a = fiber(line(0, 600, 100))
        b = fiber(np.stack([np.full(300, 620.0), np.arange(110, 710, 2.0)], 1))
        self.assertEqual(len(close_gaps([a, b])), 2)

    def test_short_debris_does_not_change_the_count(self):
        staple = 1300.0
        fibers = [fiber(line(0, 1300, 100)), fiber(line(0, 1300, 400))]
        clean = count_plate(features(fibers), staple)
        debris = [fiber(line(0, 60, 700 + 20 * i)) for i in range(30)]  # 30 x 60 px specks
        dirty = count_plate(features(fibers + debris), staple)
        self.assertEqual(clean["length"], 2)
        self.assertEqual(dirty["length"], 2)
        self.assertEqual(dirty["debris_removed"], 30)
        # Blob counting is fooled by exactly this.
        self.assertGreater(dirty["components"], clean["components"])

    def test_chain_rule_keeps_under_traced_fibers(self):
        # Three fibers each traced at 75% of a staple: pooled length rounds to 2, the
        # chain-level rule still counts 3.
        staple = 1000.0
        fibers = [fiber(line(0, 750, 100 + 300 * i)) for i in range(3)]
        result = count_plate(features(fibers), staple)
        self.assertEqual(result["length"], 2)
        self.assertEqual(result["chains_count"], 3)

    def test_chain_lengths_matches_count(self):
        fibers = [fiber(line(0, 1300, 100)), fiber(line(0, 50, 500))]
        c = chain_lengths(features(fibers))
        self.assertEqual(len(c), 2)
        self.assertAlmostEqual(float(c.max()), 1300.0, delta=2.5)


class TestStaplePrior(unittest.TestCase):
    def test_rules_out_only_short_interior_isolated_structures(self):
        prob = np.zeros((256, 256), np.float32)
        prob[128, 5:251] = 1.0             # long, reaches the border: a possible staple
        prob[40:44, 100:130] = 1.0         # short, interior, isolated: interference
        prob[135:137, 60:80] = 1.0         # short but next to the long fiber: protected
        neg = staple_negatives(prob)
        self.assertTrue(neg[42, 115])
        self.assertFalse(neg[128, 100])
        self.assertFalse(neg[136, 70])


if __name__ == "__main__":
    unittest.main()
