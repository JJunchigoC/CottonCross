"""Unit tests for the algorithmic core.

These verify the implementation, not accuracy: they check that the generator honours
the priors it claims, that labels are aligned with the images they describe, that the
augmentation transforms direction channels correctly, that the linker is order
independent and refuses ambiguous crossings, and that no evaluation path can invent a
ground truth.
"""
import unittest

import numpy as np
import torch

from cottoncross import synthgen
from cottoncross.background import naive, normalise
from cottoncross.dataset import geometric, pack_target
from cottoncross.linking import analyze, extract_graph
from cottoncross.losses import embedding_loss, supervised_loss
from cottoncross.metrics import instance_scores, point_counts, point_scores
from cottoncross.models import BACKBONES, FiberModel, OUT_CHANNELS, activate
from cottoncross.topology import branch_count, match_ports, zhang_suen


def draw(lines, size=128, width=3):
    import cv2

    image = np.zeros((size, size), np.uint8)
    for a, b in lines:
        cv2.line(image, a, b, 1, width)
    return image


class TestGenerator(unittest.TestCase):
    def test_worm_chain_has_the_requested_arc_length(self):
        rng = np.random.default_rng(0)
        for target in (200.0, 710.0):
            pts = synthgen.worm_chain(rng, (0, 0), 0.3, target, 150.0, step=2.0)
            length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
            self.assertAlmostEqual(length / target, 1.0, delta=0.02)

    def test_equal_staple_prior_gives_identical_lengths(self):
        rng = np.random.default_rng(7)
        curves, _ = synthgen.make_geometry(rng, 384, "cross", 710.0, 56.0, 84)
        lengths = [np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in curves]
        self.assertGreaterEqual(len(lengths), 2)
        self.assertLess(max(lengths) / min(lengths) - 1.0, 0.02)

    def test_disabling_the_prior_produces_unequal_lengths(self):
        spread = []
        for seed in range(12):
            rng = np.random.default_rng(seed)
            curves, _ = synthgen.make_geometry(rng, 384, "cross", 710.0, 56.0, 84,
                                               equal_length=False)
            lengths = [np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in curves]
            spread.append(max(lengths) / min(lengths))
        self.assertGreater(max(spread), 1.2)

    def test_complete_crossing_guarantee(self):
        for seed in range(12):
            rng = np.random.default_rng(1000 + seed)
            curves, records = synthgen.make_geometry(rng, 384, "cross", 710.0, 56.0, 84)
            complete = [r for r in records if r["complete"]]
            self.assertTrue(complete, f"seed {seed} produced no complete crossing")
            for r in complete:
                x, y = r["xy"]
                self.assertTrue(84 <= x < 384 - 84 and 84 <= y < 384 - 84)
                self.assertGreaterEqual(min(r["arms"]), 56.0)

    def test_near_miss_strategy_usually_avoids_a_complete_crossing(self):
        """`case` is a sampling strategy, not a label.

        The strategy aims for fibers that approach without meeting, and succeeds on the
        large majority of draws.  When two long chains do wander into each other the
        canvas is still labelled from the geometry that was drawn, so the pair stays
        consistent; only the strategy name is then a misnomer.
        """
        clean = 0
        for seed in range(20):
            rng = np.random.default_rng(500 + seed)
            _, records = synthgen.make_geometry(rng, 384, "near_miss", 710.0, 56.0, 84)
            clean += not any(r["complete"] for r in records)
        self.assertGreaterEqual(clean, 16, f"only {clean}/20 near-miss draws stayed apart")

    def test_labels_are_aligned_with_the_rendered_image(self):
        rng = np.random.default_rng(3)
        curves, records = synthgen.make_geometry(rng, 256, "cross", 400.0, 40.0, 56)
        image, parts = synthgen.render(rng, curves, 256)
        targets = synthgen.build_targets(parts, records, 256)
        self.assertEqual(image.shape, (256, 256))
        for r in records:
            x, y = int(round(r["xy"][0])), int(round(r["xy"][1]))
            window = targets["joint"][max(0, y - 4):y + 5, max(0, x - 4):x + 5]
            self.assertGreater(window.max(), 200, "crossing heatmap misses its own point")
            patch = targets["overlap"][max(0, y - 3):y + 4, max(0, x - 3):x + 4]
            self.assertGreaterEqual(int(patch.max()), 2, "overlap count misses a crossing")

    def test_instance_labels_cover_every_curve(self):
        rng = np.random.default_rng(11)
        curves, records = synthgen.make_geometry(rng, 256, "multi", 400.0, 40.0, 56)
        _, parts = synthgen.render(rng, curves, 256)
        targets = synthgen.build_targets(parts, records, 256)
        self.assertEqual(len(targets["instances"]), len(curves))
        for k, (mask, curve) in enumerate(zip(targets["instances"], curves)):
            inside = ((curve[:, 0] >= 0) & (curve[:, 0] < 256)
                      & (curve[:, 1] >= 0) & (curve[:, 1] < 256))
            if not inside.any():
                continue          # a distractor drawn wholly outside the canvas
            ys, xs = np.nonzero(mask)
            self.assertGreater(len(xs), 0, f"curve {k} rendered no centerline")
            # instances[k] must describe curves[k], not whatever was drawn k-th.
            nearest = np.hypot(curve[:, 0][None, :] - xs[:, None],
                               curve[:, 1][None, :] - ys[:, None]).min(axis=1)
            self.assertLess(float(nearest.max()), 2.0, f"instance {k} is misaligned")
        ids = set(np.unique(targets["instance_id"]).tolist()) - {0}
        # The id map is a hard assignment, so a fiber lying under another can be absent
        # from it; what must hold is that no id outside the curve set ever appears.
        self.assertTrue(ids <= set(range(1, len(curves) + 1)))
        self.assertGreaterEqual(len(ids), len(curves) - 1)


class TestBackground(unittest.TestCase):
    def test_grain_is_suppressed_relative_to_a_ridge(self):
        rng = np.random.default_rng(0)
        image = np.full((256, 256), 110, np.float32)
        image[128:, :] += rng.normal(0, 26, (128, 256))     # grainy half
        image[:128, :] += rng.normal(0, 3, (128, 256))      # smooth half
        image[60:64, :] = 210                               # a fiber-like ridge
        image = np.clip(image, 0, 255).astype(np.uint8)
        bgn = normalise(image)
        plain = naive(image)
        grain_bgn = bgn[1][160:].std() / max(bgn[1][20:50].std(), 1e-6)
        grain_plain = plain[1][160:].std() / max(plain[1][20:50].std(), 1e-6)
        self.assertLess(grain_bgn, grain_plain)

    def test_illumination_ramp_is_removed(self):
        ramp = np.linspace(40, 210, 256).astype(np.float32)
        image = np.tile(ramp, (256, 1)).astype(np.uint8)
        flat = normalise(image)[0]
        self.assertLess(float(np.abs(flat).mean()), 0.2)

    def test_output_is_deterministic_and_shaped(self):
        rng = np.random.default_rng(1)
        image = (rng.random((96, 128)) * 255).astype(np.uint8)
        a, b = normalise(image), normalise(image)
        self.assertEqual(a.shape, (3, 96, 128))
        np.testing.assert_array_equal(a, b)


class TestAugmentation(unittest.TestCase):
    def test_quarter_turn_matches_the_analytic_moment_transform(self):
        """np.rot90 maps a direction theta to theta - 90 degrees.

        Second-order moments therefore flip sign once per quarter turn and fourth-order
        moments are unchanged, which is exactly what `geometric` applies.
        """
        size = 32
        theta = 0.37
        target = np.zeros((10, size, size), np.float32)
        target[3:7] = np.array([np.cos(2 * theta), np.sin(2 * theta),
                                np.cos(4 * theta), np.sin(4 * theta)]).reshape(4, 1, 1)

        class RotateOnce:
            def integers(self, n):
                return 1

            def random(self):
                return 1.0      # above 0.5, so no flip

        _, out = geometric(np.zeros((size, size), np.uint8), target.copy(), RotateOnce())
        turned = theta - np.pi / 2
        expected = [np.cos(2 * turned), np.sin(2 * turned),
                    np.cos(4 * turned), np.sin(4 * turned)]
        for ch, want in zip(range(3, 7), expected):
            np.testing.assert_allclose(out[ch, 0, 0], want, atol=1e-6,
                                       err_msg=f"channel {ch} rotates incorrectly")

    def test_rendered_moments_agree_with_the_curve_tangent(self):
        rng = np.random.default_rng(5)
        theta = 0.4
        t = np.linspace(-40, 40, 120)
        curve = (np.stack([64 + t * np.cos(theta), 64 + t * np.sin(theta)], axis=1)
                 .astype(np.float32))
        _, parts = synthgen.render(rng, [curve], 128)
        target = pack_target(synthgen.build_targets(parts, [], 128), 128)
        valid = target[7] > 0
        self.assertGreater(valid.sum(), 100)
        np.testing.assert_allclose(target[3][valid].mean(), np.cos(2 * theta), atol=0.05)
        np.testing.assert_allclose(target[4][valid].mean(), np.sin(2 * theta), atol=0.05)

    def test_flip_negates_the_odd_moments(self):
        target = np.zeros((10, 16, 16), np.float32)
        target[3:7] = np.array([0.3, 0.6, -0.2, 0.8]).reshape(4, 1, 1)
        image = np.zeros((16, 16), np.uint8)

        class Fixed:
            def __init__(self):
                self.calls = 0

            def integers(self, n):
                return 0

            def random(self):
                return 0.0

        _, out = geometric(image, target.copy(), Fixed())
        np.testing.assert_allclose(out[3, 0, 0], 0.3, atol=1e-6)
        np.testing.assert_allclose(out[4, 0, 0], -0.6, atol=1e-6)
        np.testing.assert_allclose(out[5, 0, 0], -0.2, atol=1e-6)
        np.testing.assert_allclose(out[6, 0, 0], -0.8, atol=1e-6)


class TestTopology(unittest.TestCase):
    def test_thinning_is_idempotent_and_keeps_connectivity(self):
        from scipy import ndimage as ndi

        mask = draw([((10, 64), (118, 64)), ((64, 10), (64, 118))])
        once = zhang_suen(mask)
        twice = zhang_suen(once)
        np.testing.assert_array_equal(once, twice)
        self.assertEqual(ndi.label(once, structure=np.ones((3, 3)))[1], 1)
        self.assertLess(once.sum(), mask.sum() / 2)

    def test_crossing_number_finds_the_x_junction(self):
        sk = zhang_suen(draw([((10, 64), (118, 64)), ((64, 10), (64, 118))]))
        counts = branch_count(sk)
        self.assertGreaterEqual(int(counts.max()), 4)

    def test_joint_matching_is_order_independent(self):
        cost = np.array([[np.inf, 0.10, 0.90, 0.80],
                         [0.10, np.inf, 0.85, 0.95],
                         [0.90, 0.85, np.inf, 0.20],
                         [0.80, 0.95, 0.20, np.inf]])
        first = match_ports(cost)
        order = [2, 0, 3, 1]
        permuted = cost[np.ix_(order, order)]
        second = match_ports(permuted)
        back = {tuple(sorted((order[a], order[b]))) for a, b in second["pairs"]}
        self.assertEqual({tuple(sorted(p)) for p in first["pairs"]}, back)

    def test_ambiguous_crossing_is_refused(self):
        cost = np.array([[np.inf, 0.50, 0.50, 0.50],
                         [0.50, np.inf, 0.50, 0.50],
                         [0.50, 0.50, np.inf, 0.50],
                         [0.50, 0.50, 0.50, np.inf]])
        self.assertTrue(match_ports(cost)["ambiguous"])

    def test_greedy_can_be_beaten_by_joint_matching(self):
        # Port 0 prefers 1, but pairing them strands 2 and 3 in an expensive match.
        cost = np.array([[np.inf, 0.30, 0.34, np.inf],
                         [0.30, np.inf, np.inf, 0.34],
                         [0.34, np.inf, np.inf, 0.95],
                         [np.inf, 0.34, 0.95, np.inf]])
        joint = match_ports(cost, unmatched=0.9, margin_threshold=0.0)
        total = sum(cost[a, b] for a, b in joint["pairs"])
        self.assertAlmostEqual(total, 0.68, places=6)
        self.assertLess(total, 0.30 + 0.95)

    def test_x_crossing_resolves_into_two_trajectories(self):
        mask = draw([((6, 64), (122, 64)), ((64, 6), (64, 122))])
        _, graph = analyze(mask, None, None, method="angle_joint")
        self.assertEqual(len(graph["joints"]), 1)
        self.assertEqual(graph["joints"][0]["degree"], 4)
        self.assertEqual(len(graph["joints"][0]["accepted_pairs"]), 2)
        self.assertEqual(len(graph["fibers"]), 2)

    def test_parallel_lines_produce_no_crossing(self):
        mask = draw([((6, 40), (122, 40)), ((6, 90), (122, 90))])
        _, graph = analyze(mask, None, None, method="cfx")
        self.assertEqual(graph["crossing_candidates"], 0)
        self.assertEqual(len(graph["fibers"]), 2)

    def test_graph_extraction_reports_rejected_components(self):
        sk, segments, labels, centres, rejected = extract_graph(
            draw([((6, 64), (122, 64)), ((64, 6), (64, 122))]))
        self.assertGreaterEqual(len(segments), 4)
        self.assertEqual(len(centres), 1)
        self.assertIsInstance(rejected, int)


class TestMetrics(unittest.TestCase):
    def test_point_matching_is_one_to_one(self):
        truth = [[10, 10]]
        predicted = [[10, 10], [11, 11], [12, 12]]
        counts = point_counts(predicted, truth, tolerance=6)
        self.assertEqual(counts["tp"], 1)
        self.assertEqual(counts["fp"], 2)

    def test_point_scores_handle_empty_sets(self):
        self.assertEqual(point_scores(point_counts([], [], 6))["f1"], 1.0)
        self.assertEqual(point_scores(point_counts([[1, 1]], [], 6))["precision"], 0.0)

    def test_fragmentation_costs_precision(self):
        line = np.zeros((64, 64), np.uint8)
        line[32, 8:56] = 1
        whole = [dict(points_xy=[[x, 32] for x in range(8, 56)], closed=False)]
        halves = [dict(points_xy=[[x, 32] for x in range(8, 30)], closed=False),
                  dict(points_xy=[[x, 32] for x in range(32, 56)], closed=False)]
        self.assertEqual(instance_scores(whole, [line])["tp"], 1)
        self.assertGreaterEqual(instance_scores(halves, [line])["fp"], 1)


class TestModels(unittest.TestCase):
    def test_every_backbone_emits_the_shared_head(self):
        x = torch.randn(1, 3, 64, 64)
        for arch in BACKBONES:
            if arch == "dscnet" and not torch.cuda.is_available():
                continue
            model = FiberModel(arch, base=12 if arch == "dscnet" else 8, depth=3)
            if arch == "dscnet":
                model = model.cuda()
                x_in = x.cuda()
            else:
                x_in = x
            with torch.no_grad():
                z = model(x_in)
            self.assertEqual(z.shape[1], OUT_CHANNELS)
            self.assertEqual(z.shape[-2:], x.shape[-2:])

    def test_activation_bundle_is_in_range(self):
        z = torch.randn(2, OUT_CHANNELS, 16, 16)
        bundle = activate(z)
        self.assertTrue(bool((bundle["mask"] >= 0).all() and (bundle["mask"] <= 1).all()))
        np.testing.assert_allclose(bundle["overlap"].sum(1).numpy(), 1.0, atol=1e-5)
        np.testing.assert_allclose(
            bundle["embed"].norm(dim=1).numpy(), 1.0, atol=1e-5)

    def test_oca_can_be_switched_off(self):
        with_oca = FiberModel("cfxnet", base=8, depth=3)
        without = FiberModel("cfxnet", base=8, depth=3, use_oca=False, use_dcm=False)
        self.assertIsNotNone(with_oca.backbone.oca)
        self.assertIsNone(without.backbone.oca)


class TestLosses(unittest.TestCase):
    def _batch(self):
        torch.manual_seed(0)
        target = torch.zeros(2, 10, 32, 32)
        target[:, 0, 14:18, 4:28] = 1.0
        target[:, 1, 15:17, 4:28] = 1.0
        target[:, 2, 15:17, 14:18] = 1.0
        target[:, 7] = 1.0
        target[:, 8, 15:17, 14:18] = 2.0
        target[:, 9, 15:17, 4:16] = 1.0
        target[:, 9, 15:17, 16:28] = 2.0
        return target

    def test_supervised_loss_is_finite_and_differentiable(self):
        model = FiberModel("cfxnet", base=8, depth=3)
        x = torch.randn(2, 3, 32, 32)
        z, aux = model(x, want_aux=True)
        loss, parts = supervised_loss(z, self._batch(), None, aux)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        for key in ("mask", "center", "joint", "topology", "orientation", "overlap",
                    "embedding", "deep"):
            self.assertIn(key, parts)

    def test_embedding_loss_rewards_matching_instances(self):
        target = self._batch()
        instance = target[:, 9:10]
        centre = (target[:, 1:2] > 0.5).float()
        good = torch.zeros(2, 8, 32, 32)
        good[:, 0][instance[:, 0] == 1] = 1.0
        good[:, 1][instance[:, 0] == 2] = 1.0
        bad = torch.zeros(2, 8, 32, 32)
        bad[:, 0] = 1.0
        self.assertLess(float(embedding_loss(good, instance, centre)),
                        float(embedding_loss(bad, instance, centre)))

    def test_disabled_terms_are_absent(self):
        model = FiberModel("unet", base=8, depth=3)
        z = model(torch.randn(2, 3, 32, 32))
        _, parts = supervised_loss(z, self._batch(),
                                   dict(topology=0.0, orientation=0.0, overlap=0.0,
                                        embedding=0.0, deep=0.0), {})
        for key in ("topology", "orientation", "overlap", "embedding", "deep"):
            self.assertNotIn(key, parts)


if __name__ == "__main__":
    unittest.main()
