"""Integrity tests for the delivered data and the end-to-end pipeline.

These run against the files on disk and are skipped when a stage has not been produced
yet, so the suite is useful both before and after a full run.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cottoncross.common import load_json, save_json
from cottoncross.dataset import RealCrops, SynthDataset
from cottoncross.evaluate_real import SCHEMA, score_against_annotations

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data/real"
SYNTH = ROOT / "data/synth"


def have(path):
    return (ROOT / path).exists()


@unittest.skipUnless(have("data/real/manifest.json"), "real dataset not prepared")
class TestRealData(unittest.TestCase):
    def test_originals_hash_and_splits_are_consistent(self):
        rows = load_json(REAL / "manifest.json")
        self.assertEqual(len(rows), 75)
        by_group = {}
        for r in rows:
            digest = hashlib.sha256((REAL / r["raw"]).read_bytes()).hexdigest()
            self.assertEqual(digest, r["sha256"], f"{r['id']} raw bytes changed")
            self.assertTrue((REAL / r["roi"]).exists())
            by_group.setdefault(r["group"], set()).add(r["split"])
        self.assertTrue(all(len(s) == 1 for s in by_group.values()),
                        "an acquisition block was split across train/val/test")

    def test_tiles_inherit_the_plate_split(self):
        source = {r["id"]: r for r in load_json(REAL / "manifest.json")}
        for t in load_json(REAL / "tiles.json"):
            self.assertEqual(t["split"], source[t["source_id"]]["split"])

    @unittest.skipUnless(have("data/real/crossing_windows.json"), "windows not cached")
    def test_crossing_windows_lie_inside_their_plate(self):
        import cv2

        payload = load_json(REAL / "crossing_windows.json")
        source = {r["id"]: r for r in load_json(REAL / "manifest.json")}
        self.assertGreater(payload["total"], 100)
        shapes = {}
        for w in payload["windows"][:60]:
            plate = source[w["id"]]
            self.assertEqual(w["split"], plate["split"])
            if w["id"] not in shapes:
                data = np.fromfile(str(REAL / plate["roi"]), dtype=np.uint8)
                shapes[w["id"]] = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE).shape
            h, wd = shapes[w["id"]]
            x0, y0, x1, y1 = w["xyxy"]
            self.assertTrue(0 <= x0 < x1 <= wd and 0 <= y0 < y1 <= h)
            self.assertGreaterEqual(w["arm_sectors"], 4)

    @unittest.skipUnless(have("data/real/crossing_windows.json"), "windows not cached")
    def test_unlabelled_sampler_returns_images_only(self):
        crops = RealCrops(REAL, size=256, seed=0)
        batch = crops.sample(4)
        self.assertEqual(len(batch), 4)
        for crop in batch:
            self.assertEqual(crop.shape, (256, 256))
            self.assertEqual(crop.dtype, np.uint8)
        train_plates = {r["id"] for r in load_json(REAL / "manifest.json")
                        if r["split"] == "train"}
        self.assertTrue(set(crops.plates) <= train_plates,
                        "the adaptation branch reached a non-training plate")


@unittest.skipUnless(have("data/synth/manifest.json"), "synthetic corpus not generated")
class TestSyntheticCorpus(unittest.TestCase):
    def test_split_seeds_are_disjoint(self):
        rows = load_json(SYNTH / "manifest.json")
        seeds = {}
        for r in rows:
            seeds.setdefault(r["split"], set()).add(r["seed"])
        splits = list(seeds)
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                self.assertFalse(seeds[splits[i]] & seeds[splits[j]],
                                 f"{splits[i]} and {splits[j]} share generator seeds")

    def test_every_row_has_its_three_files(self):
        rows = load_json(SYNTH / "manifest.json")
        for r in rows[:: max(len(rows) // 80, 1)]:
            for key in ("image", "target", "meta"):
                self.assertTrue((SYNTH / r[key]).exists(), f"missing {r[key]}")

    def test_provenance_records_the_guarantees(self):
        prov = load_json(SYNTH / "provenance.json")
        self.assertEqual(prov["generation_order"], "label first, image rendered from label")
        self.assertEqual(prov["background_carrier_split"], "train only")
        guarantee = prov["complete_crossing_guarantee"]
        self.assertEqual(guarantee["satisfied"], guarantee["of"])
        self.assertTrue(prov["equal_length"])

    def test_backgrounds_never_come_from_held_out_plates(self):
        prov = load_json(SYNTH / "provenance.json")
        self.assertEqual(prov["background_carrier_split"], "train only")

    def test_crossing_centred_crops_keep_a_crossing_in_frame(self):
        ds = SynthDataset(SYNTH, "train", augment=True, seed=42, crop=256,
                          crop_mode="crossing")
        checked = 0
        for i in range(0, 60):
            row = ds.rows[i]
            if row["complete_crossings"] == 0:
                continue
            _, target = ds[i]
            self.assertGreater(float(target[2].max()), 0.6,
                               f"{row['id']} lost its crossing when cropped")
            checked += 1
        self.assertGreater(checked, 20)

    def test_labels_match_the_image_size(self):
        ds = SynthDataset(SYNTH, "test", augment=False, crop=None)
        x, y = ds[0]
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(y.shape[0], 10)
        self.assertEqual(x.shape[-2:], y.shape[-2:])


@unittest.skipUnless(have("data/calibration.json"), "calibration not run")
class TestCalibration(unittest.TestCase):
    def test_calibration_is_declared_a_lower_bound(self):
        info = load_json(ROOT / "data/calibration.json")
        self.assertEqual(info["nominal_staple_mm"], 35.0)
        self.assertEqual(info["bound"], "lower")
        self.assertGreater(info["staple_pixels"], 100)
        self.assertAlmostEqual(info["pixels_per_mm"] * 35.0, info["staple_pixels"], places=3)


class TestGroundTruthDiscipline(unittest.TestCase):
    """No path may turn model output into a ground truth."""

    def _annotation_file(self, complete):
        payload = dict(schema=SCHEMA, coordinate_system="original extracted image pixels",
                       annotations=[dict(id="p057_i01", page=57, complete=complete,
                                         points_xy_original=[[100.0, 100.0]])])
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                             encoding="utf-8")
        json.dump(payload, handle)
        handle.close()
        return handle.name

    def test_incomplete_pages_are_refused(self):
        path = self._annotation_file(complete=False)
        with self.assertRaises(ValueError):
            score_against_annotations(REAL, "does-not-exist.pt", path)

    def test_foreign_schema_is_refused(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                             encoding="utf-8")
        json.dump({"schema": "something.else", "annotations": []}, handle)
        handle.close()
        with self.assertRaises(ValueError):
            score_against_annotations(REAL, "does-not-exist.pt", handle.name)

    @unittest.skipUnless(have("results/real/main/cfxnet/metrics.json"), "real eval not run")
    def test_real_metrics_never_claim_accuracy(self):
        summary = load_json(ROOT / "results/real/main/cfxnet/metrics.json")
        self.assertIn("ground_truth", summary)
        self.assertIn("not accuracy", summary["ground_truth"])


@unittest.skipUnless(torch.cuda.is_available() or True, "")
class TestPrediction(unittest.TestCase):
    def test_tiled_prediction_matches_a_single_tile(self):
        from cottoncross.models import FiberModel
        from cottoncross.predict import predict

        torch.manual_seed(0)
        model = FiberModel("cfxnet", base=8, depth=3).eval()
        rng = np.random.default_rng(0)
        gray = (rng.random((128, 128)) * 200 + 20).astype(np.uint8)
        single = predict(gray, model, "cpu", tile=128, stride=128, amp=False)
        tiled = predict(gray, model, "cpu", tile=96, stride=48, amp=False)
        # Blending is a partition of unity, so a smooth field is reproduced closely.
        self.assertLess(float(np.abs(single["mask"] - tiled["mask"]).mean()), 0.06)
        self.assertEqual(single["embed"].shape[0], 8)
        norms = np.linalg.norm(tiled["embed"], axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)


@unittest.skipUnless(have("results/tables/leaderboard.json"), "tables not built")
class TestReportedResults(unittest.TestCase):
    def test_leaderboard_states_losses_as_well_as_wins(self):
        board = load_json(ROOT / "results/tables/leaderboard.json")
        self.assertIn("verdict", board)
        for metric, entry in board["synthetic"].items():
            self.assertIn("ranking", entry)
            self.assertEqual(entry["ours_is_best"], entry["best"] == board["ours"])


@unittest.skipUnless(have("results/synthetic/main"), "synthetic evaluation not run")
class TestResultConsistency(unittest.TestCase):
    """The comparison is only meaningful if every method was scored the same way."""

    def test_every_result_used_the_same_linker_configuration(self):
        """A shared post-processing configuration is part of the comparison protocol.

        If one method were scored with different cost weights or a different rejection
        margin, its trajectory and pairing numbers would not be comparable.
        """
        signatures = {}
        for folder in sorted((ROOT / "results/synthetic/main").iterdir()):
            metrics = folder / "metrics.json"
            if not metrics.exists():
                continue
            linker = load_json(metrics).get("linker")
            self.assertIsNotNone(linker, f"{folder.name} did not record its linker config")
            costs = linker["costs"]
            signatures[folder.name] = (
                tuple(sorted(costs.items())), linker["margin_threshold"],
                linker["angle_limit"])
        self.assertTrue(signatures, "no synthetic results to check")
        unique = set(signatures.values())
        self.assertEqual(len(unique), 1,
                         f"results were produced with different linker settings: "
                         f"{ {k: v[1] for k, v in signatures.items()} }")

    def test_crossing_peaks_are_stored_for_post_hoc_thresholds(self):
        for folder in sorted((ROOT / "results/synthetic/main").iterdir()):
            per_image = folder / "per_image.json"
            if not per_image.exists():
                continue
            first = load_json(per_image)[0]
            self.assertIn("crossing_peaks", first,
                          f"{folder.name} cannot be re-thresholded after the fact")


@unittest.skipUnless(have("results/tables/leaderboard.json"), "tables not built")
class TestReportedRuns(unittest.TestCase):
    def test_every_reported_run_recorded_its_environment(self):
        for run in sorted((ROOT / "runs/main").glob("*_s*")):
            if (run / "summary.json").exists():
                env = load_json(run / "environment.json")
                self.assertIn("gpu", env)
                self.assertIn("label_policy", env)
                self.assertIsNone(load_json(run / "summary.json")["real_accuracy"])


if __name__ == "__main__":
    unittest.main()
