# Pre-registered rule: per-image crossing count on full plates

Written before the crossing-filtered features were extracted.

## Why

On the counting benchmark (full 1536 px plates with interference) the raw crossing
heatmap over-counts by +9 to +17 crossings per image for every learned model, because it
also fires on interference and on tight bends; skeleton junctions are off by 2.3-2.8.
Neither is what a user means by "a crossing": a point where two of the **counted** fibers
meet.

## Rule (identical for every model)

1. Gap-closed chains shorter than 0.1 staple are interference (the counting debris rule);
   the rest are the counted fibers.
2. A crossing is a crossing-heatmap peak above the model's val-selected threshold
   (`crossing_threshold.json`, non-maximum suppression as in `linking.heatmap_points`)
   lying within 8 px of a counted fiber.  The training-free baseline has no heatmap and is
   scored with rule 3 only.
3. Reported alongside: skeleton junctions of degree >= 4 within 8 px of a counted fiber.

Staple used for the debris cut: the benchmark's exact 1300 px; 1400 px on the real plates
(the prior; no per-model selection enters the crossing rule).

## Scores

Counting benchmark test split, exact crossing ground truth (every intersection of two
fiber centerlines): per-image crossing-count MAE (primary), exact-count accuracy, and
localisation F1 at 8 px.  Mean over seeds.  Parameters above are fixed; nothing is
selected on the test split.

## Outcome

**Abandoned before the full extraction; no benchmark number is reported for it.**

A smoke test of the filter on the first three benchmark test images (cfxnet_s42) showed
that restricting heatmap peaks to counted fibers removes almost nothing: 17 of 18, 16 of
18 and 7 of 8 peaks lie on a counted fiber, against 1, 3 and 0 true fiber-fiber crossings.
Two likely reasons (not verified image by image):

* the benchmark's interference fragments (15-70 per plate) often lie across a fiber, which
  is a genuine crossing of two centerlines - the heatmap is right to fire, but the ground
  truth only counts fiber-fiber crossings, so the benchmark cannot score this fairly;
* on full plates the heatmap also fires along tightly waved stretches of a single fiber.

Because three test images were looked at, the rule was not re-tuned and re-run; it is
recorded here as a negative result.  The application reports skeleton junctions of degree >= 4
on counted fibers (rule 3), and per-image crossing counts are not claimed as a result.
