# Pre-registered rule: interference-aware training corpus

Written before any pilot model trained on `data/synth_if` was evaluated.

## What changes

`data/synth_if` is the main corpus regenerated with the same seeds, sizes and staple, plus
`interference=True` in `synthgen.generate`:

* short fiber-like distractor curves (16-200 px of arc) rendered with the fiber appearance
  model but absent from every label — lint and broken ends are not staple fibers;
* 0-2 faded stretches per fiber, attenuated in the image with the labels kept — the
  network must follow a fiber through a poorly imaged stretch.

Nothing else changes: same architectures, losses, steps, adaptation, linker, thresholds.

## Pilot

`unet_s42` and `cfxnet_s42` retrained on `data/synth_if` (runs/pilot_if).  Evaluated only
on selection data: the counting benchmark val split and the real train+val plates with a
high/medium reference count.  No test split is looked at.

## Adoption rule

The corpus replaces `data/synth` for **every** model (all six architectures, all three
seeds, all ablations) if and only if, averaged over the two pilot architectures,

1. counting MAE (length estimator, selection protocol of `scripts/count_fibers.py`)
   decreases on the benchmark val split, and
2. it does not increase on the real train+val plates, and
3. crossing F1 on the synthetic val split does not drop by more than 0.01.

Otherwise `data/synth` stays the main corpus and the pilot is reported as a negative
result.  Either way the outcome is recorded here.

## Outcome

**Rejected.**  Condition 2 fails for both pilot architectures (real train+val plates,
calibrated counting protocol, seed 42):

| model | counting MAE, data/synth | counting MAE, data/synth_if |
|---|---|---|
| unet_s42 | 0.211 | 0.316 |
| cfxnet_s42 | 0.368 | 0.474 |

Crossing F1 on the clean synthetic val split: unet 0.763 -> 0.829, cfxnet 0.837 -> 0.829.
The synthetic distractors did not transfer to the real interference, which is mostly
substrate grain rather than fiber-like lint (see staple_prior_rule.md).  Because
condition 2 already failed, the benchmark val split was not evaluated for the pilot.
`data/synth` remains the main corpus; `data/synth_if` and `runs/pilot_if` are kept only
as the record of this negative result.
