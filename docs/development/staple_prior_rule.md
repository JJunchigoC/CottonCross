# Pre-registered rule: staple-prior pseudo-label refinement (SPR)

Written before any SPR model was trained.

## Motivation (observed on train/val plates, no test plate involved)

Counting by the equal-staple rule needs a clean centerline.  On the real plates the
dominant error is substrate grain that the adapted network calls fiber: `cfxnet_s43`
produces a 25-44k px skeleton on plates whose fibers amount to ~4-5k px, almost all of it
grain in the lower, grainy half of the plate.  The unlabelled adaptation never tells the
network that such structures are background - its pseudo-labels only copy the teacher.

## Method

`cottoncross/prior.py`: in every 256 px adaptation crop, a teacher-positive component that
does not reach the crop border, has a skeleton shorter than 140 px (the counting debris
cut-off, 0.1 staple) and keeps 24 px of clearance from anything long or border-reaching
cannot be a 35 mm staple; it becomes confident background with its own loss term
(`unsupervised.staple_prior`, weight fixed at 0.1 without tuning).

## Runs

`cfxnet` x 3 seeds with SPR (runs/spr).  `unet` x 3 seeds with SPR as a generality check,
reported whatever it shows.  Everything else identical to runs/main.

## Adoption rule

SPR becomes part of the proposed method (CFX-Net + SPR = "ours") if, averaged over the
three seeds,

1. counting MAE (primary calibrated protocol of `scripts/count_fibers.py`) on the real
   train+val plates with a high/medium reference count is lower than for CFX-Net
   without SPR, and
2. mean crossing F1 on the synthetic val split (at each model's val-selected threshold)
   drops by no more than 0.01.

The test plates and the counting benchmark test split are not looked at for this
decision.  If the rule fails, SPR is reported as a negative result and "ours" stays
CFX-Net without SPR.

## Outcome

**Rejected; stopped early once the rule could no longer be satisfied.**

Real train+val plates, calibrated counting protocol (lower is better):

| run | counting MAE | mean skeleton per plate |
|---|---|---|
| cfxnet_s42 (main) | 0.368 | 5595 px |
| cfxnet_s42 + SPR | 1.553 | 1022 px |

The three-seed mean without SPR is 0.307; with seed 42 at 1.553, seeds 43 and 44 would
need a negative MAE for condition 1 to hold, so the remaining SPR runs (cfxnet s43/s44 and
the unet generality check) were stopped and are not reported as results.  Synthetic val
crossing F1 of cfxnet_s42 + SPR: 0.824 (main 0.837).

Failure mechanism: on the real plates a faint fiber is broken by the teacher into short
isolated pieces; SPR rules those pieces out, the student learns to suppress faint fiber,
the EMA teacher follows the student, and the next round produces even more short pieces.
The 24 px clearance protects pieces next to a long fiber, not a fiber that has already
fallen apart.  The synthetic score stayed flat (0.916 vs 0.918) while the real skeleton
collapsed by 80%: exactly the kind of drift a synthetic-only checkpoint criterion cannot
see.  The staple prior is kept where it is safe - at counting time, after segmentation
(`cottoncross/count.py`) - and not as a training signal.
