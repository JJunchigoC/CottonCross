"""Train the staple-prior (SPR) runs: runs/spr/<arch>_s<seed>.

    python scripts/tools/train_spr.py --archs cfxnet --seeds 42 43 44

Identical to the main runs except `unsupervised.staple_prior` (see
docs/development/staple_prior_rule.md).  Phase 1 (synthetic supervision only) does not
involve SPR, so each run resumes from the phase-1 checkpoint of the matching main run:
the two models share every weight up to the start of the real adaptation, and differ only
in what happens after it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_experiments as R  # noqa: E402
from cottoncross import trainer  # noqa: E402

SPR = dict(unsupervised=dict(staple_prior=0.1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="*", default=["cfxnet", "unet"])
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    a = p.parse_args()
    arch_over = dict(R.ARCHS)
    for seed in a.seeds:
        for arch in a.archs:
            name = f"{arch}_s{seed}"
            out = ROOT / "runs/spr" / name
            if (out / "summary.json").exists():
                print(f"[skip] spr/{name}", flush=True)
                continue
            over = trainer.merge(trainer.merge(arch_over[arch], dict(seed=seed)), SPR)
            cfg = trainer.merge(R.base_config(3000, 2000, 8), over)
            print(f"[train] spr/{name} (resume from main phase 1)", flush=True)
            trainer.run(None, cfg, str(ROOT / "data/synth"), str(ROOT / "data/real"), str(out),
                        resume=str(ROOT / "runs/main" / name / "phase1.pt"))


if __name__ == "__main__":
    main()
