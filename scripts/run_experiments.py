"""The full experiment matrix.

Stages (run all, or pick with --stage):

  main       6 backbones x 3 seeds, identical data / losses / budget
  ablation   CFX-Net with one component removed at a time, including the three
             data-design ablations that need their own generated corpus
  post       greedy vs joint vs embedding-guided linking on the same checkpoint
  real       label-free evaluation on the held-out real plates
  tables     collect everything into CSV tables and a summary JSON

Every training run writes runs/<group>/<name>/ and every evaluation writes
results/<group folder>/<name>/ (see cottoncross/paths.py).  Completed work is skipped, so the script can be re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cottoncross import evaluate, evaluate_real, synthgen, trainer  # noqa: E402
from cottoncross.paths import diagnostic, group_dir  # noqa: E402
from cottoncross.common import load_json, save_json  # noqa: E402

SEEDS = [42, 43, 44]

ARCHS = [
    ("unet", dict(arch="unet", base=32)),
    ("attunet", dict(arch="attunet", base=32)),
    ("unetpp", dict(arch="unetpp", base=32)),
    ("resunet", dict(arch="resunet", base=32)),
    ("dscnet", dict(arch="dscnet", base=24)),
    # oca_version is pinned: the v2 revision lost on the validation split and was
    # rejected by the pre-registered rule in docs/development/.
    ("cfxnet", dict(arch="cfxnet", base=24, model=dict(oca_version="v1"))),
]
OURS = "cfxnet"

# name -> (config overrides, training corpus, linking method used at evaluation)
ABLATIONS = {
    "no_topology": (dict(weights=dict(topology=0.0)), "data/synth", "cfx"),
    "no_orientation": (dict(weights=dict(orientation=0.0)), "data/synth", "cfx"),
    "no_overlap": (dict(weights=dict(overlap=0.0)), "data/synth", "cfx"),
    "no_embedding": (dict(weights=dict(embedding=0.0)), "data/synth", "angle_joint"),
    "no_deep": (dict(weights=dict(deep=0.0)), "data/synth", "cfx"),
    "no_oca": (dict(model=dict(use_oca=False)), "data/synth", "cfx"),
    "no_dcm": (dict(model=dict(use_dcm=False)), "data/synth", "cfx"),
    "no_adaptation": (dict(unsupervised=dict(enabled=False)), "data/synth", "cfx"),
    "raw_input": (dict(input_mode="raw"), "data/synth", "cfx"),
    "no_equal_length": (dict(val_synth="data/synth"), "data/synth_varlen", "cfx"),
    "no_complete_crossing": (dict(val_synth="data/synth"), "data/synth_nocomplete", "cfx"),
    "no_real_background": (dict(val_synth="data/synth"), "data/synth_nocarrier", "cfx"),
}

DATA_VARIANTS = {
    "data/synth_varlen": dict(equal_length=False),
    "data/synth_nocomplete": dict(require_complete=False),
    "data/synth_nocarrier": dict(carrier_probability=0.0),
}

# name -> (linker, margin override).  `cfx` uses the margin selected on the validation
# split; `cfx_conservative` is the same linker at an explicit abstention operating point.
POST_METHODS = {
    "greedy": ("greedy", None),
    "angle_joint": ("angle_joint", None),
    "cfx": ("cfx", None),
    "cfx_conservative": ("cfx", 0.12),
}


def base_config(steps, warmup, batch):
    return dict(steps=steps, warmup=warmup, batch_size=batch)


def train_one(name, group, overrides, synth, steps, warmup, batch, force=False):
    out = ROOT / "runs" / group / name
    if (out / "summary.json").exists() and not force:
        print(f"[skip] {group}/{name} already trained", flush=True)
        return load_json(out / "summary.json")
    print(f"[train] {group}/{name}  synth={synth}", flush=True)
    cfg = trainer.merge(base_config(steps, warmup, batch), overrides)
    trainer.run(None, cfg, str(ROOT / synth), str(ROOT / "data/real"), str(out))
    return load_json(out / "summary.json")


def eval_one(name, group, checkpoint, method, split, out_group, expected_length,
             data="data/synth", force=False, limit=None, margin_threshold=None):
    out = group_dir(ROOT / "results", out_group) / name
    if (out / "metrics.json").exists() and not force:
        print(f"[skip] eval {out_group}/{name}", flush=True)
        return load_json(out / "metrics.json")
    print(f"[eval ] {out_group}/{name} split={split} method={method}", flush=True)
    return evaluate.evaluate_split(
        checkpoint=str(checkpoint) if checkpoint else None, data=str(ROOT / data),
        split=split, method=method, out=str(out), limit=limit,
        expected_length=expected_length, tag=name, margin_threshold=margin_threshold)


def selected_archs(args):
    chosen = set(args.archs or [])
    return [(n, o) for n, o in ARCHS if not chosen or n in chosen]


def stage_main(args):
    for seed in SEEDS:
        for name, over in selected_archs(args):
            train_one(f"{name}_s{seed}", "main", trainer.merge(over, dict(seed=seed)),
                      "data/synth", args.steps, args.warmup, args.batch, args.force)


def stage_data_variants(args):
    calib = load_json(ROOT / "data/calibration.json")
    staple = float(calib["staple_pixels"])
    for path, kwargs in DATA_VARIANTS.items():
        target = ROOT / path
        if (target / "manifest.json").exists() and not args.force:
            print(f"[skip] corpus {path}", flush=True)
            continue
        print(f"[data ] {path} {kwargs}", flush=True)
        synthgen.generate(str(ROOT / "data/real"), str(target), args.size, args.plate_size,
                          (args.ablation_train, 128, 0), 0, staple, float(args.arm),
                          seed=args.seed, **kwargs)


def stage_ablation(args):
    stage_data_variants(args)
    for name, (over, synth, _) in ABLATIONS.items():
        cfg = trainer.merge(dict(ARCHS[-1][1]), dict(seed=SEEDS[0]))
        train_one(name, "ablation", trainer.merge(cfg, over), synth,
                  args.steps, args.warmup, args.batch, args.force)


def stage_eval(args):
    calib = load_json(ROOT / "data/calibration.json")
    staple = float(calib["staple_pixels"])
    for seed in SEEDS:
        for name, _ in selected_archs(args):
            ck = ROOT / "runs/main" / f"{name}_s{seed}" / "best.pt"
            if not ck.exists():
                print(f"[warn] missing {ck}", flush=True)
                continue
            eval_one(f"{name}_s{seed}", "main", ck, "cfx", "test", "synth", None,
                     force=args.force, limit=args.limit)
            if seed == SEEDS[0]:
                # The 1024 px plate split is an order of magnitude slower per canvas and
                # is used for whole-fiber statistics, so one seed is enough there.
                eval_one(f"{name}_s{seed}_plate", "main", ck, "cfx", "plate",
                         "synth_plate", staple, force=args.force, limit=args.limit)
    if not args.archs:
        eval_one("classical", "main", None, "angle_joint", "test", "synth", None,
                 force=args.force, limit=args.limit)
        eval_one("classical_plate", "main", None, "angle_joint", "plate", "synth_plate",
                 staple, force=args.force, limit=args.limit)


def stage_eval_ablation(args):
    for name, (_, _, method) in ABLATIONS.items():
        ck = ROOT / "runs/ablation" / name / "best.pt"
        if not ck.exists():
            print(f"[warn] missing {ck}", flush=True)
            continue
        eval_one(name, "ablation", ck, method, "test", "synth_ablation", None,
                 force=args.force, limit=args.limit)


def stage_tune(args):
    """Fix every validation-selected hyperparameter before the test split is touched.

    Two things are chosen here: the crossing-heatmap threshold per model, and the
    linker cost weights and rejection margin for our method.
    """
    import subprocess

    cmd = [sys.executable, str(ROOT / "scripts/tools/tune_threshold.py")]
    if args.force:
        cmd.append("--force")
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    ck = ROOT / "runs/main" / f"{OURS}_s{SEEDS[0]}" / "best.pt"
    target = ROOT / "configs/linker.json"
    if not ck.exists():
        print(f"[warn] cannot tune the linker, missing {ck}", flush=True)
        return
    if target.exists() and not args.force:
        print("[skip] linker already tuned", flush=True)
        return
    subprocess.run([sys.executable, str(ROOT / "scripts/tools/tune_linker.py"),
                    "--checkpoint", str(ck), "--limit", str(args.tune_images)],
                   check=True, cwd=str(ROOT))


def stage_post(args):
    ck = ROOT / "runs/main" / f"{OURS}_s{SEEDS[0]}" / "best.pt"
    for name, (method, margin) in POST_METHODS.items():
        eval_one(name, "post", ck, method, "test", "synth_post", None,
                 force=args.force, limit=args.limit, margin_threshold=margin)


def stage_real(args):
    calib = load_json(ROOT / "data/calibration.json")
    staple = float(calib["staple_pixels"])
    jobs = [(name, ROOT / "runs/main" / f"{name}_s{SEEDS[0]}" / "best.pt", "cfx")
            for name, _ in selected_archs(args)]
    if not args.archs:
        jobs.append(("classical", None, "angle_joint"))
    for name, ck, method in jobs:
        out = group_dir(ROOT / "results", "real") / name
        if (out / "metrics.json").exists() and not args.force:
            print(f"[skip] real {name}", flush=True)
            continue
        if ck is not None and not ck.exists():
            print(f"[warn] missing {ck}", flush=True)
            continue
        print(f"[real ] {name}", flush=True)
        evaluate_real.run(str(ROOT / "data/real"), str(ck) if ck else None, str(out),
                          split=args.real_split, method=method, limit=args.limit,
                          expected_length=staple, tag=name)


STAGES = dict(main=stage_main, tune=stage_tune, ablation=stage_ablation,
              eval=stage_eval, eval_ablation=stage_eval_ablation, post=stage_post,
              real=stage_real)
ORDER = ["main", "tune", "ablation", "eval", "eval_ablation", "post", "real"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", nargs="*", default=ORDER, choices=ORDER)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--plate-size", type=int, default=1024)
    p.add_argument("--arm", type=int, default=56)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--ablation-train", type=int, default=4096)
    p.add_argument("--real-split", default="test")
    p.add_argument("--tune-images", type=int, default=128)
    p.add_argument("--archs", nargs="*", help="restrict the stages to these backbones")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    began = time.time()
    save_json(ROOT / "configs/experiment_matrix.json", dict(
        seeds=SEEDS, steps=args.steps, warmup=args.warmup, batch=args.batch,
        architectures={name: over for name, over in ARCHS},
        ablations={name: dict(overrides=over, corpus=corpus, linker=linker)
                   for name, (over, corpus, linker) in ABLATIONS.items()},
        data_variants=DATA_VARIANTS,
        linkers={k: dict(method=m, margin_threshold=g) for k, (m, g) in POST_METHODS.items()},
        ours=OURS,
        note="the resolved matrix for this invocation; each run also writes its own "
             "effective config.json"))
    log = []
    for stage in ORDER:
        if stage not in args.stage:
            continue
        t = time.time()
        print(f"\n===== stage {stage} =====", flush=True)
        try:
            STAGES[stage](args)
            log.append(dict(stage=stage, seconds=round(time.time() - t, 1), status="ok"))
        except Exception as exc:  # keep later stages runnable after one failure
            traceback.print_exc()
            log.append(dict(stage=stage, seconds=round(time.time() - t, 1),
                            status="failed", error=str(exc)))
        save_json(diagnostic(ROOT / "results", "experiment_log.json"),
                  dict(stages=log, total_seconds=round(time.time() - began, 1),
                       seeds=SEEDS, steps=args.steps, warmup=args.warmup,
                       batch=args.batch))
    print(f"\nall requested stages finished in {(time.time() - began) / 60:.1f} min", flush=True)
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
