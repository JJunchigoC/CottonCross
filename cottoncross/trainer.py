"""Unified trainer for every architecture and ablation.

One configuration file drives the whole comparison, so two runs differ only where their
configs differ.  Real plates enter through the unlabelled branch only; no human label
touches training at any point.

Phase 1 (steps <= warmup)  synthetic supervision only.
Phase 2 (steps >  warmup)  synthetic supervision + unlabelled real adaptation:
    * an EMA teacher sees a weak view, the student a strong, context-masked view;
    * an anchor teacher frozen at the end of phase 1 vetoes drift;
    * a pixel is pseudo-labelled only if the EMA teacher, the anchor and the
      horizontally-flipped teacher prediction all agree beyond `confidence`;
    * an equivariance term asks the student's own predictions to transform with the
      image, which needs no teacher and no labels at all;
    * optional staple-prior refinement (SPR, `unsupervised.staple_prior` > 0): short
      structures lying wholly inside a crop cannot be a 35 mm staple and are handed to
      the student as confident background (see `prior.py`).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import load_json, save_json, seed_all
from .dataset import RealCrops, SynthDataset, real_views
from .losses import DEFAULT_WEIGHTS, consistency_loss, equivariance_loss, supervised_loss
from .models import FiberModel, count_parameters
from .prior import staple_negatives

DEFAULT_CONFIG = dict(
    arch="cfxnet", base=32, depth=4, strip=11, model={},
    input_mode="bgn", val_synth=None,
    steps=4000, warmup=2500, batch_size=8, crop=256, crop_mode="crossing",
    learning_rate=1.2e-3, weight_decay=1e-4, seed=42,
    amp=True, validate_every=250, num_workers=4,
    weights=dict(DEFAULT_WEIGHTS),
    unsupervised=dict(enabled=True, weight=0.05, confidence=0.95, ema=0.995,
                      context_mask=True, equivariance=0.02, orientation=0.05,
                      crossing_fraction=0.6, staple_prior=0.0, staple_prior_max=140,
                      staple_prior_clearance=24),
)


def merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


@torch.no_grad()
def validate(model, loader, device, amp):
    """Model-selection score: mean of occupancy Dice and centerline Dice on synthetic val."""
    model.eval()
    stats = np.zeros(6)  # inter,pred,true for mask then center
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            z = model(x)
        p = torch.sigmoid(z[:, 0:2].float()).cpu()
        for j, thr in enumerate((0.5, 0.5)):
            pred = p[:, j] > thr
            truth = y[:, j] > 0.5
            stats[3 * j + 0] += float((pred & truth).sum())
            stats[3 * j + 1] += float(pred.sum())
            stats[3 * j + 2] += float(truth.sum())
    model.train()
    dice = [(2 * stats[3 * j] + 1) / (stats[3 * j + 1] + stats[3 * j + 2] + 1) for j in (0, 1)]
    return float(np.mean(dice)), float(dice[0]), float(dice[1])


def run(config_path=None, overrides=None, synth="data/synth", real="data/real",
        out="runs/cfxnet_s42", resume=None):
    cfg = merge(DEFAULT_CONFIG, load_json(config_path) if config_path else {})
    cfg = merge(cfg, overrides or {})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["amp"]) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    mode = cfg["input_mode"]
    train_set = SynthDataset(synth, "train", augment=True, seed=cfg["seed"],
                             crop=cfg["crop"], mode=mode, crop_mode=cfg["crop_mode"])
    val_set = SynthDataset(cfg["val_synth"] or synth, "val", augment=False, mode=mode,
                           crop=None)
    workers = int(cfg["num_workers"])
    loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=workers, persistent_workers=workers > 0,
                        pin_memory=True, prefetch_factor=4 if workers else None)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], num_workers=0)

    uns = cfg["unsupervised"]
    source = (RealCrops(real, size=cfg["crop"] or train_set.rows[0]["size"], seed=cfg["seed"],
                        crossing_fraction=uns["crossing_fraction"])
              if uns["enabled"] and uns["weight"] > 0 else None)

    model = FiberModel(cfg["arch"], cfg["base"], cfg["depth"], strip=cfg["strip"],
                       **(cfg["model"] or {})).to(device)
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    anchor = None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"])

    start, best = 0, -1.0
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        teacher.load_state_dict(ck["teacher"])
        opt.load_state_dict(ck["optimizer"])
        start, best = ck["step"], ck.get("best_val", -1.0)
        if ck.get("anchor"):
            anchor = copy.deepcopy(teacher)
            anchor.load_state_dict(ck["anchor"])
            anchor.eval()

    save_json(out / "config.json", cfg)
    save_json(out / "environment.json", dict(
        python=sys.version, torch=torch.__version__, numpy=np.__version__,
        device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        amp="bfloat16" if amp else "float32", platform=platform.platform(),
        input_mode=mode,
        parameters=count_parameters(model), architecture=cfg["arch"],
        synthetic=str(synth), real=str(real), resume=resume,
        label_policy="real plates are used without any label; validation is synthetic only"))

    def save(name, step):
        torch.save(dict(model=model.state_dict(), teacher=teacher.state_dict(),
                        optimizer=opt.state_dict(),
                        anchor=anchor.state_dict() if anchor is not None else None,
                        step=step, best_val=best, config=cfg), out / name)

    rng = np.random.default_rng(cfg["seed"])
    rows = []
    iterator = iter(loader)
    began = time.time()
    model.train()
    print(f"[{cfg['arch']}] {count_parameters(model) / 1e6:.2f}M params, "
          f"{len(train_set)} synthetic canvases, device={device}", flush=True)

    for step in range(start + 1, cfg["steps"] + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            z, aux = model(x, want_aux=True)
        sup, parts = supervised_loss(z.float(), y, cfg["weights"],
                                     {k: [a.float() for a in v] if k == "deep" else v
                                      for k, v in aux.items()})
        loss = sup
        unsup_value, coverage, spr_value, spr_fraction = 0.0, 0.0, 0.0, 0.0

        if source is not None and step > cfg["warmup"]:
            if anchor is None:
                anchor = copy.deepcopy(teacher).eval()
                save("phase1.pt", step)
            images = source.sample(cfg["batch_size"])
            weak, _ = real_views(images, rng, use_mask=False, mode=mode)
            weak = weak.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                t_logits = teacher(weak).float()
                a_logits = anchor(weak).float()
                f_logits = torch.flip(teacher(torch.flip(weak, [-1])), [-1]).float()
            tp = torch.sigmoid(t_logits[:, 0:3])
            ap = torch.sigmoid(a_logits[:, 0:3])
            fp = torch.sigmoid(f_logits[:, 0:3])
            c = float(uns["confidence"])
            confident = ((tp > c) & (ap > c) & (fp > c)) | ((tp < 1 - c) & (ap < 1 - c) & (fp < 1 - c))
            spr_weight = float(uns.get("staple_prior", 0.0))
            negative = None
            if spr_weight > 0:
                ruled_out = staple_negatives(
                    tp[:, 0].detach().cpu().numpy(),
                    max_length=float(uns.get("staple_prior_max", 140)),
                    clearance=float(uns.get("staple_prior_clearance", 24)))
                if ruled_out.any():
                    negative = torch.from_numpy(ruled_out).to(device)[:, None].expand(-1, 3, -1, -1)
                    # The prior overrides the teacher: background, and confidently so.
                    t_logits = t_logits.clone()
                    t_logits[:, 0:3][negative] = -12.0
                    confident = confident | negative
            _, strong = real_views(images, rng, tp[:, 2].detach().cpu().numpy(),
                                   use_mask=bool(uns["context_mask"]), mode=mode)
            strong = strong.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                s_logits = model(strong)
            unsup = consistency_loss(s_logits.float(), t_logits, confident,
                                     float(uns["orientation"]))
            spr_loss = s_logits.float().sum() * 0
            if negative is not None:
                spr_loss = F.binary_cross_entropy_with_logits(
                    s_logits[:, 0:3].float()[negative],
                    torch.zeros_like(s_logits[:, 0:3].float()[negative]))
            if uns["equivariance"] > 0 and step % 2 == 0:
                transform = str(rng.choice(["flip_x", "flip_y", "rot90"]))
                unsup = unsup + float(uns["equivariance"]) * equivariance_loss(
                    lambda v: model(v).float(), weak, transform)
            ramp = min(1.0, (step - cfg["warmup"]) / 200.0)
            loss = loss + ramp * (float(uns["weight"]) * unsup + spr_weight * spr_loss)
            unsup_value = float(unsup.detach())
            spr_value = float(spr_loss.detach())
            spr_fraction = float(negative[:, 0].float().mean()) if negative is not None else 0.0
            coverage = float(confident.float().mean())

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        with torch.no_grad():
            decay = min(float(uns["ema"]), 1 - 1 / (step + 1))
            for t, s in zip(teacher.parameters(), model.parameters()):
                t.mul_(decay).add_(s, alpha=1 - decay)
            for t, s in zip(teacher.buffers(), model.buffers()):
                t.copy_(s)
        lr = cfg["learning_rate"] * (0.08 + 0.92 * 0.5 * (1 + np.cos(np.pi * step / cfg["steps"])))
        for group in opt.param_groups:
            group["lr"] = lr

        row = dict(step=step, loss=float(loss.detach()), learning_rate=float(lr),
                   unsupervised=unsup_value, pseudo_coverage=coverage,
                   staple_prior=spr_value, staple_prior_fraction=spr_fraction,
                   elapsed_seconds=time.time() - began, **parts)
        if step % cfg["validate_every"] == 0 or step == cfg["steps"]:
            score, mask_dice, center_dice = validate(teacher, val_loader, device, amp)
            row.update(val_score=score, val_mask_dice=mask_dice, val_center_dice=center_dice)
            if score > best:
                best = score
                save("best.pt", step)
            save("last.pt", step)
            print(f"  step {step}/{cfg['steps']} val={score:.4f} "
                  f"(mask {mask_dice:.4f}, center {center_dice:.4f}) "
                  f"loss={row['loss']:.4f} {row['elapsed_seconds']:.0f}s", flush=True)
        rows.append(row)
        if step % 100 == 0:
            save_json(out / "training_log.json", rows)

    save("last.pt", cfg["steps"])
    save_json(out / "training_log.json", rows)
    fields = sorted({k for r in rows for k in r})
    with (out / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    save_json(out / "summary.json", dict(
        architecture=cfg["arch"], seed=cfg["seed"], steps=cfg["steps"],
        parameters=count_parameters(model), best_val_score=best,
        elapsed_seconds=round(time.time() - began, 1),
        selection="teacher, mean of occupancy and centerline Dice on the synthetic val split",
        real_accuracy=None,
        real_accuracy_reason="no human ground truth exists for these plates"))
    print(f"[{cfg['arch']}] done in {(time.time() - began) / 60:.1f} min, best val {best:.4f}",
          flush=True)
    return dict(out=str(out), best_val_score=best, seconds=time.time() - began)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--set", default="{}", help="JSON dict of config overrides")
    p.add_argument("--synth", default="data/synth")
    p.add_argument("--real", default="data/real")
    p.add_argument("--out", default="runs/cfxnet_s42")
    p.add_argument("--resume")
    a = p.parse_args()
    run(a.config, json.loads(a.set), a.synth, a.real, a.out, a.resume)
