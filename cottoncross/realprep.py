"""Real-plate preparation: crossing-centred windows and the augmentation gallery.

Nothing here is an annotation.  The windows are sampling locations found by a
training-free ridge detector, cached so that (a) unlabelled training crops are
guaranteed to contain a complete crossing and (b) the exported augmentation gallery can
be inspected to confirm that guarantee visually.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .background import normalise
from .calibrate import crossing_windows
from .common import load_json, read_image, save_json, write_image
from .dataset import photometric


def build_windows(real="data/real", window=384, arm=56, per_plate=24, out=None):
    root = Path(real)
    rows = load_json(root / "manifest.json")
    records = []
    for i, r in enumerate(rows):
        gray = read_image(root / r["roi"], True)
        found = crossing_windows(gray, window, arm=arm, max_windows=per_plate)
        for w in found:
            records.append(dict(id=r["id"], page=r["page"], split=r["split"], **w))
        print(f"windows {i + 1}/{len(rows)} {r['id']}: {len(found)}", flush=True)
    payload = dict(
        window=window, arm_pixels=arm, per_plate_cap=per_plate,
        total=len(records),
        by_split={s: sum(w["split"] == s for w in records) for s in ("train", "val", "test")},
        plates_with_window=len({w["id"] for w in records}),
        plates=len(rows),
        rule=("skeleton junction of degree>=3 whose centre is interior by 28% of the "
              "window and whose arms occupy at least 4 of 8 angular sectors at radius "
              f"{arm} px; unsupervised, no human labels"),
        windows=records)
    save_json(out or (root / "crossing_windows.json"), payload)
    print(f"cached {len(records)} crossing windows over {payload['plates_with_window']} plates")
    return payload


def export_gallery(real="data/real", out="data/real_aug", window=384, plates=16,
                   variants=8, seed=42):
    """Export inspectable augmented real crops, every one containing a full crossing."""
    root = Path(real)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = load_json(root / "crossing_windows.json")
    windows = [w for w in payload["windows"] if w["split"] == "train"]
    by_plate = {}
    for w in windows:
        by_plate.setdefault(w["id"], []).append(w)
    chosen = list(by_plate)[:plates]
    rng = np.random.default_rng(seed)
    rows = []
    cells = []
    for pid in chosen:
        gray = read_image(root / f"roi/{pid}.png", True)
        w = max(by_plate[pid], key=lambda d: (d["arm_sectors"], d["fiber_fraction"]))
        x0, y0, x1, y1 = w["xyxy"]
        crop = gray[y0:y1, x0:x1]
        for v in range(variants):
            k = v % 4
            aug = np.rot90(crop, k)
            if v >= 4:
                aug = np.fliplr(aug)
            aug = photometric(np.ascontiguousarray(aug), rng, strong=True).astype(np.uint8)
            name = f"{pid}_w{y0:04d}x{x0:04d}_r{k * 90}_f{int(v >= 4)}.png"
            write_image(out / name, aug)
            rows.append(dict(file=name, plate=pid, xyxy=[x0, y0, x1, y1], rot=k * 90,
                             flip=int(v >= 4), arm_sectors=w["arm_sectors"],
                             junction_xy=w["junction_xy"], degree=w["degree"]))
            if v == 0:
                vis = cv2.cvtColor(cv2.resize(aug, (176, 176)), cv2.COLOR_GRAY2BGR)
                jx, jy = w["junction_xy"]
                cv2.circle(vis, (int(jx * 176 / window), int(jy * 176 / window)), 10,
                           (40, 60, 255), 1, cv2.LINE_AA)
                cells.append(vis)
    if cells:
        while len(cells) % 4:
            cells.append(np.zeros_like(cells[0]))
        grid = np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)])
        write_image(out / "contact.jpg", grid)
    save_json(out / "manifest.json", dict(
        count=len(rows), plates=len(chosen), window=window, variants=variants,
        guarantee="every crop is centred on a junction whose four arms are inside the crop",
        note="augmentation examples for inspection; training uses the same operations online",
        items=rows))
    print(f"exported {len(rows)} augmented real crops from {len(chosen)} plates")
    return rows


def background_report(real="data/real", out="results/diagnostics/background_report.json", limit=None):
    """Quantify what background normalisation removes, per plate."""
    root = Path(real)
    rows = load_json(root / "manifest.json")
    if limit:
        rows = rows[:limit]
    records = []
    for r in rows:
        gray = read_image(root / r["roi"], True).astype(np.float32) / 255.0
        ch = normalise((gray * 255).astype(np.uint8))
        # Illumination span before vs after flattening, and grain contrast before/after.
        detail = gray - cv2.GaussianBlur(gray, (0, 0), 6)
        h = gray.shape[0]
        # Compare the LARGE-SCALE component of each representation, both expressed as a
        # fraction of that representation's own full dynamic range, so the two are
        # directly comparable.
        def shading(channel):
            low = cv2.GaussianBlur(channel.astype(np.float32), (0, 0), 40)
            span = np.percentile(channel, 99) - np.percentile(channel, 1)
            return float((np.percentile(low, 99) - np.percentile(low, 1)) / max(span, 1e-6))

        records.append(dict(
            id=r["id"], split=r["split"],
            illumination_span_raw=shading(gray),
            illumination_span_flat=shading(ch[0]),
            illumination_span_snr=shading(ch[1]),
            grain_std_top=float(detail[: h // 2].std()),
            grain_std_bottom=float(detail[h // 2:].std()),
            grain_ratio_raw=float(detail[h // 2:].std() / max(detail[: h // 2].std(), 1e-6)),
            grain_ratio_snr=float(ch[1][h // 2:].std() / max(ch[1][: h // 2].std(), 1e-6)),
            coherence_mean=float(ch[2].mean()),
        ))
    summary = dict(
        n=len(records),
        illumination_span_raw=float(np.mean([r["illumination_span_raw"] for r in records])),
        illumination_span_flat=float(np.mean([r["illumination_span_flat"] for r in records])),
        illumination_span_snr=float(np.mean([r["illumination_span_snr"] for r in records])),
        grain_ratio_raw=float(np.mean([r["grain_ratio_raw"] for r in records])),
        grain_ratio_snr=float(np.mean([r["grain_ratio_snr"] for r in records])),
        reading=("grain_ratio is the ratio of high-frequency contrast between the grainy "
                 "lower half and the smooth upper half of each plate; a value near 1 "
                 "after normalisation means the substrate no longer looks like structure. "
                 "illumination_span is the large-scale (sigma=40 px) swing as a fraction "
                 "of each representation's own dynamic range; near 0 means flat."),
        per_plate=records)
    save_json(out, summary)
    print({k: v for k, v in summary.items() if k != "per_plate"})
    return summary


def domain_gap_report(real="data/real", synth="data/synth",
                      out="results/diagnostics/domain_gap.json", plates=12, canvases=48):
    """How far apart are synthetic and real images *after* background normalisation?

    The comparison is made in the representation the network actually sees.  Fiber
    evidence is read on the centerline (synthetic, exact) and on the training-free fiber
    mask (real, approximate); background is read everywhere else.  A small gap here is
    the argument that the analytic normalisation, not the network, absorbs the
    photometric domain shift.
    """
    from .calibrate import fiber_mask

    sroot, rroot = Path(synth), Path(real)
    rows = [r for r in load_json(sroot / "manifest.json") if r["split"] == "test"][:canvases]
    syn_fg, syn_bg = [], []
    for r in rows:
        ch = normalise(read_image(sroot / r["image"], True))
        with np.load(sroot / r["target"]) as z:
            centre = z["center"] > 128
            occupied = z["mask"] > 0
        if centre.sum() > 50:
            syn_fg.append(float(np.median(ch[1][centre])))
            syn_bg.append(float(np.percentile(np.abs(ch[1][~occupied]), 95)))

    rrows = [r for r in load_json(rroot / "manifest.json") if r["split"] == "train"][:plates]
    real_fg, real_bg = [], []
    for r in rrows:
        gray = read_image(rroot / r["roi"], True)
        ch = normalise(gray)
        mask = fiber_mask(gray)
        if mask.sum() > 200:
            real_fg.append(float(np.median(ch[1][mask])))
            real_bg.append(float(np.percentile(np.abs(ch[1][~mask]), 95)))

    def stat(values):
        return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)),
                    n=len(values))

    summary = dict(
        channel="local SNR (channel 1 of the normalised representation)",
        synthetic_foreground=stat(syn_fg), real_foreground=stat(real_fg),
        synthetic_background=stat(syn_bg), real_background=stat(real_bg),
        foreground_ratio=float(np.mean(syn_fg) / max(np.mean(real_fg), 1e-6)),
        background_ratio=float(np.mean(syn_bg) / max(np.mean(real_bg), 1e-6)),
        reading=("ratios near 1.0 mean the two domains look alike in the representation "
                 "the network sees. Real foreground is read on a training-free mask, so "
                 "it is an approximation, not a label."),
        caveat="a small photometric gap does not imply a small geometric or textural gap")
    save_json(out, summary)
    print({k: v for k, v in summary.items() if isinstance(v, (float, str))})
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real", default="data/real")
    p.add_argument("--window", type=int, default=384)
    p.add_argument("--arm", type=int, default=56)
    p.add_argument("--per-plate", type=int, default=24)
    p.add_argument("--skip-windows", action="store_true")
    a = p.parse_args()
    if not a.skip_windows:
        build_windows(a.real, a.window, a.arm, a.per_plate)
    export_gallery(a.real, window=a.window)
    background_report(a.real)
    domain_gap_report(a.real)
