"""Palette validator, ported from the data-viz skill's Node script.

This machine has no Node runtime, so the six checks are re-implemented here with the
same constants, the same Machado CVD matrices and the same OKLab distance, and run
against the figure palette in `cottoncross/figures.py`.  Run it whenever the palette
changes:  python scripts/tools/validate_palette.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}


def hex_to_srgb(h):
    h = h.strip().lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear(h):
    return to_linear(hex_to_srgb(h))


def relative_luminance(h):
    r, g, b = linear(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_linear(rgb):
    r, g, b = rgb
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s])


def oklch(h):
    L, a, b = oklab_from_linear(linear(h))
    return L, float(np.hypot(a, b))


def simulate(h, kind):
    return np.clip(MACHADO[kind] @ linear(h), 0, 1)


def delta_e(h1, h2, kind=None):
    a = oklab_from_linear(simulate(h1, kind) if kind else linear(h1))
    b = oklab_from_linear(simulate(h2, kind) if kind else linear(h2))
    return 100 * float(np.linalg.norm(a - b))


def validate(palette, mode="light", pairs="adjacent"):
    surface = SURFACE[mode]
    lo, hi = BAND[mode]
    report, ok = [], True

    off = [(c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi]
    ok &= not off
    report.append(("Lightness band", not off,
                   f"outside [{lo}, {hi}]: {off}" if off else f"all {len(palette)} inside"))

    low_c = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not low_c
    report.append(("Chroma floor", not low_c,
                   f"reads gray: {low_c}" if low_c else f"all >= {CHROMA_FLOOR}"))

    n = len(palette)
    pairlist = (list(combinations(range(n), 2)) if pairs == "all"
                else [(i, i + 1) for i in range(n - 1)])
    worst_cvd, worst_cvd_pair = 99.0, None
    for i, j in pairlist:
        d = min(delta_e(palette[i], palette[j], "protan"),
                delta_e(palette[i], palette[j], "deutan"))
        if d < worst_cvd:
            worst_cvd, worst_cvd_pair = d, (palette[i], palette[j])
    state = "pass" if worst_cvd >= CVD_TARGET else "floor" if worst_cvd >= CVD_FLOOR else "fail"
    ok &= state != "fail"
    report.append((f"CVD separation ({pairs})", state != "fail",
                   f"worst {worst_cvd:.1f} on {worst_cvd_pair} [{state}]"))

    tritan = min((delta_e(palette[i], palette[j], "tritan") for i, j in pairlist), default=99)
    report.append(("Tritan separation", True, f"worst {tritan:.1f} (informational)"))

    worst_normal, normal_pair = 99.0, None
    for i, j in pairlist:
        d = delta_e(palette[i], palette[j])
        if d < worst_normal:
            worst_normal, normal_pair = d, (palette[i], palette[j])
    ok &= worst_normal >= NORMAL_FLOOR
    report.append(("Normal-vision floor", worst_normal >= NORMAL_FLOOR,
                   f"worst {worst_normal:.1f} on {normal_pair} (floor {NORMAL_FLOOR})"))

    low = [(c, round(contrast(c, surface), 2)) for c in palette
           if contrast(c, surface) < CONTRAST_MIN]
    report.append(("Surface contrast", not low,
                   f"below {CONTRAST_MIN}:1 (needs direct labels): {low}" if low
                   else f"all >= {CONTRAST_MIN}:1"))
    return report, ok


def main():
    from cottoncross.figures import METHOD_ORDER, PALETTE

    palette = [PALETTE[m] for m in METHOD_ORDER if m in PALETTE]
    print(f"figure palette in plotting order ({len(palette)}): {palette}\n")
    everything_ok = True
    for mode in ("light",):
        for pairs in ("adjacent", "all"):
            report, ok = validate(palette, mode, pairs)
            print(f"--- mode={mode} pairs={pairs} ---")
            for name, passed, detail in report:
                print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
            print(f"  => {'OK' if ok else 'NEEDS FIX'}\n")
            if pairs == "adjacent":
                everything_ok &= ok
    return 0 if everything_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
