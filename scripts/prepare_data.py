"""Stage 1 of the pipeline: calibration, real crossing windows, synthetic corpus.

Run from the project root:  python scripts/prepare_data.py
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cottoncross import calibrate, realprep, synthgen  # noqa: E402
from cottoncross.common import load_json, save_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", default="data/real")
    p.add_argument("--synth", default="data/synth")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--plate-size", type=int, default=1024)
    p.add_argument("--counts", type=int, nargs=3, default=[4096, 256, 512])
    p.add_argument("--plate-count", type=int, default=64)
    p.add_argument("--arm", type=int, default=56)
    p.add_argument("--staple", type=float, default=0.0,
                   help="fiber arc length in pixels; 0 = use the calibrated value")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--skip", nargs="*", default=[])
    a = p.parse_args()

    began = time.time()
    steps = {}

    if "calibrate" not in a.skip:
        t = time.time()
        info = calibrate.main(a.real, "data/calibration.json")
        steps["calibrate"] = round(time.time() - t, 1)
    else:
        info = load_json("data/calibration.json")

    staple = a.staple or float(info["staple_pixels"])
    print(f"[prepare] staple length = {staple:.1f} px "
          f"({staple / calibrate.NOMINAL_STAPLE_MM:.2f} px/mm)", flush=True)

    if "windows" not in a.skip:
        t = time.time()
        realprep.build_windows(a.real, a.size, a.arm, per_plate=24)
        steps["windows"] = round(time.time() - t, 1)

    if "synth" not in a.skip:
        t = time.time()
        synthgen.generate(a.real, a.synth, a.size, a.plate_size, tuple(a.counts),
                          a.plate_count, staple, float(a.arm), seed=a.seed)
        steps["synth"] = round(time.time() - t, 1)

    if "gallery" not in a.skip:
        t = time.time()
        realprep.export_gallery(a.real, "data/real_aug", a.size)
        steps["gallery"] = round(time.time() - t, 1)

    if "background" not in a.skip:
        t = time.time()
        realprep.background_report(a.real, "results/diagnostics/background_report.json")
        steps["background"] = round(time.time() - t, 1)

    if "domain_gap" not in a.skip and Path(a.synth, "manifest.json").exists():
        t = time.time()
        realprep.domain_gap_report(a.real, a.synth, "results/diagnostics/domain_gap.json")
        steps["domain_gap"] = round(time.time() - t, 1)

    save_json("data/prepare_summary.json",
              dict(seconds=steps, total_seconds=round(time.time() - began, 1),
                   staple_pixels=staple, calibration=info))
    print(f"[prepare] done in {time.time() - began:.0f}s: {steps}", flush=True)


if __name__ == "__main__":
    main()
