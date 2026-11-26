#!/usr/bin/env python3
"""
Synthetic phase-current files in the same layout and .mat structure as the
Paderborn data, for UNIT-TESTING the pipeline only (loader, features, training,
quantization, export, RTL). Accuracy obtained on these files says nothing about
real diagnostic performance.

Signal per file (64 kHz, 4 s, two phases 120 deg apart):
    fundamental f_e (+3rd/5th harmonics) + white noise
    + for a faulty bearing, sidebands at |h*f_e +- k*f_c'|, k = 1..3, where
      f_c' = f_c * (1 - slip) (rolling elements slip ~0-2 %)
    + for an inner-race fault, an extra +-f_r modulation of those lines.
"""
import argparse
import os
import sys

import numpy as np
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paderborn as pb   # noqa: E402

POLE_PAIRS = 4           # assumption for the synthetic drive only; the pipeline estimates f_e


def make_currents(code, cond, rng, seconds=4.0, fs=pb.FS):
    t = np.arange(int(seconds * fs)) / fs
    rpm = pb.rpm_of(cond)
    fr = rpm / 60
    fe = POLE_PAIRS * fr * (1 + rng.normal(0, 1e-3))
    bpfo, bpfi = pb.fault_freqs(rpm)
    lab = pb.label_of(code)
    brng = np.random.default_rng(abs(hash(code)) % 2**32)          # per-bearing severity
    sev_db = brng.uniform(-62, -48)
    slip = brng.uniform(0.0, 0.02)
    out = []
    for ph in (0.0, -2 * np.pi / 3):
        x = np.sin(2 * np.pi * fe * t + ph)
        x += 10 ** (-30 / 20) * np.sin(3 * (2 * np.pi * fe * t + ph))
        x += 10 ** (-36 / 20) * np.sin(5 * (2 * np.pi * fe * t + ph))
        if lab:
            fc = (bpfo if lab == 1 else bpfi) * (1 - slip)
            for h, hg in ((1, 1.0), (3, 0.3)):
                for k in (1, 2, 3):
                    a = 10 ** ((sev_db - 6 * (k - 1)) / 20) * hg
                    for sgn in (-1, 1):
                        f = abs(h * fe + sgn * k * fc)
                        x += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
                        if lab == 2:                         # load-zone modulation at f_r
                            for s2 in (-1, 1):
                                x += 0.5 * a * np.sin(2 * np.pi * abs(f + s2 * fr) * t + rng.uniform(0, 2 * np.pi))
        x += rng.normal(0, 10 ** (-40 / 20), len(t))           # broadband noise / PWM residue
        out.append(x.astype(np.float32))
    return out


def write_mat(path, name, signals):
    rec = np.zeros((1, len(signals)), dtype=[("Name", "O"), ("Type", "O"), ("Unit", "O"),
                                              ("Data", "O"), ("Raster", "O")])
    for i, (sig_name, data) in enumerate(signals.items()):
        rec[0, i] = (sig_name, "data", "A", data[None, :], "HostService")
    savemat(path, {name: {"Info": "synthetic", "Y": rec}}, do_compression=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--files", type=int, default=6, help="measurements per bearing and condition")
    ap.add_argument("--conditions", default="baseline")
    a = ap.parse_args()
    conds = [pb.BASELINE] if a.conditions == "baseline" else pb.CONDITIONS
    rng = np.random.default_rng(0)
    codes = pb.TRAIN_BEARINGS + pb.TEST_BEARINGS
    for code in codes:
        os.makedirs(os.path.join(a.dest, code), exist_ok=True)
        for cond in conds:
            for k in range(1, a.files + 1):
                name = f"{cond}_{code}_{k}"
                i1, i2 = make_currents(code, cond, rng)
                write_mat(os.path.join(a.dest, code, name + ".mat"), name,
                          {"phase_current_1": i1, "phase_current_2": i2})
        print(code, end=" ", flush=True)
    print("\nwrote", a.dest)


if __name__ == "__main__":
    main()
