#!/usr/bin/env python3
"""
Compact per-file spectra for diagnosing what the current (and vibration)
signals actually contain, small enough to share (~1 MB).

    python pipeline/diagnose.py --data C:\\paderborn

Writes results/diag.npz with, for every measurement file of the train and
test bearings:
    cur_db  [files, bins]  mean phase-current spectrum, dB re fundamental, 0-1000 Hz
    env_db  [files, bins]  envelope spectrum of the vibration signal, dB, 0-1000 Hz
                           (NaN when the file has no vibration channel)
    freqs, codes, labels, conds, fe
"""
import argparse
import os
import sys

import numpy as np
from scipy.signal import hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paderborn as pb       # noqa: E402
import features as ft        # noqa: E402

F_MAX = 1000.0


def mean_spectrum(x, L=ft.L_FFT, hop=ft.HOP):
    fr = ft._frames(x, L, hop)
    fr = fr - fr.mean(axis=1, keepdims=True)
    return np.sqrt((np.abs(np.fft.rfft(fr * np.hanning(L), axis=1)) ** 2).mean(axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--conditions", default="baseline")
    a = ap.parse_args()
    conds = {"baseline": [pb.BASELINE], "all": pb.CONDITIONS}.get(a.conditions, a.conditions.split(","))
    files = pb.list_files(a.data, pb.TRAIN_BEARINGS + pb.TEST_BEARINGS, conds)
    if not files:
        raise SystemExit("no Paderborn .mat files found.\n" + pb.diagnose(a.data, pb.TRAIN_BEARINGS, conds))

    df = pb.FS / ft.L_FFT
    nb = int(F_MAX / df) + 1
    cur, env, codes, labels, cnds, fes = [], [], [], [], [], []
    for k, (path, code, cond) in enumerate(files):
        sig = pb.load_signals(path)
        ch = [v for n, v in sig.items() if "phase_current" in n.lower()]
        n = min(len(c) for c in ch)
        mag = np.sqrt(np.mean([mean_spectrum(c[:n]) ** 2 for c in ch], axis=0))
        fe = ft.estimate_fe(mag, df)
        b = int(round(fe / df))
        cur.append(20 * np.log10(mag[:nb] / mag[b - 2:b + 3].max() + 1e-12))
        vib = [v for n_, v in sig.items() if "vibration" in n_.lower()]
        if vib:
            e = np.abs(hilbert(vib[0] - vib[0].mean()))
            em = mean_spectrum(e)
            env.append(20 * np.log10(em[:nb] / (em[1:nb].max() + 1e-30) + 1e-12))
        else:
            env.append(np.full(nb, np.nan))
        codes.append(code)
        labels.append(pb.label_of(code))
        cnds.append(cond)
        fes.append(fe)
        print(f"\r{k + 1}/{len(files)}", end="", flush=True)
    print()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "diag.npz")
    np.savez_compressed(path, cur_db=np.array(cur, np.float32), env_db=np.array(env, np.float32),
                        freqs=np.arange(nb) * df, codes=np.array(codes), labels=np.array(labels),
                        conds=np.array(cnds), fe=np.array(fes))
    print("wrote", os.path.abspath(path), f"({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
