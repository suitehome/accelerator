"""
Vibration envelope features for bearing faults (32 per window).

A local defect is struck once per ball pass, so the vibration contains a train
of impacts at the fault frequency that excites structural resonances. The
ENVELOPE (|Hilbert transform|) demodulates the resonance and leaves lines at
the fault frequencies, so the envelope spectrum is read at:

    k*BPFO (k=1..6), k*BPFI (k=1..6), k*BPFI +- f_r (k=1..3)   18
    k*f_r (k=1..4), k*BSF (k=1..3), FTF, 2*FTF                  9
plus time-domain statistics of the window:
    RMS (dB), kurtosis, crest factor, skewness, envelope kurtosis 5
                                                              -- 32

Line levels are in dB above the median envelope-spectrum level in 20-500 Hz
of the same window, so they do not depend on sensor gain. No band-pass: in a
leave-one-bearing-out check on the TRAINING bearings, the raw signal (0.86
binary accuracy) beat 0.5-5 kHz (0.69), 2-10 kHz (0.71) and 5-20 kHz (0.62).
"""
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

import paderborn as pb

L_FFT = 16384
HOP = 8192
TOL_BINS = 2
BAND = None              # (lo, hi) Hz for an optional band-pass before the envelope


def fault_lines(rpm):
    fr = rpm / 60.0
    r = pb.BALL_D / pb.PITCH_D * np.cos(pb.CONTACT_ANGLE)
    bpfo, bpfi = pb.fault_freqs(rpm)
    bsf = pb.PITCH_D / (2 * pb.BALL_D) * fr * (1 - r * r)
    ftf = fr / 2 * (1 - r)
    names, f = [], []
    for k in range(1, 7):
        names.append(f"{k}BPFO"); f.append(k * bpfo)
    for k in range(1, 7):
        names.append(f"{k}BPFI"); f.append(k * bpfi)
    for k in (1, 2, 3):
        for s, sn in ((-1, "-"), (1, "+")):
            names.append(f"{k}BPFI{sn}fr"); f.append(k * bpfi + s * fr)
    for k in (1, 2, 3, 4):
        names.append(f"{k}fr"); f.append(k * fr)
    for k in (1, 2, 3):
        names.append(f"{k}BSF"); f.append(k * bsf)
    names += ["FTF", "2FTF"]; f += [ftf, 2 * ftf]
    return names, np.array(f)


STAT_NAMES = ["rms_dB", "kurtosis", "crest", "skewness", "env_kurtosis"]


def feature_names(rpm=1500.0):
    return fault_lines(rpm)[0] + STAT_NAMES


def _frames(x, L, hop):
    n = 1 + (len(x) - L) // hop
    return x[np.arange(L)[None, :] + hop * np.arange(n)[:, None]]


def _kurt(a):
    a = a - a.mean(axis=1, keepdims=True)
    return (a ** 4).mean(axis=1) / ((a ** 2).mean(axis=1) ** 2 + 1e-30)


def extract(signals: dict, cond: str, band=BAND, L=L_FFT, hop=HOP, tol=TOL_BINS):
    vib = [v for k, v in signals.items() if "vibration" in k.lower()]
    if not vib:
        raise ValueError(f"no vibration channel in {list(signals)}")
    x = vib[0] - vib[0].mean()
    if band:
        x = sosfiltfilt(butter(4, band, btype="band", fs=pb.FS, output="sos"), x)
    env = np.abs(hilbert(x))
    fx, fe = _frames(x, L, hop), _frames(env, L, hop)
    fe0 = fe - fe.mean(axis=1, keepdims=True)
    df = pb.FS / L
    S = 20 * np.log10(np.abs(np.fft.rfft(fe0 * np.hanning(L), axis=1)) + 1e-12)
    base = np.median(S[:, int(20 / df):int(500 / df)], axis=1, keepdims=True)
    names, f = fault_lines(pb.rpm_of(cond))
    bins = np.clip(np.round(f / df).astype(int), tol, S.shape[1] - tol - 1)
    lines = np.stack([S[:, b - tol:b + tol + 1].max(axis=1) for b in bins], axis=1) - base

    rms = np.sqrt((fx ** 2).mean(axis=1))
    peak = np.abs(fx).max(axis=1)
    skew = ((fx - fx.mean(axis=1, keepdims=True)) ** 3).mean(axis=1) / (rms ** 3 + 1e-30)
    stats = np.stack([20 * np.log10(rms + 1e-12), _kurt(fx), peak / (rms + 1e-12), skew, _kurt(fe)], axis=1)
    feats = np.concatenate([lines, stats], axis=1).astype(np.float32)
    return feats, dict(n_win=len(feats), names=names + STAT_NAMES)
