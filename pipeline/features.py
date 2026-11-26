"""
Current-signature features for bearing faults.

A bearing fault with characteristic frequency f_c (BPFO or BPFI) modulates the
phase current, giving sidebands at |h*f_e +- k*f_c| around the fundamental f_e
and its harmonics h. An inner-race defect also rotates in and out of the load
zone, adding lines at +-f_r around those (|h*f_e +- k*BPFI +- f_r|).

All of these scale with the shaft frequency f_r (f_e = p*f_r for a synchronous
motor with p pole pairs), so the set of target lines is fixed in ORDERS
(multiples of f_r) once p is known, independent of speed. Candidates are taken
in priority order and a candidate is DROPPED when its line
  * lies within MIN_SEP orders of a fundamental harmonic (incl. DC), where the
    leakage of the much larger harmonic would dominate, or
  * coincides (within MIN_SEP) with an already selected line.
The second rule matters: BPFO + BPFI = N*f_r, so for N = 8 balls and p = 4,
f_e - BPFO and BPFI - f_e are the SAME frequency, and that line cannot tell an
outer from an inner fault.

Feature value: spectrum level in dB relative to the fundamental, max over
+-TOL_BINS bins (rolling elements slip ~1-2 %).
"""
import numpy as np

from paderborn import FS, fault_freqs, rpm_of

L_FFT = 16384          # 0.256 s at 64 kHz, df = 3.9 Hz
HOP = 8192             # 50 % overlap -> ~30 windows per 4 s file
N_FEAT = 32
K_MAX = 4
TOL_BINS = 2
MIN_SEP = 0.6          # orders; > 2*TOL_BINS*df/f_r at 900 rpm (0.52)
FE_SEARCH = (20.0, 400.0)


def _candidates(p, bo, bi, k_max=K_MAX, hs=(1, 3, 5)):
    """Priority-ordered (name, order) candidates. bo, bi: BPFO, BPFI in orders."""
    c = []
    for h in hs:
        for k in range(1, k_max + 1):
            for fc_name, fc in (("BPFO", bo), ("BPFI", bi)):
                for sgn, s in (("-", -1), ("+", 1)):
                    c.append((f"{h}fe{sgn}{k}{fc_name}", abs(h * p + s * k * fc)))
        for k in (1, 2):                                   # inner-race load-zone lines
            for sgn, s in (("-", -1), ("+", 1)):
                for rs, r in (("-", -1), ("+", 1)):
                    c.append((f"{h}fe{sgn}{k}BPFI{rs}fr", abs(h * p + s * k * bi + r)))
    return c


def select_targets(p, rpm=1500.0, n_feat=N_FEAT, min_sep=MIN_SEP):
    """Pick n_feat target lines (in orders) for a motor with p pole pairs."""
    fr = rpm / 60.0
    bpfo, bpfi = fault_freqs(rpm)
    bo, bi = bpfo / fr, bpfi / fr
    harmonics = [h * p for h in (0, 1, 3, 5, 7, 9, 11)]    # DC and odd harmonics of f_e
    chosen, dropped = [], []
    pool = _candidates(p, bo, bi)
    seen = {n for n, _ in pool}
    pool += [(n, o) for n, o in _candidates(p, bo, bi, k_max=6, hs=(1, 3, 5, 7)) if n not in seen]   # fallback lines
    for name, o in pool:
        if min(abs(o - hp) for hp in harmonics) < min_sep:
            dropped.append((name, o, "near f_e harmonic/DC"))
        elif any(abs(o - oc) < min_sep for _, oc in chosen):
            other = next(n for n, oc in chosen if abs(o - oc) < min_sep)
            dropped.append((name, o, f"coincides with {other}"))
        else:
            chosen.append((name, o))
        if len(chosen) == n_feat:
            break
    if len(chosen) < n_feat:
        raise ValueError(f"only {len(chosen)} usable lines for p={p}")
    return chosen, dropped


def _frames(x, L, hop):
    n = 1 + (len(x) - L) // hop
    idx = np.arange(L)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def estimate_fe(mag_mean, df, search=FE_SEARCH):
    """Fundamental frequency from the strongest line in the search band,
    refined by parabolic interpolation of the log magnitude."""
    lo, hi = int(search[0] / df), int(search[1] / df)
    b = lo + int(np.argmax(mag_mean[lo:hi]))
    y0, y1, y2 = np.log(mag_mean[b - 1:b + 2] + 1e-30)
    denom = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    return (b + delta) * df


def extract(signals: dict, cond: str, p=None, L=L_FFT, hop=HOP, tol=TOL_BINS):
    """signals: {name: samples} of one measurement file.
    p: pole pairs; estimated as round(f_e / f_r) when None.
    Returns (features [n_win, N_FEAT], info)."""
    cur = [v for k, v in signals.items() if "phase_current" in k.lower()]
    if not cur:
        raise ValueError(f"no phase_current channel in {list(signals)}")
    n = min(len(c) for c in cur)
    w = np.hanning(L)
    power = 0.0
    for c in cur:                                     # average power over the current channels
        fr_ = _frames(c[:n], L, hop)
        fr_ = fr_ - fr_.mean(axis=1, keepdims=True)
        power = power + np.abs(np.fft.rfft(fr_ * w, axis=1)) ** 2
    mag = np.sqrt(power / len(cur))
    df = FS / L

    rpm = rpm_of(cond)
    fr = rpm / 60.0
    fe = estimate_fe(mag.mean(axis=0), df)
    p_est = fe / fr
    if p is None:
        p = int(round(p_est))
    targets, _ = select_targets(p, rpm)
    # scale the order-domain targets with the measured f_e (absorbs small speed error)
    f_t = np.array([o for _, o in targets]) * (fe / p)

    b_fe = int(round(fe / df))
    ref = mag[:, b_fe - tol:b_fe + tol + 1].max(axis=1, keepdims=True)
    db = 20 * np.log10(mag / ref + 1e-12)

    bins = np.clip(np.round(f_t / df).astype(int), tol, db.shape[1] - tol - 1)
    feats = np.stack([db[:, b - tol:b + tol + 1].max(axis=1) for b in bins], axis=1)
    bpfo, bpfi = fault_freqs(rpm)
    return feats.astype(np.float32), dict(fe=fe, p=p, p_est=p_est, bpfo=bpfo, bpfi=bpfi, df=df,
                                          n_win=len(feats), names=[nm for nm, _ in targets])


def feature_names(p):
    return [nm for nm, _ in select_targets(p)[0]]
