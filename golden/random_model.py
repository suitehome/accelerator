"""
Random INT8 MLP + test inputs, used to verify the RTL before the real
(trained) model exists. The scales are calibrated so activations spread over
the int8 range and some outputs saturate, which exercises the clamp logic.
"""
import json
import numpy as np

from qmlp import QLayer, quantize_multiplier, check_ranges

TARGET_STD = 40.0   # aim requantized outputs at roughly this spread before clamping


def make_random_model(dims=(32, 16, 4), seed=0, n_calib=256, pow2=False):
    """pow2=True forces M0 = 2**14 so every scale is a power of two; then many
    neurons land exactly on .5 before rounding, including negative ones, which
    is where round-half-up, round-half-even and round-away-from-zero differ."""
    rng = np.random.default_rng(seed)
    x_cal = np.clip(np.round(rng.normal(0, 40, (n_calib, dims[0]))), -128, 127).astype(np.int64)
    layers = []
    a = x_cal
    for li, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        relu = li < len(dims) - 2
        W = np.clip(np.round(rng.normal(0, 40, (dout, din))), -127, 127).astype(np.int64)
        acc = a @ W.T
        sd = float(acc.std()) or 1.0
        b = np.round(rng.normal(0, 0.3 * sd, dout)).astype(np.int64)
        M0, n = quantize_multiplier(TARGET_STD / sd)
        if pow2:
            M0 = 1 << 14
            if li == 0:                      # make layer 0 shift small: halves every 2**(n-14)
                W = np.clip(W // 16, -127, 127)
                acc = a @ W.T
                b = b // 16
            n = 14 + max(1, round(np.log2(max(float(acc.std()), 1.0) / TARGET_STD)))
        L = QLayer(W.tolist(), b.tolist(), M0, n, relu)
        layers.append(L)
        # propagate calibration data through the integer layer
        v = acc + b
        y = (v * M0 + (1 << (n - 1))) >> n
        a = np.clip(y, 0 if relu else -128, 127)
    check_ranges(layers)
    return layers


def make_inputs(in_dim, n_samples, seed=1):
    rng = np.random.default_rng(seed)
    X = np.clip(np.round(rng.normal(0, 40, (n_samples, in_dim))), -128, 127).astype(int).tolist()
    # corner cases: all max, all min, all zero, alternating extremes
    X[:0] = [[127] * in_dim, [-128] * in_dim, [0] * in_dim,
             [127 if i % 2 else -128 for i in range(in_dim)]]
    return X[:n_samples]


def save_model(layers, path):
    with open(path, "w") as f:
        json.dump([dict(W=L.W, b=L.b, M0=L.M0, n=L.n, relu=L.relu) for L in layers], f)


def load_model(path):
    with open(path) as f:
        return [QLayer(**d) for d in json.load(f)]
