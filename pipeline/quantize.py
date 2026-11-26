"""
Post-training quantization of the float MLP to the accelerator's integer format.

Scales (symmetric, per tensor):
    input   x_q = clamp(round(z / s_in), -128, 127),  s_in = P99.9(|z|) / 127
    weight  W_q = round(W / s_w),                     s_w  = max(|W|) / qmax
    act     s_a = P99.9(post-ReLU activation) / 127   (hidden), P99.9(|logit|) / 127 (last)
    bias    b_q = round(b / (s_w * s_x))
    requant M   = s_w * s_x / s_out  ->  (M0, n) via quantize_multiplier
The layer then computes y_q = clamp(round(M * (W_q x_q + b_q))) exactly as golden/qmlp.py.
"""
import numpy as np

from qmlp import QLayer, quantize_multiplier, check_ranges

PCT = 99.9


def quantize_input(z, s_in, qmax=127):
    return np.clip(np.round(z / s_in), -qmax - 1, qmax).astype(np.int64)


def ptq(mlp, Z_calib, w_bits=8, a_bits=8):
    """mlp: float MLP trained on standardized inputs Z. Returns (layers, s_in)."""
    qw = 2 ** (w_bits - 1) - 1
    qa = 2 ** (a_bits - 1) - 1
    _, acts = mlp.forward(Z_calib, keep=True)
    s_x = np.percentile(np.abs(Z_calib), PCT) / qa
    s_in = s_x
    layers = []
    nl = len(mlp.W)
    for li, (W, b) in enumerate(zip(mlp.W, mlp.b)):
        relu = li < nl - 1
        s_w = np.abs(W).max() / qw
        Wq = np.clip(np.round(W / s_w), -qw, qw).astype(int)
        bq = np.round(b / (s_w * s_x)).astype(int)
        out = acts[li + 1]
        s_out = np.percentile(out if relu else np.abs(out), PCT) / qa
        s_out = max(s_out, 1e-12)
        M0, n = quantize_multiplier(s_w * s_x / s_out)
        layers.append(QLayer(Wq.tolist(), bq.tolist(), M0, n, relu))
        s_x = s_out
    check_ranges(layers)
    return layers, float(s_in)


def int_forward(layers, Xq, qmax=127):
    """Vectorized integer inference, same arithmetic as qmlp.infer (int64 is wide
    enough: |v| < 2^33, M0 < 2^15). qmax < 127 emulates narrower activations for
    the bit-width sweep; qmax = 127 is what the hardware does."""
    a = np.asarray(Xq, dtype=np.int64)
    for L in layers:
        W = np.asarray(L.W, dtype=np.int64)
        v = a @ W.T + np.asarray(L.b, dtype=np.int64)
        y = (v * L.M0 + (1 << (L.n - 1))) >> L.n
        a = np.clip(y, 0 if L.relu else -qmax - 1, qmax)
    return a


def argmax_first(logits):
    return np.argmax(logits, axis=1)    # numpy returns the first max, same as the RTL
