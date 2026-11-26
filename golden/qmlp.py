"""
Integer golden model for the INT8 MLP accelerator.

Everything on the inference path is pure Python int (unbounded), so there is
no float rounding and no silent int32 overflow. Floats appear only offline,
in quantize_multiplier(), and both this model and the RTL use the resulting
integers (M0, n).

Requantization (per output neuron):
    v  = sum_i W[o][i] * x[i] + b[o]                 # exact integer
    y  = (v * M0 + 2**(n-1)) >> n                     # round half up, arithmetic shift
    y  = clamp(y, 0, 127)     if relu  (hidden layers)
    y  = clamp(y, -128, 127)  otherwise (last layer)

Python's >> on negative ints floors, exactly like Verilog >>> on a signed
operand, so the two agree bit for bit.
"""
from dataclasses import dataclass, field
from math import floor, log2
from typing import List, Optional, Tuple

INT8_MIN, INT8_MAX = -128, 127
M0_BITS = 15            # 2**14 <= M0 < 2**15, fits a 16-bit signed field as a positive number
N_MAX = 47              # shift amount must fit the RTL's 50-bit product


def quantize_multiplier(M: float, m0_bits: int = M0_BITS) -> Tuple[int, int]:
    """Approximate a positive real M as M0 * 2**-n with M0 in [2**(m0_bits-1), 2**m0_bits)."""
    if not M > 0:
        raise ValueError(f"multiplier must be positive, got {M}")
    n = (m0_bits - 1) - floor(log2(M))
    M0 = round(M * (1 << n)) if n >= 0 else round(M / (1 << -n))
    if M0 >= (1 << m0_bits):          # rounding pushed it up to 2**m0_bits
        M0 >>= 1
        n -= 1
    if not 1 <= n <= N_MAX:
        raise ValueError(f"M={M} needs shift n={n}, outside 1..{N_MAX}")
    return int(M0), int(n)


def requant(v: int, M0: int, n: int, relu: bool) -> int:
    y = (v * M0 + (1 << (n - 1))) >> n
    lo = 0 if relu else INT8_MIN
    return max(lo, min(INT8_MAX, y))


@dataclass
class QLayer:
    W: List[List[int]]      # [out][in], int8 values
    b: List[int]            # [out], int32 values
    M0: int
    n: int
    relu: bool

    @property
    def in_dim(self) -> int:
        return len(self.W[0])

    @property
    def out_dim(self) -> int:
        return len(self.W)


def argmax_first(vals: List[int]) -> int:
    """Index of the maximum; on a tie the FIRST index wins (RTL uses strict >)."""
    best_i, best_v = 0, vals[0]
    for i, v in enumerate(vals[1:], start=1):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def infer(layers: List[QLayer], x: List[int],
          trace: Optional[list] = None, sample: int = 0) -> Tuple[int, List[List[int]]]:
    """Run one inference. Returns (class, per-layer outputs).
    If trace is given, appends (sample, layer, o, v, y) for every requantized neuron,
    in the same order the RTL produces them."""
    a = [int(t) for t in x]
    outs = []
    for li, L in enumerate(layers):
        assert len(a) == L.in_dim, f"layer {li}: input {len(a)} != in_dim {L.in_dim}"
        y = []
        for o in range(L.out_dim):
            acc = sum(int(L.W[o][i]) * a[i] for i in range(L.in_dim))
            v = acc + int(L.b[o])
            yo = requant(v, L.M0, L.n, L.relu)
            if trace is not None:
                trace.append((sample, li, o, v, yo))
            y.append(yo)
        a = y
        outs.append(y)
    return argmax_first(a), outs


# ---------------------------------------------------------------------------
# Hardware export
# ---------------------------------------------------------------------------

def _hex(v: int, bits: int) -> str:
    return format(v & ((1 << bits) - 1), f"0{(bits + 3) // 4}x")


REQ_PIPE = 4   # rtl/mlp_accel.v: drain cycles per layer of the pipelined requant unit


def cycles_expected(layers: List[QLayer], P: int, pipe: int = REQ_PIPE) -> int:
    """Cycle count from the start pulse to done, matching the FSM in mlp_accel.v.
    pipe = 0 gives the single-cycle requant design (rtl/mlp_accel_comb.v)."""
    c = pipe * len(layers)                          # drain at every layer boundary
    for L in layers:
        groups = -(-L.out_dim // P)
        for g in range(groups):
            c += L.in_dim                       # MAC: one input per cycle
            c += min(P, L.out_dim - g * P)      # serial requant of this group's outputs
    c += layers[-1].out_dim                     # argmax, one output per cycle
    return c + 1                                # done is registered one cycle later


def export_hw(layers: List[QLayer], P: int, outdir: str) -> dict:
    """Write wrom.hex / brom.hex / cfg.hex for parallelism P and return RTL parameters.

    Weight ROM layout (output-stationary): for each layer, for each group g of P
    output neurons, for each input i: one word holding W[g*P+p][i] in lane p
    (bits [8p+7:8p]); lanes past out_dim are zero. Addresses are contiguous
    across layers, so the RTL walks the ROM with a single counter.

    cfg word (96 bits): [7:0] in_dim  [15:8] out_dim  [23:16] groups
                        [39:24] w_base  [55:40] b_base  [71:56] M0
                        [79:72] n  [80] relu
    """
    import os
    os.makedirs(outdir, exist_ok=True)
    wlines, blines, clines = [], [], []
    w_base = b_base = 0
    for L in layers:
        groups = -(-L.out_dim // P)
        for g in range(groups):
            for i in range(L.in_dim):
                word = 0
                for p in range(P):
                    o = g * P + p
                    w = L.W[o][i] if o < L.out_dim else 0
                    word |= (w & 0xFF) << (8 * p)
                wlines.append(_hex(word, 8 * P))
        for o in range(L.out_dim):
            blines.append(_hex(int(L.b[o]), 32))
        cfg = (L.in_dim | (L.out_dim << 8) | (groups << 16) | (w_base << 24)
               | (b_base << 40) | (L.M0 << 56) | (L.n << 72) | (int(L.relu) << 80))
        clines.append(_hex(cfg, 96))
        w_base += groups * L.in_dim
        b_base += L.out_dim

    for name, lines in (("wrom.hex", wlines), ("brom.hex", blines), ("cfg.hex", clines)):
        with open(os.path.join(outdir, name), "w") as f:
            f.write("\n".join(lines) + "\n")

    return {
        "P": P,
        "NL": len(layers),
        "MAX_DIM": max([layers[0].in_dim] + [L.out_dim for L in layers]),
        "W_DEPTH": len(wlines),
        "B_DEPTH": len(blines),
    }


def check_ranges(layers: List[QLayer]) -> None:
    """Assert the model fits the RTL's fixed field widths."""
    for li, L in enumerate(layers):
        for row in L.W:
            for w in row:
                assert INT8_MIN <= w <= INT8_MAX, f"layer {li}: weight {w} not int8"
        for b in L.b:
            assert -(1 << 31) <= b < (1 << 31), f"layer {li}: bias {b} not int32"
        assert (1 << (M0_BITS - 1)) <= L.M0 < (1 << M0_BITS), f"layer {li}: M0 {L.M0}"
        assert 1 <= L.n <= N_MAX, f"layer {li}: n {L.n}"
        assert L.in_dim <= 255 and L.out_dim <= 255
    for a, b in zip(layers, layers[1:]):
        assert a.out_dim == b.in_dim, "layer dims do not chain"
