#!/usr/bin/env python3
"""
Bit-exact verification: golden model vs. RTL simulation (Icarus Verilog).

    python3 run_sim.py                    # random model, P = 1 4 16, 200 samples
    python3 run_sim.py --P 4 --n 1000
    python3 run_sim.py --model model.json # a trained, quantized model

For every P it exports the ROM files, simulates the RTL, and compares
  * every requantized neuron (sample, layer, o, v, y)  -> pinpoints the first mismatch
  * final class and logits per sample
  * measured vs. expected cycle count
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "golden"))

from qmlp import infer, export_hw, cycles_expected, check_ranges   # noqa: E402
from random_model import make_random_model, make_inputs, save_model, load_model  # noqa: E402


def run_one(layers, X, P, build_root):
    bdir = os.path.join(build_root, f"P{P}")
    params = export_hw(layers, P, bdir)
    in0 = layers[0].in_dim
    with open(os.path.join(bdir, "inputs.hex"), "w") as f:
        for x in X:
            for v in x:
                f.write(format(v & 0xFF, "02x") + "\n")

    # golden
    trace, results = [], []
    for s, x in enumerate(X):
        cls, outs = infer(layers, x, trace=trace, sample=s)
        results.append((s, cls, outs[-1]))

    # simulate
    defs = dict(params, IN0=in0, OUT_LAST=layers[-1].out_dim, NS=len(X))
    cmd = ["iverilog", "-g2005", "-o", "sim.vvp"]
    cmd += [f"-Ptb.{k}={v}" for k, v in defs.items()]
    cmd += [os.path.join(ROOT, "rtl", "mlp_accel.v"), os.path.join(ROOT, "tb", "tb_mlp.v")]
    subprocess.run(cmd, cwd=bdir, check=True)
    sim = subprocess.run(["vvp", "-n", "sim.vvp"], cwd=bdir, capture_output=True, text=True)
    if "ERROR" in sim.stdout or sim.returncode != 0:
        print(sim.stdout, sim.stderr)
        return dict(P=P, ok=False)

    # compare trace
    with open(os.path.join(bdir, "rtl_trace.txt")) as f:
        rtl_trace = [tuple(int(t) for t in line.split()) for line in f if line.strip()]
    first_bad = None
    for k, (g, r) in enumerate(zip(trace, rtl_trace)):
        if g != r:
            first_bad = (k, g, r)
            break
    trace_ok = first_bad is None and len(trace) == len(rtl_trace)

    # compare results
    with open(os.path.join(bdir, "rtl_results.txt")) as f:
        rtl_res = [list(map(int, line.split())) for line in f if line.strip()]
    n_match, cycles = 0, set()
    for (s, cls, logits), r in zip(results, rtl_res):
        if r[0] == s and r[1] == cls and r[2:-1] == logits:
            n_match += 1
        cycles.add(r[-1])

    return dict(P=P, ok=trace_ok and n_match == len(X), trace_ok=trace_ok,
                rtl_classes=[r[1] for r in rtl_res],
                n_trace=len(trace), n_rtl_trace=len(rtl_trace), first_bad=first_bad,
                n_match=n_match, n=len(X), cycles=sorted(cycles),
                cycles_expected=cycles_expected(layers, P), w_depth=params["W_DEPTH"])


def coverage(layers, X):
    """Count the corner cases the test set exercises."""
    trace, ties = [], 0
    for s, x in enumerate(X):
        _, outs = infer(layers, x, trace=trace, sample=s)
        ties += outs[-1].count(max(outs[-1])) > 1
    c = dict(sat_hi=0, sat_lo=0, relu0=0, half_pos=0, half_neg=0)
    for _, l, _, v, y in trace:
        L = layers[l]
        c["sat_hi"] += y == 127
        c["sat_lo"] += (y == -128) and not L.relu
        c["relu0"] += (y == 0) and L.relu
        if (v * L.M0) & ((1 << L.n) - 1) == 1 << (L.n - 1):
            c["half_pos" if v > 0 else "half_neg"] += 1
    c["ties"] = ties
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--n", type=int, default=200, help="number of test samples")
    ap.add_argument("--model", help="quantized model .json (default: random model)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pow2", action="store_true",
                    help="power-of-two scales: many exact .5 rounding cases")
    ap.add_argument("--inputs", help="int8 input vectors .json (e.g. results/test_inputs.json)")
    ap.add_argument("--labels", help="true labels .json; prints the accuracy the RTL achieves")
    ap.add_argument("--build", default=os.path.join(ROOT, "build"))
    a = ap.parse_args()

    if a.model:
        layers = load_model(a.model)
        check_ranges(layers)
    else:
        layers = make_random_model(seed=a.seed, pow2=a.pow2)
        os.makedirs(a.build, exist_ok=True)
        save_model(layers, os.path.join(a.build, "random_model.json"))
    if a.inputs:
        with open(a.inputs) as f:
            X = json.load(f)
    else:
        X = make_inputs(layers[0].in_dim, a.n, seed=a.seed + 1)
    labels = None
    if a.labels:
        with open(a.labels) as f:
            labels = json.load(f)
        assert len(labels) == len(X), "labels and inputs differ in length"

    dims = [layers[0].in_dim] + [L.out_dim for L in layers]
    cov = coverage(layers, X)
    print(f"model {'-'.join(map(str, dims))}, {len(X)} samples")
    print("coverage: clamp@127 {sat_hi}, clamp@-128 {sat_lo}, ReLU->0 {relu0}, "
          "exact .5 rounding +{half_pos}/-{half_neg}, tied max logit {ties}".format(**cov))
    print(f"{'P':>3} {'bit-exact':>10} {'neurons':>9} {'cycles':>7} {'expected':>9} {'ROM words':>10}")
    all_ok = True
    for P in a.P:
        r = run_one(layers, X, P, a.build)
        all_ok &= r["ok"]
        if "n_match" not in r:
            print(f"{P:>3}  simulation failed")
            continue
        print(f"{P:>3} {r['n_match']:>4}/{r['n']:<5} {'ok' if r['trace_ok'] else 'FAIL':>9} "
              f"{','.join(map(str, r['cycles'])):>7} {r['cycles_expected']:>9} {r['w_depth']:>10}")
        if r["first_bad"]:
            k, g, rt = r["first_bad"]
            print(f"    first mismatch at trace line {k}: golden (s,l,o,v,y)={g}  rtl={rt}")
        elif r["n_trace"] != r["n_rtl_trace"]:
            print(f"    trace length differs: golden {r['n_trace']} rtl {r['n_rtl_trace']}")
        if labels is not None:
            acc = sum(c == t for c, t in zip(r["rtl_classes"], labels)) / len(labels)
            print(f"    RTL classification accuracy on these inputs: {acc:.3f}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
