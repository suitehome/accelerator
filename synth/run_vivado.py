#!/usr/bin/env python3
"""
Vivado synthesis + place & route of mlp_accel for several P, no board needed.

    python synth/run_vivado.py                       # P = 1 4 16, trained model, Artix-7 xc7a35t
    python synth/run_vivado.py --P 4 --period 8
    python synth/run_vivado.py --part xc7z020clg400-1 --vivado C:/Xilinx/Vivado/2023.2/bin/vivado.bat

For each P it writes the ROM .hex files and a Tcl script into <work>/P<P>, runs
Vivado in batch mode (out-of-context, so the design needs no pins), and reads
back utilization, timing and power. Results: results/synth_summary.csv and a
table on the screen.

Vivado misbehaves on paths with spaces or non-ASCII characters (OneDrive,
"바탕 화면"), so the work folder defaults to C:/nn_synth (Windows) or
~/nn_synth, which this script creates.
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "golden"))
from qmlp import export_hw, cycles_expected          # noqa: E402
from random_model import load_model, make_random_model  # noqa: E402

TCL = """\
set_param general.maxThreads 4
read_verilog mlp_accel.v
synth_design -top mlp_accel -part {part} -mode out_of_context {generics}
create_clock -name clk -period {period} [get_ports clk]
report_utilization -file util_synth.rpt
opt_design
place_design
route_design
report_utilization -file util.rpt
report_timing_summary -max_paths 5 -file timing.rpt
report_power -file power.rpt
report_timing -max_paths 1 -nworst 1 -path_type full -file critical_path.rpt
"""


def find_vivado(explicit=None):
    if explicit:
        return explicit
    for name in ("vivado.bat", "vivado"):
        p = shutil.which(name)
        if p:
            return p
    pats = [r"C:\Xilinx\Vivado\*\bin\vivado.bat", r"C:\AMDDesignTools\*\Vivado\bin\vivado.bat",
            r"D:\Xilinx\Vivado\*\bin\vivado.bat", r"D:\AMDDesignTools\*\Vivado\bin\vivado.bat",
            r"E:\AMDDesignTools\*\Vivado\bin\vivado.bat", "/tools/Xilinx/Vivado/*/bin/vivado",
            "/opt/Xilinx/Vivado/*/bin/vivado", os.path.expanduser("~/Xilinx/Vivado/*/bin/vivado")]
    found = sorted(p for pat in pats for p in glob.glob(pat))
    return found[-1] if found else None


def parse_util(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    out = {}
    for key, pat in (("LUT", r"Slice LUTs|CLB LUTs"), ("FF", r"Slice Registers|CLB Registers"),
                     ("DSP", r"DSPs"), ("BRAM", r"Block RAM Tile")):
        m = re.search(r"\|\s*(?:%s)\*?\s*\|\s*([\d.]+)" % pat, txt)
        out[key] = float(m.group(1)) if m else None
    return out


def parse_timing(path):
    lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    for i, ln in enumerate(lines):
        if "WNS(ns)" in ln and "TNS(ns)" in ln:
            for nxt in lines[i + 1:i + 4]:
                vals = nxt.split()
                if vals and re.fullmatch(r"-?\d+\.\d+", vals[0]):
                    return float(vals[0])
    return None


def parse_power(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    out = {}
    for key, pat in (("P_total_W", r"Total On-Chip Power \(W\)"), ("P_dyn_W", r"Dynamic \(W\)"),
                     ("P_static_W", r"Device Static \(W\)")):
        m = re.search(r"\|\s*%s\s*\|\s*([\d.]+)" % pat, txt)
        out[key] = float(m.group(1)) if m else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--model", default=os.path.join(ROOT, "results", "model_q.json"))
    ap.add_argument("--part", default="xc7a35tcpg236-1", help="default: Artix-7 35T (Basys3)")
    ap.add_argument("--period", type=float, default=10.0, help="clock constraint in ns")
    ap.add_argument("--work", default="C:/nn_synth" if os.name == "nt" else os.path.expanduser("~/nn_synth"))
    ap.add_argument("--vivado")
    ap.add_argument("--parse-only", action="store_true", help="re-read reports of a previous run")
    a = ap.parse_args()

    if any(ord(ch) > 127 or ch == " " for ch in os.path.abspath(a.work)):
        sys.exit(f"work folder must be ASCII without spaces: {a.work}")
    if os.path.exists(a.model):
        layers = load_model(a.model)
        src = a.model
    else:
        layers = make_random_model()
        src = "random model (no results/model_q.json found)"
    dims = [layers[0].in_dim] + [L.out_dim for L in layers]
    print(f"model {'-'.join(map(str, dims))} from {src}")

    viv = None
    if not a.parse_only:
        viv = find_vivado(a.vivado)
        if not viv:
            sys.exit("Vivado not found. Pass --vivado <path to vivado.bat>, "
                     "e.g. C:/Xilinx/Vivado/2023.2/bin/vivado.bat")
        print(f"vivado: {viv}")

    rows = []
    for P in a.P:
        wd = os.path.join(a.work, f"P{P}")
        os.makedirs(wd, exist_ok=True)
        params = export_hw(layers, P, wd)          # writes wrom.hex, brom.hex, cfg.hex
        shutil.copy(os.path.join(ROOT, "rtl", "mlp_accel.v"), wd)
        gen = " ".join(f"-generic {k}={v}" for k, v in params.items())
        with open(os.path.join(wd, "synth.tcl"), "w") as f:
            f.write(TCL.format(part=a.part, generics=gen, period=a.period))
        if not a.parse_only:
            print(f"P={P}: running Vivado (a few minutes) ...", flush=True)
            t0 = time.time()
            r = subprocess.run([viv, "-mode", "batch", "-source", "synth.tcl", "-nojournal",
                                "-log", "vivado.log"], cwd=wd, capture_output=True, text=True,
                               errors="replace")
            out = (r.stdout or "") + "\n" + (r.stderr or "")
            with open(os.path.join(wd, "console.txt"), "w", encoding="utf-8") as f:
                f.write(out)
            if r.returncode != 0 or not os.path.exists(os.path.join(wd, "timing.rpt")):
                print(f"  Vivado failed (exit {r.returncode}); console output in {os.path.join(wd, 'console.txt')}")
                errs = [l for l in out.splitlines() if "ERROR" in l or "rror" in l or "icense" in l]
                print("\n".join("    " + l for l in (errs or out.strip().splitlines()[-15:])[:20]))
                continue
            print(f"  done in {time.time() - t0:.0f} s")
        u = parse_util(os.path.join(wd, "util.rpt"))
        wns = parse_timing(os.path.join(wd, "timing.rpt"))
        pw = parse_power(os.path.join(wd, "power.rpt"))
        fmax = 1000.0 / (a.period - wns) if wns is not None else None
        cyc = cycles_expected(layers, P)
        row = dict(P=P, **u, WNS_ns=wns, Fmax_MHz=fmax, cycles=cyc,
                   latency_us=cyc / fmax if fmax else None,
                   inf_per_s=fmax * 1e6 / cyc if fmax else None, **pw)
        if fmax and pw.get("P_dyn_W") is not None:
            # dynamic energy per inference, with the power figure taken at the constrained clock
            row["E_dyn_nJ"] = pw["P_dyn_W"] * cyc / (1e9 / a.period) * 1e9
        rows.append(row)

    if not rows:
        sys.exit(1)
    out = os.path.join(ROOT, "results", "synth_summary.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def fmt(v, n=1):
        return "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))
    print(f"\npart {a.part}, clock constraint {a.period} ns, place & route results")
    print(f"{'P':>3} {'LUT':>6} {'FF':>6} {'DSP':>4} {'BRAM':>5} {'WNS ns':>7} {'Fmax MHz':>9} "
          f"{'cycles':>7} {'lat us':>7} {'inf/s':>9} {'P_dyn W':>8} {'E nJ':>7}")
    for r in rows:
        print(f"{r['P']:>3} {fmt(r['LUT'], 0):>6} {fmt(r['FF'], 0):>6} {fmt(r['DSP'], 0):>4} "
              f"{fmt(r['BRAM'], 1):>5} {fmt(r['WNS_ns'], 3):>7} {fmt(r['Fmax_MHz']):>9} {r['cycles']:>7} "
              f"{fmt(r['latency_us'], 2):>7} {fmt(r['inf_per_s'], 0):>9} {fmt(r.get('P_dyn_W'), 3):>8} "
              f"{fmt(r.get('E_dyn_nJ')):>7}")
    print(f"\nwrote {out}; reports in {a.work}/P*/ (critical_path.rpt shows the slowest path)")


if __name__ == "__main__":
    main()
