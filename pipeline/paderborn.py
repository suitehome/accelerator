"""
Paderborn (KAt-DataCenter) bearing dataset: bearing codes, labels, split and
.mat loader.

Data: https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/
License: CC BY-NC 4.0, cite Lessmeier et al., PHM Europe 2016.

Each archive <CODE>.rar unpacks to a folder <CODE>/ with 80 files
<COND>_<CODE>_<k>.mat (4 operating conditions x 20 measurements of 4 s).
"""
import glob
import os
import re

import numpy as np
from scipy.io import loadmat

BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"

HEALTHY = ["K001", "K002", "K003", "K004", "K005", "K006"]
# real damages from accelerated lifetime tests (combined KB23/KB24/KB27 left out)
REAL_OUTER = ["KA04", "KA15", "KA16", "KA22", "KA30"]
REAL_INNER = ["KI04", "KI14", "KI16", "KI17", "KI18", "KI21"]
# artificial damages (EDM, drilling, engraving) - optional extra training data
ART_OUTER = ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09"]
ART_INNER = ["KI01", "KI03", "KI05", "KI07", "KI08"]

CLASSES = ["healthy", "outer", "inner"]

# Split BY BEARING: no bearing appears in both sets, so test accuracy is not
# inflated by windows of the same bearing leaking into training.
TRAIN_BEARINGS = ["K001", "K002", "K003", "K004",
                  "KA04", "KA15", "KA22",
                  "KI04", "KI14", "KI16", "KI17"]
TEST_BEARINGS = ["K005", "K006", "KA16", "KA30", "KI18", "KI21"]

CONDITIONS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
BASELINE = "N15_M07_F10"

FS = 64000                # Hz, phase currents and vibration

# Test bearing 6203 geometry. Values as given in Lessmeier et al. (2016);
# check them against the fact sheet shipped with the download.
N_BALLS = 8
BALL_D = 6.75             # mm
PITCH_D = 28.55           # mm
CONTACT_ANGLE = 0.0       # rad, deep groove ball bearing under radial load


def label_of(code: str) -> int:
    if code.startswith("K0"):
        return 0
    if code.startswith("KA"):
        return 1
    if code.startswith("KI"):
        return 2
    raise ValueError(f"no single-fault label for bearing {code}")


def rpm_of(cond: str) -> float:
    """N15 -> 1500 rpm, N09 -> 900 rpm (speed-controlled drive)."""
    return float(int(re.match(r"N(\d+)", cond).group(1)) * 100)


def fault_freqs(rpm: float):
    """BPFO, BPFI in Hz for shaft speed rpm (see README for the derivation)."""
    fr = rpm / 60.0
    ratio = BALL_D / PITCH_D * np.cos(CONTACT_ANGLE)
    bpfo = N_BALLS / 2 * fr * (1 - ratio)
    bpfi = N_BALLS / 2 * fr * (1 + ratio)
    return bpfo, bpfi


def _walk(obj):
    """Yield every (Name, Data) record anywhere in a loadmat() result."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.names and "Name" in obj.dtype.names and "Data" in obj.dtype.names:
            for rec in obj.ravel():
                yield rec["Name"], rec["Data"]
            return
        if obj.dtype.names:
            for rec in obj.ravel():
                for f in obj.dtype.names:
                    yield from _walk(rec[f])
        elif obj.dtype == object:
            for o in obj.ravel():
                yield from _walk(o)


def _as_str(x) -> str:
    while isinstance(x, np.ndarray) and x.size == 1 and x.dtype == object:
        x = x.ravel()[0]
    if isinstance(x, np.ndarray):
        x = "".join(map(str, x.ravel()))
    return str(x).strip()


def load_signals(path: str) -> dict:
    """Return {signal name: 1-D float array} for one measurement file."""
    m = loadmat(path)
    out = {}
    for key, val in m.items():
        if key.startswith("__"):
            continue
        for name, data in _walk(val):
            out[_as_str(name)] = np.asarray(data, dtype=np.float64).ravel()
    if not out:
        raise ValueError(f"{path}: no Name/Data records found")
    return out


def diagnose(root: str, codes, conditions) -> str:
    """Explain why list_files() found nothing under root."""
    root_abs = os.path.abspath(root)
    if not os.path.isdir(root_abs):
        return f"folder does not exist: {root_abs}"
    mats = glob.glob(os.path.join(root_abs, "**", "*.mat"), recursive=True)
    rars = glob.glob(os.path.join(root_abs, "**", "*.rar"), recursive=True)
    top = sorted(os.listdir(root_abs))[:15]
    lines = [f"searched: {root_abs}",
             f"  top-level entries: {top}",
             f"  .mat files found: {len(mats)}, .rar archives found: {len(rars)}"]
    if rars and not mats:
        lines.append("  -> archives were downloaded but not unpacked. Install 7-Zip "
                     "(https://www.7-zip.org) and re-run data/download_paderborn.py with the same --dest.")
    elif mats:
        lines.append(f"  example .mat names: {[os.path.basename(m) for m in mats[:3]]}")
        lines.append(f"  expected names like {conditions[0]}_{codes[0]}_1.mat")
    else:
        lines.append("  -> nothing downloaded here. Pass the folder you used as --dest "
                     "in download_paderborn.py, e.g. --data C:\\paderborn")
    return "\n".join(lines)


def list_files(root: str, codes, conditions=(BASELINE,)):
    """[(path, code, condition)] for the given bearings and operating conditions."""
    files = []
    for code in codes:
        for cond in conditions:
            pat = os.path.join(root, "**", f"{cond}_{code}_*.mat")
            for p in sorted(glob.glob(pat, recursive=True)):
                files.append((p, code, cond))
    return files
