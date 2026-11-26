#!/usr/bin/env python3
"""
Features -> float MLP -> INT8 PTQ -> model json for the accelerator.

    python pipeline/train.py --data C:\\paderborn                      # vibration, healthy vs damaged
    python pipeline/train.py --data C:\\paderborn --task 3class
    python pipeline/train.py --data C:\\paderborn --sensor current     # MCSA features (negative result)
    python run_sim.py --model results/model_q.json --inputs results/test_inputs.json \\
                      --labels results/test_labels.json --P 1 4 16

Evaluation, in the order it should be trusted:
  1. LOBO: leave-one-bearing-out over the TRAINING bearings only. Used for every
     design choice (sensor, band, features). Never looks at the test bearings.
  2. Test: train on all training bearings, evaluate once on the held-out test
     bearings (paderborn.TEST_BEARINGS).
  3. Leaky split: windows shuffled at random, so the same bearing appears in
     train and test. Printed only to show how much a wrong split inflates accuracy.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "golden"))

import paderborn as pb                                   # noqa: E402
import features as ft_cur                                # noqa: E402
import features_vib as ft_vib                            # noqa: E402
from mlp_np import MLP                                   # noqa: E402
from quantize import ptq, quantize_input, int_forward    # noqa: E402
from qmlp import infer                                   # noqa: E402
from random_model import save_model                      # noqa: E402

TASKS = {"binary": ["healthy", "damaged"], "3class": pb.CLASSES}


def no_files(root, codes, conds):
    raise SystemExit("no Paderborn .mat files found.\n" + pb.diagnose(root, codes, conds))


def extractor(sensor, root, codes, conds, poles):
    """Return (extract(signals, cond) -> feats, feature names, key for the cache)."""
    if sensor == "vibration":
        return (lambda s, c: ft_vib.extract(s, c)[0], ft_vib.feature_names(),
                [ft_vib.L_FFT, ft_vib.HOP, ft_vib.TOL_BINS, str(ft_vib.BAND)])
    p = poles
    if p is None:
        files = pb.list_files(root, codes, conds) or no_files(root, codes, conds)
        _, info = ft_cur.extract(pb.load_signals(files[0][0]), files[0][2])
        p = info["p"]
        print(f"pole pairs: f_e/f_r = {info['p_est']:.3f} on {files[0][1]} -> p = {p}")
    return (lambda s, c: ft_cur.extract(s, c, p=p)[0], ft_cur.feature_names(p),
            [p, ft_cur.L_FFT, ft_cur.HOP, ft_cur.N_FEAT, ft_cur.K_MAX, ft_cur.TOL_BINS, ft_cur.MIN_SEP])


def build_set(root, codes, conds, fx):
    X, lab, fid, bear = [], [], [], []
    files = pb.list_files(root, codes, conds) or no_files(root, codes, conds)
    for k, (path, code, cond) in enumerate(files):
        f = fx(pb.load_signals(path), cond)
        X.append(f)
        lab += [pb.label_of(code)] * len(f)
        fid += [k] * len(f)
        bear += [code] * len(f)
        print(f"\r  {k + 1}/{len(files)} files", end="", flush=True)
    print()
    missing = sorted(set(codes) - {c for _, c, _ in files})
    return np.concatenate(X), np.array(lab), np.array(fid), np.array(bear), missing


def cached_set(out, tag, root, codes, conds, fx, fkey, use_cache=True):
    path = os.path.join(out, f"features_{tag}.npz")
    key = json.dumps([os.path.abspath(root), sorted(codes), sorted(conds)] + fkey)
    if use_cache and os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        if "key" in d.files and str(d["key"]) == key:
            return d["X"], d["y"], d["fid"], d["bear"], list(d["missing"])
    X, lab, fid, bear, missing = build_set(root, codes, conds, fx)
    np.savez(path, X=X, y=lab, fid=fid, bear=bear, missing=np.array(missing, dtype=str), key=key)
    return X, lab, fid, bear, missing


def to_task(lab, task):
    return (lab > 0).astype(int) if task == "binary" else lab


def train_mlp(X, y, ncls, hidden, epochs, seed):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    cnt = np.bincount(y, minlength=ncls)
    cw = cnt.sum() / (ncls * np.maximum(cnt, 1))
    m = MLP([X.shape[1], hidden, ncls], seed=seed).fit((X - mu) / sd, y, epochs=epochs,
                                                      class_weight=cw, seed=seed)
    return m, mu, sd


def file_vote(pred, fid, y, ncls):
    ok = [np.bincount(pred[fid == f], minlength=ncls).argmax() == y[fid == f][0] for f in np.unique(fid)]
    return float(np.mean(ok))


def confusion(pred, y, k):
    c = np.zeros((k, k), int)
    np.add.at(c, (y, pred), 1)
    return c.tolist()


def per_bearing(pred, y, bear):
    return {b: float(np.mean(pred[bear == b] == y[bear == b])) for b in dict.fromkeys(bear)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the unpacked bearing folders")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--sensor", choices=["vibration", "current"], default="vibration")
    ap.add_argument("--task", choices=list(TASKS), default="binary")
    ap.add_argument("--conditions", default="baseline", help="'baseline', 'all' or comma list")
    ap.add_argument("--artificial", action="store_true", help="add artificial damages to training")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lobo-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poles", type=int, help="motor pole pairs for --sensor current")
    ap.add_argument("--no-lobo", action="store_true", help="skip leave-one-bearing-out validation")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    conds = {"baseline": [pb.BASELINE], "all": pb.CONDITIONS}.get(a.conditions, a.conditions.split(","))
    train_codes = pb.TRAIN_BEARINGS + (pb.ART_OUTER + pb.ART_INNER if a.artificial else [])
    assert not set(train_codes) & set(pb.TEST_BEARINGS), "a bearing is in both train and test"
    classes = TASKS[a.task]
    ncls = len(classes)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    fx, names, fkey = extractor(a.sensor, a.data, train_codes, conds, a.poles)
    fkey = [a.sensor] + fkey
    print(f"sensor {a.sensor}, task {a.task} ({' / '.join(classes)}), {len(names)} features")
    Xtr, ltr, ftr, btr, miss_tr = cached_set(a.out, f"{a.sensor}_train", a.data, train_codes, conds, fx, fkey, not a.no_cache)
    Xte, lte, fte, bte, miss_te = cached_set(a.out, f"{a.sensor}_test", a.data, pb.TEST_BEARINGS, conds, fx, fkey, not a.no_cache)
    for tag, miss in (("train", miss_tr), ("test", miss_te)):
        if miss:
            print(f"WARNING: no files for {tag} bearings {miss}")
    ytr, yte = to_task(ltr, a.task), to_task(lte, a.task)
    print(f"windows: train {len(ytr)} ({len(set(btr))} bearings), test {len(yte)} ({len(set(bte))} bearings)"
          f"  [{time.time() - t0:.0f} s]")

    rep = {"sensor": a.sensor, "task": a.task, "classes": classes, "conditions": conds,
           "feature_names": names, "train_bearings": train_codes, "test_bearings": pb.TEST_BEARINGS}

    # 1. leave-one-bearing-out on the training bearings
    if not a.no_lobo:
        lobo = {}
        for b in dict.fromkeys(btr):
            tr, va = btr != b, btr == b
            m, mu, sd = train_mlp(Xtr[tr], ytr[tr], ncls, a.hidden, a.lobo_epochs, a.seed)
            lobo[b] = float(np.mean(m.predict((Xtr[va] - mu) / sd) == ytr[va]))
        rep["lobo_train"] = {"mean_acc": float(np.mean(list(lobo.values()))), "per_bearing": lobo}
        print(f"\n[1] LOBO on training bearings: {rep['lobo_train']['mean_acc']:.3f}")
        print("    " + "  ".join(f"{b} {v:.2f}" for b, v in lobo.items()))

    # 2. final model, evaluated once on the test bearings
    mlp, mu, sd = train_mlp(Xtr, ytr, ncls, a.hidden, a.epochs, a.seed)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    pf_te = mlp.predict(Zte)
    rep["float"] = {"train_acc": float(np.mean(mlp.predict(Ztr) == ytr)),
                    "test_acc": float(np.mean(pf_te == yte)),
                    "test_file_vote_acc": file_vote(pf_te, fte, yte, ncls)}

    sweep = {}
    for bits in (8, 6, 4):
        layers, s_in = ptq(mlp, Ztr, w_bits=bits, a_bits=bits)
        qmax = 2 ** (bits - 1) - 1
        pq = np.argmax(int_forward(layers, quantize_input(Zte, s_in, qmax), qmax), axis=1)
        sweep[bits] = float(np.mean(pq == yte))
        if bits == 8:
            layers8, s_in8, pq8 = layers, s_in, pq
    rep["bit_sweep_test_acc"] = sweep

    Xq_te = quantize_input(Zte, s_in8)
    gold = np.array([infer(layers8, x.tolist())[0] for x in Xq_te])
    assert np.array_equal(gold, pq8), "vectorized int inference disagrees with golden model"
    rep["int8"] = {"test_acc": float(np.mean(pq8 == yte)),
                   "test_file_vote_acc": file_vote(pq8, fte, yte, ncls),
                   "loss_vs_float_pp": 100 * (rep["float"]["test_acc"] - float(np.mean(pq8 == yte))),
                   "confusion_test": confusion(pq8, yte, ncls),
                   "per_bearing_acc": per_bearing(pq8, yte, bte)}

    # 3. leaky random split, for comparison only
    rng = np.random.default_rng(a.seed)
    Xall, yall = np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte])
    idx = rng.permutation(len(yall))
    cut = int(0.7 * len(idx))
    m, mu2, sd2 = train_mlp(Xall[idx[:cut]], yall[idx[:cut]], ncls, a.hidden, a.lobo_epochs, a.seed)
    rep["leaky_random_split_acc"] = float(np.mean(m.predict((Xall[idx[cut:]] - mu2) / sd2) == yall[idx[cut:]]))

    save_model(layers8, os.path.join(a.out, "model_q.json"))
    with open(os.path.join(a.out, "preprocess.json"), "w") as f:
        json.dump({"sensor": a.sensor, "task": a.task, "mean": mu.tolist(), "std": sd.tolist(),
                   "s_in": s_in8, "feature_names": names, "conditions": conds}, f)
    with open(os.path.join(a.out, "test_inputs.json"), "w") as f:
        json.dump(Xq_te.tolist(), f)
    with open(os.path.join(a.out, "test_labels.json"), "w") as f:
        json.dump(yte.tolist(), f)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    fl, q8 = rep["float"], rep["int8"]
    print(f"\n[2] test bearings {', '.join(pb.TEST_BEARINGS)}")
    print(f"    float : train {fl['train_acc']:.3f}  test {fl['test_acc']:.3f}  (file vote {fl['test_file_vote_acc']:.3f})")
    print(f"    INT8  : test {q8['test_acc']:.3f}  (file vote {q8['test_file_vote_acc']:.3f}), "
          f"loss vs float {q8['loss_vs_float_pp']:+.2f} %p")
    print("    bit sweep: " + ", ".join(f"{b} bit {v:.3f}" for b, v in sweep.items()))
    print(f"    confusion (rows true, cols pred; {' / '.join(classes)}):")
    for row in q8["confusion_test"]:
        print("       ", row)
    print("    per bearing: " + "  ".join(f"{b} {v:.2f}" for b, v in q8["per_bearing_acc"].items()))
    print(f"\n[3] leaky random window split (same bearings in train and test): "
          f"{rep['leaky_random_split_acc']:.3f}  <- not a valid estimate")
    print(f"\nwrote {os.path.abspath(a.out)}: model_q.json, test_inputs.json, test_labels.json, report.json")


if __name__ == "__main__":
    main()
