"""BATADAL leave-one-attack-window-out evaluation.

dataset04 is cut at midpoints between its 5 labelled attack windows. Fold k tests
on segment k and trains on dataset03 + the other dataset04 segments, so (unlike the
main chronological split) training data can come after the test period.
Hyperparameters are the ones selected on the main validation split.
Usage: python dataset_study/batadal_lowo.py
"""
import os
import time

import numpy as np
import pandas as pd

from common import CACHE, RES, load_json, save_json
from loaders import BATADAL_LOWO_CUTS
from train_eval import MODELS, binary_metrics, fit_model, make_preprocessor


def main():
    out_p = os.path.join(RES, "BATADAL", "lowo.json")
    df = pd.read_parquet(os.path.join(CACHE, "BATADAL", "clean_full.parquet"))
    meta = load_json(os.path.join(CACHE, "BATADAL", "meta.json"))
    feats = meta["features_numeric"]
    d04 = df["__part"] == "dataset04"
    cuts = [pd.Timestamp.min] + [pd.Timestamp(c) for c in BATADAL_LOWO_CUTS] + [pd.Timestamp.max]
    seg = pd.Series(-1, index=df.index)
    for k in range(5):
        m = d04 & (df["__time"] >= cuts[k]) & (df["__time"] < cuts[k + 1])
        seg[m] = k
    results = load_json(out_p) if os.path.exists(out_p) else {}
    for mname in MODELS:
        if mname in results:
            continue
        params = load_json(os.path.join(RES, "BATADAL", f"binary_{mname}.json"))["refit_params"]
        folds = []
        for k in range(5):
            tr, te = df[seg != k], df[seg == k]
            pre = make_preprocessor(feats, [], scale=mname in ("LogReg", "MLP"))
            Xtr = pre.fit_transform(tr[feats]).astype(np.float32)
            Xte = pre.transform(te[feats]).astype(np.float32)
            t0 = time.perf_counter()
            m = fit_model(mname, params, Xtr, tr["__y"].to_numpy(), None, None, 2)
            tt = time.perf_counter() - t0
            met = binary_metrics(te["__y"].to_numpy(), m.predict_proba(Xte)[:, 1])
            met.update({"fold": k + 1, "test_from": str(te["__time"].min()), "test_to": str(te["__time"].max()),
                        "train_time_s": tt})
            folds.append(met)
            print(f"  [BATADAL LOWO] {mname} window {k + 1}: F1={met['f1']:.3f} AUC={met['roc_auc']}", flush=True)
        keys = ["f1", "precision", "recall", "roc_auc", "pr_auc", "fpr", "accuracy"]
        summ = {k: {"mean": float(np.mean([f[k] for f in folds])), "std": float(np.std([f[k] for f in folds]))}
                for k in keys}
        results[mname] = {"params": params, "folds": folds, "summary": summ}
        save_json(results, out_p)
    return results


if __name__ == "__main__":
    main()
