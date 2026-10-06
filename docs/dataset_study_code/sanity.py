"""Step 3: sanity checks against misleadingly high scores.

For every dataset (and the no-dedup variants):
  - train/test exact feature-vector overlap (same label vs conflicting label)
  - single-feature separability (train ROC-AUC per raw feature; depth-2 stump test F1)
  - LightGBM top importances, and ablations without the suspicious feature(s)
  - flag any model with test accuracy > 99%
Usage: python dataset_study/sanity.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from common import MAIN_DATASETS, RES, SEED, load_json, save_json, slug
from prepare import load_split
from train_eval import MODELS, binary_metrics, run_task

SEPARABLE_AUC = 0.95


def overlap(parts, feats):
    h = {s: pd.util.hash_pandas_object(d[feats], index=False).to_numpy() for s, d in parts.items()}
    tr = pd.DataFrame({"h": h["train"], "y": parts["train"]["__y"].to_numpy()})
    lab = tr.groupby("h")["y"].agg(lambda s: set(s))
    te_h, te_y = h["test"], parts["test"]["__y"].to_numpy()
    in_train = np.isin(te_h, lab.index.to_numpy())
    same = sum(1 for hh, yy in zip(te_h[in_train], te_y[in_train]) if yy in lab[hh])
    return {"test_rows": int(len(te_h)), "test_rows_seen_in_train": int(in_train.sum()),
            "pct_seen_in_train": float(100 * in_train.mean()),
            "seen_with_same_label": int(same), "seen_with_other_label_only": int(in_train.sum() - same)}


def single_feature(parts, num, cat):
    tr, te = parts["train"], parts["test"]
    rows = []
    for c in num + cat:
        if c in cat:
            codes = {v: i for i, v in enumerate(tr[c].astype(str).unique())}
            xtr = tr[c].astype(str).map(codes).to_numpy(float)
            xte = te[c].astype(str).map(codes).fillna(-1).to_numpy(float)
        else:
            med = np.nanmedian(tr[c]) if tr[c].notna().any() else 0.0
            xtr = tr[c].fillna(med).to_numpy(float)
            xte = te[c].fillna(med).to_numpy(float)
        ytr, yte = tr["__y"].to_numpy(), te["__y"].to_numpy()
        try:
            auc = roc_auc_score(ytr, xtr)
            auc = max(auc, 1 - auc)
        except ValueError:
            auc = 0.5
        stump = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=SEED)
        stump.fit(xtr.reshape(-1, 1), ytr)
        f1 = f1_score(yte, stump.predict(xte.reshape(-1, 1)), zero_division=0)
        rows.append({"feature": c, "train_auc": float(auc), "stump_test_f1": float(f1)})
    return sorted(rows, key=lambda r: -r["train_auc"])


def raw_feature(encoded, raw_cols):
    e = encoded.replace("missingindicator_", "")
    best = [c for c in raw_cols if e == c or e.startswith(c + "_")]
    return max(best, key=len) if best else e


def check(name):
    out_p = os.path.join(RES, slug(name), "sanity.json")
    if os.path.exists(out_p):
        return load_json(out_p)
    parts, meta = load_split(name)
    num, cat = meta["features_numeric"], meta["features_categorical"]
    feats = num + cat
    rd = os.path.join(RES, slug(name))
    models = [m for m in MODELS if os.path.exists(os.path.join(rd, f"binary_{m}.json"))]
    res = {m: load_json(os.path.join(rd, f"binary_{m}.json"))["test"] for m in models}
    flagged = {m: r["accuracy"] for m, r in res.items() if r["accuracy"] > 0.99}
    out = {"dataset": name, "flag_accuracy_over_99": flagged,
           "train_test_overlap": overlap(parts, feats),
           "single_feature": single_feature(parts, num, cat)[:10]}
    lgbm = load_json(os.path.join(rd, "binary_LightGBM.json"))
    imp = lgbm["feature_importance_gain"]
    out["lightgbm_top_importance"] = imp[:10]
    # aggregate importance by raw column
    agg = {}
    for r in imp:
        k = raw_feature(r["feature"], feats)
        agg[k] = agg.get(k, 0) + r["share"]
    top_raw = sorted(agg.items(), key=lambda kv: -kv[1])
    out["lightgbm_importance_by_raw_feature"] = top_raw[:10]

    # Ablations (LightGBM, same selection procedure on validation)
    abl = {"baseline": {k: lgbm["test"][k] for k in ("accuracy", "f1", "macro_f1", "roc_auc", "pr_auc")}}
    if "@" not in name:
        top1 = top_raw[0][0]
        r = run_task(name, "binary", models=["LightGBM"], drop=(top1,), tag="_drop_top1")["LightGBM"]
        abl[f"without top-1 feature ({top1})"] = {k: r["test"][k] for k in abl["baseline"]}
        sep = [s["feature"] for s in out["single_feature"] if s["train_auc"] >= SEPARABLE_AUC]
        if sep and sep != [top1]:
            r = run_task(name, "binary", models=["LightGBM"], drop=tuple(sep), tag="_drop_separable")["LightGBM"]
            abl[f"without all single-feature-separable (AUC>={SEPARABLE_AUC}): {sep}"] = \
                {k: r["test"][k] for k in abl["baseline"]}
    out["ablations_lightgbm"] = abl

    if name == "Edge-IIoTset":
        # DDoS_UDP / MITM rows were column-misaligned in the source CSV: score without them.
        d = np.load(os.path.join(rd, "probs_binary_LightGBM.npz"))
        keep = ~parts["test"]["__type"].isin(["DDoS_UDP", "MITM"]).to_numpy()
        m = binary_metrics(d["y"][keep], d["prob"][keep, 1])
        out["edge_without_misaligned_classes"] = {
            "note": "LightGBM test metrics excluding DDoS_UDP and MITM rows (misaligned in source CSV)",
            "n_excluded": int((~keep).sum()), **{k: m[k] for k in ("accuracy", "f1", "roc_auc")}}
    save_json(out, out_p)
    return out


if __name__ == "__main__":
    for n in MAIN_DATASETS + [d + "@nodedup" for d in MAIN_DATASETS[1:]]:
        print("[sanity]", n, flush=True)
        check(n)
