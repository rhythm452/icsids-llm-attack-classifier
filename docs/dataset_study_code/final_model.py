"""Step 6: final merged detector (Edge-IIoTset + TON_IoT-Network + X-IIoTID).

Representation: event description -> all-MiniLM-L6-v2 embedding (the web app's model),
the only representation in which these three datasets merge. Binary attack/normal.
Candidates: LightGBM (study recommendation) and the logistic-regression probe used in the
study; selected on the merged validation set (mean per-dataset attack F1). Each dataset
gets equal total weight; classes are balanced within each dataset.
Tested on each dataset's held-out test sample (the same rows as the study) and, as a
domain-shift check, on the web app's procedure examples + benign sentences.
Usage: python dataset_study/final_model.py
"""
import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression

from common import CACHE, ROOT, SEED, path, save_json, slug
from cross import EMB_MODEL
from prepare import load_split
from textrep import describe, stratified_sample
from train_eval import binary_metrics
from zeroshot import benign_sentences

MERGE = ["Edge-IIoTset", "TON_IoT-Network", "X-IIoTID"]
TRAIN_N, VAL_N = 20_000, 3_000
OUT = os.path.join(ROOT, "models", "ics_ids_merged_v1")
EMB_DIR = os.path.join(CACHE, "emb_final")


def embed(model, name, split, n):
    p = os.path.join(EMB_DIR, f"{slug(name)}_{split}.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["X"], z["y"]
    parts, meta = load_split(name)
    s = stratified_sample(parts[split], n, SEED)
    texts = describe(s, name, meta["features_numeric"], meta["features_categorical"])
    X = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    np.savez_compressed(path(p), X=X, y=s["__y"].to_numpy())
    print(f"  [final] embedded {name} {split}: {len(texts)}", flush=True)
    return X, s["__y"].to_numpy()


def weights(ys):
    """Equal total weight per dataset, balanced classes inside each dataset."""
    w = []
    for y in ys:
        wy = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9)) / len(y)
        w.append(wy)
    w = np.concatenate(w)
    return w / w.mean()


def main():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(EMB_MODEL, device="cpu")
    tr = {n: embed(st, n, "train", TRAIN_N) for n in MERGE}
    va = {n: embed(st, n, "val", VAL_N) for n in MERGE}
    emb_cache = os.path.join(CACHE, "emb", slug(EMB_MODEL.replace("/", "_")))
    te = {}
    for n in MERGE:  # identical test rows to the study's cross/zero-shot sample
        z = np.load(os.path.join(emb_cache, f"{slug(n)}_test.npz"), allow_pickle=True)
        te[n] = (z["X"], z["y"])

    Xtr = np.vstack([tr[n][0] for n in MERGE])
    ytr = np.concatenate([tr[n][1] for n in MERGE])
    wtr = weights([tr[n][1] for n in MERGE])
    Xva = np.vstack([va[n][0] for n in MERGE])
    yva = np.concatenate([va[n][1] for n in MERGE])
    wva = weights([va[n][1] for n in MERGE])

    cands = {}
    t0 = time.perf_counter()
    lr = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED).fit(Xtr, ytr, sample_weight=wtr)
    cands["LogReg probe"] = (lr, time.perf_counter() - t0)
    for leaves in (31, 127):
        t0 = time.perf_counter()
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=leaves, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.5, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], eval_sample_weight=[wva],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        cands[f"LightGBM (leaves={leaves})"] = (m, time.perf_counter() - t0)

    sel = {}
    for k, (m, tt) in cands.items():
        f1s = {n: binary_metrics(va[n][1], m.predict_proba(va[n][0])[:, 1])["f1"] for n in MERGE}
        sel[k] = {"val_f1_per_dataset": f1s, "val_mean_f1": float(np.mean(list(f1s.values()))), "train_time_s": tt}
        print(f"  [final] {k}: val mean F1 {sel[k]['val_mean_f1']:.4f} {f1s}", flush=True)
    best = max(sel, key=lambda k: sel[k]["val_mean_f1"])
    model = cands[best][0]

    test = {}
    for n in MERGE:
        r = binary_metrics(te[n][1], model.predict_proba(te[n][0])[:, 1])
        test[n] = {k: r[k] for k in ("accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc",
                                     "fpr", "fnr", "confusion_matrix", "n_test", "n_attack")}
        print(f"  [final] test {n}: F1 {r['f1']:.4f} AUC {r['roc_auc']:.4f} FPR {r['fpr']:.4f}", flush=True)
    yall = np.concatenate([te[n][1] for n in MERGE])
    r = binary_metrics(yall, model.predict_proba(np.vstack([te[n][0] for n in MERGE]))[:, 1])
    test["MERGED (all three test sets pooled)"] = {k: r[k] for k in test[MERGE[0]]}
    print(f"  [final] test MERGED: F1 {r['f1']:.4f} AUC {r['roc_auc']:.4f} FPR {r['fpr']:.4f}", flush=True)
    single = {}  # study's single-dataset baselines on the same test rows, for comparison
    cross_p = os.path.join(ROOT, "dataset_study", "results", f"cross_{slug(EMB_MODEL.replace('/', '_'))}.json")
    if os.path.exists(cross_p):
        cm = json.load(open(cross_p))["embedding_matrix"]
        single = {n: {"f1": cm[n][n]["f1"], "roc_auc": cm[n][n]["roc_auc"]} for n in MERGE}

    # Domain-shift check: the web app's own prose events (all attacks) + its benign sentences.
    wa = os.path.join(ROOT, "webapp")
    techniques = json.load(open(os.path.join(wa, "techniques.json"), encoding="utf-8"))
    procs = [p["text"] for p in json.load(open(os.path.join(wa, "procedure_examples.json"), encoding="utf-8"))
             if p["technique_id"] in techniques]
    ben = benign_sentences()
    pa = model.predict_proba(st.encode(procs, normalize_embeddings=True, show_progress_bar=False))[:, 1]
    pb = model.predict_proba(st.encode(ben, normalize_embeddings=True, show_progress_bar=False))[:, 1]
    shift = {"procedure_examples_flagged_attack": float((pa >= 0.5).mean()), "n_procedure_examples": len(procs),
             "benign_sentences_flagged_attack": float((pb >= 0.5).mean()), "n_benign": len(ben),
             **binary_metrics(np.r_[np.ones(len(pa)), np.zeros(len(pb))].astype(int), np.r_[pa, pb])}
    shift.pop("confusion_matrix", None)
    print(f"  [final] web-app prose: {100 * shift['procedure_examples_flagged_attack']:.1f}% of attack examples, "
          f"{100 * shift['benign_sentences_flagged_attack']:.1f}% of benign flagged; AUC {shift['roc_auc']:.3f}")

    # inference cost per 1000 events: embedding + classifier
    sample = ["IIoT network packet: tcp flags 24, tcp flags ack 1, tcp len 349, service port 80."] * 1000
    t0 = time.perf_counter()
    model.predict_proba(st.encode(sample, batch_size=128, normalize_embeddings=True, show_progress_bar=False))
    infer = time.perf_counter() - t0

    os.makedirs(OUT, exist_ok=True)
    joblib.dump(model, os.path.join(OUT, "classifier.joblib"))
    card = {
        "name": "ics_ids_merged_v1", "version": "1.0.0",
        "trained": time.strftime("%Y-%m-%d"), "task": "binary attack vs normal",
        "datasets": MERGE, "embedding_model": EMB_MODEL, "normalize_embeddings": True,
        "representation": (
            "The three datasets share almost no columns, so each row is rendered as one short English event "
            "description (dataset prefix + up to 20 non-zero fields as 'field name value', plus HTTP method and "
            "URI path without host where present) and embedded with all-MiniLM-L6-v2 into a 384-dim normalised "
            "vector. The classifier's features are those 384 embedding dimensions; no raw dataset column is a "
            "model input."),
        "description_fields_per_dataset": {n: load_split(n)[1]["features_numeric"] + load_split(n)[1]["features_categorical"]
                                           for n in MERGE},
        "features": "384 sentence-embedding dimensions (emb_0..emb_383)",
        "input": "one event description string (template: dataset_study/textrep.py describe())",
        "test_protocol": ("held-out test rows from each dataset's test split (same rows as the dataset study); "
                          "never used for model selection, early stopping or thresholding"),
        "single_dataset_baseline_same_test_rows": single,
        "decision_threshold": 0.5, "positive_class": 1,
        "selected": best, "candidates": sel,
        "train_rows": {n: int(len(tr[n][1])) for n in MERGE}, "val_rows": {n: int(len(va[n][1])) for n in MERGE},
        "weighting": "equal total weight per dataset; classes balanced within each dataset",
        "test_per_dataset": test, "webapp_prose_check": shift,
        "inference_s_per_1000_events_cpu": infer, "seed": SEED,
    }
    save_json(card, os.path.join(OUT, "model_card.json"))
    print("[final] saved ->", OUT)


if __name__ == "__main__":
    main()
