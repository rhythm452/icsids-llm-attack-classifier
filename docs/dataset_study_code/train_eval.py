"""Step 1b/2: one shared train/eval routine for every dataset and model.

Usage: python dataset_study/train_eval.py [dataset ...]
Results: results/<dataset>/{binary,multiclass}_<model>.json + probs_*.npz
"""
import os
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from common import APPENDIX_DATASETS, CACHE, MAIN_DATASETS, RES, SEED, load_json, save_json, slug
from prepare import load_split

warnings.filterwarnings("ignore", category=lgb.basic.LGBMDeprecationWarning) if hasattr(lgb.basic, "LGBMDeprecationWarning") else warnings.filterwarnings("ignore", message=".*eval_set.*")

MODELS = ["LogReg", "RandomForest", "LightGBM", "MLP"]
GRIDS = {
    "LogReg": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
    "RandomForest": [{"max_depth": None, "min_samples_leaf": 1}, {"max_depth": None, "min_samples_leaf": 5},
                     {"max_depth": 20, "min_samples_leaf": 1}],
    "LightGBM": [{"num_leaves": 31}, {"num_leaves": 127}],
    "MLP": [{"hidden": [256, 128]}, {"hidden": [128, 64]}],
}


# ---------------------------------------------------------------- preprocessing
def make_preprocessor(num, cat, scale):
    steps = [("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                               ("sc", StandardScaler() if scale else "passthrough")]), num)]
    if cat:
        steps.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                           max_categories=30, sparse_output=False), cat))
    return ColumnTransformer(steps, verbose_feature_names_out=False)


# ---------------------------------------------------------------- MLP (torch, class-weighted)
class TorchMLP:
    def __init__(self, hidden, n_classes, class_weight, epochs=30, patience=4, lr=1e-3, batch=1024):
        self.hidden, self.k, self.cw = hidden, n_classes, class_weight
        self.epochs, self.patience, self.lr, self.batch = epochs, patience, lr, batch

    def _net(self, d):
        layers, prev = [], d
        for h in self.hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
            prev = h
        layers.append(torch.nn.Linear(prev, self.k))
        return torch.nn.Sequential(*layers)

    def fit(self, X, y, Xv=None, yv=None):
        torch.manual_seed(SEED)
        use_val = Xv is not None
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        if use_val:
            Xv, yv = torch.tensor(Xv, dtype=torch.float32), torch.tensor(yv, dtype=torch.long)
        self.net = self._net(X.shape[1])
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.cw, dtype=torch.float32))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(SEED)
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                loss_fn(self.net(X[b]), y[b]).backward()
                opt.step()
            self.net.eval()
            if not use_val:  # fixed number of epochs, keep final weights
                self.best_epoch = ep + 1
                continue
            with torch.no_grad():
                vl = float(loss_fn(self.net(Xv), yv))
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                self.best_epoch = ep + 1
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if use_val:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 65536):
                out.append(torch.softmax(self.net(torch.tensor(X[i:i + 65536], dtype=torch.float32)), 1))
        return torch.cat(out).numpy()


# ---------------------------------------------------------------- model factory
def fit_model(name, params, X, y, Xv, yv, n_classes):
    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y)
    if name == "LogReg":
        m = LogisticRegression(C=params["C"], class_weight="balanced", max_iter=2000, random_state=SEED)
        m.fit(X, y)
    elif name == "RandomForest":
        m = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=SEED, **params)
        m.fit(X, y)
    elif name == "LightGBM":
        p = dict(params)
        n_est = p.pop("n_estimators", 2000)
        m = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.05, class_weight="balanced",
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               random_state=SEED, n_jobs=-1, verbose=-1, **p)
        if Xv is not None and "n_estimators" not in params:
            m.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            m.fit(X, y)
    elif name == "MLP":
        p = dict(params)
        m = TorchMLP(p["hidden"], n_classes, cw, epochs=p.get("epochs", 30))
        m.fit(X, y, Xv, yv) if "epochs" not in p else m.fit(X, y)
    return m


def fitted_params(name, params, m):
    """Params needed to refit without a validation set (used for LOWO)."""
    p = dict(params)
    if name == "LightGBM":
        p["n_estimators"] = int(m.best_iteration_ or m.n_estimators)
    if name == "MLP":
        p["epochs"] = int(m.best_epoch)
    return p


# ---------------------------------------------------------------- metrics
def binary_metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    both = len(np.unique(y)) == 2
    return {
        "accuracy": accuracy_score(y, pred), "precision": p, "recall": r, "f1": f,
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, prob) if both else None,
        "pr_auc": average_precision_score(y, prob) if both else None,
        "fpr": fp / (fp + tn) if fp + tn else None, "fnr": fn / (fn + tp) if fn + tp else None,
        "confusion_matrix": cm.tolist(), "n_test": int(len(y)), "n_attack": int(y.sum()),
    }


def multiclass_metrics(y, prob, classes):
    pred = prob.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=np.arange(len(classes)), zero_division=0)
    try:
        auc = roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=np.arange(len(classes)))
    except ValueError:
        auc = None
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "roc_auc_ovr_macro": auc,
        "per_class": {c: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(len(classes))).tolist(),
        "classes": list(classes),
    }


def score(y, prob, task):
    if task == "binary":
        return f1_score(y, (prob[:, 1] >= 0.5).astype(int), zero_division=0)
    return f1_score(y, prob.argmax(1), average="macro", zero_division=0)


# ---------------------------------------------------------------- feature matrices
def matrices(parts, meta, scale, drop=()):
    num = [c for c in meta["features_numeric"] if c not in drop]
    cat = [c for c in meta["features_categorical"] if c not in drop]
    pre = make_preprocessor(num, cat, scale)
    Xtr = pre.fit_transform(parts["train"][num + cat])  # fit on train only
    Xva = pre.transform(parts["val"][num + cat])
    return pre, num + cat, Xtr.astype(np.float32), Xva.astype(np.float32)


def run_task(name, task, models=MODELS, drop=(), tag="", target_col=None):
    """Train/select/evaluate all models for one dataset+task. Saves per model."""
    parts, meta = load_split(name)
    out_dir = os.path.join(RES, slug(name))
    col = target_col or ("__y" if task == "binary" else "__multi")
    if task == "binary":
        classes = np.array([0, 1])
        ys = {s: d[col].to_numpy().astype(int) for s, d in parts.items()}
    else:
        le = LabelEncoder().fit(pd.concat([d[col] for d in parts.values()]))
        classes = le.classes_
        ys = {s: le.transform(d[col]) for s, d in parts.items()}
    results = {}
    for mname in models:
        res_p = os.path.join(out_dir, f"{task}{tag}_{mname}.json")
        if os.path.exists(res_p):
            results[mname] = load_json(res_p)
            continue
        scale = mname in ("LogReg", "MLP")
        pre, cols, Xtr, Xva = matrices(parts, meta, scale, drop)
        grid = GRIDS[mname]
        if task != "binary" and not tag.startswith("_search"):
            # multiclass: reuse the config selected in the binary run (no new search)
            bp = os.path.join(out_dir, f"binary_{mname}.json")
            if os.path.exists(bp):
                grid = [load_json(bp)["selected_params"]]
        search = []
        best = None
        for params in grid:
            t0 = time.perf_counter()
            m = fit_model(mname, params, Xtr, ys["train"], Xva, ys["val"], len(classes))
            tt = time.perf_counter() - t0
            s = score(ys["val"], m.predict_proba(Xva), task)
            search.append({"params": params, "val_score": s, "train_time_s": tt})
            print(f"  [{name}/{task}{tag}] {mname} {params} val={s:.4f} ({tt:.1f}s)", flush=True)
            if best is None or s > best[0]:
                best = (s, params, m, tt)
        _, params, m, tt = best
        t0 = time.perf_counter()
        Xte = pre.transform(parts["test"][cols]).astype(np.float32)
        prob = m.predict_proba(Xte)
        infer = (time.perf_counter() - t0) / len(Xte) * 1000
        yte = ys["test"]
        mets = binary_metrics(yte, prob[:, 1]) if task == "binary" else multiclass_metrics(yte, prob, classes)
        mets.update({"train_time_s": tt, "inference_s_per_1000": infer})
        r = {"dataset": name, "task": task, "tag": tag, "model": mname, "selected_params": params,
             "refit_params": fitted_params(mname, params, m), "val_search": search,
             "test": mets, "n_features_in": len(cols), "n_features_encoded": int(Xtr.shape[1]),
             "dropped_extra": list(drop)}
        if mname == "LightGBM":
            names = pre.get_feature_names_out()
            imp = m.booster_.feature_importance("gain")
            order = np.argsort(-imp)[:25]
            tot = imp.sum() or 1
            r["feature_importance_gain"] = [{"feature": names[i], "share": imp[i] / tot} for i in order]
        save_json(r, res_p)
        np.savez_compressed(os.path.join(out_dir, f"probs_{task}{tag}_{mname}.npz"), y=yte, prob=prob)
        results[mname] = r
    return results


def run_all():
    for n in MAIN_DATASETS:
        run_task(n, "binary")
        if load_split(n)[1]["multiclass"]:
            run_task(n, "multiclass")
            if n == "X-IIoTID":
                run_task(n, "multiclass", models=["LightGBM"], tag="_class1", target_col="__class1")
    for n in APPENDIX_DATASETS:
        if load_json(os.path.join(CACHE, slug(n), "meta.json")).get("evaluable"):
            run_task(n, "binary")
    for n in MAIN_DATASETS[1:]:
        run_task(n + "@nodedup", "binary", models=["LightGBM"])


if __name__ == "__main__":
    if not sys.argv[1:]:
        run_all()
    for n in sys.argv[1:]:
        run_task(n, "binary")
        if n in MAIN_DATASETS and load_split(n)[1]["multiclass"]:
            run_task(n, "multiclass")
            if n == "X-IIoTID":
                run_task(n, "multiclass", models=["LightGBM"], tag="_class1", target_col="__class1")
