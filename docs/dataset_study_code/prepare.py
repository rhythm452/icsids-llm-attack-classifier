"""Step 1a: load -> clean -> de-duplicate -> sample -> split, cached as parquet.

Usage: python dataset_study/prepare.py [dataset ...]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (APPENDIX_DATASETS, CACHE, MAIN_DATASETS, MIN_PER_CLASS, SAMPLE_CAP, SEED,
                    SPLIT, load_json, path, save_json, slug)
from loaders import BATADAL_SPLIT, DERIVED_DOC, SPECS, load



def stratified_cap(df, cap, min_per_class, seed):
    counts = df["__type"].value_counts()
    if len(df) <= cap:
        return df, {"sampled": False}
    keep = {c: min(n, min_per_class) for c, n in counts.items()}
    remaining = cap - sum(keep.values())
    extra_pool = {c: n - keep[c] for c, n in counts.items()}
    pool_total = sum(extra_pool.values())
    for c in counts.index:
        keep[c] += int(np.floor(remaining * extra_pool[c] / pool_total))
    rng = np.random.default_rng(seed)
    idx = []
    for c, k in keep.items():
        ci = np.flatnonzero(df["__type"].values == c)
        idx.append(rng.choice(ci, size=k, replace=False))
    out = df.iloc[np.sort(np.concatenate(idx))].reset_index(drop=True)
    return out, {"sampled": True, "per_class_kept": keep}


def prepare(name):
    out_dir = os.path.join(CACHE, slug(name))
    meta_p = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_p):
        print(f"[prepare] {name}: cached")
        return load_json(meta_p)
    base, _, variant = name.partition("@")  # "<dataset>@nodedup" = sensitivity variant
    spec = SPECS[base]
    print(f"[prepare] {name}: loading", flush=True)
    clean_p = os.path.join(CACHE, slug(base), "clean_full.parquet")
    if os.path.exists(clean_p):
        df, meta = pd.read_parquet(clean_p), load_json(clean_p + ".meta.json")
    else:
        df, meta = load(base)
        df.to_parquet(path(clean_p))
        save_json(meta, clean_p + ".meta.json")
    feats = [c for c in df.columns if not c.startswith("__")]  # labels/time/text-only columns
    n0 = len(df)
    types_before = df["__type"].value_counts().to_dict()

    # De-duplicate on cleaned features + finest label (identifiers already dropped).
    if variant != "nodedup":
        df = df.drop_duplicates(subset=feats + ["__type"]).reset_index(drop=True)
    n_dedup = len(df)
    types_after = df["__type"].value_counts().to_dict()
    # Feature vectors that still appear with more than one label = irreducible label conflict.
    key = pd.util.hash_pandas_object(df[feats], index=False)
    lab_n = df.groupby(key.values)["__y"].transform("nunique")
    conflict_rows = int((lab_n > 1).sum())

    if df["__y"].nunique() < 2 or df["__y"].value_counts().min() < 10:
        meta.update({"name": name, "kind": spec["kind"], "evaluable": False,
                     "reason": f"only {n_dedup} unique rows after de-duplication "
                               f"(per label: {df['__y'].value_counts().to_dict()})",
                     "rows_after_cleaning": n0, "rows_after_dedup": n_dedup,
                     "type_counts_before_dedup": types_before, "type_counts_after_dedup": types_after,
                     "conflicting_label_rows": conflict_rows})
        save_json(meta, meta_p)
        print(f"[prepare] {name}: NOT EVALUABLE - {meta['reason']}", flush=True)
        return meta

    if spec["split"] == "chronological":
        df = df.sort_values("__time").reset_index(drop=True)
        samp = {"sampled": False}
        t = df["__time"]
        split = np.where(t < pd.Timestamp(BATADAL_SPLIT["val_start"]), "train",
                         np.where(t < pd.Timestamp(BATADAL_SPLIT["test_start"]), "val", "test"))
        parts = {s: df[split == s] for s in ("train", "val", "test")}
        split_desc = (f"chronological: train < {BATADAL_SPLIT['val_start']} (dataset03 + dataset04 "
                      f"windows 1-2), val < {BATADAL_SPLIT['test_start']} (window 3), test after (windows 4-5)")
    else:
        df, samp = stratified_cap(df, SAMPLE_CAP, MIN_PER_CLASS, SEED)
        # Classes with < 10 unique rows cannot be split three ways: keep them in train only.
        vc = df["__type"].value_counts()
        tiny = vc[vc < 10].index.tolist()
        train_only = df[df["__type"].isin(tiny)]
        df = df[~df["__type"].isin(tiny)]
        samp["train_only_tiny_classes"] = {t: int(vc[t]) for t in tiny}
        strat = df["__type"]
        tr, rest = train_test_split(df, test_size=SPLIT[1] + SPLIT[2], stratify=strat, random_state=SEED)
        tr = pd.concat([tr, train_only])
        va, te = train_test_split(rest, test_size=SPLIT[2] / (SPLIT[1] + SPLIT[2]),
                                  stratify=rest["__type"], random_state=SEED)
        parts = {"train": tr, "val": va, "test": te}
        split_desc = "stratified random 70/15/15 by finest attack type (no usable time order)"

    # Constant-in-train columns carry no information.
    constant = [c for c in feats if parts["train"][c].nunique(dropna=False) <= 1]
    feats = [c for c in feats if c not in constant]
    for s, d in parts.items():
        d.reset_index(drop=True).to_parquet(path(out_dir, f"{s}.parquet"))

    cat = [c for c in spec["categorical"] if c in feats]
    num = [c for c in feats if c not in cat]
    dropped = dict(spec["drop"])
    dropped.update({c: "constant in training split" for c in constant})
    meta.update({
        "name": name, "kind": spec["kind"], "split": split_desc,
        "rows_after_cleaning": n0, "rows_after_dedup": n_dedup,
        "duplicates_removed": n0 - n_dedup,
        "type_counts_before_dedup": types_before, "type_counts_after_dedup": types_after,
        "conflicting_label_rows": conflict_rows,
        "sampling": samp, "features_numeric": num, "features_categorical": cat,
        "dropped_columns": dropped,
        "derived_features": [c for c in feats if c == "service_port" or c.startswith(("len_", "inj_"))],
        "derived_doc": DERIVED_DOC if spec.get("derive") else {},
        "text_only_columns": [c for c in df.columns if c.startswith("__txt")],
        "n_rows": {s: len(d) for s, d in parts.items()},
        "attack_ratio": {s: float(d["__y"].mean()) for s, d in parts.items()},
        "type_counts": {s: d["__type"].value_counts().to_dict() for s, d in parts.items()},
        "multiclass": spec["multiclass"] is not None, "evaluable": True,
        "variant": variant or "dedup",
    })
    save_json(meta, meta_p)
    print(f"[prepare] {name}: {n0} -> dedup {n_dedup} -> {meta['n_rows']}", flush=True)
    return meta


def load_split(name):
    d = os.path.join(CACHE, slug(name))
    meta = load_json(os.path.join(d, "meta.json"))
    parts = {s: pd.read_parquet(os.path.join(d, f"{s}.parquet")) for s in ("train", "val", "test")}
    return parts, meta


if __name__ == "__main__":
    for n in sys.argv[1:] or MAIN_DATASETS + APPENDIX_DATASETS:
        prepare(n)
