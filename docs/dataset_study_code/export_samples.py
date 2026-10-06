"""Export the web app's structured-event schema and held-out sample events.

Writes webapp/event_schema.json and webapp/samples/heldout_events.json. Samples come
only from each dataset's TEST split (never used for training, selection or tuning):
10 attack (spread over attack types) + 10 normal per merged dataset. Verifies that
webapp/event_template.describe_event() reproduces textrep.describe() exactly.
Usage: python dataset_study/export_samples.py
"""
import importlib.util
import json
import os

import numpy as np

from common import ROOT, SEED, save_json
from final_model import MERGE
from loaders import SPECS
from prepare import load_split
from textrep import describe

WEBAPP = os.path.join(ROOT, "webapp")
PER_CLASS = 10


def pick(test, rng):
    att, nor = test[test["__y"] == 1], test[test["__y"] == 0]
    types = att["__type"].unique().tolist()
    rng.shuffle(types)
    idx = []
    while len(idx) < PER_CLASS:  # round-robin over attack types for variety
        for t in types:
            pool = [i for i in att.index[att["__type"] == t] if i not in idx]
            if pool and len(idx) < PER_CLASS:
                idx.append(int(rng.choice(pool)))
    idx += [int(i) for i in rng.choice(nor.index.to_numpy(), size=PER_CLASS, replace=False)]
    return test.loc[idx]


def main():
    schema, samples, rng = {}, [], np.random.default_rng(SEED)
    for name in MERGE:
        parts, meta = load_split(name)
        num, cat = meta["features_numeric"], meta["features_categorical"]
        schema[name] = {"prefix": SPECS[name]["text_prefix"], "categorical": cat, "numeric": num,
                        "has_http": "__txt_http" in parts["test"].columns}
        rows = pick(parts["test"], rng)
        texts = describe(rows, name, num, cat)
        for (i, r), text in zip(rows.iterrows(), texts):
            fields = {c: r[c] for c in cat if r[c] not in ("na", "0", "")}
            fields.update({c: float(r[c]) for c in num if not np.isnan(r[c]) and r[c] != 0})
            http = None
            if "__txt_http" in rows.columns and r["__txt_http"]:
                import re
                http = re.sub(r"[a-zA-Z]+://[^/\s]+", "", r["__txt_http"]).strip() or None
            samples.append({"id": f"{name}-test-{i}", "source": name, "split": "test (held-out)",
                            "fields": fields, "http_request": http,
                            "true_label": "ATTACK" if r["__y"] == 1 else "BENIGN",
                            "true_type": str(r["__type"]), "_expected_text": text})
    save_json(schema, os.path.join(WEBAPP, "event_schema.json"))

    spec = importlib.util.spec_from_file_location("event_template", os.path.join(WEBAPP, "event_template.py"))
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)
    bad = [s["id"] for s in samples
           if et.describe_event(s["source"], s["fields"], s["http_request"])[0] != s["_expected_text"]]
    if bad:
        raise SystemExit(f"template mismatch for {len(bad)} samples: {bad[:5]}")
    for s in samples:
        s.pop("_expected_text")
    save_json({"note": "Held-out test-split rows from the dataset study; never used for training or tuning.",
               "samples": samples}, os.path.join(WEBAPP, "samples", "heldout_events.json"))
    print(f"[export] {len(samples)} samples, template matches training text for all of them")


if __name__ == "__main__":
    main()
