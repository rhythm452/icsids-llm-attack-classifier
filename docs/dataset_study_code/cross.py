"""Step 4: cross-dataset analysis.

  1. Schema concept map (which kinds of features each dataset carries)
  2. Generalisation matrix train A -> test B in the shared text-embedding space
  3. Row-level train A -> test B for the one schema-compatible pair
     (X-IIoTID <-> TON_IoT-Network, harmonised Zeek-style flow features)
  4. Merge experiments (embedding space + row-level), each tested per dataset
Usage: python dataset_study/cross.py
The embedding model defaults to a stand-in until the web app's model is known;
set ICS_EMB_MODEL to override. Results are stored per model name.
"""
import os
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import ROOT, CACHE, EMB_TEST_N, EMB_TRAIN_N, MAIN_DATASETS, RES, SEED, load_json, save_json, slug
from prepare import load_split
from textrep import describe, stratified_sample
from train_eval import binary_metrics, make_preprocessor

def _webapp_model():
    """MODEL_NAME as declared in the web app's ics_llm_ids.py (read as text, not executed)."""
    p = os.path.join(ROOT, "webapp", "ics_llm_ids.py")
    if os.path.exists(p):
        m = re.search(r'^MODEL_NAME\s*=\s*["\']([^"\']+)["\']', open(p, encoding="utf-8").read(), re.M)
        if m:
            return m.group(1)
    return None


STAND_IN = "sentence-transformers/all-MiniLM-L6-v2"
WEBAPP_MODEL = _webapp_model()
EMB_MODEL = os.environ.get("ICS_EMB_MODEL", WEBAPP_MODEL or STAND_IN)
IS_STAND_IN = WEBAPP_MODEL is None

CONCEPTS = {
    "Transport protocol": {"Edge-IIoTset": "tcp.flags, tcp.connection.*", "X-IIoTID": "Protocol",
                           "TON_IoT-Network": "proto"},
    "Application service": {"Edge-IIoTset": "implied by http/mqtt/dns/mbtcp field groups",
                            "X-IIoTID": "Service", "TON_IoT-Network": "service"},
    "Flow duration": {"X-IIoTID": "Duration", "TON_IoT-Network": "duration"},
    "Bytes per direction": {"Edge-IIoTset": "tcp.len (per packet only)",
                            "X-IIoTID": "Scr_bytes, Des_bytes, Scr_ip_bytes, Des_ip_bytes",
                            "TON_IoT-Network": "src_bytes, dst_bytes, src_ip_bytes, dst_ip_bytes"},
    "Packets per direction": {"X-IIoTID": "Scr_pkts, Des_pkts", "TON_IoT-Network": "src_pkts, dst_pkts"},
    "TCP connection state / flags": {"Edge-IIoTset": "tcp.connection.syn/fin/rst/synack, tcp.flags.ack",
                                     "X-IIoTID": "Conn_state, is_syn_only, Is_SYN_ACK, is_pure_ack, FIN or RST",
                                     "TON_IoT-Network": "conn_state"},
    "DNS fields": {"Edge-IIoTset": "dns.qry.qu, dns.retransmission, ...", "TON_IoT-Network": "dns_qclass, dns_qtype, dns_rcode, dns_AA/RD/RA"},
    "HTTP fields": {"Edge-IIoTset": "http.request.method, http.response, ...", "TON_IoT-Network": "http_method, http_status_code, ..."},
    "MQTT fields": {"Edge-IIoTset": "mqtt.msgtype, mqtt.hdrflags, ...", "X-IIoTID": "(Service = mqtt only)"},
    "Modbus": {"Edge-IIoTset": "mbtcp.len, mbtcp.unit_id (framing only)",
               "TON_IoT-Modbus": "FC1-FC4 register values", "X-IIoTID": "(Service = modbus only)"},
    "Host CPU / memory / disk": {"X-IIoTID": "Avg_/Std_ user, system, iowait, tps, kbmemused, ..."},
    "Host / IDS alerts": {"X-IIoTID": "OSSEC_alert(_level), anomaly_alert, Login_attempt, File_activity, ..."},
    "Physical process values": {"BATADAL": "tank levels, pump flows/status, valve flow/status, junction pressures",
                                "TON_IoT-Modbus": "register values (semantics undocumented)"},
}
GRANULARITY = {"BATADAL": "hourly plant snapshot", "Edge-IIoTset": "single packet",
               "X-IIoTID": "connection flow + host window", "TON_IoT-Network": "connection flow",
               "TON_IoT-Modbus": "Modbus poll (4 register values)"}

HARMONISED = {  # name: (X-IIoTID column, TON_IoT-Network column)
    "proto": ("Protocol", "proto"), "service": ("Service", "service"), "duration": ("Duration", "duration"),
    "src_bytes": ("Scr_bytes", "src_bytes"), "dst_bytes": ("Des_bytes", "dst_bytes"),
    "missed_bytes": ("missed_bytes", "missed_bytes"), "src_pkts": ("Scr_pkts", "src_pkts"),
    "src_ip_bytes": ("Scr_ip_bytes", "src_ip_bytes"), "dst_pkts": ("Des_pkts", "dst_pkts"),
    "dst_ip_bytes": ("Des_ip_bytes", "dst_ip_bytes")}

MERGE_GROUPS = {
    "Edge+X-IIoTID": ["Edge-IIoTset", "X-IIoTID"],
    "Edge+TON-Net": ["Edge-IIoTset", "TON_IoT-Network"],
    "X-IIoTID+TON-Net": ["X-IIoTID", "TON_IoT-Network"],
    "Network trio": ["Edge-IIoTset", "X-IIoTID", "TON_IoT-Network"],
    "BATADAL+TON-Modbus": ["BATADAL", "TON_IoT-Modbus"],
    "All 5": MAIN_DATASETS,
}
MET = ("f1", "precision", "recall", "roc_auc", "pr_auc", "accuracy")


def _m(y, p):
    r = binary_metrics(y, p)
    return {k: r[k] for k in MET}


# ---------------------------------------------------------------- embeddings
def embed_samples():
    from sentence_transformers import SentenceTransformer
    d = os.path.join(CACHE, "emb", slug(EMB_MODEL.replace("/", "_")))
    os.makedirs(d, exist_ok=True)
    model = None
    data = {}
    for name in MAIN_DATASETS:
        parts, meta = load_split(name)
        num, cat = meta["features_numeric"], meta["features_categorical"]
        out = {}
        for split, n in (("train", EMB_TRAIN_N), ("test", EMB_TEST_N)):
            p = os.path.join(d, f"{slug(name)}_{split}")
            if not os.path.exists(p + ".npz"):
                s = stratified_sample(parts[split], n, SEED)
                texts = describe(s, name, num, cat)
                model = model or SentenceTransformer(EMB_MODEL, device="cpu")
                e = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
                np.savez_compressed(p + ".npz", X=e, y=s["__y"].to_numpy(), type=s["__type"].to_numpy())
                pd.Series(texts).to_csv(p + "_texts.csv", index=False, header=["text"])
                print(f"  [emb] {name} {split}: {len(texts)} rows", flush=True)
            z = np.load(p + ".npz", allow_pickle=True)
            out[split] = (z["X"], z["y"], z["type"])
        data[name] = out
    return data


def linear_probe(Xtr, ytr):
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED).fit(Xtr, ytr)


def embedding_matrix(data):
    mat = {}
    for a in MAIN_DATASETS:
        clf = linear_probe(data[a]["train"][0], data[a]["train"][1])
        mat[a] = {b: _m(data[b]["test"][1], clf.predict_proba(data[b]["test"][0])[:, 1]) for b in MAIN_DATASETS}
    return mat


def embedding_merges(data):
    out = {}
    for g, members in MERGE_GROUPS.items():
        X = np.vstack([data[m]["train"][0] for m in members])
        y = np.concatenate([data[m]["train"][1] for m in members])
        clf = linear_probe(X, y)
        out[g] = {"members": members,
                  "test": {m: _m(data[m]["test"][1], clf.predict_proba(data[m]["test"][0])[:, 1]) for m in members}}
    return out


# ---------------------------------------------------------------- row-level (harmonised flows)
def harmonised(name):
    parts, _ = load_split(name)
    idx = 0 if name == "X-IIoTID" else 1
    out = {}
    for s, d in parts.items():
        h = pd.DataFrame({k: d[v[idx]] for k, v in HARMONISED.items()})
        h["__y"] = d["__y"].to_numpy()
        out[s] = h
    return out


def rowlevel():
    names = ["X-IIoTID", "TON_IoT-Network"]
    H = {n: harmonised(n) for n in names}
    num = [k for k in HARMONISED if k not in ("proto", "service")]
    cat = ["proto", "service"]
    service_vocab = {n: sorted(H[n]["train"]["service"].unique().tolist()) for n in names}

    def fit(train_sets, val_sets):
        tr = pd.concat(train_sets, ignore_index=True)
        va = pd.concat(val_sets, ignore_index=True)
        pre = make_preprocessor(num, cat, scale=False)
        Xtr = pre.fit_transform(tr[num + cat])
        m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                               n_jobs=-1, verbose=-1)
        m.fit(Xtr, tr["__y"], eval_set=[(pre.transform(va[num + cat]), va["__y"])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        return pre, m

    def ev(pre, m, n):
        te = H[n]["test"]
        return _m(te["__y"].to_numpy(), m.predict_proba(pre.transform(te[num + cat]))[:, 1])

    res = {"features": HARMONISED, "service_vocab": service_vocab, "matrix": {}, "merge": {}}
    for a in names:
        pre, m = fit([H[a]["train"]], [H[a]["val"]])
        res["matrix"][a] = {b: ev(pre, m, b) for b in names}
    pre, m = fit([H[n]["train"] for n in names], [H[n]["val"] for n in names])
    res["merge"] = {b: ev(pre, m, b) for b in names}
    return res


def main():
    out_p = os.path.join(RES, f"cross_{slug(EMB_MODEL.replace('/', '_'))}.json")
    res = load_json(out_p) if os.path.exists(out_p) else {}
    res.update({"embedding_model": EMB_MODEL, "embedding_model_is_stand_in": IS_STAND_IN,
                "concepts": CONCEPTS, "granularity": GRANULARITY,
                "sample_sizes": {"train": EMB_TRAIN_N, "test": EMB_TEST_N}})
    if "embedding_matrix" not in res:
        data = embed_samples()
        res["embedding_matrix"] = embedding_matrix(data)
        res["embedding_merges"] = embedding_merges(data)
        res["embedding_sample_counts"] = {n: {s: int(len(data[n][s][1])) for s in ("train", "test")}
                                          for n in MAIN_DATASETS}
        save_json(res, out_p)
    if "rowlevel" not in res:
        res["rowlevel"] = rowlevel()
        save_json(res, out_p)
    print("[cross] done ->", out_p)
    return res


if __name__ == "__main__":
    main()
