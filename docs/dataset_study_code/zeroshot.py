"""Project zero-shot method, reproduced from the web app (webapp/app.py, ics_llm_ids.py).

Row -> short event description -> embedding (web app model) -> cosine similarity to
the 97 ATT&CK ICS technique texts ("<name>. <description>" from techniques.json).
Verdict ATTACK if max similarity >= the web app threshold (results.json
task_b_threshold), exactly as app.classify() does. The threshold is also recomputed
from procedure_examples.json + the benign sentences in ics_llm_ids.py (read as text,
not executed) to confirm it matches.

Reported per dataset on the same stratified test sample used in cross.py:
  - web-app threshold (as deployed)
  - threshold tuned for F1 on that dataset's training sample (calibrated variant)
  - ROC-AUC / PR-AUC of the raw max-similarity score (threshold-free)
  - top-1 technique vs the analyst ATT&CK mapping (mitre_map.py) for attack rows
Usage: python dataset_study/zeroshot.py   (after cross.py has cached the samples)
"""
import ast
import json
import os
import time

import numpy as np
from sklearn.metrics import precision_recall_curve

from common import MAIN_DATASETS, RES, ROOT, save_json, slug
from cross import EMB_MODEL, embed_samples
from mitre_map import MAP, T
from prepare import load_split
from textrep import describe, stratified_sample
from train_eval import binary_metrics

WEBAPP = os.path.join(ROOT, "webapp")


def benign_sentences():
    src = open(os.path.join(WEBAPP, "ics_llm_ids.py"), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "benign_samples" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("benign_samples not found in ics_llm_ids.py")


def metrics_at(y, score, thr):
    m = binary_metrics(y, (score >= thr).astype(float))
    full = binary_metrics(y, score)  # threshold-free AUCs from the raw score
    m["roc_auc"], m["pr_auc"] = full["roc_auc"], full["pr_auc"]
    m["threshold"] = float(thr)
    return m


def main():
    from sentence_transformers import SentenceTransformer
    techniques = json.load(open(os.path.join(WEBAPP, "techniques.json"), encoding="utf-8"))
    procs = json.load(open(os.path.join(WEBAPP, "procedure_examples.json"), encoding="utf-8"))
    deployed_thr = json.load(open(os.path.join(WEBAPP, "results.json")))["task_b_threshold"]
    tech_ids = list(techniques)
    tech_names = [techniques[t]["name"] for t in tech_ids]
    tech_texts = [f'{techniques[t]["name"]}. {techniques[t]["description"]}' for t in tech_ids]

    model = SentenceTransformer(EMB_MODEL, device="cpu")
    tech_emb = model.encode(tech_texts, normalize_embeddings=True, show_progress_bar=False)
    ev = [p for p in procs if p["technique_id"] in techniques]
    pe = model.encode([p["text"] for p in ev], normalize_embeddings=True, show_progress_bar=False)
    be = model.encode(benign_sentences(), normalize_embeddings=True, show_progress_bar=False)
    recomputed = float(((pe @ tech_emb.T).max(1).mean() + (be @ tech_emb.T).max(1).mean()) / 2)

    name_to_code = {v: k for k, v in T.items()}
    unmapped = sorted({n for n in name_to_code if n not in tech_names})
    data = embed_samples()  # cached stratified train/test samples, same model
    out = {"embedding_model": EMB_MODEL, "n_techniques": len(tech_ids),
           "threshold_deployed": deployed_thr, "threshold_recomputed": recomputed,
           "threshold_matches": abs(recomputed - deployed_thr) < 1e-3,
           "analyst_techniques_missing_from_techniques_json": unmapped, "datasets": {}}

    for name in MAIN_DATASETS:
        Xtr, ytr, _ = data[name]["train"]
        Xte, yte, tte = data[name]["test"]
        s_tr = (Xtr @ tech_emb.T).max(1)
        sims = Xte @ tech_emb.T
        s_te = sims.max(1)
        # calibrated variant: F1-optimal threshold on the training sample (labels from train only)
        p, r, th = precision_recall_curve(ytr, s_tr)
        f1 = 2 * p * r / np.maximum(p + r, 1e-12)
        tuned = float(th[np.argmax(f1[:-1])])
        # inference cost: describe + embed + similarity for 1000 fresh test rows
        parts, meta = load_split(name)
        sample = stratified_sample(parts["test"], 1000, 7)
        t0 = time.perf_counter()
        e = model.encode(describe(sample, name, meta["features_numeric"], meta["features_categorical"]),
                         batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        _ = (e @ tech_emb.T).max(1)
        infer = (time.perf_counter() - t0) / len(sample) * 1000
        # technique agreement on attack rows
        top1 = sims.argmax(1)
        agree, n_att, top_by_type = 0, 0, {}
        for i in np.flatnonzero(yte == 1):
            label = str(tte[i]) if name != "X-IIoTID" else None
            tname = tech_names[top1[i]]
            if name == "X-IIoTID":
                # X-IIoTID map is keyed by class2; samples carry class1 types -> use the class2 notes
                label = next((k for k, v in MAP[name].items() if str(tte[i]) in v[2]), str(tte[i]))
            exp = MAP[name].get(label) or MAP[name].get("attack")
            if exp:
                n_att += 1
                agree += name_to_code.get(tname) in exp[0]
            top_by_type.setdefault(str(tte[i]), {}).setdefault(tname, 0)
            top_by_type[str(tte[i])][tname] += 1
        out["datasets"][name] = {
            "n_test": int(len(yte)), "n_attack": int(yte.sum()),
            "deployed_threshold": metrics_at(yte, s_te, deployed_thr),
            "tuned_threshold": metrics_at(yte, s_te, tuned),
            "mean_max_sim": {"normal": float(s_te[yte == 0].mean()), "attack": float(s_te[yte == 1].mean())},
            "pct_rows_flagged_attack_at_deployed": float(100 * (s_te >= deployed_thr).mean()),
            "inference_s_per_1000": infer,
            "technique_agreement_attack_rows": agree / n_att if n_att else None,
            "top1_technique_by_type": {k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:3])
                                       for k, v in top_by_type.items()},
        }
        d = out["datasets"][name]
        print(f"[zeroshot] {name}: deployed F1={d['deployed_threshold']['f1']:.3f} "
              f"tuned F1={d['tuned_threshold']['f1']:.3f} AUC={d['deployed_threshold']['roc_auc']}", flush=True)
    save_json(out, os.path.join(RES, "zeroshot.json"))
    return out


if __name__ == "__main__":
    main()
