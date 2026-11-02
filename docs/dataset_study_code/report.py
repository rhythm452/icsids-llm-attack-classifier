"""Step 5: assemble report.md + self-contained report.html from saved results.

All numbers are read from results/*.json. The judgement sections (recommendation,
limitations) come from results/recommendation.md, written after reading the numbers.
Usage: python dataset_study/report.py
"""
import base64
import glob
import os
import re

from markdown_it import MarkdownIt

import plots
from common import (APPENDIX_DATASETS, CACHE, EMB_TEST_N, EMB_TRAIN_N, MAIN_DATASETS, MIN_PER_CLASS, RES,
                    SAMPLE_CAP, STUDY, load_json, slug)
from mitre_map import MAP, T
from train_eval import MODELS


def f(v, d=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def img(p, alt):
    return f"![{alt}](plots/{os.path.basename(p)})"


def meta(n):
    return load_json(os.path.join(CACHE, slug(n), "meta.json"))


def res(n, task, m, tag=""):
    p = os.path.join(RES, slug(n), f"{task}{tag}_{m}.json")
    return load_json(p) if os.path.exists(p) else None


def section_summary(L):
    prof = load_json(os.path.join(RES, "step0_profile.json"))
    raw_rows = {"BATADAL": sum(prof[k]["rows"] for k in prof if k.startswith("BATADAL/"))}
    L.append("## 1. Dataset summary\n")
    L.append(f"Sampling policy (same for every dataset): after removing duplicates, each dataset is capped at "
             f"{SAMPLE_CAP:,} rows, stratified by its finest attack-type label, keeping at least "
             f"{MIN_PER_CLASS:,} rows per class (or all rows if fewer). Sub-sampled after de-duplication: "
             f"**{', '.join(n for n in MAIN_DATASETS if meta(n)['sampling'].get('sampled')) or 'none'}**; all "
             f"other main datasets are used in full. Splits: 70/15/15 stratified unless noted. Classes with fewer "
             f"than 10 unique rows after de-duplication go to the training split only.\n")
    rows = []
    for n in MAIN_DATASETS:
        m = meta(n)
        rows.append([n, m["kind"], m["rows_raw"] if "rows_raw" in m else "", f"{m['rows_after_dedup']:,}",
                     f"{m['duplicates_removed']:,} ({100 * m['duplicates_removed'] / m['rows_after_cleaning']:.1f}%)",
                     f"{m['conflicting_label_rows']:,}",
                     " / ".join(f"{m['n_rows'][s]:,}" for s in ("train", "val", "test")),
                     " / ".join(f"{100 * m['attack_ratio'][s]:.1f}" for s in ("train", "val", "test")),
                     f"{len(m['features_numeric'])} + {len(m['features_categorical'])}",
                     len(m["dropped_columns"]), "chronological" if m["split"].startswith("chrono") else "stratified"])
    for r in rows:
        r[2] = f"{r[2]:,}" if isinstance(r[2], int) else r[2]
    L.append(table(["Dataset", "Type", "Raw rows", "Unique rows", "Duplicates removed", "Rows with conflicting labels",
                    "Train / val / test", "Attack % (tr/va/te)", "Features (num + cat)", "Cols dropped", "Split"], rows))
    L.append("\n**Duplicates are counted after identifier columns are removed** (a row that differs only by "
             "timestamp/IP/port/sequence number is a duplicate). 'Conflicting labels' = rows whose feature vector "
             "also occurs with the other label (irreducible error).\n")
    L.append("### Fields kept per dataset\n")
    L.append("Same policy for every network dataset: identifiers (IPs, timestamps), per-packet counters and "
             "checksums are dropped; the two raw port numbers become one `service_port` (lower, server-side port); "
             "raw strings (payloads, URIs, queries, DNS names, certificates, user agents) become length and "
             "injection-character-count summaries. HTTP method + URI (first 80 chars) is kept only for the text "
             "descriptions used by the embedding/zero-shot methods, never as a tabular feature.\n")
    for n in MAIN_DATASETS:
        m = meta(n)
        der = set(m.get("derived_features", []))
        orig = [c for c in m["features_numeric"] + m["features_categorical"] if c not in der]
        L.append(f"**{n}** ({len(orig) + len(der)} features)\n")
        L.append(f"- Original fields kept: {', '.join(f'`{c}`' for c in orig)}")
        if der:
            L.append(f"- Derived fields: {', '.join(f'`{c}`' for c in m['derived_features'])}")
        if m.get("text_only_columns"):
            L.append(f"- Text-only (descriptions, not tabular): {', '.join(f'`{c}`' for c in m['text_only_columns'])}")
        L.append("")
    doc = next((meta(n)["derived_doc"] for n in MAIN_DATASETS if meta(n).get("derived_doc")), {})
    if doc:
        L.append(table(["Derived field", "Definition"], [[f"`{k}`", v] for k, v in doc.items()]) + "\n")
    L.append("### Columns dropped and why\n")
    for n in MAIN_DATASETS:
        m = meta(n)
        by_reason = {}
        for c, r in m["dropped_columns"].items():
            by_reason.setdefault(r, []).append(f"`{c}`")
        L.append(f"**{n}**\n")
        L += [f"- {', '.join(cs)}: {r}" for r, cs in by_reason.items()]
        L.append("")
    L.append("### Per-class counts before and after de-duplication\n")
    for n in MAIN_DATASETS[1:]:
        m = meta(n)
        b, a = m["type_counts_before_dedup"], m["type_counts_after_dedup"]
        L.append(f"<details><summary>{n}</summary>\n")
        L.append(table(["Type", "Before", "After"], [[k, f"{b[k]:,}", f"{a.get(k, 0):,}"]
                                                     for k in sorted(b, key=lambda k: -b[k])]))
        tiny = m["sampling"].get("train_only_tiny_classes")
        if tiny:
            L.append(f"\nTrain-only (fewer than 10 unique rows): {tiny}")
        L.append("\n</details>\n")


def binary_rows(n, models=MODELS, tag=""):
    rows = []
    for mname in models:
        r = res(n, "binary", mname, tag)
        if not r:
            rows.append([mname] + ["not run"] * 11)
            continue
        t = r["test"]
        rows.append([mname, f(t["accuracy"]), f(t["precision"]), f(t["recall"]), f(t["f1"]), f(t["macro_f1"]),
                     f(t["roc_auc"]), f(t["pr_auc"]), f(t["fpr"]), f(t["fnr"]), f(t["train_time_s"], 1),
                     f(t["inference_s_per_1000"] * 1000, 2)])
    return rows


BIN_HEAD = ["Model", "Acc", "Prec", "Recall", "F1 (attack)", "Macro-F1", "ROC-AUC", "PR-AUC", "FPR", "FNR",
            "Train s", "Infer ms/1k rows"]


def section_metrics(L):
    L.append("## 2. Per-dataset metrics (held-out test set)\n")
    L.append("Models: Logistic Regression (baseline), Random Forest, LightGBM, MLP (PyTorch). Class weights "
             "'balanced'; no oversampling. Light hyperparameter search scored on the validation split (attack F1 "
             "for binary, macro-F1 for multiclass); multiclass reuses the binary-selected configuration. "
             "Decision threshold 0.5. Seed 42 throughout. Inference time includes preprocessing.\n")
    zs = load_json(os.path.join(RES, "zeroshot.json")) if os.path.exists(os.path.join(RES, "zeroshot.json")) else {}
    for n in MAIN_DATASETS:
        m = meta(n)
        L.append(f"### {n}\n")
        L.append(f"Split: {m['split']}. Test rows: {m['n_rows']['test']:,} ({100 * m['attack_ratio']['test']:.1f}% attack).\n")
        L.append(table(BIN_HEAD, binary_rows(n)))
        zd = zs.get("datasets", {}).get(n)
        if zd:
            L.append(f"\nProject zero-shot method (web app model `{zs['embedding_model']}`, 97 technique texts), "
                     f"on a stratified test sample of {zd['n_test']:,} rows ({zd['n_attack']:,} attack); "
                     f"ROC/PR-AUC use the raw max-similarity score. Inference time includes embedding on CPU.\n")
            rows = []
            for label, k in (("Zero-shot, deployed threshold", "deployed_threshold"),
                             ("Zero-shot, threshold tuned on train sample", "tuned_threshold")):
                t = zd[k]
                rows.append([f"{label} ({t['threshold']:.3f})", f(t["accuracy"]), f(t["precision"]), f(t["recall"]),
                             f(t["f1"]), f(t["macro_f1"]), f(t["roc_auc"]), f(t["pr_auc"]), f(t["fpr"]), f(t["fnr"]),
                             "none", f(zd["inference_s_per_1000"] * 1000, 0)])
            L.append(table(BIN_HEAD, rows) + "\n")
        for p in plots.per_dataset(n):
            L.append(img(p, os.path.basename(p)) + "\n")
        if n == "BATADAL":
            section_lowo(L)
        if m.get("multiclass"):
            section_multiclass(L, n)
    L.append("### Appendix: other TON_IoT IoT device files (low-signal)\n")
    L.append("Each file has only 1-3 sensor features. Marked low-signal: most feature vectors repeat, and the "
             "attack/normal rows were captured at different times, so these results say little about real "
             "detection ability.\n")
    rows = []
    for n in APPENDIX_DATASETS:
        m = meta(n)
        if not m.get("evaluable"):
            rows.append([n, "not evaluable: " + m["reason"]] + [""] * 6)
            continue
        for mname in MODELS:
            r = res(n, "binary", mname)
            t = r["test"]
            rows.append([n, mname, f"{m['rows_after_dedup']:,}", f(t["accuracy"]), f(t["f1"]), f(t["roc_auc"]),
                         f(t["fpr"]), f(t["fnr"])])
    L.append(table(["Device file", "Model", "Unique rows", "Acc", "F1", "ROC-AUC", "FPR", "FNR"], rows) + "\n")


def section_lowo(L):
    p = os.path.join(RES, "BATADAL", "lowo.json")
    if not os.path.exists(p):
        return
    lo = load_json(p)
    L.append("#### BATADAL leave-one-attack-window-out (LOWO)\n")
    L.append("dataset04 is cut at the midpoints between its 5 labelled attack windows; each fold tests on one "
             "segment and trains on dataset03 + the other segments (so training data may come *after* the test "
             "period). Hyperparameters fixed to the main-split selection. Caveat: `-999` rows are treated as "
             "normal, but some may be unlabelled attacks.\n")
    rows = []
    for mname, r in lo.items():
        fs = r["folds"]
        rows.append([mname] + [f"{x['f1']:.2f} / {f(x['roc_auc'], 2)}" for x in fs] +
                    [f"{r['summary']['f1']['mean']:.3f} ± {r['summary']['f1']['std']:.3f}",
                     f"{r['summary']['roc_auc']['mean']:.3f} ± {r['summary']['roc_auc']['std']:.3f}"])
    L.append(table(["Model"] + [f"Window {k} F1 / AUC" for k in range(1, 6)] + ["Mean F1 ± sd", "Mean AUC ± sd"], rows) + "\n")


def section_multiclass(L, n):
    L.append(f"#### {n}: multiclass\n")
    rows = []
    for mname in MODELS:
        r = res(n, "multiclass", mname)
        if r:
            t = r["test"]
            rows.append([mname, f(t["accuracy"]), f(t["macro_f1"]), f(t["weighted_f1"]), f(t["roc_auc_ovr_macro"])])
    L.append(table(["Model", "Acc", "Macro-F1", "Weighted-F1", "ROC-AUC (OvR macro)"], rows) + "\n")
    best = max((res(n, "multiclass", mm) for mm in MODELS if res(n, "multiclass", mm)),
               key=lambda r: r["test"]["macro_f1"])
    L.append(f"Per-class results, best model by macro-F1 ({best['model']}):\n")
    pc = best["test"]["per_class"]
    L.append(table(["Class", "Precision", "Recall", "F1", "Test support"],
                   [[c, f(v["precision"]), f(v["recall"]), f(v["f1"]), v["support"]] for c, v in pc.items()]) + "\n")
    if n == "X-IIoTID":
        r = res(n, "multiclass", "LightGBM", "_class1")
        if r:
            full = meta(n)["type_counts_after_dedup"]
            L.append("Extra: X-IIoTID `class1` (19 types), LightGBM. Classes with under 100 unique rows are "
                     "flagged *unstable*.\n")
            L.append(f"Accuracy {f(r['test']['accuracy'])}, macro-F1 {f(r['test']['macro_f1'])}.\n")
            L.append(table(["class1", "Unique rows", "Precision", "Recall", "F1", "Test support", "Flag"],
                           [[c, full.get(c, 0), f(v["precision"]), f(v["recall"]), f(v["f1"]), v["support"],
                             "unstable (<100 rows)" if full.get(c, 0) < 100 else ""]
                            for c, v in r["test"]["per_class"].items()]) + "\n")


def section_sanity(L):
    L.append("## 3. Sanity checks and leakage\n")
    rows = []
    for n in MAIN_DATASETS:
        s = load_json(os.path.join(RES, slug(n), "sanity.json"))
        o = s["train_test_overlap"]
        top = s["single_feature"][0]
        imp = s["lightgbm_importance_by_raw_feature"][0]
        rows.append([n, ", ".join(f"{k} {v:.4f}" for k, v in s["flag_accuracy_over_99"].items()) or "none",
                     f"{o['test_rows_seen_in_train']:,} ({o['pct_seen_in_train']:.1f}%)",
                     f"`{top['feature']}` AUC {top['train_auc']:.3f}, stump F1 {top['stump_test_f1']:.3f}",
                     f"`{imp[0]}` ({100 * imp[1]:.0f}% of gain)"])
    L.append(table(["Dataset", "Models with test acc > 99%", "Test rows whose features occur in train",
                    "Most separating single feature", "Top LightGBM feature"], rows) + "\n")
    L.append("### Ablations (LightGBM, retrained without the suspicious feature(s))\n")
    for n in MAIN_DATASETS:
        s = load_json(os.path.join(RES, slug(n), "sanity.json"))
        L.append(f"**{n}**\n")
        L.append(table(["Variant", "Acc", "F1", "Macro-F1", "ROC-AUC", "PR-AUC"],
                       [[k, f(v["accuracy"]), f(v["f1"]), f(v["macro_f1"]), f(v["roc_auc"]), f(v["pr_auc"])]
                        for k, v in s["ablations_lightgbm"].items()]) + "\n")
        sf = s["single_feature"][:5]
        L.append("Top single features: " + "; ".join(
            f"`{x['feature']}` (train AUC {x['train_auc']:.3f}, stump test F1 {x['stump_test_f1']:.3f})" for x in sf) + "\n")
        if "edge_without_misaligned_classes" in s:
            e = s["edge_without_misaligned_classes"]
            L.append(f"Edge-IIoTset excluding the column-misaligned DDoS_UDP/MITM rows ({e['n_excluded']} test rows "
                     f"removed): acc {f(e['accuracy'])}, F1 {f(e['f1'])}, ROC-AUC {f(e['roc_auc'])}.\n")
    L.append("### Effect of de-duplication (LightGBM, no-dedup variant vs main)\n")
    L.append("The no-dedup variant keeps every row (as most published results do). It shows how much of a "
             "published-style score comes from identical feature vectors in train and test.\n")
    rows = []
    for n in MAIN_DATASETS[1:]:
        a, b = res(n, "binary", "LightGBM"), res(n + "@nodedup", "binary", "LightGBM")
        s = load_json(os.path.join(RES, slug(n + "@nodedup"), "sanity.json"))
        if a and b:
            rows.append([n, f(b["test"]["accuracy"], 4), f(b["test"]["f1"], 4),
                         f"{s['train_test_overlap']['pct_seen_in_train']:.1f}%",
                         f(a["test"]["accuracy"], 4), f(a["test"]["f1"], 4)])
    L.append(table(["Dataset", "No-dedup acc", "No-dedup F1", "No-dedup test rows seen in train",
                    "Dedup acc", "Dedup F1"], rows) + "\n")


def section_zeroshot(L):
    p = os.path.join(RES, "zeroshot.json")
    if not os.path.exists(p):
        return
    z = load_json(p)
    L.append("### Project zero-shot method: summary\n")
    L.append(f"Reproduced from `webapp/app.py`: verdict ATTACK when the max cosine similarity to the "
             f"{z['n_techniques']} ATT&CK ICS technique texts is >= {z['threshold_deployed']:.4f} "
             f"(`results.json`). Recomputed from `procedure_examples.json` and the benign sentences: "
             f"{z['threshold_recomputed']:.4f} ({'matches' if z['threshold_matches'] else 'DOES NOT match'}). "
             f"Techniques used in the analyst mapping but absent from `techniques.json`: "
             f"{', '.join(z['analyst_techniques_missing_from_techniques_json']) or 'none'}.\n")
    rows = []
    for n, d in z["datasets"].items():
        top = {}
        for v in d["top1_technique_by_type"].values():
            for k, c in v.items():
                top[k] = top.get(k, 0) + c
        top3 = ", ".join(k for k, _ in sorted(top.items(), key=lambda kv: -kv[1])[:3])
        agree = d["technique_agreement_attack_rows"]
        rows.append([n, f"{d['mean_max_sim']['normal']:.3f} / {d['mean_max_sim']['attack']:.3f}",
                     f"{d['pct_rows_flagged_attack_at_deployed']:.1f}%", f(d["deployed_threshold"]["f1"]),
                     f(d["deployed_threshold"]["roc_auc"]),
                     f"{100 * agree:.1f}%" if agree is not None else "n/a", top3])
    L.append(table(["Dataset", "Mean max-sim normal / attack", "Rows flagged at deployed threshold", "F1",
                    "ROC-AUC", "Top-1 technique agrees with analyst map", "Most frequent top-1 techniques"], rows) + "\n")


def section_cross(L):
    p = sorted(glob.glob(os.path.join(RES, "cross_*.json")))
    if not p:
        L.append("## 4. Cross-dataset analysis\n\nNot run.\n")
        return
    c = load_json(p[0])
    L.append("## 4. Cross-dataset analysis\n")
    L.append("### Feature-schema comparison\n")
    L.append(table(["Concept"] + MAIN_DATASETS,
                   [[k] + [v.get(d, "") for d in MAIN_DATASETS] for k, v in c["concepts"].items()]))
    L.append("\n" + table(["Dataset", "Row granularity"], [[k, v] for k, v in c["granularity"].items()]) + "\n")
    L.append("Row-level merging is only possible for **X-IIoTID + TON_IoT-Network**: both carry Zeek-style flow "
             "fields (protocol, service, duration, bytes and packets per direction, missed bytes). Edge-IIoTset is "
             "per-packet, BATADAL is an hourly plant snapshot and TON_IoT-Modbus is 4 register values, so no shared "
             "columns exist. All other pairs can only be compared through the shared text-embedding representation.\n")
    stand = c["embedding_model_is_stand_in"]
    L.append(f"### Generalisation matrix: text-embedding representation\n")
    L.append(f"Rows rendered as short event descriptions, embedded with **`{c['embedding_model']}`**"
             + (" (**stand-in**: the web app's model was not available, see Limitations)" if stand else "")
             + f", classified with a class-weighted logistic-regression probe. {EMB_TRAIN_N:,} training rows and "
               f"{EMB_TEST_N:,} test rows per dataset (stratified by attack type; BATADAL test uses all "
               f"{c['embedding_sample_counts']['BATADAL']['test']} rows). The diagonal is the in-domain score.\n")
    hp = plots.cross_heatmap(c["embedding_matrix"], "Attack F1, train on row dataset, test on column dataset (embeddings)",
                             "cross_embedding_f1.png")
    L.append(img(hp, "cross-dataset F1 heatmap") + "\n")
    L.append(table(["Train \\ Test"] + MAIN_DATASETS,
                   [[a] + [f"{c['embedding_matrix'][a][b]['f1']:.3f} (AUC {f(c['embedding_matrix'][a][b]['roc_auc'], 2)})"
                           for b in MAIN_DATASETS] for a in MAIN_DATASETS]) + "\n")
    r = c["rowlevel"]
    L.append("### Generalisation matrix: row-level, harmonised flow features (X-IIoTID and TON_IoT-Network)\n")
    L.append("LightGBM on the 10 shared flow features: " + ", ".join(f"`{k}`" for k in r["features"]) + ".\n")
    names = list(r["matrix"])
    L.append(table(["Train \\ Test"] + names,
                   [[a] + [f"{r['matrix'][a][b]['f1']:.3f} (AUC {f(r['matrix'][a][b]['roc_auc'], 2)})" for b in names]
                    for a in names] + [["Merged (both)"] + [f"{r['merge'][b]['f1']:.3f} (AUC {f(r['merge'][b]['roc_auc'], 2)})"
                                                            for b in names]]) + "\n")
    L.append("Service vocabularies differ (" + "; ".join(f"{k}: {', '.join(v)}" for k, v in r["service_vocab"].items())
             + "), which limits how far the shared `service` feature actually aligns.\n")

    L.append("## 5. Merge experiments\n")
    L.append("Each group's training samples are concatenated (equal rows per dataset); the model is tested on each "
             "member's own held-out test sample and compared with that dataset's single-dataset model in the same "
             "representation (embedding diagonal) and with its best native-feature model from Section 2.\n")
    single = {d: c["embedding_matrix"][d][d] for d in MAIN_DATASETS}
    mp = plots.merge_bars(c["embedding_merges"], single, "Attack F1 per dataset: single vs merged training (embeddings)",
                          "merge_embedding_f1.png")
    L.append(img(mp, "merge experiment bars") + "\n")
    rows = []
    for g, v in c["embedding_merges"].items():
        for d, mt in v["test"].items():
            nat = max((res(d, "binary", m) for m in MODELS if res(d, "binary", m)), key=lambda x: x["test"]["f1"])
            rows.append([g, d, f(mt["f1"]), f(single[d]["f1"]), f"{mt['f1'] - single[d]['f1']:+.3f}",
                         f"{f(nat['test']['f1'])} ({nat['model']})"])
    L.append(table(["Merge group", "Tested on", "Merged F1 (emb)", "Single F1 (emb)", "Change", "Best native single F1"], rows) + "\n")
    L.append("Row-level merge X-IIoTID + TON_IoT-Network (harmonised flows): " + "; ".join(
        f"{b}: merged F1 {r['merge'][b]['f1']:.3f} vs single {r['matrix'][b][b]['f1']:.3f}" for b in names) + ".\n")


def section_coverage(L):
    L.append("## 5b. Label noise, imbalance and attack coverage (MITRE ATT&CK for ICS)\n")
    L.append("Technique mapping by the study author (nearest technique; 'loose' = IT-side attack with only an "
             "approximate ICS technique). Review before treating as ground truth.\n")
    def dup(n):
        m = meta(n)
        return f"{100 * m['duplicates_removed'] / m['rows_after_cleaning']:.1f}% feature-duplicates"
    noise = {
        "BATADAL": "-999 rows treated as normal; some dataset04 attacks were deliberately left unlabelled",
        "Edge-IIoTset": f"{dup('Edge-IIoTset')}; DDoS_UDP/MITM rows column-misaligned in source",
        "X-IIoTID": f"{dup('X-IIoTID')}; 2.7M '-' placeholders in flow fields (non-TCP/host-only records)",
        "TON_IoT-Network": f"{dup('TON_IoT-Network')}; protocol-specific fields mostly '-'",
        "TON_IoT-Modbus": f"{dup('TON_IoT-Modbus')}; 'time' formatting differs by label (dropped)",
    }
    for n in MAIN_DATASETS:
        noise[n] += f"; {meta(n)['conflicting_label_rows']:,} conflicting-label rows"
    rows = []
    for n in MAIN_DATASETS:
        m = meta(n)
        types = m["type_counts_after_dedup"]
        att = {k: v for k, v in types.items() if k.lower() not in ("normal",)}
        mp = MAP[n]
        tech = sorted({t for v in mp.values() for t in v[0]})
        rows.append([n, f"{100 * sum(att.values()) / sum(types.values()):.1f}%", len(att),
                     f"{min(att.values()):,} / {max(att.values()):,}" if att else "", noise[n],
                     ", ".join(f"{t} {T[t]}" for t in tech)])
    L.append(table(["Dataset", "Attack share (unique rows)", "Attack types", "Smallest / largest type",
                    "Label-quality notes", "ATT&CK for ICS techniques covered"], rows) + "\n")
    L.append("<details><summary>Per-label technique mapping</summary>\n")
    for n in MAIN_DATASETS:
        L.append(f"\n**{n}**\n")
        L.append(table(["Label", "Techniques", "Fit", "Note"],
                       [[k, ", ".join(f"{t} {T[t]}" for t in v[0]), v[1], v[2]] for k, v in MAP[n].items()]))
    L.append("\n</details>\n")


def section_outcome(L):
    """Section 8: the final merged model, from its saved model card (skipped if not trained yet)."""
    p = os.path.join(STUDY, "..", "models", "ics_ids_merged_v1", "model_card.json")
    if not os.path.exists(p):
        return
    c = load_json(p)
    s = c["webapp_prose_check"]
    L.append("## 8. Outcome: final merged model\n")
    L.append(f"The recommendation was accepted: **{', '.join(c['datasets'])}** were merged and BATADAL and "
             f"TON_IoT-Modbus left out. The final model `{c['name']}` v{c.get('version', '?')} is a {c['selected']} "
             f"classifier on {c['embedding_model']} embeddings of the event descriptions, the only representation in "
             "which the three datasets' different columns merge. Selection used the validation splits only; the test "
             "rows below were never used for tuning.\n")
    L.append(table(["Held-out test set", "Rows", "F1 (attack)", "ROC-AUC", "False-positive rate"],
                   [[k, f"{v['n_test']:,}", f(v["f1"]), f(v["roc_auc"]), f(v["fpr"])]
                    for k, v in c["test_per_dataset"].items()]) + "\n")
    L.append(f"On the web app's own handwritten sentences ({s['n_procedure_examples']} procedure examples and "
             f"{s['n_benign']} benign sentences) the model is near chance: it flags "
             f"{100 * s['procedure_examples_flagged_attack']:.1f}% of the attack examples and "
             f"{100 * s['benign_sentences_flagged_attack']:.1f}% of the benign ones (ROC-AUC {s['roc_auc']:.3f}). "
             "The web app therefore uses it only for structured events and keeps free-text analysis as an "
             "experimental, low-confidence mode.\n")


def build():
    L = ["# ICS-IDS dataset study: comparison and merge recommendation\n"]
    L.append("Generated by `dataset_study/report.py` from the saved run results. Reproduce with "
             "`python dataset_study/run_all.py`.\n")
    rec = os.path.join(RES, "recommendation.md")
    rec_txt = open(rec, encoding="utf-8").read() if os.path.exists(rec) else ""
    exec_summary, _, rest = rec_txt.partition("<!-- split -->")
    if exec_summary.strip():
        L.append(exec_summary.strip() + "\n")
    section_summary(L)
    section_metrics(L)
    section_sanity(L)
    section_zeroshot(L)
    section_cross(L)
    section_coverage(L)
    L.append(rest.strip() + "\n" if rest.strip() else "## 6. Recommendation\n\n(pending)\n")
    section_outcome(L)
    md = "\n".join(L)
    with open(os.path.join(RES, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    html = MarkdownIt("commonmark", {"html": True}).enable("table").render(md)

    def embed(m):
        p = os.path.join(RES, m.group(1))
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        return f'src="data:image/png;base64,{b64}"'
    html = re.sub(r'src="(plots/[^"]+)"', embed, html)
    css = """body{font-family:system-ui,Segoe UI,sans-serif;max-width:1180px;margin:0 auto;padding:24px 16px;
color:#0b0b0b;background:#fcfcfb;line-height:1.5}table{border-collapse:collapse;font-size:12.5px;margin:8px 0;
display:block;overflow-x:auto}th,td{border:1px solid #e4e3df;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f0efec}img{max-width:100%}code{background:#f0efec;padding:0 3px;border-radius:3px;font-size:90%}
h2{border-bottom:1px solid #e4e3df;padding-bottom:4px;margin-top:2em}details{margin:6px 0}"""
    with open(os.path.join(RES, "report.html"), "w", encoding="utf-8") as fh:
        fh.write(f"<!doctype html><html><head><meta charset='utf-8'><title>ICS-IDS dataset study</title>"
                 f"<style>{css}</style></head><body>{html}</body></html>")
    print("[report] ->", os.path.join(RES, "report.md"), "and report.html")


if __name__ == "__main__":
    build()
