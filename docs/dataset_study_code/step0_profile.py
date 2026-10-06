"""Step 0: profile every raw dataset file (read-only, chunked).

Usage: python -I dataset_study/step0_profile.py <project_root>
Writes dataset_study/results/step0_profile.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(ROOT, "dataset_study", "results", "step0_profile.json")
CHUNK = 250_000
TON = "data/extracted/ton_iot/Train_Test_datasets"

# (family, part name, relative path, binary label col, attack-type col(s), kind)
FILES = [
    ("BATADAL", "dataset03", "data/raw/BATADAL_dataset03.csv", "ATT_FLAG", [], "process/sensor telemetry"),
    ("BATADAL", "dataset04", "data/raw/BATADAL_dataset04.csv", "ATT_FLAG", [], "process/sensor telemetry"),
    ("Edge-IIoTset", "DNN-EdgeIIoT", "data/extracted/edge_iiotset/DNN-EdgeIIoT-dataset.csv",
     "Attack_label", ["Attack_type"], "network traffic (packet-level, incl. Modbus/MQTT fields)"),
    ("X-IIoTID", "X-IIoTID", "data/extracted/x_iiotid/X-IIoTID dataset.csv",
     "class3", ["class2", "class1"], "network flow + host telemetry"),
]
for f in ["Fridge", "Garage_Door", "GPS_Tracker", "Modbus", "Motion_Light", "Thermostat", "Weather"]:
    FILES.append(("TON_IoT", f"IoT_{f}", f"{TON}/Train_Test_IoT_dataset/Train_Test_IoT_{f}.csv",
                  "label", ["type"], "IoT/IIoT device telemetry" + (" (Modbus register values)" if f == "Modbus" else "")))
FILES += [
    ("TON_IoT", "Network", f"{TON}/Train_Test_Network_dataset/train_test_network.csv", "label", ["type"], "network traffic (Zeek flows)"),
    ("TON_IoT", "Linux_disk", f"{TON}/Train_Test_Linux_dataset/Train_test_linux_disk.csv", "attack", ["type"], "host telemetry (Linux)"),
    ("TON_IoT", "Linux_memory", f"{TON}/Train_Test_Linux_dataset/Train_test_linux_memory.csv", "label", ["type"], "host telemetry (Linux)"),
    ("TON_IoT", "Linux_process", f"{TON}/Train_Test_Linux_dataset/Train_Test_Linux_process.csv", "label", ["type"], "host telemetry (Linux)"),
    ("TON_IoT", "Windows_10", f"{TON}/Train_Test_Windows_dataset/Train_Test_Windows_10.csv", "label", ["type"], "host telemetry (Windows)"),
    ("TON_IoT", "Windows_7", f"{TON}/Train_Test_Windows_dataset/Train_Test_Windows_7.csv", "label", ["type"], "host telemetry (Windows)"),
]

PLACEHOLDERS = {"-", "", " "}


def profile(family, part, rel, label, types, kind):
    path = os.path.join(ROOT, rel)
    hashes, n, cols = [], 0, None
    missing = None
    placeholder = None
    inf_count = 0
    label_counts, type_counts = {}, {t: {} for t in types}
    first_ts = last_ts = None
    for chunk in pd.read_csv(path, chunksize=CHUNK, dtype=str, keep_default_na=False,
                             encoding="utf-8-sig", low_memory=False):
        chunk.columns = [c.strip() for c in chunk.columns]
        if cols is None:
            cols = list(chunk.columns)
            missing = pd.Series(0, index=cols)
            placeholder = pd.Series(0, index=cols)
            first_ts = chunk.iloc[0, 0]
        stripped = chunk.apply(lambda s: s.str.strip())
        lower = stripped.apply(lambda s: s.str.lower())
        missing += lower.isin({"", "nan", "null", "none", "na"}).sum()
        placeholder += stripped.isin({"-"}).sum()
        inf_count += int(lower.isin({"inf", "-inf", "infinity", "-infinity"}).sum().sum())
        for k, v in stripped[label].value_counts().items():
            label_counts[k] = label_counts.get(k, 0) + int(v)
        for t in types:
            for k, v in stripped[t].value_counts().items():
                type_counts[t][k] = type_counts[t].get(k, 0) + int(v)
        hashes.append(pd.util.hash_pandas_object(stripped, index=False).to_numpy())
        n += len(chunk)
        last_ts = chunk.iloc[-1, 0]
    h = np.concatenate(hashes)
    dup = int(n - np.unique(h).size)
    return {
        "family": family, "part": part, "path": rel, "format": "CSV",
        "size_mb": round(os.path.getsize(path) / 1e6, 2),
        "rows": n, "columns": len(cols), "column_names": cols,
        "label_column": label, "attack_type_columns": types,
        "label_distribution": label_counts,
        "attack_type_distribution": type_counts,
        "missing_total": int(missing.sum()),
        "missing_by_column": {k: int(v) for k, v in missing.items() if v},
        "dash_placeholder_total": int(placeholder.sum()),
        "dash_placeholder_by_column": {k: int(v) for k, v in placeholder.items() if v},
        "inf_values": inf_count,
        "duplicate_rows_exact": dup,
        "duplicate_pct": round(100 * dup / n, 2),
        "first_col_first_value": first_ts, "first_col_last_value": last_ts,
        "data_kind": kind,
    }


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for spec in FILES:
        key = f"{spec[0]}/{spec[1]}"
        if key in results:
            continue
        print("profiling", key, flush=True)
        results[key] = profile(*spec)
        with open(OUT, "w") as f:  # save after each file
            json.dump(results, f, indent=1)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
