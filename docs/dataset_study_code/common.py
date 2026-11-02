"""Shared paths, constants and small helpers for the dataset study."""
import json
import os

import numpy as np

SEED = 42
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STUDY = os.path.join(ROOT, "dataset_study")
RES = os.path.join(STUDY, "results")
CACHE = os.path.join(STUDY, "cache")
PLOTS = os.path.join(RES, "plots")

# Sampling policy (identical for every dataset): after de-duplication, cap at
# SAMPLE_CAP rows, stratified by the finest attack-type label, keeping at least
# MIN_PER_CLASS rows of every class (or all of them if the class is smaller).
SAMPLE_CAP = 250_000
MIN_PER_CLASS = 1_000
SPLIT = (0.70, 0.15, 0.15)

# Text-embedding experiments: per-dataset sample sizes (stratified by type).
EMB_TRAIN_N = 5_000
EMB_TEST_N = 2_000

MAIN_DATASETS = ["BATADAL", "Edge-IIoTset", "X-IIoTID", "TON_IoT-Network", "TON_IoT-Modbus"]
APPENDIX_DATASETS = ["TON_IoT-IoT_Fridge", "TON_IoT-IoT_Garage_Door", "TON_IoT-IoT_GPS_Tracker",
                     "TON_IoT-IoT_Motion_Light", "TON_IoT-IoT_Thermostat", "TON_IoT-IoT_Weather"]


def path(*parts):
    p = os.path.join(*parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def save_json(obj, p):
    tmp = path(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_default)
    os.replace(tmp, p)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def slug(name):
    return name.replace("/", "_")
