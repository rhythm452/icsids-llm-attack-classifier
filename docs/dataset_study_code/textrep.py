"""Shared natural-language representation of rows (for embeddings / zero-shot)."""
import re

import numpy as np
import pandas as pd

from loaders import SPECS

MAX_FIELDS = 20
BATADAL_GLOSSARY = [(r"^L_T(\d+)$", r"tank \1 water level"), (r"^F_PU(\d+)$", r"pump \1 flow"),
                    (r"^S_PU(\d+)$", r"pump \1 status"), (r"^F_V(\d+)$", r"valve \1 flow"),
                    (r"^S_V(\d+)$", r"valve \1 status"), (r"^P_J(\d+)$", r"junction \1 pressure")]


def humanize(col):
    for pat, rep in BATADAL_GLOSSARY:
        if re.match(pat, col):
            return re.sub(pat, rep, col)
    c = re.sub(r"[._/]+", " ", col).replace("Scr", "source").replace("Des", "destination")
    c = re.sub(r"\bsrc\b", "source", c)
    c = re.sub(r"\bdst\b", "destination", c)
    return re.sub(r"\s+", " ", c).strip().lower()


def fmt(v):
    if abs(v) >= 1e4 or (0 < abs(v) < 1e-3):
        return f"{v:.3g}"
    return f"{v:.4g}"


def describe(df, name, num, cat):
    """One short event description per row: dataset kind + non-default fields."""
    prefix = SPECS[name]["text_prefix"]
    names = {c: humanize(c) for c in num + cat}
    numv = df[num].to_numpy(dtype=float) if num else np.zeros((len(df), 0))
    catv = df[cat].to_numpy(dtype=object) if cat else np.zeros((len(df), 0), dtype=object)
    # strip scheme + host (an IP in these testbeds) so no identifier enters the text
    http = (df["__txt_http"].str.replace(r"[a-zA-Z]+://[^/\s]+", "", regex=True).str.strip().to_numpy()
            if "__txt_http" in df else None)
    out = []
    for i in range(len(df)):
        fields = [f"http request {http[i]}"] if http is not None and http[i] else []
        for j, c in enumerate(cat):
            v = catv[i, j]
            if v not in ("na", "0", "", None):
                fields.append(f"{names[c]} {v}")
        for j, c in enumerate(num):
            v = numv[i, j]
            if not np.isnan(v) and v != 0:
                fields.append(f"{names[c]} {fmt(v)}")
        out.append(f"{prefix}: " + (", ".join(fields[:MAX_FIELDS]) if fields else "all fields zero") + ".")
    return out


def stratified_sample(df, n, seed):
    if len(df) <= n:
        return df.reset_index(drop=True)
    frac = n / len(df)
    rng = np.random.default_rng(seed)
    idx = []
    for _, g in df.groupby("__type"):
        k = max(1, int(round(len(g) * frac)))
        idx.extend(rng.choice(g.index.to_numpy(), size=min(k, len(g)), replace=False))
    return df.loc[sorted(idx)].reset_index(drop=True)
