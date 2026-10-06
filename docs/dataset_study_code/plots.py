"""Step 2/4 figures: confusion matrices, ROC/PR curves, cross-dataset heatmaps, merge bars."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from common import MAIN_DATASETS, PLOTS, RES, slug  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"LogReg": "#2a78d6", "RandomForest": "#eb6834", "LightGBM": "#1baf7a", "MLP": "#eda100"}
BLUES = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    p = os.path.join(PLOTS, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def _heat(ax, M, xl, yl, fmt="{:d}", vmax=None):
    ax.grid(False)
    im = ax.imshow(M, cmap=BLUES, vmin=0, vmax=vmax if vmax is not None else max(M.max(), 1e-9))
    ax.set_xticks(range(len(xl)), xl, rotation=30, ha="right")
    ax.set_yticks(range(len(yl)), yl)
    hi = (vmax if vmax is not None else M.max()) * 0.55
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > hi else INK)
    return im


def per_dataset(name, models=("LogReg", "RandomForest", "LightGBM", "MLP")):
    rd = os.path.join(RES, slug(name))
    avail = [m for m in models if os.path.exists(os.path.join(rd, f"probs_binary_{m}.npz"))]
    if not avail:
        return []
    out = []
    # confusion matrices
    fig, axes = plt.subplots(1, len(avail), figsize=(2.6 * len(avail), 2.6))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, avail):
        d = np.load(os.path.join(rd, f"probs_binary_{m}.npz"))
        pred = (d["prob"][:, 1] >= 0.5).astype(int)
        cm = np.array([[np.sum((d["y"] == a) & (pred == b)) for b in (0, 1)] for a in (0, 1)])
        _heat(ax, cm, ["pred normal", "pred attack"], ["normal", "attack"])
        ax.set_title(m)
    fig.suptitle(f"{name}: test confusion matrices (binary)", x=0.01, ha="left", fontsize=11)
    out.append(_save(fig, f"{slug(name)}_confusion.png"))
    # ROC + PR
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for m in avail:
        d = np.load(os.path.join(rd, f"probs_binary_{m}.npz"))
        y, p = d["y"], d["prob"][:, 1]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        a1.plot(fpr, tpr, color=SERIES[m], label=m)
        a2.plot(rc, pr, color=SERIES[m], label=m)
    a1.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    a1.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC", xlim=(0, 1), ylim=(0, 1.01))
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall", xlim=(0, 1), ylim=(0, 1.01))
    a2.legend(loc="lower left")
    fig.suptitle(f"{name}: test ROC and PR curves", x=0.01, ha="left", fontsize=11)
    out.append(_save(fig, f"{slug(name)}_roc_pr.png"))
    return out


def cross_heatmap(matrix, title, fname, metric="f1"):
    names = list(matrix)
    cols = list(next(iter(matrix.values())))
    M = np.array([[matrix[a][b][metric] if matrix[a][b][metric] is not None else np.nan for b in cols]
                  for a in names], dtype=float)
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 2, 1.0 * len(names) + 1.4))
    _heat(ax, M, [f"test: {c}" for c in cols], [f"train: {a}" for a in names], fmt="{:.2f}", vmax=1.0)
    ax.set_title(title, loc="left")
    return _save(fig, fname)


def merge_bars(merges, single, title, fname):
    """Grouped bars: single-dataset vs each merge group that contains the dataset."""
    groups = list(merges)
    ds = [d for d in MAIN_DATASETS if any(d in merges[g]["members"] for g in groups)]
    series = ["single"] + groups
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.8 / len(series)
    for i, s in enumerate(series):
        xs, vs = [], []
        for j, d in enumerate(ds):
            v = single[d]["f1"] if s == "single" else merges[s]["test"].get(d, {}).get("f1")
            if v is not None:
                xs.append(j + (i - (len(series) - 1) / 2) * w)
                vs.append(v)
        ax.bar(xs, vs, width=w * 0.9, color=colors[i % len(colors)], label=s if s != "single" else "single dataset")
    ax.set_xticks(range(len(ds)), ds)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Attack-class F1 (test)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    ax.set_title(title, loc="left")
    return _save(fig, fname)
