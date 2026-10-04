"""Consistent, publication-oriented figures (English labels, colour-blind-safe palettes)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
})

# Okabe-Ito + extended tab20 for many categories
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def palette(n: int) -> list:
    if n <= len(OKABE_ITO):
        return OKABE_ITO[:n]
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def spatial_panels(coords: np.ndarray, values, sections=None, title: str = "", max_panels: int = 12,
                   categorical: bool | None = None, s: float = 2.0, cmap: str = "viridis"):
    """One 2D panel per section (or a single panel). Categorical values get a shared legend."""
    values = np.asarray(values)
    if categorical is None:
        categorical = values.dtype.kind in "OUSb" or pd.api.types.is_categorical_dtype(values)
    secs = pd.unique(np.asarray(sections).astype(str)) if sections is not None else np.array(["all"])
    secs = secs[:max_panels]
    n = len(secs)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
    if categorical:
        cats = pd.unique(values.astype(str))
        cats = sorted(cats)
        colors = dict(zip(cats, palette(len(cats))))
    for ax, sec in zip(axes.ravel(), secs):
        m = np.ones(len(values), bool) if sections is None else (np.asarray(sections).astype(str) == sec)
        if categorical:
            for c in cats:
                mm = m & (values.astype(str) == c)
                ax.scatter(coords[mm, 0], coords[mm, 1], s=s, color=colors[c], label=c, linewidths=0)
        else:
            sc = ax.scatter(coords[m, 0], coords[m, 1], c=values[m].astype(float), s=s, cmap=cmap, linewidths=0)
            fig.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_title(str(sec) if sections is not None else title)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if categorical:
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=4, frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def barh_counts(series: pd.Series, title: str, xlabel: str = "Observations", top: int = 25):
    s = series.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.25 * len(s))))
    ax.barh(s.index.astype(str), s.values, color=OKABE_ITO[4])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def heatmap(df: pd.DataFrame, title: str, cmap: str = "RdBu_r", center: float | None = 0.0, fmt_labels: bool = True):
    fig, ax = plt.subplots(figsize=(max(4, 0.4 * df.shape[1] + 2), max(3, 0.3 * df.shape[0] + 1.5)))
    v = df.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(v)) if center is not None else None
    im = ax.imshow(v, aspect="auto", cmap=cmap, vmin=-vmax if vmax else None, vmax=vmax)
    ax.set_xticks(range(df.shape[1]), df.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(range(df.shape[0]), df.index.astype(str))
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(title)
    fig.tight_layout()
    return fig
