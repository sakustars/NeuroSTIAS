"""Data loading with consistent conventions across modalities.

Spatial conventions (shared by every spatial skill):

* ``obsm['spatial']``     2D coordinates (pixels or µm, per dataset)
* ``obsm['spatial_3d']``  3D coordinates in µm after reconstruction
* ``obs['section']``      section/slice identifier (string)
* ``obs['replicate']``    biological unit (animal, donor, organoid) used for statistics
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HUMAN_HINTS = ("GAPDH", "ACTB", "SNAP25", "MBP", "GFAP", "MT-CO1")
MOUSE_HINTS = ("Gapdh", "Actb", "Snap25", "Mbp", "Gfap", "mt-Co1")


def read_h5ad(path: str | Path, backed: bool = False):
    import anndata as ad
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return ad.read_h5ad(path, backed="r" if backed else None)


def detect_species(var_names) -> str:
    names = set(map(str, var_names))
    h = sum(g in names for g in HUMAN_HINTS)
    m = sum(g in names for g in MOUSE_HINTS)
    if h == m == 0:
        upper = np.mean([str(g).isupper() for g in list(names)[:2000]]) if names else 0
        return "human" if upper > 0.8 else "mouse"
    return "human" if h >= m else "mouse"


def gene_symbols(adata) -> pd.Index:
    """Return gene symbols, preferring a symbol column when var_names are Ensembl IDs."""
    for col in ("gene_symbol", "gene_symbols", "feature_name", "gene_name", "symbol"):
        if col in adata.var.columns:
            return pd.Index(adata.var[col].astype(str))
    return pd.Index(adata.var_names.astype(str))


def use_symbols(adata):
    """Set var_names to gene symbols (made unique) in place and return adata."""
    syms = gene_symbols(adata)
    if not syms.equals(pd.Index(adata.var_names.astype(str))):
        adata.var["original_id"] = adata.var_names.astype(str)
        adata.var_names = syms
    adata.var_names_make_unique()
    return adata


def coords(adata, prefer_3d: bool = True) -> np.ndarray:
    keys = ("spatial_3d", "spatial") if prefer_3d else ("spatial", "spatial_3d")
    for k in keys:
        if k in adata.obsm:
            return np.asarray(adata.obsm[k], dtype=float)
    raise KeyError("No spatial coordinates in obsm ('spatial_3d' or 'spatial').")


def counts_matrix(adata):
    """Raw counts if available (layers['counts'] or .raw), else X."""
    if "counts" in adata.layers:
        return adata.layers["counts"]
    if adata.raw is not None and adata.raw.shape == adata.shape:
        return adata.raw.X
    return adata.X


def summarize(adata) -> dict:
    out = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns": list(map(str, adata.obs.columns)),
        "obsm_keys": list(map(str, adata.obsm.keys())),
        "layers": list(map(str, adata.layers.keys())),
        "species_guess": detect_species(gene_symbols(adata)),
    }
    for col in ("section", "slice_id", "sample_id", "replicate", "donor", "cell_type", "layer_guess"):
        if col in adata.obs.columns:
            vc = adata.obs[col].astype(str).value_counts()
            out[f"{col}_counts"] = {str(k): int(v) for k, v in vc.head(50).items()}
    for k in ("spatial", "spatial_3d"):
        if k in adata.obsm:
            c = np.asarray(adata.obsm[k])
            out[f"{k}_dims"] = int(c.shape[1])
            out[f"{k}_range"] = [[float(c[:, i].min()), float(c[:, i].max())] for i in range(c.shape[1])]
    return out


# ---------------------------------------------------------------------------
# SO3D CSV triplet (expression, coordinates, metadata) - the 3D-STIAS V4.4.1 input
# format. Ported from ``validate_so3d_triplet`` / ``_create_anndata_from_spatial``:
# the three files are an atomic input and are aligned strictly by ID, never by
# row order.
# ---------------------------------------------------------------------------

ID_COLUMNS = {"cell_id", "spot_id", "barcode", "cell", "spot", "unnamed: 0"}


def _read_table(path, nrows=None):
    p = str(path).lower()
    sep = "\t" if p.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")) else ","
    return pd.read_csv(path, sep=sep, nrows=nrows)


def _id_col(frame):
    return next((c for c in frame.columns if str(c).strip().lower() in ID_COLUMNS), None)


def validate_triplet(expression_file, coordinates_file, metadata_file) -> tuple[bool, list[str]]:
    paths = [expression_file, coordinates_file, metadata_file]
    errors = []
    if any(not p for p in paths):
        errors.append("expression, coordinates and metadata files must all be provided")
    for label, p in zip(("expression", "coordinates", "metadata"), paths):
        if p and not Path(p).is_file():
            errors.append(f"{label} file not found: {p}")
    if len({str(Path(p).resolve()) for p in paths if p}) != 3:
        errors.append("the three inputs must be three different files")
    if errors:
        return False, errors
    try:
        expr = _read_table(expression_file, nrows=5)
        crd = _read_table(coordinates_file, nrows=1000)
        meta = _read_table(metadata_file, nrows=1000)
    except Exception as exc:  # noqa: BLE001
        return False, [f"cannot read files as CSV/TSV: {exc}"]
    lower = {str(c).strip().lower(): c for c in crd.columns}
    missing = [a for a in ("x", "y", "z") if a not in lower]
    if missing:
        errors.append(f"coordinates file lacks column(s): {', '.join(missing)}")
    cid, mid = _id_col(crd), _id_col(meta)
    if cid is None:
        errors.append("coordinates file lacks a cell_id/spot_id/barcode column")
    if mid is None:
        errors.append("metadata file lacks a cell_id/spot_id/barcode column")
    if expr.shape[1] < 2:
        errors.append("expression matrix must have genes in the first column and cells/spots in later columns")
    if not missing:
        xyz = crd[[lower["x"], lower["y"], lower["z"]]].apply(pd.to_numeric, errors="coerce")
        if xyz.isna().any().any():
            errors.append("x/y/z coordinates must all be finite numbers")
    if cid is not None and crd[cid].astype(str).duplicated().any():
        errors.append("duplicate IDs in coordinates file")
    if mid is not None and meta[mid].astype(str).duplicated().any():
        errors.append("duplicate IDs in metadata file")
    return not errors, errors


def load_triplet(expression_file, coordinates_file, metadata_file):
    """Load the three files into AnnData aligned by ID (raises on any inconsistency)."""
    import anndata as ad
    ok, errors = validate_triplet(expression_file, coordinates_file, metadata_file)
    if not ok:
        raise ValueError("; ".join(errors))
    expr = _read_table(expression_file)
    crd = _read_table(coordinates_file)
    meta = _read_table(metadata_file)
    gene_col = expr.columns[0]
    cid, mid = _id_col(crd), _id_col(meta)
    lower = {str(c).strip().lower(): c for c in crd.columns}
    crd = crd.set_index(crd[cid].astype(str))
    meta = meta.set_index(meta[mid].astype(str))
    cells = [c for c in map(str, expr.columns[1:]) if c in crd.index and c in meta.index]
    if len(cells) < 2:
        raise ValueError("fewer than 2 IDs shared by all three files")
    X = expr.set_index(expr[gene_col].astype(str))[cells].apply(pd.to_numeric, errors="raise").to_numpy().T
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.obs_names = cells
    adata.var_names = expr[gene_col].astype(str).to_numpy()
    obs = meta.loc[cells].drop(columns=[mid])
    for c in obs.columns:
        adata.obs[str(c)] = obs[c].to_numpy() if pd.api.types.is_numeric_dtype(obs[c]) else obs[c].astype(str).to_numpy()
    adata.obsm["spatial_3d"] = crd.loc[cells, [lower["x"], lower["y"], lower["z"]]].to_numpy(float)
    adata.obsm["spatial"] = adata.obsm["spatial_3d"][:, :2]
    adata.uns["source_files"] = {"expression": str(expression_file), "coordinates": str(coordinates_file),
                                 "metadata": str(metadata_file)}
    adata.var_names_make_unique()
    return adata
