"""Offline enrichment: hypergeometric ORA and weighted pre-ranked GSEA (ported from V4.4.1).

No external API is called. Gene-set libraries:

* ``neuro_v1``: curated neuroscience programmes (all marker panels in
  ``atlas/markers/brain_markers.yaml`` plus synaptic, myelination,
  blood–brain-barrier, ion-channel and neuroinflammation sets below);
* ``local_v441``: the 14 exploratory panels of V4.4.1;
* any ``.gmt`` file supplied by the user (e.g. downloaded GO/Reactome releases).

Names describe biological themes; these are not complete GO/KEGG terms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..atlas import markers
from ..core.stats import bh

LOCAL_V441 = {
    'Neuronal identity': {'RBFOX3', 'TUBB3', 'MAP2', 'NEFL', 'NEFM', 'STMN2', 'ELAVL3', 'ELAVL4'},
    'Oligodendrocyte differentiation': {'MBP', 'PLP1', 'MOG', 'MAG', 'MOBP', 'CNP', 'CLDN11', 'OLIG1', 'OLIG2', 'SOX10'},
    'Astrocyte homeostasis': {'GFAP', 'AQP4', 'ALDH1L1', 'SLC1A2', 'SLC1A3', 'GLUL', 'SOX9'},
    'Microglial immune program': {'C1QA', 'C1QB', 'C1QC', 'TYROBP', 'AIF1', 'LST1', 'CTSS', 'FCER1G'},
    'Vascular endothelium': {'PECAM1', 'VWF', 'KDR', 'EMCN', 'CLDN5', 'RAMP2', 'EGFL7'},
    'Extracellular matrix': {'COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'COL4A2', 'DCN', 'LUM', 'FN1'},
    'Immediate early response': {'FOS', 'FOSB', 'JUN', 'JUNB', 'EGR1', 'EGR2', 'ATF3', 'DUSP1', 'IER2'},
    'Interferon response': {'STAT1', 'STAT2', 'IRF7', 'ISG15', 'IFIT1', 'IFIT2', 'IFIT3', 'MX1', 'OAS1'},
    'Hypoxia response': {'HIF1A', 'VEGFA', 'CA9', 'SLC2A1', 'LDHA', 'PDK1', 'BNIP3', 'NDRG1'},
    'Cell cycle': {'MKI67', 'TOP2A', 'PCNA', 'CDK1', 'CCNB1', 'CCNB2', 'UBE2C', 'MCM2', 'MCM5'},
    'Apoptosis and stress': {'BAX', 'BAK1', 'BCL2', 'CASP3', 'CASP8', 'TP53', 'DDIT3', 'HSPA1A'},
    'TGF beta fibroblast response': {'TGFB1', 'TGFBR1', 'TGFBR2', 'SMAD2', 'SMAD3', 'TGFBI', 'FN1', 'COL1A1'},
    'Pericyte vascular support': {'RGS5', 'CSPG4', 'MCAM', 'PDGFRB', 'NOTCH3', 'KCNJ8', 'ABCC9'},
    'Myofibroblast activation': {'ACTA2', 'TAGLN', 'DES', 'TGFBI', 'FN1', 'COL1A1', 'COL3A1'},
}

NEURO_EXTRA = {
    'Presynaptic vesicle cycle': {'SYN1', 'SYN2', 'SYP', 'SYT1', 'SNAP25', 'STX1A', 'VAMP2', 'RAB3A', 'CPLX1', 'CPLX2',
                                  'NSF', 'STXBP1', 'UNC13A', 'SV2A', 'SYNGR1'},
    'Postsynaptic density': {'DLG4', 'SHANK1', 'SHANK2', 'SHANK3', 'HOMER1', 'GRIN1', 'GRIN2B', 'GRIA1', 'GRIA2',
                             'CAMK2A', 'DLGAP1', 'NLGN1', 'SYNGAP1', 'ARC'},
    'Myelination': {'MBP', 'PLP1', 'MAG', 'MOG', 'MOBP', 'MYRF', 'CNP', 'CLDN11', 'UGT8', 'FA2H', 'GAL3ST1', 'ERMN'},
    'Blood-brain barrier': {'CLDN5', 'OCLN', 'TJP1', 'SLC2A1', 'ABCB1', 'MFSD2A', 'LSR', 'ABCG2', 'SLC7A5'},
    'Voltage-gated sodium channels': {'SCN1A', 'SCN2A', 'SCN3A', 'SCN8A', 'SCN1B', 'SCN2B', 'SCN4B'},
    'Voltage-gated potassium channels': {'KCNA1', 'KCNA2', 'KCNB1', 'KCNC1', 'KCNC2', 'KCND2', 'KCNQ2', 'KCNQ3', 'KCNH7'},
    'Voltage-gated calcium channels': {'CACNA1A', 'CACNA1B', 'CACNA1C', 'CACNA1D', 'CACNA1E', 'CACNA1G', 'CACNB2'},
    'Complement / synaptic pruning': {'C1QA', 'C1QB', 'C1QC', 'C3', 'C4A', 'C4B', 'ITGAM', 'ITGB2', 'CD47', 'SIRPA'},
    'Neuroinflammation (cytokines)': {'IL1B', 'TNF', 'IL6', 'CCL2', 'CXCL10', 'NFKBIA', 'PTGS2', 'NLRP3', 'CASP1'},
    'Axon initial segment / node': {'ANK3', 'SPTBN4', 'NFASC', 'CNTNAP1', 'SCN8A', 'KCNQ2', 'CNTN1'},
    'Adult neurogenesis': {'DCX', 'SOX2', 'NES', 'ASCL1', 'EOMES', 'NEUROD1', 'PROX1', 'CALB2', 'TBR1'},
    'Heme / iron handling': {'HMOX1', 'FTL', 'FTH1', 'TFRC', 'SLC40A1', 'CD163', 'HPX', 'HP', 'LCN2'},
}


def neuro_library(species: str = "human") -> dict[str, set[str]]:
    lib: dict[str, set[str]] = {}
    for section in markers.SECTIONS:
        for name, genes in markers.panels(section, species).items():
            lib[f"{section}: {name}"] = set(genes)
    conv = (lambda g: g) if species == "human" else (lambda g: g[:1].upper() + g[1:].lower())
    for name, genes in NEURO_EXTRA.items():
        lib[name] = {conv(g) for g in genes}
    return lib


def read_gmt(path: str | Path) -> dict[str, set[str]]:
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 2:
            out[parts[0]] = set(parts[2:])
    return out


def get_library(name: str = "neuro_v1", species: str = "human") -> dict[str, set[str]]:
    if name == "neuro_v1":
        return neuro_library(species)
    if name == "local_v441":
        return LOCAL_V441
    return read_gmt(name)


def ora(genes, universe, library: dict[str, set[str]], min_size: int = 3) -> pd.DataFrame:
    genes = {g.upper() for g in genes}
    uni = {g.upper() for g in universe}
    genes &= uni
    rows = []
    for name, members in library.items():
        m = {g.upper() for g in members} & uni
        if len(m) < min_size:
            continue
        k = len(genes & m)
        p = stats.hypergeom.sf(k - 1, len(uni), len(m), len(genes)) if k > 0 else 1.0
        rows.append({"term": name, "set_size": len(m), "overlap": k, "query_size": len(genes),
                     "overlap_genes": ";".join(sorted(genes & m)), "pval": float(p)})
    df = pd.DataFrame(rows)
    if len(df):
        df["padj"] = bh(df["pval"])
        df = df.sort_values("pval").reset_index(drop=True)
    return df


def gsea_preranked(scores: pd.Series, library: dict[str, set[str]], min_size: int = 5, max_size: int = 500,
                   n_permutations: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Weighted (p=1) running-sum enrichment with gene-label permutation null."""
    s = scores.dropna()
    s.index = s.index.astype(str).str.upper()
    s = s[~s.index.duplicated()].sort_values(ascending=False)
    genes = s.index.to_numpy()
    w = np.abs(s.to_numpy())
    rng = np.random.default_rng(seed)

    def es(mask):
        hit = np.where(mask, w, 0.0)
        nh = hit.sum()
        if nh == 0:
            return 0.0
        miss = (~mask).astype(float) / max((~mask).sum(), 1)
        run = np.cumsum(hit / nh - miss)
        return float(run[np.argmax(np.abs(run))])

    rows = []
    for name, members in library.items():
        mask = np.isin(genes, [g.upper() for g in members])
        if not (min_size <= mask.sum() <= max_size):
            continue
        obs = es(mask)
        null = np.array([es(rng.permutation(mask)) for _ in range(n_permutations)])
        same = null[np.sign(null) == np.sign(obs)] if obs != 0 else null
        p = (1 + np.sum(np.abs(same) >= abs(obs))) / (len(same) + 1)
        nes = obs / np.mean(np.abs(same)) if len(same) and np.mean(np.abs(same)) > 0 else 0.0
        rows.append({"term": name, "set_size": int(mask.sum()), "es": obs, "nes": float(nes), "pval": float(p)})
    df = pd.DataFrame(rows)
    if len(df):
        df["padj"] = bh(df["pval"])
        df = df.sort_values("pval").reset_index(drop=True)
    return df
