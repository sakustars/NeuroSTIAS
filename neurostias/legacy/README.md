# Legacy: 3D-STIAS V4.4.1 port

3D-STIAS V4.4.1 (original authors: Haolan Li and Haolin Wu) was a single interactive,
Chinese-language script (12,026 lines). This package ports its analyses to English,
non-interactive library functions. Each function's docstring states what V4.4.1 did and
what changed.

| V4.4.1 module | Port | Change for neuroscience |
|---|---|---|
| preprocessing | `core/preprocess.py` | none (same defaults); only real integer counts are labelled `counts` |
| cell annotation (generic immune/skin/brain panels) | `atlas/annotate.py` | brain-only hierarchical panels; enrichment test vs expression-matched controls; Unresolved when unsupported |
| reference mapping (NNLS) | `legacy/misc.py: deconvolve_nnls` | brain class basis |
| spatial variability (Moran's I) | `spatial3d/svg.py` | vectorised; anisotropic z |
| spatial niches | `spatial3d/niches.py` | anisotropic z |
| spatial GNN | `legacy/misc.py: gcn_embedding` | none |
| metabolic activity | `legacy/misc.py: metabolic_scores` | adds lactate shuttle, glutamate-glutamine cycle, myelin cholesterol synthesis |
| community (k-means on coordinates) | `legacy/community.py` | kept for comparison; superseded by `spatial3d/domains.py` |
| communication (40 generic LR pairs) | `spatial3d/communication.py` | neurotransmitter/neuropeptide database; replicate consistency |
| GRN | `legacy/grn.py` | neural TF list by default |
| stemness | `legacy/development.py` | adds neurodevelopmental stage scoring |
| trajectory (DPT) | `legacy/trajectory.py` | biological root via neurodevelopmental labels |
| machine learning (RF) | `legacy/ml.py` | none |
| differential expression | `legacy/de.py` | adds pseudobulk replicate-aware DE as default |
| enrichment (offline ORA/GSEA) | `legacy/enrichment.py` | neuroscience gene-set library |
| virtual intervention | `legacy/misc.py: virtual_intervention` | none (heuristic, exploratory) |
| colocalisation | `legacy/misc.py: colocalization_candidates` | none |
| interactive menu / installer / PDF | `menu.py`, `install.sh`, `pipeline.py` | English; skill-based |

Regression against V4.4.1 was performed on a privately shared dataset and is not included in the public release.

Source files (not committed; SHA-256 for provenance):

- `三维空间转录组交互式分析系统V4.4.1_对外(3).py`: `cb3c666eb8ddfe05e559782579bc7b2cc36bb4f04afb514868d34d90cd21dc29`
- `program_snapshot_at_run_start.py` (used by the archived 2026-08-28 run): `0b1491a678a9843ab18994ed62545ca06a7de1db30181bf7a058af89012b85ef`
