# NeuroSTIAS

**Neuro**science **S**patial **T**ranscriptomics **I**ntegrated **A**nalysis **S**ystem: a plug-and-play
analysis system for neuroscience. It is built around 3D spatial transcriptomics and also covers
brain cell-type atlasing, brain organoids, electrophysiology, neuron and circuit modelling, neuron
morphology, and connectome analysis.

NeuroSTIAS grew out of 3D-STIAS V4.4.1, a general-purpose, Chinese-language, 12,000-line
interactive script. It has been rebuilt as an English Python package with:

- **Skills**: every analysis is a self-describing plug-in (`skill.yaml` + a function).
  Put a new skill folder in `skills/` and it is discovered automatically.
- **Provenance by default**: every run writes `provenance.json` with input checksums, git commit,
  parameters, seed, package versions and hardware.
- **Replicate-aware statistics**: animals, donors and organoids are the statistical unit, not spots.
- **Reproducible benchmark experiments** (`experiments/`) against established tools on public data.
- **An optional LLM assistant plugin** (`plugins/llm_assistant/`). It works with any local model
  (LM Studio, Ollama, llama.cpp) or, if you opt in, a cloud model. The core never needs it.

## Install

```bash
./install.sh          # core + analysis extras (ephys, modelling, morphology, imaging)
./install.sh full     # + benchmark baselines + LLM plugin
source .venv/bin/activate
neurostias doctor
```

## Use

```bash
neurostias skills list                         # what can it do?
neurostias skills show atlas.annotate          # inputs/outputs of one skill
neurostias run core.inspect input=my.h5ad      # run a skill -> runs/<skill>_<time>/
neurostias data list                           # declared datasets
neurostias experiment E01                      # rerun a benchmark experiment
neurostias report                              # rebuild paper tables/figures from results
neurostias menu                                # interactive menu
neurostias ask "Which layers are enriched for Sst interneurons?" --data my.h5ad   # optional plugin
```

## Layout

| Folder | Contents |
|---|---|
| `neurostias/core` | I/O conventions, provenance, skill registry and runner, replicate-aware statistics |
| `neurostias/spatial3d` | serial-section registration and 3D reconstruction, 3D graphs, domains, niches, communication |
| `neurostias/atlas` | curated brain marker panels, cell-type and cortical-layer annotation |
| `neurostias/organoid` | organoid fidelity and maturation against fetal references |
| `neurostias/ephys` | spike-train and LFP/EEG analysis |
| `neurostias/modeling` | Hodgkin–Huxley / LIF neurons, networks, virtual perturbations |
| `neurostias/morphology` | SWC morphology, Sholl analysis, morphometrics |
| `neurostias/imaging` | fMRI connectivity and graph metrics |
| `neurostias/legacy` | translated V4.4.1 modules (GRN, trajectory, stemness, ML, enrichment, ...) |
| `neurostias/skills`, `skills/` | built-in and drop-in skill manifests |
| `plugins/` | optional extras (LLM assistant) |
| `experiments/` | benchmark experiments, each with `run.py`, results and an auto-generated report |
| `data/` | `manifest.yaml` (sources, licences, checksums); data files themselves are not committed |
| `paper/` | landscape/gap analysis and auto-generated tables and figures |

## Data integrity

Results labelled `real` come from real measured data. Tests that apply a known perturbation to
real data (for example, a known rigid transform to check registration) are labelled
`semi_synthetic` everywhere they appear. Report numbers are generated from result files and are
never typed by hand.

## Benchmark experiments

| ID | Question | Data | Compared with |
|---|---|---|---|
| E01 | Cortical-layer (domain) recovery | DLPFC, 12 Visium sections, 3 donors | Scanpy Leiden, k-means, BANKSY, SpaGCN, GraphST, V4.4.1 |
| E02 | Do 3D cross-section graphs beat 2D graphs? | DLPFC adjacent pairs, GSE214349 (5 mice), MERFISH (12 sections) | per-section 2D graphs incl. degree-matched 2D; z-weight sweep |
| E03 | Registration accuracy (semi-synthetic) | real DLPFC / GSE214349 sections with known transforms | PASTE, centroid alignment |
| E04 | Brain cell-type and layer annotation | MERFISH (author classes), DLPFC (manual layers) | V4.4.1 panels, argmax, CellTypist |
| E05 | False positives from pseudoreplication | DLPFC null section splits; GSE214349 design audit | spot-level Wilcoxon / t-test (V4.4.1) |
| E06 | Neurotransmitter-aware communication | MERFISH, DLPFC | generic V4.4.1 database, LIANA |
| E07 | Organoid maturation and regional identity | HNOCA time-course subset | V4.4.1 stemness, genes-detected (CytoTRACE principle) |
| E08 | Spike-train / EEG validation | DANDI 000006, EEGBCI | Elephant, MNE CSP+LDA |
| E09 | Simulator correctness | HH, LIF, Brunel network | Brian2, closed form |
| E10 | Morphometrics and cell-class classification | NeuroMorpho (Allen Cell Types) | NeuroM |
| E11 | Functional connectome | nilearn development fMRI (155 subjects) | nilearn reference pipeline, motion-only baseline, motion-regressed connectome |
| E12 | (optional, not run) LLM assistant routing and grounding | 154 pre-registered questions | local Qwen 9B / 27B |

Each experiment folder holds `run.py`, `results/` (tables, figures, `summary.json`, `provenance.json`) and
an auto-generated `REPORT.md`. `neurostias report` gathers them into `paper/`.
