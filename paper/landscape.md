# Landscape and gap analysis

*Compiled 2026-10-04 from web searches during Phase 1. Every claim about another tool links to its
source. Before quoting anything here in the paper, check it against the primary publication.*

## 1. Existing solutions by area

| Area | Established tools | What they do well | Relevant limitation (for this project) |
|---|---|---|---|
| Spatial domains (2D) | Scanpy/Leiden, SpaGCN, STAGATE, GraphST, SEDR, BANKSY | DLPFC cortical layers are the standard benchmark (ARI/NMI against manual layers) [1,2]. BANKSY augments each cell with its neighbourhood transcriptome and scales to millions of cells [3]. | Benchmarks pool spots within sections; results are rarely reported per donor with confidence intervals. |
| 3D spatial / multi-section | PASTE, PASTE2 (alignment), STitch3D, Spateo, SPACEL, STAIR, STAGATE-3D, SpatialLeiden-3D [4,5,6,7] | PASTE2 handles partial overlap between slices [4]. STAGATE-3D combines within-section networks with networks between adjacent sections [5]. SpatialLeiden has a 3D guide [6]. | **3D graphs already exist**, so 3D support alone is not novel. Open questions: (i) inter-section edges usually use one radius on aligned coordinates, ignoring that section spacing (z) differs from in-plane spot spacing; (ii) little held-out-section evaluation of whether 3D context improves prediction over 2D; (iii) sections from the same animal are often treated as independent. |
| Neural cell–cell communication | CellChat, CellPhoneDB, LIANA, COMMOT; **NeuronChat** (neural-specific) [8] | NeuronChat models neurotransmitter signalling, including synthesis enzymes and vesicular transporters (373 ligand–target pairs, human and mouse) and can use spatial data [8]. | NeuronChat is R-based and centred on single-cell groups. It does not run inside a 3D spatial pipeline with distance-constrained, replicate-aware testing. Generic databases (CellPhoneDB/LIANA) under-represent small-molecule transmitters. |
| Differential expression | Scanpy `rank_genes_groups` (cell/spot-level Wilcoxon), Seurat FindMarkers; pseudobulk (DESeq2/edgeR) | Squair et al. 2021 showed that methods ignoring variation between biological replicates are biased and can report hundreds of DE genes with no real biological difference [9]. | Spot-level tests remain the default in spatial pipelines, including 3D-STIAS V4.4.1. |
| Brain cell-type annotation | Allen MapMyCells (SEA-AD MTG, whole mouse brain taxonomies) [10,11], CellTypist brain models | MapMyCells reports supertype-level F1 ≈ 0.86 (mean) on held-out SEA-AD MTG data [11]. | Built for dissociated sc/snRNA-seq. Visium spots (55 µm) are mixtures, and imaging panels cover few genes. |
| Brain organoids | HNOCA (1.7M cells, 36 datasets, 26 protocols) [12]; VoxHunt (map organoids to spatial/single-cell brain references) [13] | Quantitative comparison of organoid states with developing human brain references [12]. | Mostly R / large reference models. Organoid analysis is separate from spatial/3D tissue analysis and from the rest of a neuroscience workflow. |
| Electrophysiology | Elephant, Neo (EBRAINS) [14], SpikeInterface, MNE, Brainstorm [15] | Mature, validated implementations. | Interoperability across segments of neurophysiology workflows is limited [15]. These are separate ecosystems from transcriptomics tools. |
| Modelling | Brian2, NEURON, NEST | Reference simulators. | Not connected to the data-analysis side (for example, perturbations informed by expression of channel genes). |
| Morphology / imaging | NeuroM, navis; nilearn, bctpy; siibra (EBRAINS atlases) [14] | Mature domain tools. | Separate data models; no shared provenance with other modalities. |
| AI agents | Biomni (general biomedical; 150 tools, 59 databases) [16], SpatialAgent (Plan-Act-Conclude, verification modules, brain datasets) [17], CellAgent [18], SpaCellAgent [19], CellVoyager, Neuroagent (EPFL), Omega (napari plugin), Brainstorm GPT plugin, PiEEG Agent (local or cloud) | SpatialAgent audits generated claims and was evaluated against baselines and experts [17]. | Most depend on hosted LLMs. In mature tools the LLM is an optional plugin (napari, Brainstorm). |

## 2. Gaps NeuroSTIAS addresses, and the experiment that tests each one

| # | Gap (stated narrowly enough to test) | Our approach | Test | Expected risk |
|---|---|---|---|---|
| G1 | 3D graphs between sections ignore anisotropic section spacing and are rarely evaluated on held-out sections | Anisotropy-aware 3D graph (z-distance scaled by section spacing / in-plane spacing; z-weight chosen by held-out sections) | **E02**: leave-one-section-out prediction, 3D vs 2D vs STAGATE-3D-style single-radius graph | 3D may give only small gains at Visium resolution. If so, we report that. |
| G2 | Spatial domain methods are rarely reported per donor with uncertainty | Domain detection with neighbourhood augmentation plus brain priors; per-section ARI with donor-level CIs | **E01** on DLPFC (12 sections, 3 donors) vs Leiden, BANKSY, SpaGCN, GraphST/STAGATE | Deep models may beat us on ARI; we position on speed, CPU-only use and calibration. |
| G3 | Registration accuracy for serial sections is hard to measure without ground truth | Deterministic rigid/affine registration (from the GSE214349 importer) | **E03** semi-synthetic: known transforms applied to real sections; vs PASTE/PASTE2 | Labelled `semi_synthetic`. |
| G4 | Generic annotation under-resolves brain cell types in spatial data | Curated brain marker panels (neurons by transmitter, astro, oligo, OPC, microglia, vascular, layers) plus reference mapping | **E04** vs V4.4.1 generic, CellTypist, plain marker scoring; macro-F1 against author labels | Spot mixtures limit F1 on Visium; report MERFISH and Visium separately. |
| G5 | Spot-level DE inflates false discoveries | Pseudobulk by animal/donor (`neurostias.core.stats`) | **E05** null splits within animal/donor: empirical FDR vs Scanpy Wilcoxon | Low power with only 3–5 animals; we report power too. |
| G6 | Neurotransmitter signalling is missing from 3D spatial communication analysis | Neurotransmitter/neuropeptide database with synthesis + transporter genes (NeuronChat-style ligand definition), distance-constrained in 3D, replicate-aware permutation | **E06** vs generic LR database; spatial-proximity enrichment and known projections | Ground truth for communication is weak; frame as consistency evidence, not proof. |
| G7 | Organoid fidelity tools are separate from tissue analysis | Reference mapping and maturation scoring in the same framework | **E07**: maturation score vs organoid age (Spearman); region agreement vs VoxHunt-style correlation | We cannot reuse HNOCA's full models; compare on a public subset. |
| G8 | Neuroscience tooling is fragmented across modalities | One data model, skill registry and provenance trail across modalities, using MNE, Elephant, NeuroM and nilearn internally where they are the reference | **E08–E11** validate agreement with reference tools | Validation only, not a claim of superiority. |
| G9 | Agents depend on hosted LLMs and can make up numbers | Optional, model-agnostic plugin (local by default); guardrail that ties every number to tool output | **E12** (optional): routing accuracy and numeric hallucination rate per model | Small local models may route poorly; we report it either way. |

## 3. Data used to test the gaps

DLPFC spatialLIBD (12 Visium sections, 3 donors, manual layers); GSE214349 (mouse IVH, 19 serial
Visium sections from 5 mice, already reconstructed locally); MERFISH mouse hypothalamus
(Moffitt et al. 2018, multiple Bregma slices, author cell types); a public organoid time course (HNOCA subset, Paulsen et al. 2022 samples); DANDI/Allen NWB
ephys and MNE EEGBCI; NeuroMorpho.org SWC files; nilearn development fMRI. Sources and checksums
are in `data/manifest.yaml`.

## References

1. Recent DLPFC benchmark comparisons (SpaGCN, STAGATE, GraphST, SEDR, DeepST): SemanticST, arXiv 2506.11491 — https://arxiv.org/pdf/2506.11491 ; STMGAC, arXiv 2408.06377 — https://arxiv.org/pdf/2408.06377
2. Dong & Zhang, STAGATE, *Nat Commun* 13, 1739 (2022) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8976049
3. Singhal et al., BANKSY, *Nat Genet* 56, 431–441 (2024) — https://eurobioc2024.bioconductor.org/abstracts/paper39
4. PASTE2 / 3D reconstruction overview, *Genome Biol* (2024) — https://link.springer.com/article/10.1186/s13059-024-03361-0
5. STAGATE Tutorial 5: 3D spatial domain identification — https://stagate.readthedocs.io/en/latest/T5_3D.html
6. SpatialLeiden 3D guide — https://spatialleiden.readthedocs.io/latest/guides/3D.html
7. STitch3D / joint modelling of multiple slices — https://www.biorxiv.org/content/10.1101/2023.02.02.526814.full.pdf
8. Zhao et al., NeuronChat, *Nat Commun* (2023) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9974942/
9. Squair et al., Confronting false discoveries in single-cell differential expression, *Nat Commun* 12, 5692 (2021) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8479118
10. Allen Institute, cell type references and algorithms — https://portal.brain-map.org/explore/cell-type-references-and-algorithms
11. Allen Institute, cell type mapping evaluation — https://brain-map.org/our-research/cell-types-taxonomies/cell-type-mapping-evaluation
12. He et al., An integrated transcriptomic cell atlas of human neural organoids, *Nature* 635 (2024) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11578878
13. Fleck et al., Resolving organoid brain region identities by mapping single-cell genomic data to reference atlases (VoxHunt) — https://iob.ch/research/publications/resolving-organoid-brain-region-identities-by-mapping-single-cell-genomic-data-to-reference-atlases/
14. EBRAINS tools (Elephant, Neo, siibra) — https://wiki.ebrains.eu/bin/view/Collabs/documentation/tools-and-services-documentation ; https://ebrains.eu/service/elephant/
15. Integrated open-source software for multiscale electrophysiology (Brainstorm) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6814804/
16. Biomni (Huang et al.) — https://www.broadinstitute.org/node/5559861
17. SpatialAgent, bioRxiv 2025.04.03.646459 — https://www.biorxiv.org/content/10.1101/2025.04.03.646459v1
18. CellAgent, ICLR 2026 — https://proceedings.iclr.cc/paper_files/paper/2026/hash/7c482acecdd1b3386a3a11acc536a22a-Abstract-Conference.html
19. SpaCellAgent — https://www.alphaxiv.org/abs/2607.07467
20. Omega (napari-chatgpt), *Nat Methods* (2024) — https://biohub.org/news/omega-tool-can-rapidly-analyze-complex-biological-images/ ; Brainstorm GPT plugin — https://neuroimage.usc.edu/brainstorm/Tutorials/GptPlugin ; Neuroagent — https://pypi.org/project/neuroagent ; PiEEG Agent — https://www.pieeg.com/agent
