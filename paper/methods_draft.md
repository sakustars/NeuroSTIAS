# Methods draft (auto-generated from provenance)

## E01

Parameters: `{"k_nn": 6, "lam": 0.8, "hops": 2, "n_pcs": 30, "n_hvg": 3000, "refine": true, "seeds": [0, 1, 2, 3, 4]}`

## E02

Parameters: `{"um_per_px_dlpfc": 0.7299270072992701, "z_scales": [0.25, 0.5, 1, 2, 4, 8], "k_nn": 6, "k_between": 3, "seeds": [0, 1, 2, 3, 4]}`

## E03

Parameters: `{"drop": 0.1, "jitter_um": 10, "thinning": 0.5, "max_shift_um": 500, "paste_alpha": 0.1, "seeds": [0, 1, 2, 3, 4]}`

## E04

Parameters: `{"alpha": 0.05, "n_null": 1000, "seeds": [0, 1, 2, 3, 4]}`

## E05

Parameters: `{"layers": ["L3", "L5", "WM"], "n_effect_genes": 200, "thinning": 0.5, "min_cells": 10, "seeds": [0, 1, 2, 3, 4]}`

## E06

Parameters: `{"n_permutations": 200, "k_merfish": 10, "k_dlpfc": 6, "canonical_list": [["PDGFB_PDGFRB", "PDGFB_PDGFRB", ["Endothelial"], ["Pericyte"]], ["CX3CL1_CX3CR1", null, ["Excitatory neuron", "Inhibitory neuron"], ["Microglia"]], ["IL34_CSF1R", null, ["Excitatory neuron", "Inhibitory neuron"], ["Microglia"]], ["CSF1_CSF1R", "CSF1_CSF1R", ["Astrocyte", "Oligodendrocyte", "OPC", "Excitatory neuron", "Inhibitory neuron"], ["Microglia"]], ["GABA_A", null, ["Inhibitory neuron"], ["Excitatory neuron", "Inhibitory neuron"]], ["GLU_AMPA", null, ["Excitatory neuron"], ["Excitatory neuron", "Inhibitory neuron"]], ["GLU_NMDA_2B", null, ["Excitatory neuron"], ["Excitatory neuron", "Inhibitory neuron"]], ["VEGFA_R", "VEGFA_KDR", ["Astrocyte"], ["Endothelial"]], ["PTN_PTPRZ1", null, ["Excitatory neuron", "Inhibitory neuron", "Astrocyte"], ["OPC", "Astrocyte"]]], "seeds": [0, 1, 2, 3, 4]}`

## E07

Parameters: `{"alpha": 0.05, "n_null": 1000, "seeds": [0, 1, 2, 3, 4]}`

## E08

Parameters: `{"fano_window_s": 1.0, "eeg_band_hz": [8, 30], "cv": "ShuffleSplit(10, 0.2)", "seeds": [0, 1, 2, 3, 4]}`

## E09

Parameters: `{"hh_dt_ms": 0.01, "currents": [2, 4, 6, 6.5, 7, 8, 10, 15, 20, 30], "brunel_regimes": [[5, 2], [4.5, 0.9], [6, 4]], "seeds": [0, 1, 2, 3, 4]}`

## E10

Parameters: `{"cv": "10-fold x 5", "seeds": [0, 1, 2, 3, 4]}`

## E11

Parameters: `{"density": 0.15, "cv": "StratifiedKFold(5) x 10", "seeds": [0, 1, 2, 3, 4]}`

## Software

| package | version |
|---|---|
| anndata | 0.12.19 |
| brian2 | 2.9.0 |
| elephant | 1.2.1 |
| mne | 1.13.2 |
| networkx | 3.6.1 |
| neurom | 4.0.6 |
| nilearn | 0.14.1 |
| numpy | 2.4.6 |
| pandas | 2.3.3 |
| scanpy | 1.11.5 |
| scikit-learn | 1.9.1 |
| scipy | 1.17.1 |
| squidpy | 1.8.2 |
| statsmodels | 0.15.0 |

Baseline tools ran in isolated environments; exact versions are in `benchmarks/envs/baselines_freeze.txt` and `benchmarks/envs/brian2_freeze.txt`.

## Input files (SHA-256 prefix)

| input | sha256 | bytes |
|---|---|---|
| GSE214349_reconstructed_3d | 4f76f1212564348e | 2312751450 |
| development_fmri_msdl_timeseries | e8005df60496ed5f | 8140740 |
| dlpfc_151507 | 28763fa953bc47fc | 99037548 |
| dlpfc_151508 | a66a4dadf8806f58 | 87266932 |
| dlpfc_151509 | cb8036d74d4b77a4 | 113620312 |
| dlpfc_151510 | c02189da94dea2f2 | 105189644 |
| dlpfc_151669 | 91e44f3bf9463f5b | 109083944 |
| dlpfc_151670 | 923d53981fd1be1b | 97363628 |
| dlpfc_151671 | dc7217496b40584d | 125892124 |
| dlpfc_151672 | ff35fb0248041879 | 116613072 |
| dlpfc_151673 | ad87c4880bbf61b1 | 131966928 |
| dlpfc_151674 | 606c891c98b9fb72 | 162178720 |
| dlpfc_151675 | 70098e3100ab49d9 | 107494612 |
| dlpfc_151676 | e57d5c674a5dcc42 | 110816820 |
| dlpfc_all | 0d135ba65586f2e1 | 1331528080 |
| hnoca_timecourse_subset | a9ef2e45db30391a | 1189034700 |
| merfish | 371723d48413ba76 | 51577387 |
| merfish_moffitt2018 | 371723d48413ba76 | 51577387 |
| neuromorpho_metadata | ff698c2c20c2cf21 | 78033 |
| sub-anm369962_ses-20170309.nwb | 8d1e87886816241d | 796920 |
| sub-anm369962_ses-20170310.nwb | d01c43576e3aecee | 6644036 |
| sub-anm369962_ses-20170313.nwb | 09ce236489f9f3b6 | 11476076 |
| sub-anm369962_ses-20170314.nwb | 6a94a1a71bc9489e | 7456488 |
| sub-anm369962_ses-20170316.nwb | c35c73d0e2b2f3a8 | 609568 |
| sub-anm369962_ses-20170317.nwb | 35ccbeed5575d77e | 8194336 |
| sub-anm369963_ses-20170226.nwb | ebcdf64353eeb9b6 | 6393196 |
| sub-anm369963_ses-20170227.nwb | c2b5876e40f4bcd6 | 6459808 |
| sub-anm369963_ses-20170228.nwb | 796630b7941cd4e4 | 276632 |
| sub-anm369963_ses-20170301.nwb | 2435b672abaa3731 | 493516 |
| sub-anm369963_ses-20170302.nwb | e5c0361eaa9693f4 | 281012 |
| sub-anm369963_ses-20170306.nwb | e4df2ef55c749fdb | 6666716 |
| sub-anm369963_ses-20170309.nwb | 4aed9101446df3c9 | 450860 |
| sub-anm369963_ses-20170310.nwb | fc4314ad75b53442 | 604844 |
| sub-anm369964_ses-20170320.nwb | 130ee7635334db52 | 7660100 |
| sub-anm369964_ses-20170321.nwb | d09645a90a677f34 | 499864 |
| sub-anm369964_ses-20170322.nwb | eecbd6c12f365d26 | 5047600 |
| sub-anm369964_ses-20170323.nwb | ee96f2fdb95f6cf9 | 426444 |
| sub-anm372793_ses-20170504.nwb | cc14bf369f827ddd | 357824 |
| sub-anm372793_ses-20170508.nwb | 13e42ab4f9deb1d1 | 357424 |
| sub-anm372793_ses-20170512.nwb | 7b839400c33648da | 470808 |
| sub-anm372793_ses-20170513.nwb | 175428b13cda15b6 | 5519112 |
| sub-anm372793_ses-20170514.nwb | 59c1a56ce21fcad2 | 5459688 |
| sub-anm372794_ses-20170621.nwb | 1a377918f5847f57 | 654184 |
| sub-anm372794_ses-20170622.nwb | d298916ebfd88955 | 602808 |
| sub-anm372794_ses-20170624.nwb | 1af69b8e29f0a74b | 4751624 |
| sub-anm372794_ses-20170625.nwb | 23a58889cccc5147 | 5684876 |
| sub-anm372794_ses-20170626.nwb | a51d3da08afbd56b | 4929288 |
| sub-anm372794_ses-20170627.nwb | 0e1c9d92b2d35b05 | 553420 |
| sub-anm372795_ses-20170714.nwb | 6248053ecc33dab0 | 465540 |
| sub-anm372795_ses-20170715.nwb | c8680027abe87c9f | 323156 |
| sub-anm372795_ses-20170716.nwb | 6c01807bfaaa9113 | 813220 |
| sub-anm372795_ses-20170718.nwb | 1464f84e61ab4d2d | 452348 |
| sub-anm372797_ses-20170615.nwb | cf5944209e3c58be | 5970808 |
| sub-anm372797_ses-20170617.nwb | 8369d5cb84303876 | 338864 |
| sub-anm372904_ses-20170615.nwb | 6aaaab2311b2f588 | 388212 |
| sub-anm372904_ses-20170616.nwb | 1bdfc68052424794 | 362600 |
| sub-anm372904_ses-20170617.nwb | 0d5a4f8956068f10 | 384920 |
| sub-anm372904_ses-20170618.nwb | 393454dcddec5ecb | 9378480 |
| sub-anm372904_ses-20170619.nwb | 8dd6b41adc3de119 | 4100256 |
| sub-anm372905_ses-20170715.nwb | 2eab5196f0d989c9 | 344024 |
| sub-anm372905_ses-20170716.nwb | 4d3e2e34de2cb2b9 | 554044 |
| sub-anm372905_ses-20170717.nwb | eec6ab07e54cef86 | 448436 |
| sub-anm372906_ses-20170608.nwb | 166bcd11f08a239e | 6942912 |
| sub-anm372906_ses-20170610.nwb | 6f8d6b4a2f4fc42d | 374788 |
| sub-anm372906_ses-20170611.nwb | a846ffaeeca5590f | 338652 |
| sub-anm372906_ses-20170612.nwb | 301ce1094de83b31 | 324348 |
| sub-anm372907_ses-20170608.nwb | a39b5d369f70b7d0 | 438876 |
| sub-anm372907_ses-20170610.nwb | e18a828267b77aed | 349728 |
| sub-anm372907_ses-20170612.nwb | bbd6e82fee268f3e | 444520 |
| sub-anm372907_ses-20170613.nwb | 974be4e6bb774eee | 258992 |
| sub-anm372909_ses-20170520.nwb | 6b88c9d7f093a363 | 352892 |
| sub-anm372909_ses-20170522.nwb | 141d1d9b3b12b079 | 5670812 |
