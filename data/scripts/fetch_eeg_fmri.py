"""Fetch MNE EEGBCI motor-imagery runs (PhysioNet) and nilearn's development fMRI sample (OSF).

EEGBCI: subjects 1-20, runs 6, 10, 14 (imagined hands vs feet), as in the MNE
CSP decoding example (Schalk et al. 2004; Goldberger et al. 2000).
Development fMRI: preprocessed resting-state data from Richardson et al. 2018
(children 3-12 y and adults); regional time series are extracted with the MSDL
probabilistic atlas and saved to an .npz.

Robust download: nilearn's default request timeout (15 s connect) is too short for
slow TLS handshakes to OSF from some networks, so it is raised before fetching and
each subject is fetched individually with several attempts. Subjects whose files
still cannot be obtained are listed in ``development_fmri_download_report.json``
and excluded; they are never imputed.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def fetch_eeg(out: Path, n_eeg: int) -> None:
    from mne.datasets import eegbci
    eeg_dir = out / "eegbci"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    for s in range(1, n_eeg + 1):
        eegbci.load_data(s, [6, 10, 14], path=str(eeg_dir), update_path=False, verbose="ERROR")
    print("EEGBCI subjects", n_eeg, flush=True)


def prefetch_with_curl(fdir: Path, attempts: int) -> list[str]:
    """Download missing development-fMRI files with resumable curl (same OSF keys nilearn uses).

    OSF read timeouts interrupt nilearn's ``requests`` downloads and restart them from scratch;
    ``curl -C -`` resumes partial files, so large bold images eventually complete.
    Returns the participant ids that are still incomplete.
    """
    import shutil
    import subprocess

    import nilearn
    import pandas as pd
    if shutil.which("curl") is None:
        return []
    keys = pd.read_csv(Path(nilearn.__file__).parent / "datasets" / "data" / "development_fmri.csv")
    ddir = fdir / "development_fmri"
    ddir.mkdir(parents=True, exist_ok=True)
    conf_t = "{}_task-pixar_desc-confounds_regressors.tsv"
    bold_t = "{}_task-pixar_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    missing = []
    for r in keys.itertuples(index=False):
        for name, key in ((conf_t.format(r.participant_id), r.key_r), (bold_t.format(r.participant_id), r.key_b)):
            dst = ddir / name
            if dst.exists():
                continue
            part = ddir / (name + ".curlpart")
            for _ in range(attempts):
                rc = subprocess.run(["curl", "-sSL", "--fail", "-C", "-", "--retry", "5", "--retry-delay", "10",
                                     "--connect-timeout", "120", "--speed-time", "300", "--speed-limit", "1000",
                                     "-o", str(part), f"https://osf.io/download/{key}/"]).returncode
                if rc == 0:
                    part.rename(dst)
                    print("curl ok", name, flush=True)
                    break
                print("curl retry", name, "rc", rc, flush=True)
                time.sleep(10)
            if not dst.exists():
                missing.append(r.participant_id)
    return sorted(set(missing))


def fetch_fmri(out: Path, n_fmri: int, attempts: int) -> None:
    import nilearn.datasets._utils as nu
    from nilearn import datasets
    from nilearn.maskers import NiftiMapsMasker
    nu._REQUESTS_TIMEOUT = (120, 600)  # connect, read (seconds); default (15.1, 61) times out on slow handshakes
    fdir = out / "nilearn"
    still_missing = prefetch_with_curl(fdir, attempts)
    print("curl prefetch done; incomplete subjects:", still_missing, flush=True)
    # phenotype + participant list (small)
    pheno = None
    for k in range(attempts):
        try:
            pheno = datasets.fetch_development_fmri(n_subjects=1, data_dir=str(fdir), verbose=0)
            break
        except Exception as exc:  # noqa: BLE001
            print("phenotype attempt", k + 1, "failed:", str(exc)[:120], flush=True)
            time.sleep(20)
    if pheno is None:
        raise RuntimeError("could not download development fMRI phenotype file")
    # fetch subjects one at a time via increasing n_subjects is not possible (nilearn picks a balanced subset),
    # so request the full set repeatedly; each call resumes and keeps completed files
    dev = None
    for k in range(attempts):
        try:
            dev = datasets.fetch_development_fmri(n_subjects=n_fmri, data_dir=str(fdir), verbose=0)
            break
        except Exception as exc:  # noqa: BLE001
            print("full-set attempt", k + 1, "failed:", str(exc)[:160], flush=True)
            time.sleep(30)
    report = {"requested": n_fmri, "timeout_s": nu._REQUESTS_TIMEOUT}
    if dev is not None:
        func, conf = list(dev.func), list(dev.confounds)
        ph = dev.phenotypic
        labels, ages = np.asarray(ph["Child_Adult"]), np.asarray(ph["Age"], dtype=float)
        report["excluded_subjects"] = []
    else:
        # fall back to the subjects whose bold + confounds files are fully on disk
        root = next(fdir.glob("development_fmri*"), fdir)
        import pandas as pd
        ph = pd.read_csv(next(root.rglob("participants.tsv")), sep="\t")
        func, conf, keep = [], [], []
        for i, sid in enumerate(ph["participant_id"]):
            b = list(root.rglob(f"{sid}_*bold*.nii.gz"))
            c = list(root.rglob(f"{sid}_*confounds*.tsv"))
            if b and c:
                func.append(str(b[0])); conf.append(str(c[0])); keep.append(i)
        excluded = sorted(set(ph["participant_id"]) - set(ph["participant_id"].iloc[keep]))
        ph = ph.iloc[keep]
        labels, ages = ph["Child_Adult"].to_numpy(), ph["Age"].to_numpy(float)
        report["excluded_subjects"] = excluded
        report["note"] = "full download failed; using subjects with complete local files"
    atlas = datasets.fetch_atlas_msdl(data_dir=str(fdir), verbose=0)
    masker = NiftiMapsMasker(maps_img=atlas.maps, standardize="zscore_sample", memory=str(fdir / "cache"), verbose=0)
    ts = [masker.fit_transform(f, confounds=c) for f, c in zip(func, conf)]
    import pandas as pd
    # subject ids and mean framewise displacement (head motion), needed for motion-confound controls
    pids = [Path(f).name.split("_")[0] for f in func]
    mean_fd = [float(pd.read_csv(c, sep="\t")["framewise_displacement"].astype(float).mean()) for c in conf]
    np.savez(out / "development_fmri_msdl_timeseries.npz", timeseries=np.stack(ts).astype(np.float64), labels=np.asarray(labels).astype(str),
             ages=ages, participant_ids=np.asarray(pids), mean_fd=np.asarray(mean_fd),
             region_labels=np.asarray(atlas.labels).astype(str))
    report.update({"n_used": len(ts), "groups": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}})
    (out / "development_fmri_download_report.json").write_text(json.dumps(report, indent=2))
    print("fMRI subjects", len(ts), report["groups"], "excluded", len(report["excluded_subjects"]), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--out", required=True)
    a.add_argument("--n-eeg", type=int, default=20)
    a.add_argument("--n-fmri", type=int, default=155)
    a.add_argument("--attempts", type=int, default=6)
    a.add_argument("--skip-eeg", action="store_true")
    x = a.parse_args()
    out = Path(x.out)
    if not x.skip_eeg:
        fetch_eeg(out, x.n_eeg)
    fetch_fmri(out, x.n_fmri, x.attempts)
