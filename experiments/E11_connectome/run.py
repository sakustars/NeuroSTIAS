"""E11 - Functional connectome analysis (validation + practical result, real data).

nilearn development fMRI (Richardson et al. 2018; children 3-12 y and adults), MSDL atlas time
series. Part A: child-vs-adult classification from connectivity (correlation, partial
correlation, tangent) with LinearSVC, stratified 5-fold CV repeated 10 times - the setup of
the nilearn reference example (Dadi et al. 2019 found tangent best). Part B: graph metrics
(15% density) compared between groups (Welch t, BH-FDR) - descriptive.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neurostias import data  # noqa: E402
from neurostias.core.stats import bh  # noqa: E402
from neurostias.experiment import Experiment, md_table  # noqa: E402
from neurostias.imaging.connectome import connectivity, graph_metrics, vectorize  # noqa: E402


def main():
    exp = Experiment("E11", "Functional connectome: group classification and graph metrics", "Does the NeuroSTIAS connectome "
                     "module reproduce reference child-vs-adult classification accuracy, and which graph metrics differ "
                     "between groups?", params={"density": 0.15, "cv": "StratifiedKFold(5) x 10"})
    p = data.path("ephys_imaging")
    exp.prov.add_input(p, "development_fmri_msdl_timeseries")
    z = np.load(p, allow_pickle=True)
    ts, y = [np.asarray(t, dtype=float) for t in z["timeseries"]], np.asarray(z["labels"]).astype(str)
    fd = np.asarray(z["mean_fd"], dtype=float)  # mean framewise displacement (mm): children move more
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    chance = float(pd.Series(y).value_counts(normalize=True).max())
    rows = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    splits = list(cv.split(np.zeros(len(y)), y))
    # motion-only baseline: how well does head motion alone separate the groups?
    acc, bal = [], []
    for tr, te in splits:
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(fd[tr, None], y[tr])
        pr = clf.predict(fd[te, None])
        acc.append(float(np.mean(pr == y[te])))
        bal.append(balanced_accuracy_score(y[te], pr))
    rows.append({"features": "mean FD only (motion baseline)", "accuracy_mean": float(np.mean(acc)),
                 "accuracy_sd": float(np.std(acc)), "balanced_accuracy_mean": float(np.mean(bal)),
                 "balanced_accuracy_sd": float(np.std(bal)), "chance": chance, "n_subjects": len(y)})
    from nilearn.connectome import ConnectivityMeasure
    for kind in ("correlation", "partial correlation", "tangent"):
        for deconf in (False, True):
            acc, bal = [], []
            for tr, te in splits:
                # connectivity estimator fitted on the training fold only (tangent uses a group mean)
                cm = ConnectivityMeasure(kind=kind, vectorize=True, discard_diagonal=True)
                Xtr = cm.fit_transform([ts[i] for i in tr])
                Xte = cm.transform([ts[i] for i in te])
                if deconf:  # regress mean FD out of every edge, fitted on the training fold only
                    lr = LinearRegression().fit(fd[tr, None], Xtr)
                    Xtr, Xte = Xtr - lr.predict(fd[tr, None]), Xte - lr.predict(fd[te, None])
                clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=20000, random_state=0)).fit(Xtr, y[tr])
                pr = clf.predict(Xte)
                acc.append(float(np.mean(pr == y[te])))
                bal.append(balanced_accuracy_score(y[te], pr))
            rows.append({"features": f"{kind}" + (" (mean FD regressed out in-fold)" if deconf else ""),
                         "accuracy_mean": float(np.mean(acc)), "accuracy_sd": float(np.std(acc)),
                         "balanced_accuracy_mean": float(np.mean(bal)), "balanced_accuracy_sd": float(np.std(bal)),
                         "chance": chance, "n_subjects": len(y)})
            print(rows[-1]["features"], rows[-1]["balanced_accuracy_mean"], flush=True)
    cls = pd.DataFrame(rows)
    exp.table(cls, "e11_classification.csv")
    mats = connectivity(ts, kind="correlation")
    gm = pd.DataFrame([graph_metrics(m, density=0.15) for m in mats])
    gm["group"] = y
    exp.table(gm, "e11_graph_metrics.csv")
    from scipy.stats import t as tdist
    from scipy.stats import ttest_ind
    tests = []
    # mean degree is fixed by the proportional threshold (same edge count for every subject), so it is not tested
    adult = (gm.group == "adult").to_numpy().astype(float)
    D = np.column_stack([np.ones(len(gm)), adult, (fd - fd.mean()) / fd.std()])
    for c in ("clustering", "global_efficiency", "modularity", "n_modules"):
        a, b = gm.loc[gm.group == "adult", c], gm.loc[gm.group == "child", c]
        t = ttest_ind(a, b, equal_var=False)
        # OLS metric ~ group + mean FD: group effect adjusted for head motion
        yv = gm[c].to_numpy(float)
        beta, *_ = np.linalg.lstsq(D, yv, rcond=None)
        resid = yv - D @ beta
        dof = len(yv) - D.shape[1]
        se = np.sqrt(np.diag(np.linalg.inv(D.T @ D)) * (resid @ resid) / dof)
        t_adj = beta[1] / se[1]
        tests.append({"metric": c, "adult_mean": a.mean(), "child_mean": b.mean(), "t": float(t.statistic),
                      "pval": float(t.pvalue), "adult_effect_adj_fd": float(beta[1]), "t_adj_fd": float(t_adj),
                      "pval_adj_fd": float(2 * tdist.sf(abs(t_adj), dof))})
    tests = pd.DataFrame(tests)
    tests["padj"] = bh(tests["pval"])
    tests["padj_adj_fd"] = bh(tests["pval_adj_fd"])
    exp.table(tests, "e11_graph_metric_tests.csv")
    exp.summary.update({"classification": cls.to_dict("records"), "graph_tests": tests.to_dict("records"),
                        "group_counts": pd.Series(y).value_counts().to_dict(),
                        "mean_fd_by_group": pd.Series(fd).groupby(y).mean().to_dict()})
    exp.note("Groups are imbalanced (122 children, 33 adults); balanced accuracy is the primary metric and 'chance' is "
             "the majority-class rate for plain accuracy (balanced-accuracy chance = 0.5).")
    exp.note("Children move more than adults (mean framewise displacement %.2f vs %.2f mm), so a motion-only baseline and "
             "an in-fold motion-regressed connectome are reported, and graph-metric group effects are also given "
             "adjusted for mean FD (OLS)." % (fd[y == "child"].mean(), fd[y == "adult"].mean()))
    exp.note("Mean degree is not tested: proportional thresholding at density 0.15 gives every subject the same edge count.")
    exp.note("Connectivity estimators (including the tangent-space reference) are fitted inside each training fold to "
             "avoid leakage from test subjects.")
    exp.finish([("Child vs adult classification", md_table(cls)), ("Graph metrics, adult vs child", md_table(tests))])
    print(cls); print(tests)


if __name__ == "__main__":
    main()
