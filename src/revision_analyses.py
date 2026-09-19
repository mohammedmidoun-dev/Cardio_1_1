"""
revision_analyses.py
Les quatre analyses demandées par la relecture, en une exécution.

    python src/revision_analyses.py                 # tout
    python src/revision_analyses.py --steps calib   # une étape
    python src/revision_analyses.py --quick         # bootstrap réduit

CE QUE FAIT CE SCRIPT
---------------------
A. DÉCOMPOSITION DE LA PÉNALITÉ DE TRANSFERT.
   L'objection principale des relecteurs est que la pénalité de 0,052 mélange
   changement de mesure et changement de définition d'événement. La décomposition
   les sépare, en utilisant un contraste où tout est tenu constant sauf la mesure :
   les trois cohortes UCI, mêmes patients et même définition d'événement, sous
   schéma mince puis sous schéma riche.

B. ANALYSE SANS KAGGLE.
   Kaggle représente 93 % des patients et sa provenance est mal documentée.
   Toutes les quantités principales sont recalculées sans lui.

C. EXPÉRIENCE DE RECALIBRATION (l'analyse manquante).
   Pour chaque modèle : sans pondération de classes, avec pondération, et avec
   pondération suivie d'un recalibrage. Cela transforme une observation
   accidentelle en expérience contrôlée, et distingue « la pondération nuit »
   de « la pondération décalibre, et le décalibrage nuit ».

D. CALIBRATION COMPLÈTE ET INCERTITUDE.
   Pente et ordonnée de calibration, score de Brier, ECE à 10 et 20 classes,
   et intervalles de confiance bootstrap sur les trois chiffres clés de
   l'article : gain algorithmique, gain informationnel, pénalité de transfert.

ENTRÉES : data/harmonized*/, results/models_v2/ (prédictions hors-échantillon)
SORTIES : results/tables/rev_*.csv, results/figures/rev_*.png
"""
from __future__ import annotations
import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DIRS = {"L1": ROOT / "data" / "harmonized", "L2": ROOT / "data" / "harmonized_rich"}
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MODELS = ROOT / "results" / "models_v2"
for d in (TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "target"
UCI = ["cleveland", "hungarian", "va_longbeach"]
ALGOS = ["logreg", "nb", "dtree", "knn", "svm", "rf", "extra", "hgb", "xgb", "lgbm"]
N_BOOT = 1000
DCA_T = np.linspace(0.01, 0.60, 60)


# =============================================================== métriques
def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() / len(y)
                     for b in range(bins) if (idx == b).any()))


def calibration_slope_intercept(y, p, eps=1e-6):
    """Régression logistique de l'issue sur le logit des prédictions.

    Pente 1 et ordonnée 0 = calibration parfaite. Une pente inférieure à 1
    signale des prédictions trop extrêmes ; une ordonnée négative, une
    surestimation systématique du risque (calibration-in-the-large).
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(lp, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def net_benefit(y, p, thresholds=DCA_T):
    y, p = np.asarray(y), np.asarray(p)
    N, pos = len(y), y.sum()
    out = []
    for t in thresholds:
        pred = p >= t
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        out.append(tp / N - (fp / N) * (t / (1 - t)))
    return np.array(out)


def boot_ci(fn, n=N_BOOT, seed=SEED, *arrays):
    rng = np.random.default_rng(seed)
    n_obs = len(arrays[0])
    vals = []
    for _ in range(n):
        i = rng.integers(0, n_obs, n_obs)
        try:
            v = fn(*[a[i] for a in arrays])
            if v == v:
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# =============================================================== données
def load(schema, cohort=None):
    out = {}
    for f in sorted(DIRS[schema].glob("*.csv")):
        if cohort and f.stem != cohort:
            continue
        df = pd.read_csv(f)
        feats = [c for c in df.columns if c != TARGET]
        out[f.stem] = (df[feats], df[TARGET].values.astype(int))
    return out


def oof(tag, cohort, algo):
    f = MODELS / f"oof_{tag}_{cohort}_{algo}.npy"
    return np.load(f) if f.exists() else None


def transfer_matrix(tag):
    f = TABLES / f"v2_{tag}_transfer_mean.csv"
    return pd.read_csv(f, index_col=0) if f.exists() else None


def penalty(m):
    v = m.values.astype(float)
    off = v.copy(); np.fill_diagonal(off, np.nan)
    return float(np.nanmean(np.diag(v))), float(np.nanmean(off)), \
           float(np.nanmean(np.diag(v)) - np.nanmean(off))


# =============================================================== A. décomposition
def step_decompose():
    """Sépare l'effet de la mesure de celui de la définition d'événement."""
    print("\n=== A. Décomposition de la pénalité de transfert ===")
    m1, m2 = transfer_matrix("L1"), transfer_matrix("L2")
    if m1 is None:
        print("  [skip] matrice L1 absente"); return pd.DataFrame()
    rows = []
    d, o, p = penalty(m1)
    rows.append({"regime": "A. Five cohorts, thin schema", "n_cohorts": len(m1),
                 "outcome_definition": "mixed", "schema": "thin",
                 "within": round(d, 4), "transferred": round(o, 4), "penalty": round(p, 4)})
    uci = [c for c in m1.index if c in UCI]
    if len(uci) == 3:
        d, o, p = penalty(m1.loc[uci, uci])
        rows.append({"regime": "B. Three UCI cohorts, thin schema", "n_cohorts": 3,
                     "outcome_definition": "identical", "schema": "thin",
                     "within": round(d, 4), "transferred": round(o, 4), "penalty": round(p, 4)})
    if m2 is not None:
        d, o, p = penalty(m2)
        rows.append({"regime": "C. Three UCI cohorts, rich schema", "n_cohorts": len(m2),
                     "outcome_definition": "identical", "schema": "rich",
                     "within": round(d, 4), "transferred": round(o, 4), "penalty": round(p, 4)})
    # transferts croisant la frontière de définition d'événement
    others = [c for c in m1.index if c not in UCI]
    if uci and others:
        cross = [m1.loc[s, t] for s in uci for t in others] + \
                [m1.loc[t, s] for s in uci for t in others]
        local = np.mean([m1.loc[c, c] for c in list(uci) + list(others)])
        rows.append({"regime": "D. Across outcome definitions, thin schema",
                     "n_cohorts": len(m1), "outcome_definition": "different",
                     "schema": "thin", "within": round(float(local), 4),
                     "transferred": round(float(np.mean(cross)), 4),
                     "penalty": round(float(local - np.mean(cross)), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "rev_A_transfer_decomposition.csv", index=False)
    print(df.to_string(index=False))
    if len(df) >= 3:
        b = df[df.regime.str.startswith("B")].penalty.iloc[0]
        c = df[df.regime.str.startswith("C")].penalty.iloc[0]
        print(f"\n  B − C = {b - c:+.4f}  <- attribuable à la MESURE seule "
              f"(mêmes cohortes, même définition d'événement)")
    return df


# =============================================================== B. sans Kaggle
def step_nokaggle():
    print("\n=== B. Analyse sans la cohorte Kaggle ===")
    rows = []
    m = transfer_matrix("L1")
    if m is not None:
        keep = [c for c in m.index if "kaggle" not in c]
        for lab, mm in [("all cohorts", m), ("without Kaggle", m.loc[keep, keep])]:
            d, o, p = penalty(mm)
            rows.append({"analysis": "transfer penalty", "subset": lab,
                         "value": round(p, 4), "within": round(d, 4),
                         "transferred": round(o, 4)})
    w = TABLES / "v2_L1_within.csv"
    if w.exists():
        d = pd.read_csv(w).drop_duplicates(["cohort", "algo"], keep="last")
        for lab, dd in [("all cohorts", d), ("without Kaggle", d[~d.cohort.str.contains("kaggle")])]:
            g = dd.pivot(index="cohort", columns="algo", values="auroc")
            if "logreg" not in g:
                continue
            gain = (g.max(axis=1) - g["logreg"])
            rows.append({"analysis": "best algo − logistic regression (mean)",
                         "subset": lab, "value": round(float(gain.mean()), 4),
                         "within": round(float(gain.max()), 4), "transferred": np.nan})
            rows.append({"analysis": "AUROC range across algorithms (mean)",
                         "subset": lab, "value": round(float((g.max(axis=1) - g.min(axis=1)).mean()), 4),
                         "within": np.nan, "transferred": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "rev_B_without_kaggle.csv", index=False)
    print(df.to_string(index=False))
    return df


# =============================================================== C. recalibration
def make_model(kind, weighted):
    cw = "balanced" if weighted else None
    if kind == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight=cw, random_state=SEED)
    elif kind == "dtree":
        clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                     class_weight=cw, random_state=SEED)
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                     n_jobs=4, random_state=SEED,
                                     class_weight="balanced_subsample" if weighted else None)
    elif kind == "extra":
        clf = ExtraTreesClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                   n_jobs=4, class_weight=cw, random_state=SEED)
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=250, max_depth=6, learning_rate=0.06,
                                             class_weight=cw, random_state=SEED)
    else:
        raise ValueError(kind)
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("clf", clf)])


def recalibrate(y, p, seed=SEED):
    """Recalibrage isotonique honnête : appris et appliqué hors-échantillon."""
    y, p = np.asarray(y), np.asarray(p)
    out = np.zeros_like(p, dtype=float)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in cv.split(p.reshape(-1, 1), y):
        iso = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr])
        out[te] = iso.predict(p[te])
    return out


def step_recalibration(schema="L1", n_boot=200):
    """L'expérience manquante : sans pondération / avec / avec + recalibrage."""
    print("\n=== C. Expérience de recalibration ===")
    data = load(schema)
    rows = []
    for cohort, (X, y) in data.items():
        prev = float(y.mean())
        print(f"\n  {cohort} (prévalence {prev:.3f})")
        for kind in ["logreg", "dtree", "rf", "extra", "hgb"]:
            preds = {}
            for weighted in (False, True):
                cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
                p = cross_val_predict(make_model(kind, weighted), X, y, cv=cv,
                                      method="predict_proba")[:, 1]
                preds["weighted" if weighted else "unweighted"] = p
            preds["weighted + recalibrated"] = recalibrate(y, preds["weighted"])
            for arm, p in preds.items():
                sl, ic = calibration_slope_intercept(y, p)
                nb = net_benefit(y, p)
                rows.append({
                    "schema": schema, "cohort": cohort, "prevalence": round(prev, 4),
                    "algo": kind, "arm": arm,
                    "auroc": round(roc_auc_score(y, p), 4),
                    "brier": round(brier_score_loss(y, p), 4),
                    "ece10": round(ece(y, p, 10), 4), "ece20": round(ece(y, p, 20), 4),
                    "calib_slope": round(sl, 3), "calib_intercept": round(ic, 3),
                    "net_benefit_at_0.10": round(float(nb[np.argmin(abs(DCA_T - 0.10))]), 4),
                    "net_benefit_at_0.20": round(float(nb[np.argmin(abs(DCA_T - 0.20))]), 4),
                    "nb_positive_up_to": round(float(DCA_T[nb > 0][-1]) if (nb > 0).any() else 0.0, 3)})
            r = {a: rows[-3 + i] for i, a in enumerate(preds)}
            print(f"    {kind:7s} AUROC {r['unweighted']['auroc']:.3f}/"
                  f"{r['weighted']['auroc']:.3f}/{r['weighted + recalibrated']['auroc']:.3f}"
                  f"   ECE {r['unweighted']['ece10']:.3f}/{r['weighted']['ece10']:.3f}/"
                  f"{r['weighted + recalibrated']['ece10']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / f"rev_C_recalibration_{schema}.csv", index=False)

    piv = df.groupby("arm")[["auroc", "brier", "ece10", "calib_slope", "calib_intercept"]].mean().round(4)
    print("\n  Moyennes sur toutes les cohortes et tous les modèles :")
    print(piv.to_string())

    low = df.loc[df.prevalence.idxmin(), "cohort"]
    d = df[df.cohort == low]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    order = ["unweighted", "weighted", "weighted + recalibrated"]
    for ax, metric, lab in zip(axes, ["auroc", "ece10", "net_benefit_at_0.20"],
                               ["AUROC", "Expected calibration error", "Net benefit at threshold 0.20"]):
        vals = [d[d.arm == a][metric].values for a in order]
        ax.boxplot(vals, labels=["unweighted", "weighted", "weighted +\nrecalibrated"],
                   patch_artist=True, widths=.55,
                   boxprops=dict(facecolor="#cfe2f3", edgecolor="#2E75B6"))
        ax.set_title(lab, fontsize=10); ax.grid(alpha=.25, axis="y")
        if metric == "net_benefit_at_0.20":
            ax.axhline(0, color="k", ls="--", lw=1)
    fig.suptitle(f"Class weighting, calibration and net benefit — {low} "
                 f"(prevalence {d.prevalence.iloc[0]:.3f})", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES / f"rev_C_recalibration_{schema}.png", dpi=300)
    plt.close(fig)
    return df


# =============================================================== D. calibration + IC
def step_calibration(schema="L1", n_boot=N_BOOT):
    print("\n=== D. Calibration complète et intervalles de confiance ===")
    data = load(schema)
    rows = []
    for cohort, (X, y) in data.items():
        for algo in ALGOS:
            p = oof(schema, cohort, algo)
            if p is None:
                continue
            sl, ic = calibration_slope_intercept(y, p)
            a = roc_auc_score(y, p)
            lo, hi = boot_ci(lambda yy, pp: roc_auc_score(yy, pp), n_boot, SEED, y, p)
            rows.append({"schema": schema, "cohort": cohort, "algo": algo,
                         "auroc": round(a, 4), "auroc_lo": round(lo, 4), "auroc_hi": round(hi, 4),
                         "brier": round(brier_score_loss(y, p), 4),
                         "ece10": round(ece(y, p, 10), 4), "ece20": round(ece(y, p, 20), 4),
                         "calib_slope": round(sl, 3), "calib_intercept": round(ic, 3)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / f"rev_D_calibration_{schema}.csv", index=False)
    print(df.groupby("cohort")[["calib_slope", "calib_intercept", "brier", "ece10"]]
            .agg(["min", "max"]).round(3).to_string())

    # IC sur les trois chiffres clés de l'article
    key = {}
    for cohort, (X, y) in data.items():
        pr = {a: oof(schema, cohort, a) for a in ALGOS}
        pr = {a: p for a, p in pr.items() if p is not None}
        if "logreg" not in pr or len(pr) < 2:
            continue
        best = max((a for a in pr if a != "logreg"), key=lambda a: roc_auc_score(y, pr[a]))
        d = roc_auc_score(y, pr[best]) - roc_auc_score(y, pr["logreg"])
        lo, hi = boot_ci(lambda yy, pb, pl: roc_auc_score(yy, pb) - roc_auc_score(yy, pl),
                         n_boot, SEED, y, pr[best], pr["logreg"])
        key[f"algorithmic_gain_{cohort}"] = {"best_algo": best, "delta": round(d, 4),
                                             "ci": [round(lo, 4), round(hi, 4)]}
    # gain informationnel L1 -> L2, avec IC apparié par cohorte
    d1, d2 = load("L1"), load("L2")
    for c in set(d1) & set(d2):
        pa = [oof("L1", c, a) for a in ALGOS]
        pb = [oof("L2", c, a) for a in ALGOS]
        pa = [p for p in pa if p is not None]; pb = [p for p in pb if p is not None]
        if not pa or not pb:
            continue
        y = d1[c][1]
        ma, mb = np.mean(pa, axis=0), np.mean(pb, axis=0)
        g = roc_auc_score(y, mb) - roc_auc_score(y, ma)
        lo, hi = boot_ci(lambda yy, x2, x1: roc_auc_score(yy, x2) - roc_auc_score(yy, x1),
                         n_boot, SEED, y, mb, ma)
        key[f"information_gain_{c}"] = {"delta": round(g, 4), "ci": [round(lo, 4), round(hi, 4)]}
    with open(TABLES / "rev_D_key_effects.json", "w") as fh:
        json.dump(key, fh, indent=2)
    print("\n  Effets clés avec IC bootstrap :")
    print(json.dumps(key, indent=2))
    return df


# =============================================================== driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all",
                    help="all | decompose | nokaggle | recalib | calib")
    ap.add_argument("--schema", default="L1")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    nb = 200 if a.quick else N_BOOT
    if a.steps in ("all", "decompose"):
        step_decompose()
    if a.steps in ("all", "nokaggle"):
        step_nokaggle()
    if a.steps in ("all", "recalib"):
        step_recalibration(a.schema, nb)
    if a.steps in ("all", "calib"):
        step_calibration(a.schema, nb)
    print(f"\nTerminé. Tableaux dans {TABLES}, figure dans {FIGURES}.")


if __name__ == "__main__":
    main()
