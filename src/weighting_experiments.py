"""
weighting_experiments.py
Les trois expériences qui font passer la pondération de classes du statut
d'observation à celui de résultat principal.

    python src/weighting_experiments.py                 # tout
    python src/weighting_experiments.py --steps gradient
    python src/weighting_experiments.py --quick         # version rapide

POURQUOI CES TROIS EXPÉRIENCES
------------------------------
L'analyse précédente montrait, sur cinq cohortes, que la pondération de classes
laisse l'AUROC intacte, dégrade la calibration proportionnellement à l'écart
entre la prévalence et 0,50 (r = 0,99), rend le bénéfice net négatif, et que le
recalibrage restaure tout. Trois faiblesses subsistent :

  1. cinq cohortes = cinq points, et elles diffèrent par tout autre chose que
     la prévalence. La relation dose-effet est donc observationnelle.
  2. seule la pondération de sklearn a été testée, alors que la pratique la plus
     répandue dans la littérature médicale est le suréchantillonnage SMOTE.
  3. le recalibrage isotonique coûte de l'AUROC sur les petites cohortes, sans
     qu'on sache si Platt ferait mieux.

D'où :

A. GRADIENT DE PRÉVALENCE (expérience contrôlée). Une seule population, la plus
   grande cohorte, sous-échantillonnée à prévalence croissante et à effectif
   constant. Tout est tenu fixe sauf la prévalence : la relation dose-effet
   devient causale au lieu d'être une corrélation entre cohortes hétérogènes.

B. CINQ CORRECTIONS DU DÉSÉQUILIBRE. Aucune, pondération, suréchantillonnage
   aléatoire, sous-échantillonnage aléatoire, SMOTE. Si le dommage est commun
   aux cinq, le résultat concerne la pratique entière et non une option logicielle.

C. DEUX MÉTHODES DE RECALIBRAGE selon la taille d'échantillon. Platt contre
   isotonique, pour savoir laquelle recommander et à partir de quel effectif.

SORTIES : results/tables/wexp_*.csv, results/figures/wexp_*.png
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMB = True
except ImportError:
    HAS_IMB = False

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "harmonized"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
for d in (TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "target"
MODELS = ["logreg", "rf", "hgb"]
GRADIENT_PREV = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
GRADIENT_N = 6000
GRADIENT_REPS = 3
DCA_T = np.linspace(0.01, 0.60, 60)


# =============================================================== métriques
def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    idx = np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1])
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() / len(y)
                     for b in range(bins) if (idx == b).any()))


def slope_intercept(y, p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(lp, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def net_benefit(y, p, t):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= t
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return tp / len(y) - (fp / len(y)) * (t / (1 - t))


def metrics(y, p, prev):
    sl, ic = slope_intercept(y, p)
    nb = np.array([net_benefit(y, p, t) for t in DCA_T])
    return {"auroc": round(roc_auc_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4),
            "ece10": round(ece(y, p, 10), 4), "ece20": round(ece(y, p, 20), 4),
            "calib_slope": round(sl, 3), "calib_intercept": round(ic, 3),
            "mean_predicted": round(float(np.mean(p)), 4),
            "observed_rate": round(float(np.mean(y)), 4),
            "nb_at_prevalence": round(net_benefit(y, p, max(prev, 0.02)), 4),
            "nb_at_0.20": round(net_benefit(y, p, 0.20), 4),
            "nb_positive_up_to": round(float(DCA_T[nb > 0][-1]) if (nb > 0).any() else 0.0, 3)}


# =============================================================== modèles
def base_estimator(kind, weighted):
    cw = "balanced" if weighted else None
    if kind == "logreg":
        return LogisticRegression(C=1.0, max_iter=3000, class_weight=cw, random_state=SEED)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=5,
                                      n_jobs=4, random_state=SEED,
                                      class_weight="balanced_subsample" if weighted else None)
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.06,
                                              class_weight=cw, random_state=SEED)
    raise ValueError(kind)


def build(kind, correction):
    """correction ∈ {none, class_weight, oversample, undersample, smote}"""
    pre = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    if correction in ("none", "class_weight"):
        est = base_estimator(kind, correction == "class_weight")
        return Pipeline(pre + [("clf", est)])
    if not HAS_IMB:
        return None
    sampler = {"oversample": RandomOverSampler(random_state=SEED),
               "undersample": RandomUnderSampler(random_state=SEED),
               "smote": SMOTE(random_state=SEED)}[correction]
    return ImbPipeline(pre + [("sampler", sampler), ("clf", base_estimator(kind, False))])


def recalibrate(y, p, method, seed=SEED):
    """Recalibrage appris hors-échantillon, jamais sur les données évaluées."""
    y, p = np.asarray(y), np.asarray(p)
    out = np.zeros_like(p, dtype=float)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in cv.split(p.reshape(-1, 1), y):
        if method == "isotonic":
            m = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr])
            out[te] = m.predict(p[te])
        else:  # Platt
            lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            m = LogisticRegression(max_iter=1000).fit(lp[tr], y[tr])
            out[te] = m.predict_proba(lp[te])[:, 1]
    return out


def oof(model, X, y, seed=SEED):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")[:, 1]


# =============================================================== données
def load_all():
    out = {}
    for f in sorted(DATA.glob("*.csv")):
        df = pd.read_csv(f)
        feats = [c for c in df.columns if c != TARGET]
        out[f.stem] = (df[feats].reset_index(drop=True), df[TARGET].values.astype(int))
    return out


def subsample_to_prevalence(X, y, prev, n, rng):
    """Sous-échantillon à prévalence imposée et effectif constant."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = int(round(n * prev)); n_neg = n - n_pos
    if n_pos > len(pos) or n_neg > len(neg):
        k = min(len(pos) / prev, len(neg) / (1 - prev))
        n_pos, n_neg = int(k * prev), int(k * (1 - prev))
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                          rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx]


# =============================================================== A. gradient
def step_gradient(data, reps=GRADIENT_REPS, n=GRADIENT_N):
    """Une seule population, prévalence seule variable : la dose-réponse devient
    une expérience contrôlée et non une corrélation entre cohortes."""
    print("\n=== A. Gradient de prévalence (expérience contrôlée) ===")
    big = max(data, key=lambda c: len(data[c][1]))
    X0, y0 = data[big]
    print(f"  population source : {big} (n = {len(y0)}, prévalence {y0.mean():.3f})")
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(SEED + rep)
        for prev in GRADIENT_PREV:
            X, y = subsample_to_prevalence(X0, y0, prev, n, rng)
            for kind in MODELS:
                p_un = oof(build(kind, "none"), X, y, SEED + rep)
                p_w = oof(build(kind, "class_weight"), X, y, SEED + rep)
                arms = {"unweighted": p_un, "weighted": p_w,
                        "weighted + Platt": recalibrate(y, p_w, "platt", SEED + rep),
                        "weighted + isotonic": recalibrate(y, p_w, "isotonic", SEED + rep)}
                for arm, p in arms.items():
                    rows.append({"rep": rep, "target_prevalence": prev,
                                 "actual_prevalence": round(float(y.mean()), 4),
                                 "n": len(y), "algo": kind, "arm": arm, **metrics(y, p, prev)})
            print(f"    rep {rep}  prévalence {prev:.2f}  ({len(y)} patients)")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "wexp_A_prevalence_gradient.csv", index=False)

    g = df.groupby(["target_prevalence", "arm"])[["auroc", "ece10", "calib_intercept",
                                                  "nb_at_0.20"]].mean().round(4)
    print("\n  Moyennes par prévalence et par bras :")
    print(g.to_string())

    piv = df.pivot_table(index="target_prevalence", columns="arm", values="ece10")
    if {"unweighted", "weighted"} <= set(piv.columns):
        from scipy.stats import pearsonr
        d = abs(piv.index.values - 0.5)
        inc = (piv["weighted"] - piv["unweighted"]).values
        r, p = pearsonr(d, inc)
        print(f"\n  Dose-réponse contrôlée : r = {r:+.3f} (p = {p:.4f}) entre "
              f"|prévalence − 0,50| et la hausse d'ECE")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, metric, lab in zip(axes, ["auroc", "ece10", "nb_at_0.20"],
                               ["AUROC", "Expected calibration error",
                                "Net benefit at threshold 0.20"]):
        for arm in ["unweighted", "weighted", "weighted + Platt", "weighted + isotonic"]:
            s = df[df.arm == arm].groupby("target_prevalence")[metric].mean()
            ax.plot(s.index, s.values, marker="o", ms=4, lw=1.5, label=arm)
        ax.set_xlabel("Outcome prevalence"); ax.set_title(lab, fontsize=10)
        ax.grid(alpha=.25)
        if metric == "nb_at_0.20":
            ax.axhline(0, color="k", ls="--", lw=1)
    axes[0].legend(fontsize=7.5)
    fig.suptitle(f"Class weighting across a controlled prevalence gradient "
                 f"({big}, n = {n} per level)", fontsize=11)
    plt.tight_layout(); plt.savefig(FIGURES / "wexp_A_prevalence_gradient.png", dpi=300)
    plt.close(fig)
    return df


# =============================================================== B. corrections
def step_corrections(data, cap=20000):
    """Le dommage est-il propre à la pondération, ou commun à toutes les
    corrections du déséquilibre — dont SMOTE, la plus répandue ?"""
    print("\n=== B. Cinq corrections du déséquilibre ===")
    if not HAS_IMB:
        print("  [avertissement] imbalanced-learn absent : seules 'none' et "
              "'class_weight' seront testées (pip install imbalanced-learn)")
    corrections = ["none", "class_weight"] + (["oversample", "undersample", "smote"] if HAS_IMB else [])
    rows = []
    for cohort, (X, y) in data.items():
        if len(y) > cap:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(len(y), cap, replace=False)
            X, y = X.iloc[idx].reset_index(drop=True), y[idx]
            note = f" (sous-échantillonnée à {cap})"
        else:
            note = ""
        prev = float(y.mean())
        print(f"\n  {cohort}{note} — prévalence {prev:.3f}")
        for kind in MODELS:
            for corr in corrections:
                m = build(kind, corr)
                if m is None:
                    continue
                try:
                    p = oof(m, X, y)
                except Exception as e:
                    print(f"    [warn] {kind}/{corr}: {e}"); continue
                rows.append({"cohort": cohort, "prevalence": round(prev, 4), "n": len(y),
                             "algo": kind, "correction": corr, **metrics(y, p, prev)})
            r = [x for x in rows if x["algo"] == kind and x["cohort"] == cohort]
            print("    " + kind.ljust(7) + "  " +
                  "  ".join(f"{x['correction']}: AUROC {x['auroc']:.3f} ECE {x['ece10']:.3f}"
                            for x in r))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "wexp_B_corrections.csv", index=False)
    print("\n  Moyennes par correction (toutes cohortes, tous modèles) :")
    print(df.groupby("correction")[["auroc", "brier", "ece10", "calib_slope",
                                    "calib_intercept", "nb_at_0.20"]].mean().round(4).to_string())
    return df


# =============================================================== C. recalibrage
def step_recalibration_method(data):
    """Platt ou isotonique, et à partir de quel effectif ?"""
    print("\n=== C. Méthode de recalibrage selon la taille d'échantillon ===")
    rows = []
    for cohort, (X, y) in data.items():
        prev = float(y.mean())
        for kind in MODELS:
            p_w = oof(build(kind, "class_weight"), X, y)
            arms = {"weighted": p_w,
                    "weighted + Platt": recalibrate(y, p_w, "platt"),
                    "weighted + isotonic": recalibrate(y, p_w, "isotonic")}
            for arm, p in arms.items():
                rows.append({"cohort": cohort, "n": len(y), "prevalence": round(prev, 4),
                             "algo": kind, "arm": arm, **metrics(y, p, prev)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "wexp_C_recalibration_method.csv", index=False)
    piv = df.pivot_table(index=["cohort", "n"], columns="arm",
                         values=["auroc", "ece10"]).round(4)
    print(piv.sort_index(level="n").to_string())
    d = df.pivot_table(index=["cohort", "n"], columns="arm", values="auroc")
    if {"weighted", "weighted + Platt", "weighted + isotonic"} <= set(d.columns):
        d["cost_platt"] = d["weighted"] - d["weighted + Platt"]
        d["cost_isotonic"] = d["weighted"] - d["weighted + isotonic"]
        print("\n  Coût en AUROC du recalibrage, par taille de cohorte :")
        print(d[["cost_platt", "cost_isotonic"]].sort_index(level="n").round(4).to_string())
    return df


# =============================================================== driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all",
                    help="all | gradient | corrections | recalib")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    global GRADIENT_REPS, GRADIENT_N
    if a.quick:
        GRADIENT_REPS, GRADIENT_N = 1, 2000
    data = load_all()
    if not data:
        print(f"Aucune cohorte dans {DATA}"); return
    print(f"{len(data)} cohortes chargées : " +
          ", ".join(f"{c} (n={len(y)}, prév {y.mean():.3f})" for c, (_, y) in data.items()))
    if a.steps in ("all", "gradient"):
        step_gradient(data, GRADIENT_REPS, GRADIENT_N)
    if a.steps in ("all", "corrections"):
        step_corrections(data)
    if a.steps in ("all", "recalib"):
        step_recalibration_method(data)
    print(f"\nTerminé. Tableaux dans {TABLES}, figures dans {FIGURES}.")


if __name__ == "__main__":
    main()
