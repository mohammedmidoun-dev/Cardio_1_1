"""
prior_correction.py — les expériences qui manquent à l'article.

Ce fichier n'implémente aucune métrique, aucun modèle et aucun
sous-échantillonnage : il importe les vôtres depuis weighting_experiments.py.
Toute divergence de graine, d'hyperparamètre ou de définition est donc
impossible par construction, et les résultats sont directement comparables
à ceux de wexp_A.

Placez-le dans src/, à côté de weighting_experiments.py.

    python src/prior_correction.py --steps plugin
    python src/prior_correction.py --steps ci
    python src/prior_correction.py --steps mechanism
    python src/prior_correction.py --steps all --reps 10

Étapes
------
plugin     applique la correction analytique aux prédictions hors échantillon
           et mesure ce qu'elle laisse : test direct de la sur-correction que
           l'article déduit aujourd'hui de la pente.
ci         intervalles de confiance sur les facteurs d'atténuation.
mechanism  fait varier la régularisation à structure constante, pour séparer
           « propriété de la procédure d'ajustement » de « propriété de la
           classe de modèle ».
second     reconstruit un gradient réduit dans un deuxième jeu de données.
"""

import argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from weighting_experiments import (
    SEED, TABLES, MODELS, GRADIENT_PREV, GRADIENT_N, GRADIENT_REPS,
    load_all, subsample_to_prevalence, build, oof, recalibrate,
    metrics, slope_intercept, ece,
)

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def prior_shift(p, prev, alpha=1.0):
    """Correction analytique du prior.

    Un modèle entraîné comme si le prior valait 1/2 doit voir ses log-odds
    déplacés de logit(prévalence) pour revenir à l'échelle d'origine.
    alpha = 1 est la correction théorique complète, telle que publiée.
    """
    return expit(logit(p) + alpha * logit(prev))


def fit_alpha(sub):
    """Pente de l'ordonnée de calibration contre logit(prévalence)."""
    m = sub.groupby("target_prevalence").calib_intercept.mean()
    if len(m) < 3:
        return np.nan
    return float(np.polyfit(logit(np.asarray(m.index, float)), m.values, 1)[0])


# ============================================================ D. plug-in
def step_plugin(data, reps, n):
    """Gradient identique à wexp_A, plus deux bras de correction analytique."""
    print("\n=== D. Correction analytique du prior ===")
    big = max(data, key=lambda c: len(data[c][1]))
    X0, y0 = data[big]
    print(f"  population source : {big} (n = {len(y0)}, prévalence {y0.mean():.3f})")

    store, rows = {}, []
    for rep in range(reps):
        rng = np.random.default_rng(SEED + rep)
        for prev in GRADIENT_PREV:
            X, y = subsample_to_prevalence(X0, y0, prev, n, rng)
            for kind in MODELS:
                p_un = oof(build(kind, "none"), X, y, SEED + rep)
                p_w = oof(build(kind, "class_weight"), X, y, SEED + rep)
                store[(rep, prev, kind)] = (y, p_w, float(y.mean()))
                for arm, p in {"unweighted": p_un, "weighted": p_w}.items():
                    rows.append({"rep": rep, "target_prevalence": prev, "algo": kind,
                                 "arm": arm, **metrics(y, p, prev)})
            print(f"    rep {rep}  prévalence {prev:.2f}  ({len(y)} patients)")
    df = pd.DataFrame(rows)

    print("\n  Contrôle de reproduction (ordonnée, bras pondéré)")
    ref = TABLES / "wexp_A_prevalence_gradient.csv"
    if ref.exists():
        old = pd.read_csv(ref)
        o = old[old.arm == "weighted"].groupby("target_prevalence").calib_intercept.mean()
        nw = df[df.arm == "weighted"].groupby("target_prevalence").calib_intercept.mean()
        for prev in sorted(set(o.index) & set(nw.index)):
            d = abs(o[prev] - nw[prev])
            print(f"    prévalence {prev:.2f} : wexp_A {o[prev]:+.3f}   ici {nw[prev]:+.3f}"
                  f"   écart {d:.3f} {'ok' if d <= 0.05 else 'ECART'}")
    else:
        print(f"    {ref.name} introuvable, contrôle ignoré")

    alphas = {"all": fit_alpha(df[df.arm == "weighted"])}
    for kind, g in df[df.arm == "weighted"].groupby("algo"):
        alphas[kind] = fit_alpha(g)
    print("\n  Fraction du déplacement théorique exprimée")
    for k, v in alphas.items():
        print(f"    {k:8s} alpha = {v:.3f}   sur-correction si alpha=1 : {100*(1-v)/v:5.1f} %")

    extra = []
    for (rep, prev, kind), (y, p_w, actual) in store.items():
        held = df[(df.arm == "weighted") & (df.algo == kind) &
                  (df.target_prevalence != prev)]
        a_loo = fit_alpha(held)
        arms = {"weighted + prior_full": prior_shift(p_w, actual, 1.0),
                "weighted + Platt": recalibrate(y, p_w, "platt", SEED + rep),
                "weighted + isotonic": recalibrate(y, p_w, "isotonic", SEED + rep)}
        if a_loo == a_loo:
            arms["weighted + prior_fitted"] = prior_shift(p_w, actual, a_loo)
        for arm, p in arms.items():
            r = {"rep": rep, "target_prevalence": prev, "algo": kind, "arm": arm,
                 **metrics(y, p, prev)}
            if arm == "weighted + prior_fitted":
                r["alpha_used"] = round(a_loo, 4)
            extra.append(r)
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    df.to_csv(TABLES / "wexp_D_prior_correction.csv", index=False)

    print("\n  Ordonnée de calibration résiduelle (0 = calibré)")
    print(df.pivot_table(index="target_prevalence", columns="arm",
                         values="calib_intercept").round(3).to_string())
    print("\n  Erreur de calibration par famille et par bras")
    print(df.pivot_table(index="algo", columns="arm", values="ece10").round(4).to_string())
    print("\n  AUROC par bras")
    print(df.groupby("arm").auroc.mean().round(4).to_string())
    print(f"\n  Écrit : {TABLES / 'wexp_D_prior_correction.csv'}")
    print("\n  Lecture : une ordonnée résiduelle POSITIVE après prior_full, là où le")
    print("  bras pondéré en laissait une négative, signifie que la correction")
    print("  théorique a retiré davantage que le modèle n'avait déplacé.")
    return df


# ============================================================ D2. incertitude
def step_ci(df=None, B=2000):
    """Intervalles de confiance sur les facteurs d'atténuation, par bootstrap
    sur les répétitions à l'intérieur de chaque niveau de prévalence."""
    print("\n=== D2. Incertitude des facteurs d'atténuation ===")
    if df is None:
        src = TABLES / "wexp_D_prior_correction.csv"
        if not src.exists():
            src = TABLES / "wexp_A_prevalence_gradient.csv"
        df = pd.read_csv(src)
        print(f"  source : {src.name}")
    w = df[df.arm == "weighted"]
    reps = sorted(w.rep.unique())
    if len(reps) < 3:
        print(f"  {len(reps)} répétition(s) : intervalle peu interprétable, "
              "relancez avec --reps 10 au minimum")
    rng = np.random.default_rng(SEED)
    rows = []
    for kind in ["all"] + sorted(w.algo.unique()):
        sub = w if kind == "all" else w[w.algo == kind]
        point = fit_alpha(sub)
        draws = []
        for _ in range(B):
            parts = []
            for _prev, g in sub.groupby("target_prevalence"):
                idx = rng.choice(len(g), len(g), replace=True)
                parts.append(g.iloc[idx])
            a = fit_alpha(pd.concat(parts))
            if a == a:
                draws.append(a)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append({"algo": kind, "alpha": round(point, 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "n_obs": len(sub), "B": B})
        print(f"  {kind:8s} alpha = {point:.3f}  IC 95 % [{lo:.3f}, {hi:.3f}]")
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "wexp_D2_alpha_ci.csv", index=False)
    print(f"\n  Écrit : {TABLES / 'wexp_D2_alpha_ci.csv'}")
    print("  Si les intervalles des familles se chevauchent, l'ordonnancement")
    print("  logreg > boosting > forêt n'est pas établi et doit être présenté")
    print("  comme une tendance.")
    return out


# ============================================================ D3. mécanisme
def _grid_estimator(kind, setting, weighted):
    cw = "balanced" if weighted else None
    if kind == "logreg":
        return LogisticRegression(C=setting, max_iter=3000, class_weight=cw,
                                  random_state=SEED)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=250, max_depth=setting,
                                      min_samples_leaf=5, n_jobs=4, random_state=SEED,
                                      class_weight="balanced_subsample" if weighted else None)
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                              learning_rate=setting, class_weight=cw,
                                              random_state=SEED)
    raise ValueError(kind)


def _grid_pipeline(kind, setting, weighted):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("clf", _grid_estimator(kind, setting, weighted))])


GRID = {"logreg": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "rf": [3, 6, 12, None],
        "hgb": [0.02, 0.06, 0.2]}


def step_mechanism(data, n):
    """L'atténuation vient-elle de la régularisation ou de la classe de modèle ?

    Pour chaque réglage, le gradient complet est refait avec une répétition et
    le facteur d'atténuation réestimé. Si alpha ne bouge pas quand la
    régularisation varie, l'atténuation est une propriété de la classe.
    """
    print("\n=== D3. Régularisation ou classe de modèle ? ===")
    big = max(data, key=lambda c: len(data[c][1]))
    X0, y0 = data[big]
    rows = []
    for kind, settings in GRID.items():
        for st in settings:
            recs = []
            rng = np.random.default_rng(SEED)
            for prev in GRADIENT_PREV:
                X, y = subsample_to_prevalence(X0, y0, prev, n, rng)
                p_w = oof(_grid_pipeline(kind, st, True), X, y, SEED)
                _sl, ic = slope_intercept(y, p_w)
                recs.append({"target_prevalence": prev, "calib_intercept": ic})
            a = fit_alpha(pd.DataFrame(recs))
            rows.append({"algo": kind, "setting": str(st), "alpha": round(a, 4)})
            print(f"  {kind:7s} réglage {str(st):>6s}  alpha = {a:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "wexp_D3_mechanism.csv", index=False)
    print(f"\n  Écrit : {TABLES / 'wexp_D3_mechanism.csv'}")
    print("  Lecture : un alpha stable sur toute la grille d'un modèle indique")
    print("  que l'atténuation ne s'explique pas par la force de régularisation.")
    return out


# ============================================================ D4. second jeu
def step_second(data, n=2000, levels=(0.05, 0.10, 0.15, 0.20, 0.30)):
    """Gradient réduit dans un deuxième jeu, pour éprouver la généralité."""
    print("\n=== D4. Gradient dans un deuxième jeu de données ===")
    big = max(data, key=lambda c: len(data[c][1]))
    cands = {k: v for k, v in data.items() if k != big and len(v[1]) >= 1500}
    if not cands:
        print("  aucun autre jeu assez grand")
        return None
    name = max(cands, key=lambda c: len(cands[c][1]))
    X0, y0 = cands[name]
    print(f"  jeu : {name} (n = {len(y0)}, prévalence {y0.mean():.3f}, n cible {n})")
    rows = []
    rng = np.random.default_rng(SEED)
    for prev in levels:
        X, y = subsample_to_prevalence(X0, y0, prev, n, rng)
        for kind in MODELS:
            p_w = oof(build(kind, "class_weight"), X, y, SEED)
            _sl, ic = slope_intercept(y, p_w)
            rows.append({"dataset": name, "target_prevalence": prev,
                         "actual_prevalence": round(float(y.mean()), 4), "n": len(y),
                         "algo": kind, "calib_intercept": round(ic, 3),
                         "ece10": round(ece(y, p_w), 4)})
        print(f"    prévalence {prev:.2f}  ({len(y)} patients)")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "wexp_D4_second_dataset.csv", index=False)
    print("\n  Facteur d'atténuation dans ce jeu")
    for kind, g in df.groupby("algo"):
        print(f"    {kind:7s} alpha = {fit_alpha(g):.3f}")
    print(f"\n  Écrit : {TABLES / 'wexp_D4_second_dataset.csv'}")
    return df


# ============================================================ programme
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="plugin",
                    choices=["plugin", "ci", "mechanism", "second", "all"])
    ap.add_argument("--reps", type=int, default=GRADIENT_REPS)
    ap.add_argument("--n", type=int, default=GRADIENT_N)
    a = ap.parse_args()

    data = load_all()
    print(f"{len(data)} jeux : " + ", ".join(f"{k} (n={len(v[1])})" for k, v in data.items()))

    df = None
    if a.steps in ("plugin", "all"):
        df = step_plugin(data, a.reps, a.n)
    if a.steps in ("ci", "all"):
        step_ci(df)
    if a.steps in ("mechanism", "all"):
        step_mechanism(data, a.n)
    if a.steps in ("second", "all"):
        step_second(data)
    print(f"\nTerminé. Tableaux dans {TABLES}.")


if __name__ == "__main__":
    main()
