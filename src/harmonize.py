"""
harmonize.py
Harmonisation des 4 cohortes CVD vers un schéma commun.

CORRECTIONS APPORTÉES
---------------------
1. BUG statlog : la cible 'heart-disease' est codée {1 = absence, 2 = présence}
   dans le jeu Statlog, alors que les 3 autres cohortes utilisent {0, 1}.
   Symptôme observé : "prevalence=1.444" puis plantage de roc_curve
   ("y_true takes value in {1, 2}"). -> ramené à {0, 1}.

2. Cholestérol mis sur une échelle commune. Cleveland/Framingham donnent des
   mg/dL (150-400) tandis que Kaggle donne des catégories {1,2,3}. Empiler les
   deux dans la même colonne fausse la MATRICE DE TRANSFERT (la chute d'AUROC
   serait un artefact d'unités, pas une perte de généralisation réelle).
   -> tout est ramené aux catégories cliniques NCEP/ATP III :
        1 = désirable (<200 mg/dL), 2 = limite (200-239), 3 = élevé (>=240).
   Mettre CHOL_AS_CATEGORY = False pour revenir au comportement précédent.

3. Garde-fous : vérification du codage de la cible et rapport de complétude
   par cohorte (dbp/smoke/bmi sont absents de Cleveland et Statlog — à
   documenter dans les Méthodes du manuscrit).
"""
import numpy as np
import pandas as pd

TARGET = "target"

# Aligner le cholestérol sur l'échelle catégorielle commune (cf. point 2).
CHOL_AS_CATEGORY = True

# Seuils NCEP/ATP III (mg/dL)
CHOL_BINS = [-np.inf, 200, 240, np.inf]
CHOL_LABELS = [1, 2, 3]

FEATURES = ["age", "sex", "sbp", "dbp", "chol", "smoke", "diabetes", "bmi"]


def _chol_to_category(series_mgdl: pd.Series) -> pd.Series:
    """mg/dL -> {1, 2, 3}. Les NaN restent NaN."""
    s = pd.to_numeric(series_mgdl, errors="coerce")
    cat = pd.cut(s, bins=CHOL_BINS, labels=CHOL_LABELS, right=False)
    return pd.to_numeric(cat, errors="coerce").astype(float)


# =========================
# 1. DETECTION DATASET
# =========================
def detect_dataset(df):
    cols = set(df.columns)

    if "trestbps" in cols:
        return "cleveland"

    if "rest-bp" in cols:
        return "statlog"

    raise ValueError(f"Unknown dataset format: {df.columns}")


# =========================
# 2. CLEVELAND MAPPING
# =========================
def harmonize_cleveland(df):
    out = pd.DataFrame()

    out["age"] = df["age"].astype(float)
    out["sex"] = df["sex"].astype(int)

    out["sbp"] = df["trestbps"].astype(float)
    out["dbp"] = np.nan

    chol = df["chol"].astype(float)
    out["chol"] = _chol_to_category(chol) if CHOL_AS_CATEGORY else chol

    out["smoke"] = np.nan

    out["diabetes"] = (df["fbs"] > 0).astype(float)

    out["bmi"] = np.nan

    out[TARGET] = (df["num"].astype(int) > 0).astype(int)

    return out


# =========================
# 3. STATLOG MAPPING
# =========================
def harmonize_statlog(df):
    out = pd.DataFrame()

    out["age"] = df["age"].astype(float)
    out["sex"] = df["sex"].astype(int)

    out["sbp"] = df["rest-bp"].astype(float)
    out["dbp"] = np.nan

    chol = df["serum-chol"].astype(float)
    out["chol"] = _chol_to_category(chol) if CHOL_AS_CATEGORY else chol

    out["smoke"] = np.nan

    out["diabetes"] = (df["fasting-blood-sugar"] > 0).astype(float)

    out["bmi"] = np.nan

    # === CORRECTIF PRINCIPAL ===
    # Statlog : 1 = absence de maladie, 2 = présence  ->  {0, 1}
    hd = pd.to_numeric(df["heart-disease"], errors="coerce")
    if set(pd.unique(hd.dropna())) <= {1, 2}:
        out[TARGET] = (hd == 2).astype(int)
    else:                       # déjà en {0,1} (fichier corrigé en amont)
        out[TARGET] = (hd > 0).astype(int)

    return out


# =========================
# 4. FRAMINGHAM MAPPING
# =========================
def harmonize_framingham(df):
    out = pd.DataFrame()

    out["age"] = df["age"].astype(float)
    out["sex"] = df["male"].astype(int)
    out["sbp"] = df["sysBP"].astype(float)
    out["dbp"] = df["diaBP"].astype(float)

    chol = df["totChol"].astype(float)
    out["chol"] = _chol_to_category(chol) if CHOL_AS_CATEGORY else chol

    out["smoke"] = df["currentSmoker"].astype(float)
    out["diabetes"] = df["diabetes"].astype(float)
    out["bmi"] = df["BMI"].astype(float)
    out[TARGET] = df["TenYearCHD"].astype(int)

    return out


# =========================
# 5. KAGGLE CVD70K MAPPING
# =========================
def harmonize_kaggle_cvd70k(df):
    out = pd.DataFrame()

    # Âge en jours -> années
    out["age"] = (df["age"] / 365.25).astype(float)

    # Sexe: 1=femme, 2=homme -> 0=femme, 1=homme
    out["sex"] = (df["gender"] - 1).astype(int)

    out["sbp"] = df["ap_hi"].astype(float)
    out["dbp"] = df["ap_lo"].astype(float)

    # Déjà catégoriel {1,2,3} : c'est l'échelle de référence commune.
    out["chol"] = df["cholesterol"].astype(float)

    out["smoke"] = df["smoke"].astype(float)

    # glucose > 1 = diabète
    out["diabetes"] = (df["gluc"] > 1).astype(float)

    # BMI
    height_m = df["height"].astype(float) / 100
    out["bmi"] = df["weight"].astype(float) / (height_m ** 2)

    out[TARGET] = df["cardio"].astype(int)

    return out


# =========================
# 6. GARDE-FOUS
# =========================
def _check_target(out: pd.DataFrame, name: str):
    """La cible doit être binaire {0,1}. Attrape en 2 s ce qui a coûté 45 min."""
    u = set(pd.unique(out[TARGET].dropna()))
    if not u <= {0, 1}:
        raise ValueError(f"[{name}] cible mal codée : valeurs {sorted(u)} "
                         f"(attendu {{0, 1}})")
    prev = float(out[TARGET].mean())
    if not 0.0 < prev < 1.0:
        raise ValueError(f"[{name}] prévalence aberrante : {prev:.3f}")
    return prev


def _completeness(out: pd.DataFrame) -> dict:
    """Taux de complétude par variable — à reporter dans les Méthodes."""
    return {c: round(float(out[c].notna().mean()), 3)
            for c in FEATURES if c in out.columns}


# =========================
# 7. MAIN ENTRY
# =========================
def run_all():
    from pathlib import Path

    data_dir = Path("data")
    harmonized_dir = data_dir / "harmonized"
    harmonized_dir.mkdir(exist_ok=True)

    # (fichier source, fonction, séparateur, nom sortie)
    files = [
        ("uci_heart_cleveland_raw.csv", harmonize_cleveland, ",", "uci_cleveland.csv"),
        ("statlog_heart_raw.csv", harmonize_statlog, ",", "statlog.csv"),
        ("framingham.csv", harmonize_framingham, ",", "framingham.csv"),
        ("cardio_train.csv", harmonize_kaggle_cvd70k, ";", "kaggle_cvd70k.csv"),
    ]

    report = []
    for fname, fn, sep, out_name in files:
        path = data_dir / fname

        if not path.exists():
            print(f"  [skip] {fname}: file not found")
            continue

        print(f"Processing {fname}...")
        df = pd.read_csv(path, sep=sep)
        out = fn(df)

        prev = _check_target(out, out_name)          # <- garde-fou
        comp = _completeness(out)

        out_path = harmonized_dir / out_name
        out.to_csv(out_path, index=False)

        print(f"saved -> {out_path}   n={len(out)}  prevalence={prev:.3f}")
        empty = [c for c, v in comp.items() if v == 0.0]
        if empty:
            print(f"    [note] variables entièrement absentes : {', '.join(empty)}")
        report.append({"cohort": out_name.replace('.csv', ''),
                       "n": len(out), "prevalence": round(prev, 4), **comp})

    if report:
        rep = pd.DataFrame(report)
        rep_path = Path("results") / "tables" / "harmonization_report.csv"
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep.to_csv(rep_path, index=False)
        print(f"\nRapport de complétude -> {rep_path}")
        print(rep.to_string(index=False))


if __name__ == "__main__":
    run_all()
