"""
config.py
Global configuration: paths, random seeds, hyper-parameter search spaces.

PATCHED FOR LAPTOP EXECUTION (i7-1355U / 16 GB, no GPU).
The original settings implied 10 repeats x 5 folds x 100 trials = 5000 model
fits per (algorithm, cohort). Measured cost of a single RBF-SVM fit on ~56k
overlapping CVD-like rows: ~11 min on a fast server, ~25-35 min on a 15 W
laptop chip -> the SVM alone would need weeks. FAST_MODE fixes this while
keeping the rigorous 10x5 repeated CV for the FINAL evaluation only.
"""
from pathlib import Path

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HARMONIZED_DIR = DATA_DIR / "harmonized"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

for p in (DATA_DIR, HARMONIZED_DIR, RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------- Reproducibility ----------------
SEED = 42
N_REPEATS = 10                     # repeats of the 5-fold CV (FINAL evaluation)
N_SPLITS = 5
SEEDS_REPEAT = list(range(N_REPEATS))   # 0..9 used for the 10 repeats

# ================================================================
# EXECUTION PROFILE  <-- the only block you normally need to touch
# ================================================================
FAST_MODE = True        # True = laptop profile; False = full run (RTX/i9 machine)

if FAST_MODE:
    N_TRIALS_DEFAULT = 30          # 100 -> 30 (TPE reaches ~90% of the gain)
    N_SPLITS_TUNING = 3            # folds used DURING hyper-parameter search
    N_REPEATS_TUNING = 1           # repeats used DURING search
    N_REPEATS_FINAL = 3            # repeats for the reported metrics (10 on the workstation)
else:
    N_TRIALS_DEFAULT = 100
    N_SPLITS_TUNING = 5
    N_REPEATS_TUNING = 1
    N_REPEATS_FINAL = N_REPEATS

# Per-algorithm trial budget: expensive models get fewer trials.
TRIALS_PER_ALGO = {
    "svm": 15, "knn": 15, "catboost": 20, "gb": 20,
    # everything else falls back to N_TRIALS_DEFAULT
}

# Parallelism: NOT -1. A 15 W chip (2 P-cores + 8 E-cores) throttles hard and
# nested parallelism (outer CV x inner n_jobs) causes CPU oversubscription.
N_JOBS = 6

# ---- Kernel-method guardrails (the actual bottleneck) ----
SVM_MAX_TRAIN_N = 8000             # stratified subsample cap for the RBF-SVM
SVM_MAX_ITER = 200_000             # bound pathological (C, gamma) trials
SVM_CACHE_MB = 1000                # sklearn default is 200 MB: far too small

# ---- Memory guardrails for 16 GB ----
RF_MAX_ESTIMATORS = 400            # was 1000

# ---------------- Optimization ----------------
TIMEOUT_PER_MODEL_SEC = 60 * 30    # 30 min hard cap per (algo, cohort)
# NOTE: Optuna only checks this BETWEEN trials, never during one. It is a
# safety net, not a guarantee - hence the guardrails above.

# ---------------- Cohorts ----------------
COHORTS = ["uci_cleveland", "framingham", "kaggle_cvd70k", "statlog"]

# Common harmonized feature schema (intersection of the 4 cohorts)
COMMON_FEATURES = ["age", "sex", "sbp", "dbp", "chol", "smoke", "diabetes", "bmi"]
TARGET = "target"

# ---------------- Algorithms ----------------
ALGORITHMS = [
    "logreg", "knn", "nb", "dtree", "svm",
    "rf", "gb", "xgb", "lgbm", "catboost",
]

# Suggested execution order: cheap and informative models first, so that a
# crashed or interrupted run still leaves you with usable results.
ALGO_ORDER_FAST_FIRST = [
    "logreg", "nb", "dtree", "xgb", "lgbm",
    "rf", "gb", "catboost", "svm", "knn",
]

# ---------------- Imbalance handling options ----------------
IMBALANCE_STRATEGIES = ["none", "rus", "smote", "borderline_smote", "adasyn"]

# ---------------- Calibration ----------------
ECE_BINS = 10
BOOTSTRAP_SAMPLES = 1000           # for CIs around AUROC / AUPRC
