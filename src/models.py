"""
models.py
Factory + Optuna hyper-parameter spaces for the 10 classical algorithms.

PATCHED FOR LAPTOP EXECUTION (i7-1355U / 16 GB, no GPU).
Changes vs original, all self-contained (pipeline.py needs NO modification):

  1. SVM  -> FastSVC wrapper: stratified subsampling + Platt scaling on a
     single held-out split instead of probability=True (which triggers an
     internal 5-fold CV, i.e. a hidden 5x cost). Weeks -> minutes.
  2. GB   -> HistGradientBoostingClassifier (same family, 10-50x faster).
  3. RF   -> n_estimators capped at 400 (was 1000): memory + time on 16 GB.
  4. kNN  -> n_jobs added (prediction is the bottleneck on 70k rows).
  5. All  -> n_jobs bounded (N_JOBS) rather than -1, to avoid CPU
     oversubscription and thermal throttling on a 15 W laptop chip.
"""
import numpy as np
import optuna
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import SEED

# Optional knobs; fall back to safe defaults if config.py is the original one.
try:
    from config import N_JOBS
except ImportError:
    N_JOBS = 6
try:
    from config import SVM_MAX_TRAIN_N, SVM_MAX_ITER, SVM_CACHE_MB
except ImportError:
    SVM_MAX_TRAIN_N, SVM_MAX_ITER, SVM_CACHE_MB = 8000, 200_000, 1000
try:
    from config import RF_MAX_ESTIMATORS
except ImportError:
    RF_MAX_ESTIMATORS = 400


# --------------------------------------------------------------------------
# Fast, tractable RBF-SVM
# --------------------------------------------------------------------------
class FastSVC(ClassifierMixin, BaseEstimator):
    # NOTE: ClassifierMixin MUST come first (sklearn >= 1.6 tag resolution),
    # otherwise the estimator is treated as a regressor and scoring fails.
    """RBF-SVM made tractable on large cohorts.

    Two costs are removed relative to sklearn's SVC(probability=True):

      * kernel-matrix blow-up: training is capped at `max_train_n` rows via a
        stratified subsample (SVM training is O(n^2)-O(n^3));
      * Platt calibration: sklearn refits the SVM 5 times internally when
        probability=True. Here the sigmoid is fitted once on a held-out
        split, so probabilities remain available at ~1.2x cost instead of 5x.

    Reporting note for the manuscript: "Owing to the quadratic complexity of
    kernel methods, the RBF-SVM was trained on a stratified subsample of
    N patients, with Platt scaling fitted on a held-out split."
    """

    def __init__(self, C=1.0, gamma=0.1, max_train_n=SVM_MAX_TRAIN_N,
                 calib_frac=0.2, cache_size=SVM_CACHE_MB,
                 max_iter=SVM_MAX_ITER, random_state=SEED):
        self.C = C
        self.gamma = gamma
        self.max_train_n = max_train_n
        self.calib_frac = calib_frac
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # 1. stratified subsample if the cohort is too large for a kernel method
        if self.max_train_n and len(X) > self.max_train_n:
            X, _, y, _ = train_test_split(
                X, y, train_size=int(self.max_train_n),
                stratify=y, random_state=self.random_state)

        # 2. hold out a slice for Platt scaling (single split, not 5-fold CV)
        n_min = np.bincount(y.astype(int)).min() if len(self.classes_) == 2 else len(y)
        can_calibrate = n_min >= 10 and len(y) >= 50
        if can_calibrate:
            X_fit, X_cal, y_fit, y_cal = train_test_split(
                X, y, test_size=self.calib_frac,
                stratify=y, random_state=self.random_state)
        else:
            X_fit, y_fit, X_cal, y_cal = X, y, None, None

        self.svc_ = SVC(C=self.C, gamma=self.gamma, kernel="rbf",
                        probability=False,               # no hidden 5-fold CV
                        cache_size=self.cache_size,      # 200 MB default is too small
                        max_iter=self.max_iter,          # bound pathological trials
                        class_weight="balanced",
                        random_state=self.random_state)
        self.svc_.fit(X_fit, y_fit)

        # 3. sigmoid on the decision function
        if can_calibrate:
            d = self.svc_.decision_function(X_cal).reshape(-1, 1)
            self.platt_ = LogisticRegression(max_iter=1000).fit(d, y_cal)
        else:
            self.platt_ = None
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(np.asarray(X, dtype=np.float32))

    def predict_proba(self, X):
        d = self.decision_function(X).reshape(-1, 1)
        if self.platt_ is not None:
            return self.platt_.predict_proba(d)
        p = 1.0 / (1.0 + np.exp(-d.ravel()))          # fallback
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


# --------------------------------------------------------------------------
# Optuna spaces
# --------------------------------------------------------------------------
def make_logreg(trial):
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = "liblinear"
    return LogisticRegression(C=C, penalty=penalty, solver=solver,
                              max_iter=1000, random_state=SEED)


def make_knn(trial):
    n = trial.suggest_int("n_neighbors", 3, 30)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    metric = trial.suggest_categorical("metric", ["minkowski", "manhattan"])
    return KNeighborsClassifier(n_neighbors=n, weights=weights, metric=metric,
                                n_jobs=N_JOBS)          # PATCH: parallel predict


def make_nb(trial):
    vs = trial.suggest_float("var_smoothing", 1e-12, 1e-2, log=True)
    return GaussianNB(var_smoothing=vs)


def make_dtree(trial):
    return DecisionTreeClassifier(
        max_depth=trial.suggest_int("max_depth", 3, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        random_state=SEED,
    )


def make_svm(trial):
    # PATCH: FastSVC instead of SVC(probability=True)
    return FastSVC(
        C=trial.suggest_float("C", 1e-2, 1e2, log=True),
        gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        max_train_n=SVM_MAX_TRAIN_N,
        random_state=SEED,
    )


def make_rf(trial):
    return RandomForestClassifier(
        # PATCH: 1000 -> RF_MAX_ESTIMATORS (400) and step 50 -> 100
        n_estimators=trial.suggest_int("n_estimators", 100, RF_MAX_ESTIMATORS, step=100),
        max_depth=trial.suggest_int("max_depth", 3, 30),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        n_jobs=N_JOBS, random_state=SEED,               # PATCH: bounded n_jobs
    )


def make_gb(trial):
    # PATCH: GradientBoostingClassifier -> HistGradientBoostingClassifier.
    # Same boosting family, histogram-based, 10-50x faster on 70k rows.
    return HistGradientBoostingClassifier(
        max_iter=trial.suggest_int("n_estimators", 100, 600, step=50),
        learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 1.0, log=True),
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=SEED,
    )


def make_xgb(trial):
    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        tree_method="hist",                              # PATCH: explicit hist
        eval_metric="logloss", n_jobs=N_JOBS, random_state=SEED, verbosity=0,
    )


def make_lgbm(trial):
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        n_jobs=N_JOBS, random_state=SEED, verbose=-1,
    )


def make_catboost(trial):
    return CatBoostClassifier(
        # PATCH: 800 -> 600 max, thread_count bounded
        iterations=trial.suggest_int("iterations", 200, 600, step=100),
        learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        depth=trial.suggest_int("depth", 4, 8),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        thread_count=N_JOBS,
        random_seed=SEED, verbose=False, allow_writing_files=False,
    )


FACTORY = {
    "logreg":   make_logreg,
    "knn":      make_knn,
    "nb":       make_nb,
    "dtree":    make_dtree,
    "svm":      make_svm,
    "rf":       make_rf,
    "gb":       make_gb,
    "xgb":      make_xgb,
    "lgbm":     make_lgbm,
    "catboost": make_catboost,
}
