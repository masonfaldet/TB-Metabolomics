#!/usr/bin/env python
"""
final_classifier

Fit and evaluate final classifiers for TB omics after union_feature_selector.

This module is designed to be called from an external driver script. It expects:

    1. A TB star-schema database in Parquet form:
         - samples.parquet
         - features.parquet
         - abundances.parquet

       produced by `tb_peak_to_parquets.py` and queried via
       `make_tb_dataset` from `tb_query_and_preprocess.py`.

    2. A selected-features CSV from `union_feature_selector.py`, e.g.:

         selected_features_union__20251206_140001.csv

       with columns:
         - compared_conditions (e.g. "control_vs_activated")
         - feature_id
         - iteration
         - weight
         - balanced_accuracy

For each unique `compared_conditions` value in the CSV, this script:

  1. Parses the corresponding condition pair (e.g. "control_vs_activated"
     -> ("control", "activated")).

  2. Queries the TB DB for the train and test splits restricted to this
     pair via `make_tb_dataset` (one call with split="train", another
     with split="test").

  3. For each pair, defines multiple models:

        (a) One model per *iteration* using the features discovered at
            that iteration, with `model = "iteration {k}"`.

        (b) An "ensemble" model combining the per-iteration classifiers
            via majority vote over their predictions.

        (c) A "union" model using the union of all features across all
            iterations (the original behavior).

  4. For each model, performs hyperparameter selection over a grid of C
     values for the chosen classifier family:

        - If classifier_type == "svc": linear SVC
        - If classifier_type == "logreg": logistic regression (L2)

     using Stratified K-fold cross-validation on the **train** split only.
     Inside each fold (for a given feature set), we now use a
     **preprocess → project** pattern:

        a. Fit the TB preprocessing pipeline on the **raw** train fold
           (log2, normalization, imputation, optional scaling), using all
           available feature columns (including any internal standard).

        b. Transform the raw train and validation folds with this
           pipeline.

        c. Project the preprocessed train and validation matrices onto
           the selected feature set. The internal-standard column, if
           present, may be dropped inside the pipeline and is never used
           as a predictive feature.

        d. Optionally apply RandomUnderSampler (RUS) and then
           BorderlineSMOTE to the *train fold only* (never to val).

        e. Fit the candidate classifier on the resampled train fold and
           evaluate balanced accuracy on the validation fold.

  5. For each model, fits a final classifier with the best C on the
     *entire* train split, again using the **preprocess → project**
     pattern:

        raw train  -> preprocessing pipeline fitted on raw train
                    -> transform raw test with the same pipeline
                    -> project preprocessed train/test to selected
                       feature IDs
                    -> optional RUS + BorderlineSMOTE on train only
                       (no resampling on test)
                    -> classifier fit on resampled train

  6. Evaluates each model on:
        - the train split (predictions on original, non-resampled train)
        - the test split from the DB (projected and preprocessed using
          the train-fitted pipeline)

     Metrics (binary, treating the second condition as the positive class):
        - balanced accuracy
        - precision
        - recall
        - F1 score

Outputs
-------
A CSV with one row per (condition_pair, model) and columns:

    condition_pair,
    model,              # "iteration {k}", "ensemble", or "union"
    best_C,
    n_features,
    train_bacc, train_precision, train_recall, train_f1,
    test_bacc,  test_precision,  test_recall,  test_f1

plus a JSON sidecar containing the configuration used.

Example usage (driver)
----------------------
    from final_classifier import (
        FinalClassifierConfig,
        run_final_classifiers,
    )
    from tb_query_and_preprocess import TBPreprocessConfig

    preprocess_cfg = TBPreprocessConfig(
        use_standard_scaler=True,
        # ... other preprocess knobs
    )

    config = FinalClassifierConfig(
        db_root="tb_parquets",
        selected_features_csv="results/union_fs/selected_features_union__20251206_140001.csv",
        out_dir="results/final_classifiers",
        classifier_type="svc",  # or "logreg"
        preprocess=preprocess_cfg,
        C_grid=[0.01, 0.1, 1.0, 10.0],
        use_borderline_smote=True,
        use_random_undersampler=True,
        rus_target_majority_ratio=5.0,
    )

    results_df = run_final_classifiers(config)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from Utilities.query_and_preprocess import (
    TBPreprocessConfig,
    make_tb_dataset,
    make_tb_preprocess_pipeline,
    filter_groupwise_missingness,
)

# Optional imbalanced-learn dependencies
try:
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:  # pragma: no cover
    BorderlineSMOTE = None
    RandomUnderSampler = None


PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class FinalClassifierConfig:
    """
    Configuration for final classifier fitting and evaluation.

    Attributes
    ----------
    db_root :
        Root directory containing the TB Parquet star-schema:
            samples.parquet, features.parquet, abundances.parquet.
    selected_features_csv :
        Path to the selected-features CSV produced by
        `union_feature_selector.run_union_feature_selection`, e.g.
        "selected_features_union__20251206_140001.csv".
    out_dir :
        Directory where the summary CSV and JSON sidecar will be written.

    label_col :
        Column in the samples table to use as the binary label,
        typically "condition" or "symptomatic".
        For each condition pair, labels are re-encoded to {0, 1} with
        the *second* condition treated as the positive class (1).

    classifier_type :
        "svc" for linear SVC, or "logreg" for logistic regression.
    cls_class_weight :
        "balanced" for unbalanced data (redundent if using SMOTE), None for standard weights
    cls_max_iter:
        -1 <-- no cap on iterations, stopping critera based only on convergence
    cls_tol:
        convergence criteria for fitting classifier
    preprocess :
        TBPreprocessConfig controlling log2, normalization, imputation,
        and optional standard scaling.

    C_grid :
        List of C values to search over in cross-validation. The best C
        is selected by mean balanced accuracy.

    n_folds :
        Number of folds in StratifiedKFold for hyperparameter search.
    random_state :
        Random seed for CV splitting and resampling; can be None.

    SMOTE / RUS configuration
    -------------------------
    use_borderline_smote :
        If True, apply BorderlineSMOTE *after* any RUS step.
    borderline_smote_sampling_strategy :
        Sampling strategy for BorderlineSMOTE (e.g. "auto", float, dict).
    borderline_smote_k_neighbors :
        Number of nearest neighbors for SMOTE.
    borderline_smote_m_neighbors :
        Number of nearest neighbors to determine "danger" samples.

    use_random_undersampler :
        If True, apply RandomUnderSampler *before* SMOTE, to enforce an
        upper bound on the majority:minority ratio.
    rus_sampling_strategy :
        Fallback sampling strategy for RandomUnderSampler (e.g. "auto",
        float, dict). Used for non-binary problems; ignored in the common
        binary case where `rus_target_majority_ratio` is used instead.
    rus_target_majority_ratio :
        Maximum allowed (majority : minority) ratio after RUS in the binary
        case. If the observed ratio is <= this value, RUS is skipped even
        when `use_random_undersampler` is True.

    Groupwise missingness filter
    ----------------------------
    groupwise_min_prop :
        Minimum proportion of observed values required in each group.
        If None, groupwise missingness filtering is skipped.
    groupwise_min_group_n :
        Minimum group size to enforce the groupwise missingness criterion.
    groupwise_require_all_groups :
        If True, a feature must pass in all sufficiently large groups;
        if False, it is kept if it passes in at least one.
    groupwise_cols :
        Columns in samples table defining groups. If None, defaults
        to (label_col,).

    run_id :
        Optional string identifying the run. If None, a timestamp-based ID
        is generated when `run_final_classifiers` is called.
    """

    db_root: PathLike
    selected_features_csv: PathLike
    out_dir: PathLike

    label_col: str = "condition"
    classifier_type: str = "svc"  # "svc" or "logreg"
    cls_class_weight: Optional[str] = "balanced"
    cls_max_iter: int = -1
    cls_tol: float = 1e-4

    preprocess: TBPreprocessConfig = field(default_factory=TBPreprocessConfig)

    C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0)

    n_folds: int = 5
    random_state: Optional[int] = 42

    # SMOTE / RUS
    use_borderline_smote: bool = False
    borderline_smote_sampling_strategy: Union[str, float, dict] = "auto"
    borderline_smote_k_neighbors: int = 5
    borderline_smote_m_neighbors: int = 10

    use_random_undersampler: bool = False
    rus_sampling_strategy: Union[str, float, dict] = "auto"
    rus_target_majority_ratio: float = 5.0

    # Groupwise missingness filter (optional; None -> skip)
    groupwise_min_prop: Optional[float] = 0.6
    groupwise_min_group_n: int = 3
    groupwise_require_all_groups: bool = False
    groupwise_cols: Optional[Sequence[str]] = None

    run_id: Optional[str] = None

    def to_serializable_dict(self) -> dict:
        """
        Convert config (including nested TBPreprocessConfig) to a
        JSON-serializable dictionary.
        """
        d = asdict(self)
        d["db_root"] = str(self.db_root)
        d["selected_features_csv"] = str(self.selected_features_csv)
        d["out_dir"] = str(self.out_dir)
        return d


# ---------------------------------------------------------------------
# Resampling helpers (RUS + BorderlineSMOTE)
# ---------------------------------------------------------------------


def _apply_resampling_np(
    X: np.ndarray, y: np.ndarray, config: FinalClassifierConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply RandomUnderSampler (RUS) and/or BorderlineSMOTE to (X, y) in numpy form.

    New order and semantics (binary case)
    -------------------------------------
    1. Optional RandomUnderSampler (if `use_random_undersampler` is True):

         - Compute the current majority:minority ratio r.
         - If r <= config.rus_target_majority_ratio, **skip RUS**.
         - If r > config.rus_target_majority_ratio, under-sample the
           majority class so that the post-RUS ratio is approximately
           `rus_target_majority_ratio : 1`.

       Example: target_majority_ratio = 5
         • n_majority = 20, n_minority = 10  → r = 2 ≤ 5
             => skip RUS.
         • n_majority = 100, n_minority = 10 → r = 10 > 5
             => under-sample majority to 50 (ratio 5:1).

    2. Optional BorderlineSMOTE (if `use_borderline_smote` is True):

         - Applied **after** any RUS step, to the (possibly under-sampled)
           training data.
         - With the default sampling_strategy="auto", the minority class
           is oversampled to match the majority, yielding a balanced set.

    Parameters
    ----------
    X, y :
        Arrays with shape (n_samples, n_features) and (n_samples,).
    config :
        Configuration controlling use and parameters of RUS/SMOTE.

    Returns
    -------
    X_res, y_res :
        Resampled arrays.
    """
    X_res, y_res = X, y

    # ---- 1. Optional RUS first ----
    if config.use_random_undersampler:
        if RandomUnderSampler is None:
            raise ImportError(
                "imblearn is required for RandomUnderSampler but is not installed. "
                "Install with `pip install imbalanced-learn` or disable "
                "use_random_undersampler in the config."
            )

        classes, counts = np.unique(y_res, return_counts=True)

        if len(classes) == 2 and config.rus_target_majority_ratio is not None:
            maj_idx = int(np.argmax(counts))
            min_idx = 1 - maj_idx
            maj_class = classes[maj_idx]
            min_class = classes[min_idx]
            n_maj = int(counts[maj_idx])
            n_min = int(counts[min_idx])

            if n_min > 0:
                ratio = n_maj / n_min
                target_ratio = float(config.rus_target_majority_ratio)

                if ratio > target_ratio:
                    # Desired majority count after RUS
                    n_maj_new = int(round(target_ratio * n_min))
                    n_maj_new = max(1, min(n_maj_new, n_maj))

                    sampling_strategy = {
                        min_class: n_min,
                        maj_class: n_maj_new,
                    }
                    rus = RandomUnderSampler(
                        sampling_strategy=sampling_strategy,
                        random_state=config.random_state,
                    )
                    X_res, y_res = rus.fit_resample(X_res, y_res)
            # If n_min == 0, dataset is degenerate; skip RUS
        else:
            # Fallback: non-binary / custom behavior
            rus = RandomUnderSampler(
                sampling_strategy=config.rus_sampling_strategy,
                random_state=config.random_state,
            )
            X_res, y_res = rus.fit_resample(X_res, y_res)

    # ---- 2. Optional BorderlineSMOTE after RUS ----
    if config.use_borderline_smote:
        if BorderlineSMOTE is None:
            raise ImportError(
                "imblearn is required for BorderlineSMOTE but is not installed. "
                "Install with `pip install imbalanced-learn` or disable "
                "use_borderline_smote in the config."
            )
        sm = BorderlineSMOTE(
            sampling_strategy=config.borderline_smote_sampling_strategy,
            random_state=config.random_state,
            k_neighbors=config.borderline_smote_k_neighbors,
            m_neighbors=config.borderline_smote_m_neighbors,
        )
        X_res, y_res = sm.fit_resample(X_res, y_res)

    return X_res, y_res


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _to_preprocessed_df(
    X_proc, X_raw: pd.DataFrame, config: FinalClassifierConfig
) -> pd.DataFrame:
    """
    Convert the output of the preprocessing pipeline into a DataFrame
    with appropriate column names.

    Handles the common case where the pipeline drops the internal-standard
    feature (e.g. feature_id = 9500) when
    `config.preprocess.drop_internal_standard` is True.

    Parameters
    ----------
    X_proc :
        Output of preproc_pipe.fit_transform or .transform (numpy array
        or DataFrame).
    X_raw :
        Raw input DataFrame before preprocessing. Used as a template for
        column names.
    config :
        FinalClassifierConfig containing preprocess settings.

    Returns
    -------
    X_proc_df :
        DataFrame with shape (n_samples, n_features_proc) and appropriate
        column names.
    """
    if isinstance(X_proc, pd.DataFrame):
        return X_proc

    # At this point X_proc is a numpy array
    n_samples_raw, n_features_raw = X_raw.shape
    n_samples_proc, n_features_proc = X_proc.shape

    if n_samples_proc != n_samples_raw:
        raise ValueError(
            f"Preprocess pipeline changed number of samples: "
            f"{n_samples_raw} -> {n_samples_proc}"
        )

    cols = list(X_raw.columns)

    # 1–1 mapping: just reuse the raw columns
    if n_features_proc == n_features_raw:
        return pd.DataFrame(X_proc, index=X_raw.index, columns=cols)

    # Common case: internal-standard feature dropped
    if (
        config.preprocess.drop_internal_standard
        and config.preprocess.internal_standard_feature_id in cols
        and n_features_proc == n_features_raw - 1
    ):
        internal_id = config.preprocess.internal_standard_feature_id
        cols = [c for c in cols if c != internal_id]
        return pd.DataFrame(X_proc, index=X_raw.index, columns=cols)

    # Anything else is unexpected
    raise ValueError(
        "Preprocess pipeline changed feature dimension in an unexpected way: "
        f"raw had {n_features_raw} cols, preprocessed has {n_features_proc}."
    )


def _parse_condition_pair(pair_label: str) -> Tuple[str, str]:
    """
    Parse a pair label of the form "control_vs_activated" into ("control", "activated").
    """
    if "_vs_" not in pair_label:
        raise ValueError(
            f"Invalid compared_conditions format: {pair_label!r}. "
            "Expected something like 'control_vs_activated'."
        )
    left, right = pair_label.split("_vs_", 1)
    return left, right


def _build_classifier(classifier_type: str, C: float, config: FinalClassifierConfig) -> Union[SVC, LogisticRegression]:
    """
    Instantiate a classifier of the requested family with regularization
    parameter C.

    Parameters
    ----------
    classifier_type :
        "svc" or "logreg".
    C :
        Inverse regularization strength.

    Returns
    -------
    clf :
        A scikit-learn estimator instance.
    """
    if classifier_type == "svc":
        return SVC(
            kernel="linear",
            C=C,
            class_weight=config.cls_class_weight,
            max_iter=config.cls_max_iter,
            tol=config.cls_tol,
        )
    elif classifier_type == "logreg":
        return LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=C,
            class_weight=config.cls_class_weight,
            max_iter=config.cls_max_iter,
            tol=config.cls_tol
        )
    else:
        raise ValueError(f"Unknown classifier_type={classifier_type!r}")


def _cv_score_for_C(
    X_raw: pd.DataFrame,
    y_int: np.ndarray,
    samples_df: pd.DataFrame,
    feature_ids: Sequence[int],
    config: FinalClassifierConfig,
    C: float,
) -> float:
    """
    Compute mean balanced accuracy for a given C via StratifiedKFold
    on the TRAIN split.

    Updated preprocess → project behavior
    ------------------------------------
    For each fold:

      1. Fit the TB preprocessing pipeline on the raw train fold
         (log2, normalization, imputation, optional scaling), using all
         available feature columns.

      2. Transform the raw train and validation folds with this pipeline
         and convert to DataFrames (handling optional internal-standard
         drop inside `_to_preprocessed_df`).

      3. Keep only the selected feature_ids that survive preprocessing
         for this fold.

      4. Optionally apply RUS + BorderlineSMOTE to the selected *train*
         data only (never to val).

      5. Fit a classifier with parameter C; evaluate balanced accuracy
         on the validation fold.

    Parameters
    ----------
    X_raw :
        Raw samples × features DataFrame (areas) for the TRAIN split,
        after any earlier filtering (e.g., missingness). Columns are
        feature_ids.
    y_int :
        Integer labels aligned with X_raw rows (0/1).
    samples_df :
        Samples metadata table for the TRAIN split, aligned to X_raw
        via sample_id; used by TBNormalizer inside the preprocessing
        pipeline.
    feature_ids :
        Sequence of feature_ids to project onto after preprocessing.
    config :
        FinalClassifierConfig (for n_folds, random_state, preprocess, classifier_type).
    C :
        Inverse regularization strength for the classifier.

    Returns
    -------
    mean_bacc :
        Mean balanced accuracy across folds.
    """
    feature_ids = sorted(set(int(fid) for fid in feature_ids))
    if len(feature_ids) == 0:
        return 0.0

    skf = StratifiedKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    scores: List[float] = []

    for train_idx, val_idx in skf.split(X_raw, y_int):
        X_tr_raw = X_raw.iloc[train_idx]
        X_val_raw = X_raw.iloc[val_idx]
        y_tr = y_int[train_idx]
        y_val = y_int[val_idx]

        # Build a fresh preprocessing pipeline for this fold
        preproc_pipe = make_tb_preprocess_pipeline(
            samples_df=samples_df,
            config=config.preprocess,
        )

        # Fit on raw train-only, transform raw train and val
        X_tr_proc = preproc_pipe.fit_transform(X_tr_raw)
        X_val_proc = preproc_pipe.transform(X_val_raw)

        # Convert to DataFrames with correct columns, allowing for
        # optional internal-standard drop.
        X_tr_proc = _to_preprocessed_df(X_tr_proc, X_tr_raw, config)
        X_val_proc = _to_preprocessed_df(X_val_proc, X_val_raw, config)

        # Keep only the selected feature columns that survive preprocessing
        cls_cols_in_fold = [fid for fid in feature_ids if fid in X_tr_proc.columns]
        if not cls_cols_in_fold:
            # Degenerate: no usable features in this fold
            scores.append(0.5)
            continue

        X_tr_sel = X_tr_proc.loc[:, cls_cols_in_fold]
        X_val_sel = X_val_proc.loc[:, cls_cols_in_fold]

        # Apply RUS/SMOTE on TRAIN ONLY
        X_tr_np = X_tr_sel.to_numpy(dtype=np.float32)
        X_val_np = X_val_sel.to_numpy(dtype=np.float32)

        X_tr_np_res, y_tr_res = _apply_resampling_np(X_tr_np, y_tr, config)

        clf = _build_classifier(config.classifier_type, C=C, config=config)
        clf.fit(X_tr_np_res, y_tr_res)
        y_pred = clf.predict(X_val_np)
        score = balanced_accuracy_score(y_val, y_pred)
        scores.append(float(score))

    return float(np.mean(scores)) if scores else 0.0


def _fit_model_for_feature_ids(
    *,
    config: FinalClassifierConfig,
    pair_label: str,
    model_label: str,
    X_train_filtered: pd.DataFrame,
    X_test_filtered: pd.DataFrame,
    samples_train: pd.DataFrame,
    y_train_int: np.ndarray,
    y_test_int: np.ndarray,
    feature_ids: Sequence[int],
) -> Tuple[dict, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit and evaluate a single model (for a specific feature set) for one
    condition pair.

    Returns both the result row and the train/test predictions so that
    higher-level callers can build ensembles over multiple models.

    New behavior:
      - Preprocessing is always fit on the *raw* (filtered) train matrix
        (all features) and applied to the raw test matrix.
      - Projection onto the selected `feature_ids` happens *after*
        preprocessing.
      - RUS/SMOTE is applied only to the preprocessed train data, never
        to the test data.
    """
    # Ensure unique, sorted feature_ids and restrict to columns present in TRAIN
    feature_ids = sorted(set(int(fid) for fid in feature_ids))
    kept_cols = list(X_train_filtered.columns)
    feature_ids = [fid for fid in feature_ids if fid in kept_cols]

    if len(feature_ids) == 0:
        # No usable features
        result = {
            "condition_pair": pair_label,
            "model": model_label,
            "best_C": np.nan,
            "n_features": 0,
            "train_bacc": np.nan,
            "train_precision": np.nan,
            "train_recall": np.nan,
            "train_f1": np.nan,
            "test_bacc": np.nan,
            "test_precision": np.nan,
            "test_recall": np.nan,
            "test_f1": np.nan,
        }
        return result, None, None

    # ---- Hyperparameter search over C using CV on TRAIN only ----
    best_C = None
    best_score = -np.inf

    for C in config.C_grid:
        cv_score = _cv_score_for_C(
            X_raw=X_train_filtered,
            y_int=y_train_int,
            samples_df=samples_train,
            feature_ids=feature_ids,
            config=config,
            C=float(C),
        )
        if cv_score > best_score or (cv_score == best_score and best_C is not None and C < best_C):
            best_score = cv_score
            best_C = float(C)

    if best_C is None:
        # Degenerate case: no CV scores computed
        result = {
            "condition_pair": pair_label,
            "model": model_label,
            "best_C": np.nan,
            "n_features": 0,
            "train_bacc": np.nan,
            "train_precision": np.nan,
            "train_recall": np.nan,
            "train_f1": np.nan,
            "test_bacc": np.nan,
            "test_precision": np.nan,
            "test_recall": np.nan,
            "test_f1": np.nan,
        }
        return result, None, None

    # ---- Fit final model on FULL train data with best C ----
    # New behavior: preprocess full train/test first, then project to feature_ids
    preproc_pipe = make_tb_preprocess_pipeline(
        samples_df=samples_train,
        config=config.preprocess,
    )

    # Fit on full raw (filtered) train, transform raw test
    X_train_proc = preproc_pipe.fit_transform(X_train_filtered)
    X_test_proc = preproc_pipe.transform(X_test_filtered)

    # Convert to DataFrames, handling optional internal-standard drop
    X_train_proc = _to_preprocessed_df(X_train_proc, X_train_filtered, config)
    X_test_proc = _to_preprocessed_df(X_test_proc, X_test_filtered, config)

    # Project preprocessed matrices onto selected feature_ids only
    cols_final = [fid for fid in feature_ids if fid in X_train_proc.columns]
    if not cols_final:
        result = {
            "condition_pair": pair_label,
            "model": model_label,
            "best_C": best_C,
            "n_features": 0,
            "train_bacc": np.nan,
            "train_precision": np.nan,
            "train_recall": np.nan,
            "train_f1": np.nan,
            "test_bacc": np.nan,
            "test_precision": np.nan,
            "test_recall": np.nan,
            "test_f1": np.nan,
        }
        return result, None, None

    Xtr_sel_df = X_train_proc.loc[:, cols_final]
    Xte_sel_df = X_test_proc.loc[:, cols_final]

    Xtr_sel = Xtr_sel_df.to_numpy(dtype=np.float32)
    Xte_sel = Xte_sel_df.to_numpy(dtype=np.float32)

    n_features = len(cols_final)

    # Apply RUS/SMOTE on FULL TRAIN ONLY for final fit
    Xtr_res, ytr_res = _apply_resampling_np(Xtr_sel, y_train_int, config)

    clf = _build_classifier(config.classifier_type, C=best_C, config=config)
    clf.fit(Xtr_res, ytr_res)

    # ---- Compute metrics on train (original) and test ----
    y_train_pred = clf.predict(Xtr_sel)
    y_test_pred = clf.predict(Xte_sel)

    train_bacc = balanced_accuracy_score(y_train_int, y_train_pred)
    test_bacc = balanced_accuracy_score(y_test_int, y_test_pred)

    # Precision, recall, F1: treat cond_pos (label 1) as the positive class
    train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
        y_train_int, y_train_pred, average="binary", pos_label=1, zero_division=0
    )
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_test_int, y_test_pred, average="binary", pos_label=1, zero_division=0
    )

    result = {
        "condition_pair": pair_label,
        "model": model_label,
        "best_C": best_C,
        "n_features": n_features,
        "train_bacc": float(train_bacc),
        "train_precision": float(train_prec),
        "train_recall": float(train_rec),
        "train_f1": float(train_f1),
        "test_bacc": float(test_bacc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
    }

    return result, y_train_pred.astype(int), y_test_pred.astype(int)


def _fit_and_evaluate_for_pair(
    config: FinalClassifierConfig,
    pair_label: str,
    features_by_iter: dict,
) -> List[dict]:
    """
    Fit and evaluate multiple models for a single condition pair:

        - One model per iteration ("iteration {k}")
        - An ensemble model over all iterations ("ensemble")
        - A union-of-features model ("union")

    All models now use a **preprocess → project** pattern:

        raw train/test  -> preprocessing pipeline
                        -> project preprocessed matrices onto the
                           relevant feature set
                        -> optional RUS/SMOTE on train only
                        -> classifier fit / evaluation
    """
    cond_neg, cond_pos = _parse_condition_pair(pair_label)

    # ---- 1. Query train and test splits for this pair ----
    X_train_raw, y_train_raw, samples_train, _ = make_tb_dataset(
        root=config.db_root,
        conditions=[cond_neg, cond_pos],
        split="train",
        label_col=config.label_col,
    )
    X_test_raw, y_test_raw, samples_test, _ = make_tb_dataset(
        root=config.db_root,
        conditions=[cond_neg, cond_pos],
        split="test",
        label_col=config.label_col,
    )

    # After the two make_tb_dataset calls
    samples_all = (
        pd.concat([samples_train, samples_test], axis=0, ignore_index=True)
        .drop_duplicates(subset=["sample_id"])
    )

    # Optional groupwise missingness filter on TRAIN ONLY
    if config.groupwise_min_prop is not None:
        group_cols = (
            tuple(config.groupwise_cols)
            if config.groupwise_cols is not None
            else (config.label_col,)
        )
        X_train_filtered = filter_groupwise_missingness(
            X_df=X_train_raw,
            samples_df=samples_all,
            group_cols=group_cols,
            min_prop=config.groupwise_min_prop,
            min_group_n=config.groupwise_min_group_n,
            require_all_groups=config.groupwise_require_all_groups,
        )
        kept_cols = list(X_train_filtered.columns)

        # Restrict TEST to the same feature set learned from TRAIN
        X_test_filtered = X_test_raw.loc[:, [c for c in kept_cols if c in X_test_raw.columns]]
    else:
        X_train_filtered = X_train_raw
        X_test_filtered = X_test_raw
        kept_cols = list(X_train_raw.columns)

    # Encode labels as 0/1 with cond_neg -> 0, cond_pos -> 1
    label_to_int = {cond_neg: 0, cond_pos: 1}
    y_train_int = pd.Series(y_train_raw).map(label_to_int).to_numpy()
    y_test_int = pd.Series(y_test_raw).map(label_to_int).to_numpy()

    if np.any(pd.isna(y_train_int)) or np.any(pd.isna(y_test_int)):
        raise ValueError(
            f"Unexpected labels encountered for pair {pair_label}. "
            f"Train labels: {set(np.unique(y_train_raw))}, "
            f"Test labels:  {set(np.unique(y_test_raw))}"
        )

    # Build union of all features across iterations and intersect with kept_cols
    all_feature_ids = []
    for it, fids in features_by_iter.items():
        all_feature_ids.extend(list(fids))
    union_feature_ids = sorted(set(int(fid) for fid in all_feature_ids))
    union_feature_ids = [fid for fid in union_feature_ids if fid in kept_cols]

    results_rows: List[dict] = []

    # Store preds for iteration models for ensemble
    iter_train_preds: List[np.ndarray] = []
    iter_test_preds: List[np.ndarray] = []

    # ---- 2. Per-iteration models ----
    for it in sorted(features_by_iter.keys()):
        model_label = f"iteration {int(it)}"
        feature_ids_it = features_by_iter[it]

        row_it, ytr_pred_it, yte_pred_it = _fit_model_for_feature_ids(
            config=config,
            pair_label=pair_label,
            model_label=model_label,
            X_train_filtered=X_train_filtered,
            X_test_filtered=X_test_filtered,
            samples_train=samples_all,
            y_train_int=y_train_int,
            y_test_int=y_test_int,
            feature_ids=feature_ids_it,
        )
        results_rows.append(row_it)

        if ytr_pred_it is not None and yte_pred_it is not None:
            iter_train_preds.append(ytr_pred_it)
            iter_test_preds.append(yte_pred_it)

    # ---- 3. Union-of-features model ----
    if len(union_feature_ids) > 0:
        row_union, ytr_union, yte_union = _fit_model_for_feature_ids(
            config=config,
            pair_label=pair_label,
            model_label="union",
            X_train_filtered=X_train_filtered,
            X_test_filtered=X_test_filtered,
            samples_train=samples_all,
            y_train_int=y_train_int,
            y_test_int=y_test_int,
            feature_ids=union_feature_ids,
        )
        results_rows.append(row_union)
        union_n_features = row_union.get("n_features", 0)
    else:
        # No union features; still record a row
        row_union = {
            "condition_pair": pair_label,
            "model": "union",
            "best_C": np.nan,
            "n_features": 0,
            "train_bacc": np.nan,
            "train_precision": np.nan,
            "train_recall": np.nan,
            "train_f1": np.nan,
            "test_bacc": np.nan,
            "test_precision": np.nan,
            "test_recall": np.nan,
            "test_f1": np.nan,
        }
        results_rows.append(row_union)
        union_n_features = 0

    # ---- 4. Ensemble over iteration models ----
    if iter_train_preds:
        preds_tr = np.stack(iter_train_preds, axis=0)  # (M, n_train)
        preds_te = np.stack(iter_test_preds, axis=0)   # (M, n_test)
        M = preds_tr.shape[0]

        # Majority vote (ties broken in favor of the positive class)
        votes_tr = preds_tr.sum(axis=0)
        votes_te = preds_te.sum(axis=0)
        y_train_pred_ens = (votes_tr >= (M / 2.0)).astype(int)
        y_test_pred_ens = (votes_te >= (M / 2.0)).astype(int)

        train_bacc = balanced_accuracy_score(y_train_int, y_train_pred_ens)
        test_bacc = balanced_accuracy_score(y_test_int, y_test_pred_ens)

        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            y_train_int, y_train_pred_ens, average="binary", pos_label=1, zero_division=0
        )
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            y_test_int, y_test_pred_ens, average="binary", pos_label=1, zero_division=0
        )

        row_ens = {
            "condition_pair": pair_label,
            "model": "ensemble",
            "best_C": np.nan,                 # no single C for an ensemble
            "n_features": union_n_features,   # effective feature budget ~ union
            "train_bacc": float(train_bacc),
            "train_precision": float(train_prec),
            "train_recall": float(train_rec),
            "train_f1": float(train_f1),
            "test_bacc": float(test_bacc),
            "test_precision": float(test_prec),
            "test_recall": float(test_rec),
            "test_f1": float(test_f1),
        }
    else:
        # No usable iteration models for ensemble
        row_ens = {
            "condition_pair": pair_label,
            "model": "ensemble",
            "best_C": np.nan,
            "n_features": union_n_features,
            "train_bacc": np.nan,
            "train_precision": np.nan,
            "train_recall": np.nan,
            "train_f1": np.nan,
            "test_bacc": np.nan,
            "test_precision": np.nan,
            "test_recall": np.nan,
            "test_f1": np.nan,
        }

    results_rows.append(row_ens)

    return results_rows


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def run_final_classifiers(config: FinalClassifierConfig) -> pd.DataFrame:
    """
    Run final classifier fitting and evaluation for all condition pairs
    present in the selected-features CSV.

    This function:
      - Reads the selected-features CSV.
      - For each unique compared_conditions value:
          * parses the condition pair,
          * groups selected features by iteration,
          * fits one model per iteration ("iteration {k}"),
          * fits an ensemble model over all iterations ("ensemble"),
          * fits a union-of-features model ("union"),
          * for each model, performs CV hyperparameter search for the
            chosen classifier family (svc or logreg) with fold-wise
            **preprocess → project** behavior and optional RUS/SMOTE on
            train folds only, then fits a final model on the full train
            split (again preprocess → project) and evaluates on train and
            DB test splits.
      - Aggregates all results into a single DataFrame.
      - Writes a CSV and JSON sidecar to `config.out_dir`.

    Parameters
    ----------
    config :
        FinalClassifierConfig instance.

    Returns
    -------
    results_df :
        DataFrame with one row per (condition_pair, model) and columns:

            condition_pair,
            model, best_C, n_features,
            train_bacc, train_precision, train_recall, train_f1,
            test_bacc,  test_precision, test_recall, test_f1
    """
    db_root = Path(config.db_root)
    if not db_root.exists():
        raise FileNotFoundError(f"db_root does not exist: {db_root}")

    selected_path = Path(config.selected_features_csv)
    if not selected_path.exists():
        raise FileNotFoundError(
            f"selected_features_csv does not exist: {selected_path}"
        )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Assign a run_id if not already set
    if config.run_id is None:
        config.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Read selected features
    sel = pd.read_csv(selected_path)
    required_cols = {"compared_conditions", "feature_id", "iteration"}
    if not required_cols.issubset(sel.columns):
        raise ValueError(
            f"selected_features_csv must contain columns {required_cols}, "
            f"found {set(sel.columns)}"
        )

    results_rows: List[dict] = []

    for pair_label, grp in sel.groupby("compared_conditions"):
        print(f"[INFO] Fitting final classifiers for {pair_label}")

        # Map iteration -> feature_ids for this pair
        features_by_iter = {
            int(it): grp.loc[grp["iteration"] == it, "feature_id"].astype(int).unique()
            for it in sorted(grp["iteration"].unique())
        }

        pair_rows = _fit_and_evaluate_for_pair(
            config=config,
            pair_label=pair_label,
            features_by_iter=features_by_iter,
        )
        results_rows.extend(pair_rows)

    if results_rows:
        results_df = pd.DataFrame(results_rows)
    else:
        results_df = pd.DataFrame(
            columns=[
                "condition_pair",
                "model",
                "best_C",
                "n_features",
                "train_bacc",
                "train_precision",
                "train_recall",
                "train_f1",
                "test_bacc",
                "test_precision",
                "test_recall",
                "test_f1",
            ]
        )

    # Write CSV and JSON sidecar
    base_name = f"final_classifiers__{config.run_id}"
    csv_path = out_dir / f"{base_name}.csv"
    json_path = out_dir / f"{base_name}.json"

    results_df.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(config.to_serializable_dict(), f, indent=2)

    print(f"[INFO] Wrote final classifier results to: {csv_path}")
    print(f"[INFO] Wrote config sidecar to:         {json_path}")

    return results_df


if __name__ == "__main__":
    raise SystemExit(
        "final_classifier.py is intended to be imported and used from a "
        "driver script. See module docstring for example usage."
    )
