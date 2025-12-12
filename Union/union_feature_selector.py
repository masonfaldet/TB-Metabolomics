#!/usr/bin/env python
"""
version 1 adds RUS with target majority ratio before SMOTE

union_feature_selector

Iterative union-of-sparse-solutions feature selection for TB omics, with
optional BorderlineSMOTE and random undersampling (RUS).

This module is designed to be called from an external driver script and
expects the TB star-schema utilities defined in `tb_query_and_preprocess.py`:

    - make_tb_dataset
    - filter_groupwise_missingness
    - make_tb_preprocess_pipeline
    - TBPreprocessConfig

High-level algorithm (per condition pair)
-----------------------------------------
For each pair of conditions, e.g. ("control", "activated"):

  1. Query the TB star-schema to obtain the TRAIN split only, restricted
     to these two conditions:

        X_df : samples × features (areas), indexed by sample_id
        y    : labels in {"control","activated"} (or other pair)

  2. Optionally filter features by group-wise missingness using
     `filter_groupwise_missingness` with group = label column
     (e.g. "condition"). This step is controlled via config. If the
     missingness threshold is None, this step is skipped.

  3. Build a preprocessing pipeline using `make_tb_preprocess_pipeline`,
     configured via `TBPreprocessConfig` stored inside this script's
     config. This applies:

        - log2 transform with train-aware pseudocount
        - osmolality + internal-standard normalization
        - label-agnostic 2-step imputation

  4. Optionally apply **BorderlineSMOTE** and/or **RUS**:

        • For feature selection:
            X_proc (preprocessed full train) -> SMOTE -> RUS

        • For evaluation inside CV:
            For each fold:
              - Fit preprocess pipe on raw train data only.
              - Transform train & val.
              - Apply SMOTE/RUS to *train only*.
              - Train classifier on resampled train; evaluate on val.

  5. Iteratively perform sparse feature selection using either:

        - L1ProxSVM (L1-penalized SVM via proximal gradient; y in {+1,-1})
        - Elastic-net logistic regression with a strong L1 component

     At each iteration:

        (a) Fit the sparse selector on the current feature set
            (after preprocess + optional SMOTE/RUS).
        (b) Record features with non-zero weights.
        (c) Evaluate those features via K-fold CV on raw data with
            fold-wise preprocessing + (optional) SMOTE/RUS on train folds.
        (d) If mean balanced accuracy >= threshold:
              • Add these features to the union with weight, iteration,
                and balanced accuracy.
              • Drop these features from the current feature set and repeat.
            Otherwise:
              • Stop iterating for this condition pair.

Outputs
-------
The main entry point is:

    run_union_feature_selection(config: UnionFeatureSelectorConfig) -> pd.DataFrame

Side effects (written to config.out_dir):

  - A CSV file (timestamped) with columns:
        compared_conditions, feature_id, weight, iteration, balanced_accuracy

  - For each condition pair, a PNG figure:
        iteration vs mean balanced accuracy

  - A JSON sidecar with the same base name as the CSV, storing the
    configuration used to generate the results (including SMOTE/RUS settings).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from Utilities.query_and_preprocess import (
    TBPreprocessConfig,
    filter_groupwise_missingness,
    make_tb_dataset,
    make_tb_preprocess_pipeline,
)

# Optional imbalanced-learn dependencies
try:
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    BorderlineSMOTE = None
    RandomUnderSampler = None


PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class UnionFeatureSelectorConfig:
    """
    Configuration for union-of-sparse-solutions feature selection.

    Attributes
    ----------
    db_root :
        Root directory containing the TB Parquet star-schema:
            samples.parquet, features.parquet, abundances.parquet.
    out_dir :
        Directory where CSV, JSON sidecar, and plots will be written.
    condition_pairs :
        List of condition pairs to discriminate between, e.g.
            [("control", "activated"), ("incident", "prevalent")].
        The label column is controlled by `label_col`.
    label_col :
        Column in the samples table to use as the binary label,
        typically "condition" or "symptomatic".

    Group-wise missingness filter
    -----------------------------
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

    Preprocessing configuration (TBPreprocessConfig)
    ------------------------------------------------
    preprocess :
        TBPreprocessConfig controlling log2, normalization, and imputation.

    Sparse selector specification
    -----------------------------
    selector_type :
        Either "l1_prox_svm" or "elastic_net". Controls which sparse model
        is used for feature selection and which standard classifier is used
        for evaluation (SVC or LR).

    L1ProxSVM hyperparameters
    -------------------------
    lambda_ :
        L1 regularization strength for L1ProxSVM.
    step_size :
        Optional fixed step size for L1ProxSVM; if None, a default based on
        data norm is used.
    l1_svm_max_iter :
        Maximum proximal-gradient iterations for L1ProxSVM.
    l1_svm_delta_tol :
        Convergence tolerance for L1ProxSVM.
    l1_svm_fit_intercept :
        Whether L1ProxSVM fits an intercept.

    Elastic-net selector hyperparameters
    ------------------------------------
    elasticnet_C :
        Inverse regularization strength for elastic-net LogisticRegression.
    elasticnet_l1_ratio :
        l1_ratio parameter for elastic-net LogisticRegression (0–1).
    elasticnet_max_iter :
        Max iterations for elastic-net LogisticRegression.

    Evaluation classifier hyperparameters
    -------------------------------------
    svc_C :
        C parameter for linear SVC used when selector_type="l1_prox_svm".
    lr_C :
        C parameter for standard LogisticRegression used when
        selector_type="elastic_net".

    CV / iteration / stopping
    -------------------------
    n_folds :
        Number of folds in StratifiedKFold for evaluating balanced accuracy.
    balanced_accuracy_threshold :
        Minimum mean balanced accuracy required to accept a set of features
        and move to the next iteration. If the threshold is not met, we stop
        iterating for that condition pair.
    max_iterations :
        Maximum number of feature-discovery iterations per condition pair.
    weight_tol :
        Threshold on absolute weight to consider a feature as "selected"
        (non-zero).
    random_state :
        Random seed for CV splitting and resampling; can be None.

    SMOTE / RUS configuration
    -------------------------
    use_borderline_smote :
        If True, apply BorderlineSMOTE *after* any RUS step.
    borderline_smote_sampling_strategy :
        Sampling strategy for BorderlineSMOTE (e.g. "auto", float, dict).
        In the common binary case with RUS enabled, this is usually left
        as "auto" to fully balance the classes.
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

    Bookkeeping
    -----------
    run_id :
        Optional string identifying the run. If None, a timestamp-based ID
        is generated when `run_union_feature_selection` is called.
    """

    db_root: PathLike
    out_dir: PathLike
    condition_pairs: List[Tuple[str, str]]

    label_col: str = "condition"

    # groupwise missingness (None -> skip)
    groupwise_min_prop: Optional[float] = 0.6
    groupwise_min_group_n: int = 3
    groupwise_require_all_groups: bool = False
    groupwise_cols: Optional[Sequence[str]] = None

    # preprocessing config
    preprocess: TBPreprocessConfig = field(default_factory=TBPreprocessConfig)

    # selector choice
    selector_type: str = "l1_prox_svm"  # or "elastic_net"

    # L1ProxSVM hyperparameters
    lambda_: float = 1e-3
    step_size: Optional[float] = None
    l1_svm_max_iter: int = 2000
    l1_svm_delta_tol: float = 1e-6
    l1_svm_fit_intercept: bool = True

    # elastic-net hyperparameters
    elasticnet_C: float = 0.1
    elasticnet_l1_ratio: float = 0.9
    elasticnet_max_iter: int = 5000

    # evaluation models
    svc_C: float = 1.0
    lr_C: float = 1.0

    # CV / iteration / stopping
    n_folds: int = 5
    balanced_accuracy_threshold: float = 0.6
    max_iterations: int = 10
    weight_tol: float = 1e-6
    random_state: Optional[int] = 42

    # SMOTE / RUS
    use_borderline_smote: bool = False
    borderline_smote_sampling_strategy: Union[str, float, dict] = "auto"
    borderline_smote_k_neighbors: int = 5
    borderline_smote_m_neighbors: int = 10

    use_random_undersampler: bool = False
    rus_sampling_strategy: Union[str, float, dict] = "auto"
    rus_target_majority_ratio: float = 5.0

    run_id: Optional[str] = None

    notes : str = ""

    def to_serializable_dict(self) -> dict:
        """
        Convert config (including nested TBPreprocessConfig) to a
        JSON-serializable dictionary.
        """
        d = asdict(self)
        d["db_root"] = str(self.db_root)
        d["out_dir"] = str(self.out_dir)
        return d


# ---------------------------------------------------------------------
# L1ProxSVM: L1-penalized linear SVM via proximal gradient
# ---------------------------------------------------------------------


class L1ProxSVM(BaseEstimator, ClassifierMixin):
    """
    Linear SVM with L1 penalty via proximal gradient on squared hinge loss:

        L(w, b) = (1/n) * sum_i max(0, 1 - y_i * (w·x_i + b))^2 + λ ||w||_1

    where y_i ∈ {+1, -1}. The L1 penalty is applied only to w; b is
    unregularized. Optimization uses proximal gradient:

        (w_{t+1}, b_{t+1}) = prox_{αλ||·||_1}( w_t - α∇_w f, b_t - α∇_b f )

    with convergence criterion:

        || [w_{t+1} - w_t; b_{t+1} - b_t] ||_2 <= delta_tol

    Parameters
    ----------
    lambda_ :
        L1 regularization strength λ.
    step_size :
        Fixed step size α. If None, uses α = 1 / (2 * R^2) with
        R = max_i ||x_i||_2.
    max_iter :
        Maximum number of proximal-gradient iterations.
    delta_tol :
        Tolerance on the change in parameters between iterations.
    fit_intercept :
        Whether to fit an intercept term b.
    record_history :
        If True, store per-iteration objective and parameter change
        in self.history_.

    Attributes
    ----------
    coef_ :
        Array of shape (1, n_features) with the learned weights.
    intercept_ :
        Scalar intercept b.
    history_ :
        Optional dict with keys "obj" and "delta" containing lists of
        objective values and parameter changes per iteration.
    """

    def __init__(
        self,
        lambda_: float = 1e-3,
        step_size: Optional[float] = None,
        max_iter: int = 2000,
        delta_tol: float = 1e-6,
        fit_intercept: bool = True,
        record_history: bool = False,
    ):
        self.lambda_ = float(lambda_)
        self.step_size = None if step_size is None else float(step_size)
        self.max_iter = int(max_iter)
        self.delta_tol = float(delta_tol)
        self.fit_intercept = bool(fit_intercept)
        self.record_history = bool(record_history)

        # learned params
        self.coef_: np.ndarray | None = None  # shape (1, p)
        self.intercept_: float = 0.0
        self.history_: dict | None = None

    # ---- helpers ----

    def _squared_hinge_grads(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
    ) -> Tuple[np.ndarray, float]:
        """
        Gradient of smooth part f(w, b) = (1/n) ∑ max(0, 1 - y * (Xw + b))^2.

        Returns
        -------
        grad_w, grad_b
        """
        n = X.shape[0]
        margins = y * (X @ w + (b if self.fit_intercept else 0.0))
        mask = margins < 1.0
        if not np.any(mask):
            return np.zeros_like(w), 0.0

        # For active points: loss_i = (1 - margin)^2
        r = 1.0 - margins[mask]  # shape (m,)
        y_mask = y[mask]
        X_mask = X[mask]

        # grad_w = -2/n ∑ y_i * r_i * x_i
        grad_w = -(2.0 / n) * (X_mask.T @ (y_mask * r))
        # grad_b = -2/n ∑ y_i * r_i
        grad_b = -(2.0 / n) * float(np.sum(y_mask * r))
        return grad_w, grad_b

    def _objective(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
        n = X.shape[0]
        margins = y * (X @ w + (b if self.fit_intercept else 0.0))
        losses = np.maximum(0.0, 1.0 - margins) ** 2
        return float(np.mean(losses) + self.lambda_ * np.sum(np.abs(w)))

    # ---- public API ----

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the L1-penalized SVM.

        Parameters
        ----------
        X :
            Array of shape (n_samples, n_features).
        y :
            Array of shape (n_samples,) with entries in {+1, -1}.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if set(np.unique(y)) - {-1.0, 1.0}:
            raise ValueError("L1ProxSVM expects y in {+1, -1}.")

        n, p = X.shape
        w = np.zeros(p, dtype=np.float32)
        b = 0.0

        # Step size: if unspecified, use 1/(2 R^2) with R = max_i ||x_i||_2
        if self.step_size is None:
            norms = np.linalg.norm(X, axis=1)
            R = float(np.max(norms)) if norms.size > 0 else 1.0
            alpha = 1.0 / (2.0 * R * R + 1e-12)
        else:
            alpha = self.step_size

        if self.record_history:
            history = {"obj": [], "delta": []}
        else:
            history = None

        for _ in range(self.max_iter):
            w_prev = w.copy()
            b_prev = float(b)

            grad_w, grad_b = self._squared_hinge_grads(X, y, w, b)

            # Gradient step
            w_temp = w - alpha * grad_w
            b_temp = b - alpha * grad_b

            # Proximal step for L1 on w only
            thresh = alpha * self.lambda_
            w = np.sign(w_temp) * np.maximum(np.abs(w_temp) - thresh, 0.0)
            b = b_temp  # no prox on intercept

            delta = float(
                np.linalg.norm(w - w_prev)
                + (abs(b - b_prev) if self.fit_intercept else 0.0)
            )

            if history is not None:
                obj = self._objective(X, y, w, b)
                history["obj"].append(obj)
                history["delta"].append(delta)

            if delta <= self.delta_tol:
                break

        self.coef_ = w.reshape(1, -1)
        self.intercept_ = float(b)
        self.history_ = history
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return X @ self.coef_.ravel() + (self.intercept_ if self.fit_intercept else 0.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

# ---------------------------------------------------------------------
# Resampling helpers (RUS + BorderlineSMOTE)
# ---------------------------------------------------------------------


def _apply_resampling_np(
    X: np.ndarray, y: np.ndarray, config: UnionFeatureSelectorConfig
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
            # Binary case with ratio-based behavior
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


def _apply_resampling_df(
    X_df: pd.DataFrame, y: np.ndarray, config: UnionFeatureSelectorConfig
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    DataFrame wrapper around `_apply_resampling_np`, preserving column names.

    Parameters
    ----------
    X_df :
        samples × features DataFrame.
    y :
        1D array of labels aligned with rows of X_df.

    Returns
    -------
    X_res_df, y_res :
        Resampled DataFrame and labels.
    """
    X_np = X_df.to_numpy(dtype=np.float32)
    X_res, y_res = _apply_resampling_np(X_np, y, config)
    X_res_df = pd.DataFrame(X_res, columns=X_df.columns)
    return X_res_df, y_res


def _to_preprocessed_df(
    X_proc, X_raw: pd.DataFrame, config: UnionFeatureSelectorConfig
) -> pd.DataFrame:
    """
    Convert the output of the preprocessing pipeline into a DataFrame
    with appropriate column names.

    Handles the common case where the pipeline drops the internal standard
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
        UnionFeatureSelectorConfig containing preprocess settings.

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

    if n_features_proc == n_features_raw:
        # One-to-one mapping: just reuse the raw columns
        return pd.DataFrame(X_proc, index=X_raw.index, columns=cols)

    # Handle the common case where internal standard was dropped
    if (
        config.preprocess.drop_internal_standard
        and config.preprocess.internal_standard_feature_id in cols
        and n_features_proc == n_features_raw - 1
    ):
        internal_id = config.preprocess.internal_standard_feature_id
        cols = [c for c in cols if c != internal_id]
        return pd.DataFrame(X_proc, index=X_raw.index, columns=cols)

    # If we get here, something else changed the feature dimension
    raise ValueError(
        "Preprocess pipeline changed feature dimension in an unexpected way: "
        f"raw had {n_features_raw} cols, preprocessed has {n_features_proc}."
    )

# ---------------------------------------------------------------------
# CV evaluation with fold-wise preprocess + optional SMOTE/RUS
# ---------------------------------------------------------------------


def _evaluate_iteration(
    X_raw: pd.DataFrame,
    y_int: np.ndarray,
    samples_df: pd.DataFrame,
    selected_cols: Sequence[int],
    selector_type: str,
    config: UnionFeatureSelectorConfig,
) -> float:
    """
    Evaluate a set of selected features via K-fold balanced accuracy,
    with preprocessing (log2, normalization, imputation) and optional
    SMOTE/RUS fit *inside* each CV fold to avoid data leakage.

    Workflow (per fold)
    -------------------
    1. Split X_raw, y_int into train / val.
    2. Build a TB preprocessing pipeline with make_tb_preprocess_pipeline.
    3. Fit the pipeline on the *raw* train data only; transform both train
       and val.
    4. Project the preprocessed train/val onto the selected feature columns.
    5. Optionally apply BorderlineSMOTE and/or RUS to the projected train
       data only (never to val).
    6. Fit a standard classifier on resampled train; evaluate balanced
       accuracy on val.

    Parameters
    ----------
    X_raw :
        Raw samples × features DataFrame (areas), after any missingness
        filtering. Columns are feature_ids.
    y_int :
        Integer labels aligned with X_raw rows (e.g., 0/1).
    samples_df :
        Samples metadata table aligned to X_raw (same sample_ids); used
        by the TBNormalizer inside the preprocessing pipeline.
    selected_cols :
        Sequence of feature_ids selected at this iteration (columns of
        X_raw / preprocessed X).
    selector_type :
        "l1_prox_svm" -> use SVC; "elastic_net" -> use LogisticRegression.
    config :
        Full configuration (used for CV, classifier, and resampling).

    Returns
    -------
    float
        Mean balanced accuracy across folds.
    """
    selected_cols = list(selected_cols)
    if len(selected_cols) == 0:
        return 0.0

    # Only keep selected columns that actually exist in raw X
    cols_in_raw = [c for c in selected_cols if c in X_raw.columns]
    if not cols_in_raw:
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

        # Fit on train-only, transform train and val
        X_tr_proc = preproc_pipe.fit_transform(X_tr_raw)
        X_val_proc = preproc_pipe.transform(X_val_raw)

        # Convert to DataFrames with correct columns, allowing for
        # optional internal-standard drop.
        X_tr_proc = _to_preprocessed_df(X_tr_proc, X_tr_raw, config)
        X_val_proc = _to_preprocessed_df(X_val_proc, X_val_raw, config)

        # Columns may have changed (e.g., internal standard dropped)
        cols_in_fold = [c for c in cols_in_raw if c in X_tr_proc.columns]
        if not cols_in_fold:
            # No surviving features in this fold; treat as chance-level
            scores.append(0.5)
            continue

        X_tr_sel = X_tr_proc.loc[:, cols_in_fold]
        X_val_sel = X_val_proc.loc[:, cols_in_fold]

        # Apply SMOTE/RUS on TRAIN ONLY
        X_tr_np = X_tr_sel.to_numpy(dtype=np.float32)
        X_tr_np_res, y_tr_res = _apply_resampling_np(X_tr_np, y_tr, config)
        X_val_np = X_val_sel.to_numpy(dtype=np.float32)

        # Evaluation classifier
        if selector_type == "l1_prox_svm":
            clf = SVC(
                kernel="linear",
                C=config.svc_C,
                class_weight="balanced",
            )
        elif selector_type == "elastic_net":
            clf = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                C=config.lr_C,
                class_weight="balanced",
                max_iter=1000,
            )
        else:
            raise ValueError(f"Unknown selector_type={selector_type!r}")

        clf.fit(X_tr_np_res, y_tr_res)
        y_pred = clf.predict(X_val_np)
        score = balanced_accuracy_score(y_val, y_pred)
        scores.append(float(score))

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------
# Core routine per condition pair
# ---------------------------------------------------------------------


def _run_for_pair(
    config: UnionFeatureSelectorConfig,
    cond_pair: Tuple[str, str],
) -> Tuple[pd.DataFrame, List[Tuple[int, float]]]:
    """
    Run iterative sparse feature discovery for a single condition pair.

    Parameters
    ----------
    config :
        Full configuration object.
    cond_pair :
        Tuple (cond_neg, cond_pos), i.e. the two levels of `label_col`
        to discriminate between.

    Returns
    -------
    selected_df :
        DataFrame of selected features for this pair with columns:
        [compared_conditions, feature_id, weight, iteration, balanced_accuracy].
    iter_stats :
        List of (iteration, mean_balanced_accuracy) for plotting.
    """
    cond_neg, cond_pos = cond_pair
    pair_label = f"{cond_neg}_vs_{cond_pos}"

    # ---- 1. Query DB for train split and these conditions ----
    X_df, y_raw, samples_sub, _features = make_tb_dataset(
        root=config.db_root,
        conditions=[cond_neg, cond_pos],
        split="train",
        label_col=config.label_col,
    )

    # Convert y to 0/1 (for evaluation) and ±1 (for L1ProxSVM)
    label_to_int = {cond_neg: 0, cond_pos: 1}
    y_int = pd.Series(y_raw).map(label_to_int).to_numpy()
    if np.any(pd.isna(y_int)):
        raise ValueError(
            f"Unexpected labels for pair {cond_pair}: "
            f"found labels {set(np.unique(y_raw))}"
        )

    # ---- 2. Optional groupwise missingness filter ----
    if config.groupwise_min_prop is not None:
        group_cols = (
            tuple(config.groupwise_cols)
            if config.groupwise_cols is not None
            else (config.label_col,)
        )
        X_df = filter_groupwise_missingness(
            X_df=X_df,
            samples_df=samples_sub,
            group_cols=group_cols,
            min_prop=config.groupwise_min_prop,
            min_group_n=config.groupwise_min_group_n,
            require_all_groups=config.groupwise_require_all_groups,
        )
        if X_df.shape[1] == 0:
            # No features left; return empty
            return pd.DataFrame(
                columns=[
                    "compared_conditions",
                    "feature_id",
                    "weight",
                    "iteration",
                    "balanced_accuracy",
                ]
            ), []

    # ---- 3. Preprocessing pipeline (no resampling yet) ----
    preproc_pipe = make_tb_preprocess_pipeline(
        samples_df=samples_sub,
        config=config.preprocess,
    )
    X_proc = preproc_pipe.fit_transform(X_df)
    X_proc = _to_preprocessed_df(X_proc, X_df, config)


    # ---- 4. Iterative sparse selection (with resampling) ----
    X_curr = X_proc.copy()
    selected_rows: List[dict] = []
    iter_stats: List[Tuple[int, float]] = []

    for iteration in range(1, config.max_iterations + 1):
        if X_curr.shape[1] == 0:
            break

        # Apply SMOTE/RUS on full current training data
        X_res_df, y_res_int = _apply_resampling_df(X_curr, y_int, config)

        # 4a. Fit sparse selector
        if config.selector_type == "l1_prox_svm":
            y_res_svm = 2 * y_res_int - 1
            selector = L1ProxSVM(
                lambda_=config.lambda_,
                step_size=config.step_size,
                max_iter=config.l1_svm_max_iter,
                delta_tol=config.l1_svm_delta_tol,
                fit_intercept=config.l1_svm_fit_intercept,
                record_history=False,
            )
            selector.fit(X_res_df.to_numpy(dtype=np.float32), y_res_svm)
            weights = selector.coef_.ravel()
        elif config.selector_type == "elastic_net":
            selector = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=config.elasticnet_C,
                l1_ratio=config.elasticnet_l1_ratio,
                class_weight="balanced",
                max_iter=config.elasticnet_max_iter,
            )
            selector.fit(X_res_df.to_numpy(dtype=np.float32), y_res_int)
            weights = selector.coef_.ravel()
        else:
            raise ValueError(f"Unknown selector_type={config.selector_type!r}")

        # Map weights back to X_curr columns (same order as X_res_df)
        abs_w = np.abs(weights)
        idx_nonzero = np.where(abs_w > config.weight_tol)[0]
        if idx_nonzero.size == 0:
            break

        selected_cols = list(X_curr.columns[idx_nonzero])

        # 4b. Evaluate with standard classifier using raw data +
        #     fold-wise preprocessing and SMOTE/RUS on train folds only
        mean_bacc = _evaluate_iteration(
            X_raw=X_df,
            y_int=y_int,
            samples_df=samples_sub,
            selected_cols=selected_cols,
            selector_type=config.selector_type,
            config=config,
        )
        iter_stats.append((iteration, mean_bacc))

        # 4c. Check threshold
        if mean_bacc < config.balanced_accuracy_threshold:
            # Stop iterating for this pair
            break

        # 4d. Record selected features
        for j, col in zip(idx_nonzero, selected_cols):
            selected_rows.append(
                {
                    "compared_conditions": pair_label,
                    "feature_id": int(col),
                    "weight": float(weights[j]),
                    "iteration": int(iteration),
                    "balanced_accuracy": float(mean_bacc),
                }
            )

        # 4e. Remove selected features from current matrix and continue
        X_curr = X_curr.drop(columns=selected_cols)

    if not selected_rows:
        selected_df = pd.DataFrame(
            columns=[
                "compared_conditions",
                "feature_id",
                "weight",
                "iteration",
                "balanced_accuracy",
            ]
        )
    else:
        selected_df = pd.DataFrame(selected_rows)

    return selected_df, iter_stats


# ---------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------


def _plot_iteration_curve(
    iter_stats: List[Tuple[int, float]],
    pair_label: str,
    out_path: Path,
) -> None:
    """
    Plot iteration vs mean balanced accuracy for a single condition pair.

    Parameters
    ----------
    iter_stats :
        List of (iteration, mean_balanced_accuracy).
    pair_label :
        String label used for the title (e.g. "control_vs_activated").
    out_path :
        Path (PNG) where the figure will be written.
    """
    if not iter_stats:
        return

    iters, baccs = zip(*iter_stats)
    plt.figure()
    plt.plot(iters, baccs, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Mean balanced accuracy")
    plt.title(pair_label)
    plt.grid(True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def run_union_feature_selection(config: UnionFeatureSelectorConfig) -> pd.DataFrame:
    """
    Run union-of-sparse-solutions feature selection for all condition pairs
    specified in `config.condition_pairs`.

    This function:
      - Queries the TB star-schema.
      - Applies preprocessing (log2, normalization, imputation).
      - Optionally applies BorderlineSMOTE and RUS.
      - Performs iterative sparse selection per condition pair.
      - Evaluates each iteration via K-fold balanced accuracy with
        fold-wise preprocessing and SMOTE/RUS on train folds only.
      - Aggregates selected features across pairs.
      - Writes:
          * a CSV of selected features,
          * per-pair iteration vs balanced-accuracy plots,
          * a JSON sidecar storing the configuration.

    Parameters
    ----------
    config :
        UnionFeatureSelectorConfig instance.

    Returns
    -------
    selected_all :
        Concatenated selected-features DataFrame across all condition pairs.
        Columns: compared_conditions, feature_id, weight, iteration,
        balanced_accuracy.
    """
    db_root = Path(config.db_root)
    if not db_root.exists():
        raise FileNotFoundError(f"db_root does not exist: {db_root}")

    # Assign a run_id if not already set
    if config.run_id is None:
        config.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path(f"{config.out_dir}/run_id__{config.run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_selected: List[pd.DataFrame] = []

    for cond_pair in config.condition_pairs:
        cond_neg, cond_pos = cond_pair
        pair_label = f"{cond_neg}_vs_{cond_pos}"
        print(f"[INFO] Running union feature selection for {pair_label}")

        selected_df, iter_stats = _run_for_pair(config, cond_pair)
        all_selected.append(selected_df)

        # Plot iteration curve
        plot_path = out_dir / f"bacc_vs_iter__{pair_label}__{config.run_id}.png"
        _plot_iteration_curve(iter_stats, pair_label, plot_path)

    if all_selected:
        selected_all = pd.concat(all_selected, axis=0, ignore_index=True)
    else:
        selected_all = pd.DataFrame(
            columns=[
                "compared_conditions",
                "feature_id",
                "weight",
                "iteration",
                "balanced_accuracy",
            ]
        )

    # Write CSV and JSON sidecar
    base_name = f"selected_features_union__{config.run_id}"
    csv_path = out_dir / f"{base_name}.csv"
    json_path = out_dir / f"{base_name}.json"

    selected_all.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(config.to_serializable_dict(), f, indent=2)

    print(f"[INFO] Wrote selected features to: {csv_path}")
    print(f"[INFO] Wrote config sidecar to:    {json_path}")

    return selected_all


if __name__ == "__main__":
    raise SystemExit(
        "union_feature_selector.py is intended to be imported and used from a "
        "driver script. See module docstring for example usage."
    )
