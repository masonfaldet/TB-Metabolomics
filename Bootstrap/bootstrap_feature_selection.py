#!/usr/bin/env python
"""
bootstrap_stability_selection

Bootstrap-based stability selection for TB omics.

High-level algorithm (per condition pair)
-----------------------------------------
For each (cond_a, cond_b) in `config.condition_pairs`:

  1. Query the TB star-schema *train split* restricted to these two
     conditions using `make_tb_dataset`.

  2. Optionally filter features by group-wise missingness using
     `filter_groupwise_missingness` with groups defined by
     `config.groupwise_cols` (defaults to (label_col,)). This step is
     skipped if `config.groupwise_min_prop is None`.

  3. Build a preprocessing pipeline with `make_tb_preprocess_pipeline`,
     fit it on the (optionally filtered) raw train data, and transform
     the entire train split once to obtain a preprocessed matrix
     X_proc (samples × features). This is the matrix on which all
     bootstraps are performed.

  4. For b in range(config.bootstrap):
        4a. Draw a bootstrap sample of the rows of X_proc (with
            replacement) and the corresponding labels y.
        4b. Optionally apply random undersampling (RUS) and/or
            BorderlineSMOTE to the bootstrap sample, using the same
            helper as in `union_feature_selector` so behavior is
            consistent.
        4c. Fit either:
                - L1ProxSVM  (selector_type = "l1_prox_svm")
                - Elastic-net logistic regression
            on the (resampled) bootstrap data.
        4d. Record which features have |weight| > config.weight_tol
            and increment their selection counts.

  5. After all bootstraps, write a *selected-features* CSV with
        condition_pair, feature_id, n_selections
     and a JSON sidecar containing the serialized configuration.

  6. For k = 1, ..., max(n_selections):
        6a. Define a feature set F_k consisting of all features that
            were selected in at least k bootstraps.
        6b. Using StratifiedKFold with `config.val_folds`, perform
            CV on the *raw* filtered data:
              - In each fold, fit a fresh TB preprocessing pipeline
                on the train fold only.
              - Transform train and val, then project onto F_k.
              - Optionally apply RUS/BorderlineSMOTE to the train
                data only (never val).
              - Fit an evaluation classifier:
                    * if selector_type="l1_prox_svm": linear SVC
                    * if selector_type="elastic_net" : logistic regression
              - Compute balanced accuracy on both the train and val
                fold (using the *original* train/val data, not the
                resampled train).
        6c. Average balanced accuracy across folds to obtain
            train_mean_bacc and val_mean_bacc for this k.

     Results across all k are written to a *validation* CSV with
        condition_pair, min_selections, train_mean_bacc, val_mean_bacc
     and a per-pair PNG plot of
        min_selections vs train_mean_bacc  (solid line)
        min_selections vs val_mean_bacc    (dashed line)

All CSVs, JSONs, and plots are written to `config.out_dir` with
`config.run_id` appended to their base names to distinguish runs.

This module is intended to be imported and called from a driver script,
not executed directly.
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

from Union.union_feature_selector import (
    L1ProxSVM,
    _apply_resampling_np,
    _apply_resampling_df,
    _to_preprocessed_df,
)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class BootstrapStabilitySelectionConfig:
    """
    Configuration for bootstrap-based stability selection.

    Attributes
    ----------
    db_root :
        Root directory containing the TB Parquet star-schema:
            samples.parquet, features.parquet, abundances.parquet.
    out_dir :
        Directory where CSVs, JSON sidecar, and plots will be written.
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
        TBPreprocessConfig controlling normalization, imputation,
        and log transform.

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

    Bootstrapping and CV
    --------------------
    bootstrap :
        Number of bootstrap iterations for stability selection.
    val_folds :
        Number of folds in StratifiedKFold for evaluating balanced accuracy
        as a function of min_selections.
    weight_tol :
        Threshold on absolute weight to consider a feature as "selected"
        (non-zero).
    random_state :
        Random seed for bootstrapping, CV splitting, and resampling.

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

    Bookkeeping
    -----------
    run_id :
        Optional string identifying the run. If None, a timestamp-based ID
        is generated when `run_bootstrap_stability_selection` is called.
    notes :
        Free-form notes about the run (e.g., dataset version, comments).
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
    elasticnet_tol: float = 1e-5

    # evaluation models
    svc_C: float = 1.0
    lr_C: float = 1.0

    # bootstrapping / CV / selection
    bootstrap: int = 100
    val_folds: int = 5
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

    notes: str = ""

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
# CV helper for a fixed feature set
# ---------------------------------------------------------------------


def _evaluate_feature_set_cv(
    X_raw: pd.DataFrame,
    y_int: np.ndarray,
    samples_df: pd.DataFrame,
    feature_ids: Sequence[int],
    selector_type: str,
    config: BootstrapStabilitySelectionConfig,
) -> Tuple[float, float]:
    """
    Evaluate a fixed feature set via K-fold balanced accuracy on both
    train and validation folds.

    Workflow (per fold)
    -------------------
    1. Split X_raw, y_int into train / val indices using StratifiedKFold.
    2. Build a TB preprocessing pipeline with make_tb_preprocess_pipeline.
    3. Fit the pipeline on the *raw* train data only; transform both
       train and val.
    4. Project the preprocessed train/val onto the selected feature IDs.
    5. Optionally apply BorderlineSMOTE and/or RUS to the projected train
       data only (never to val).
    6. Fit a standard classifier on resampled train; evaluate balanced
       accuracy on:
          - original train fold (pre-resampling),
          - validation fold.

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
    feature_ids :
        Sequence of feature_ids defining the feature set.
    selector_type :
        "l1_prox_svm" -> use SVC; "elastic_net" -> logistic regression.
    config :
        Full configuration (used for CV, classifier, and resampling).

    Returns
    -------
    train_mean_bacc, val_mean_bacc :
        Mean balanced accuracies across folds on train and validation sets.
    """
    feature_ids = list(feature_ids)
    if len(feature_ids) == 0:
        return 0.0, 0.0

    skf = StratifiedKFold(
        n_splits=config.val_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    train_scores: List[float] = []
    val_scores: List[float] = []

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
        cols_in_fold = [c for c in feature_ids if c in X_tr_proc.columns]
        if not cols_in_fold:
            # No surviving features in this fold; treat as chance-level
            train_scores.append(0.5)
            val_scores.append(0.5)
            continue

        X_tr_sel = X_tr_proc.loc[:, cols_in_fold]
        X_val_sel = X_val_proc.loc[:, cols_in_fold]

        # Apply SMOTE/RUS on TRAIN ONLY (same helper as union_feature_selector)
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

        # Evaluate on original (un-resampled) train and val folds
        y_tr_pred = clf.predict(X_tr_np)
        y_val_pred = clf.predict(X_val_np)

        train_scores.append(balanced_accuracy_score(y_tr, y_tr_pred))
        val_scores.append(balanced_accuracy_score(y_val, y_val_pred))

    train_mean = float(np.mean(train_scores)) if train_scores else 0.0
    val_mean = float(np.mean(val_scores)) if val_scores else 0.0
    return train_mean, val_mean


# ---------------------------------------------------------------------
# Core routine per condition pair
# ---------------------------------------------------------------------


def _run_for_pair(
    config: BootstrapStabilitySelectionConfig,
    cond_pair: Tuple[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run bootstrap-based stability selection and validation for a single
    condition pair.

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
        [condition_pair, feature_id, n_selections].
    val_grid_df :
        DataFrame of validation results for this pair with columns:
        [condition_pair, min_selections, train_mean_bacc, val_mean_bacc].
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

    # Convert y to 0/1 for classifiers
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
            empty_selected = pd.DataFrame(
                columns=["condition_pair", "feature_id", "n_selections"]
            )
            empty_val = pd.DataFrame(
                columns=[
                    "condition_pair",
                    "min_selections",
                    "train_mean_bacc",
                    "val_mean_bacc",
                ]
            )
            return empty_selected, empty_val

    # ---- 3. Preprocessing pipeline (fit once on full train) ----
    preproc_pipe = make_tb_preprocess_pipeline(
        samples_df=samples_sub,
        config=config.preprocess,
    )
    X_proc = preproc_pipe.fit_transform(X_df)
    X_proc = _to_preprocessed_df(X_proc, X_df, config)

    n_samples, n_features = X_proc.shape
    if n_features == 0:
        empty_selected = pd.DataFrame(
            columns=["condition_pair", "feature_id", "n_selections"]
        )
        empty_val = pd.DataFrame(
            columns=[
                "condition_pair",
                "min_selections",
                "train_mean_bacc",
                "val_mean_bacc",
            ]
        )
        return empty_selected, empty_val

    feature_ids = np.array(X_proc.columns)
    selection_counts = np.zeros(n_features, dtype=int)

    rng = np.random.default_rng(config.random_state)

    # ---- 4. Bootstrap stability selection ----
    for b in range(config.bootstrap):
        # 4a. Bootstrap rows with replacement
        idx_boot = rng.integers(0, n_samples, size=n_samples)
        X_boot = X_proc.iloc[idx_boot].reset_index(drop=True)
        y_boot = y_int[idx_boot]

        # Skip degenerate bootstrap samples
        if np.unique(y_boot).size < 2:
            continue

        # 4b. Optional RUS / BorderlineSMOTE (dataframe helper)
        X_boot_res, y_boot_res = _apply_resampling_df(X_boot, y_boot, config)

        # 4c. Fit sparse selector
        if config.selector_type == "l1_prox_svm":
            y_boot_svm = 2 * y_boot_res - 1  # {0,1} -> {-1,+1}
            selector = L1ProxSVM(
                lambda_=config.lambda_,
                step_size=config.step_size,
                max_iter=config.l1_svm_max_iter,
                delta_tol=config.l1_svm_delta_tol,
                fit_intercept=config.l1_svm_fit_intercept,
                record_history=False,
            )
            selector.fit(X_boot_res.to_numpy(dtype=np.float32), y_boot_svm)
            weights = selector.coef_.ravel()
        elif config.selector_type == "elastic_net":
            selector = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=config.elasticnet_C,
                l1_ratio=config.elasticnet_l1_ratio,
                class_weight="balanced", # BORDERLINE SMOTE CONFLICT ?!?!?!?!?!?!
                tol = config.elasticnet_tol,
                max_iter=config.elasticnet_max_iter,
            )
            selector.fit(X_boot_res.to_numpy(dtype=np.float32), y_boot_res)
            weights = selector.coef_.ravel()
        else:
            raise ValueError(f"Unknown selector_type={config.selector_type!r}")

        # 4d. Update counts for non-zero weights
        abs_w = np.abs(weights)
        idx_nonzero = np.where(abs_w > config.weight_tol)[0]
        if idx_nonzero.size > 0:
            selection_counts[idx_nonzero] += 1

    # Build selected-features DataFrame (only features selected at least once)
    selection_series = pd.Series(selection_counts, index=feature_ids, name="n_selections")
    selection_series = selection_series[selection_series > 0]

    if selection_series.empty:
        selected_df = pd.DataFrame(
            columns=["condition_pair", "feature_id", "n_selections"]
        )
        val_grid_df = pd.DataFrame(
            columns=[
                "condition_pair",
                "min_selections",
                "train_mean_bacc",
                "val_mean_bacc",
            ]
        )
        return selected_df, val_grid_df

    selected_df = (
        selection_series.reset_index()
        .rename(columns={"index": "feature_id"})
        .assign(condition_pair=pair_label)
        .loc[:, ["condition_pair", "feature_id", "n_selections"]]
    )

    # ---- 5. Validation grid over min_selections ----
    max_sel = int(selection_series.max())
    rows: List[dict] = []

    for min_sel in range(1, max_sel + 1):
        feat_ids_k = selection_series.index[selection_series >= min_sel].tolist()
        train_mean, val_mean = _evaluate_feature_set_cv(
            X_raw=X_df,
            y_int=y_int,
            samples_df=samples_sub,
            feature_ids=feat_ids_k,
            selector_type=config.selector_type,
            config=config,
        )
        rows.append(
            {
                "condition_pair": pair_label,
                "min_selections": int(min_sel),
                "train_mean_bacc": float(train_mean),
                "val_mean_bacc": float(val_mean),
            }
        )

    val_grid_df = pd.DataFrame(rows)
    return selected_df, val_grid_df


# ---------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------


def _plot_min_selections_curve(
    val_grid_df: pd.DataFrame,
    pair_label: str,
    out_path: Path,
) -> None:
    """
    Plot min_selections vs mean balanced accuracy (train & val) for a
    single condition pair.

    Parameters
    ----------
    val_grid_df :
        DataFrame with columns [min_selections, train_mean_bacc,
        val_mean_bacc] for this condition pair.
    pair_label :
        String label used for the title (e.g. "control_vs_activated").
    out_path :
        Path (PNG) where the figure will be written.
    """
    if val_grid_df.empty:
        return

    ks = val_grid_df["min_selections"].to_numpy()
    train_bacc = val_grid_df["train_mean_bacc"].to_numpy()
    val_bacc = val_grid_df["val_mean_bacc"].to_numpy()

    plt.figure()
    plt.plot(ks, train_bacc, marker="o", label="Train")
    plt.plot(ks, val_bacc, marker="s", linestyle="--", label="Validation")
    plt.xlabel("Minimum # bootstrap selections (k)")
    plt.ylabel("Mean balanced accuracy")
    plt.title(pair_label)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def run_bootstrap_stability_selection(
    config: BootstrapStabilitySelectionConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run bootstrap-based stability selection for all condition pairs
    specified in `config.condition_pairs`.

    This function:
      - Queries the TB star-schema train split.
      - Applies optional group-wise missingness filtering.
      - Applies preprocessing (normalization, imputation, log2).
      - Optionally applies BorderlineSMOTE and RUS inside each bootstrap
        and inside each CV fold.
      - Performs bootstrap stability selection per condition pair.
      - Evaluates downstream classifiers as a function of the minimum
        number of bootstrap selections.
      - Aggregates selected features and validation curves across pairs.
      - Writes:
          * a CSV of selected features,
          * a CSV of validation results,
          * per-pair min_selections vs balanced-accuracy plots,
          * a JSON sidecar storing the configuration.

    Parameters
    ----------
    config :
        BootstrapStabilitySelectionConfig instance.

    Returns
    -------
    selected_all :
        Concatenated selected-features DataFrame across all condition pairs.
        Columns: condition_pair, feature_id, n_selections.
    val_all :
        Concatenated validation grid across all condition pairs.
        Columns: condition_pair, min_selections, train_mean_bacc,
        val_mean_bacc.
    """
    db_root = Path(config.db_root)
    if not db_root.exists():
        raise FileNotFoundError(f"db_root does not exist: {db_root}")

    # Assign a run_id if not already set
    if config.run_id is None:
        config.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_selected: List[pd.DataFrame] = []
    all_val: List[pd.DataFrame] = []

    for cond_pair in config.condition_pairs:
        cond_neg, cond_pos = cond_pair
        pair_label = f"{cond_neg}_vs_{cond_pos}"
        print(f"[INFO] Running bootstrap stability selection for {pair_label}")

        selected_df, val_grid_df = _run_for_pair(config, cond_pair)
        all_selected.append(selected_df)
        all_val.append(val_grid_df)

        # Plot curve for this pair
        plot_path = out_dir / f"bacc_vs_min_selections__{pair_label}__{config.run_id}.png"
        _plot_min_selections_curve(val_grid_df, pair_label, plot_path)

    if all_selected:
        selected_all = pd.concat(all_selected, axis=0, ignore_index=True)
    else:
        selected_all = pd.DataFrame(
            columns=["condition_pair", "feature_id", "n_selections"]
        )

    if all_val:
        val_all = pd.concat(all_val, axis=0, ignore_index=True)
    else:
        val_all = pd.DataFrame(
            columns=[
                "condition_pair",
                "min_selections",
                "train_mean_bacc",
                "val_mean_bacc",
            ]
        )

    # Write CSVs and JSON sidecar
    base_sel = f"bootstrap_stability_selected_features__{config.run_id}"
    sel_csv_path = out_dir / f"{base_sel}.csv"
    sel_json_path = out_dir / f"{base_sel}.json"

    base_val = f"bootstrap_stability_validation__{config.run_id}"
    val_csv_path = out_dir / f"{base_val}.csv"

    selected_all.to_csv(sel_csv_path, index=False)
    val_all.to_csv(val_csv_path, index=False)

    with sel_json_path.open("w", encoding="utf-8") as f:
        json.dump(config.to_serializable_dict(), f, indent=2)

    print(f"[INFO] Wrote selected features to: {sel_csv_path}")
    print(f"[INFO] Wrote validation grid to:   {val_csv_path}")
    print(f"[INFO] Wrote config sidecar to:    {sel_json_path}")

    return selected_all, val_all


if __name__ == "__main__":
    raise SystemExit(
        "bootstrap_stability_selection.py is intended to be imported and used "
        "from a driver script, not executed directly."
    )
