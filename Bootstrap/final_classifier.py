#!/usr/bin/env python
"""
final_classification

Downstream classifier tuning and evaluation for TB omics after
bootstrap-based stability selection.

High-level algorithm (per condition pair)
-----------------------------------------
For each (cond_a, cond_b) in `config.condition_pairs`:

  1. Query the TB star-schema *train* and *test* splits restricted to
     these two conditions using `make_tb_dataset`.

  2. Load a *selected-features* CSV produced by
     `bootstrap_stability_selection.py`, with columns:
         condition_pair, feature_id, n_selections

  3. For k = 1, ..., max(n_selections):

       3a. Define a feature set F_k consisting of all features with
           n_selections >= k for this condition pair.

       3b. Run CV grid search on the *train* split to find the best
           regularization strength C for the final classifier:

             - Use StratifiedKFold with `config.val_folds`.
             - For each fold:
                 * Fit a fresh TB preprocessing pipeline on the
                   fold's raw train data (using make_tb_preprocess_pipeline).
                 * Transform both train and validation folds.
                 * Convert to DataFrames via `_to_preprocessed_df`.
                 * Project to the current feature set F_k.
                 * Fit a classifier of type `config.model_type`
                   ("svc" or "logistic") with the current C.
                 * Compute balanced accuracy on the validation fold.
             - Average validation balanced accuracy across folds.
             - Select the C with highest mean validation balanced
               accuracy. Record this best C and its mean validation
               balanced accuracy.

       3c. If `config.fit_final_classifier` is True:

             - Fit a fresh TB preprocessing pipeline on the full raw
               train split.
             - Transform the full train and test splits.
             - Convert to DataFrames via `_to_preprocessed_df`.
             - Project to F_k.
             - Fit the final classifier with the chosen best C on the
               preprocessed train data.
             - Evaluate on the preprocessed test data, recording:
                   * balanced accuracy
                   * precision
                   * recall
                   * F1 score

  4. Append a row to a results DataFrame with columns:
        condition_pair, min_selections, mean_val_bacc, best_C,
        test_bacc, test_precision, test_recall, test_f1

  5. For each condition pair, save a PNG plot of:
        min_selections vs mean_val_bacc   (solid line)
        min_selections vs test_bacc       (dashed line)

Across all pairs, this module writes:

  - A single CSV of final results:
        final_classification__{run_id}.csv

  - A JSON sidecar with the serialized configuration:
        final_classification__{run_id}.json

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
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from Utilities.query_and_preprocess import (
    TBPreprocessConfig,
    make_tb_dataset,
    make_tb_preprocess_pipeline,
)

from Union.union_feature_selector import _to_preprocessed_df

PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class FinalClassificationConfig:
    """
    Configuration for final classifier tuning and evaluation.

    Attributes
    ----------
    db_root :
        Root directory containing the TB Parquet star-schema:
            samples.parquet, features.parquet, abundances.parquet.
    out_dir :
        Directory where the results CSV, JSON sidecar, and plots
        will be written.
    selected_features_csv :
        Path to the CSV produced by `bootstrap_stability_selection.py`
        with columns:
            condition_pair, feature_id, n_selections

    condition_pairs :
        List of condition pairs to discriminate between, e.g.
            [("control", "activated"), ("incident", "prevalent")].
        The label column is controlled by `label_col`.
    label_col :
        Column in the samples table to use as the binary label,
        typically "condition" or "symptomatic".

    Preprocessing configuration
    ---------------------------
    preprocess :
        TBPreprocessConfig controlling normalization, imputation,
        and log transform. Used to build TB preprocessing pipelines
        via `make_tb_preprocess_pipeline`.

    Final model specification
    -------------------------
    model_type :
        "svc"       -> linear SVC with L2 penalty
        "logistic"  -> LogisticRegression with L2 penalty
    C_grid :
        Sequence of C values (inverse regularization strength) to
        explore in the CV grid search for the final classifier.
    logistic_max_iter :
        Maximum iterations for LogisticRegression when model_type is
        "logistic".

    CV and randomness
    -----------------
    val_folds :
        Number of folds in StratifiedKFold for grid search.
    random_state :
        Random seed for CV splitting.

    Final fit options
    -----------------
    fit_final_classifier :
        If True, fit a final classifier on the full train split using
        the best C found by CV, and evaluate on the test split.
        If False, only CV metrics (mean_val_bacc, best_C) are
        computed; test_* columns in the output CSV are left as NaN.

    Bookkeeping
    -----------
    run_id :
        Optional string identifying the run. If None, a timestamp-based
        ID is generated when `run_final_classification` is called.
    notes :
        Free-form notes about the run (e.g., dataset version, comments).
    """

    db_root: PathLike
    out_dir: PathLike
    selected_features_csv: PathLike

    condition_pairs: List[Tuple[str, str]]
    label_col: str = "condition"

    # preprocessing config
    preprocess: TBPreprocessConfig = field(default_factory=TBPreprocessConfig)

    # model choice
    model_type: str = "svc"  # "svc" or "logistic"
    C_grid: Sequence[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0]
    )
    logistic_max_iter: int = 5000

    # CV / randomness
    val_folds: int = 5
    random_state: Optional[int] = 42

    # final fit
    fit_final_classifier: bool = True

    # bookkeeping
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
        d["selected_features_csv"] = str(self.selected_features_csv)
        # Ensure C_grid is a plain list for JSON
        d["C_grid"] = list(self.C_grid)
        return d


# ---------------------------------------------------------------------
# Helper: classifier factory
# ---------------------------------------------------------------------


def _make_classifier(C: float, config: FinalClassificationConfig):
    """
    Instantiate a classifier for the given C and config.model_type.
    """
    if config.model_type == "svc":
        return SVC(
            kernel="linear",
            C=float(C),
            class_weight="balanced",
        )
    elif config.model_type == "logistic":
        return LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=float(C),
            class_weight="balanced",
            max_iter=config.logistic_max_iter,
        )
    else:
        raise ValueError(f"Unknown model_type={config.model_type!r}, "
                         "expected 'svc' or 'logistic'.")


# ---------------------------------------------------------------------
# CV grid search for a fixed feature set
# ---------------------------------------------------------------------


def _cv_grid_search_for_feature_set(
    X_train_raw: pd.DataFrame,
    y_train_int: np.ndarray,
    samples_train: pd.DataFrame,
    feature_ids: Sequence[int],
    config: FinalClassificationConfig,
) -> Tuple[float, float]:
    """
    Run CV grid search over C for a fixed feature set.

    Parameters
    ----------
    X_train_raw :
        Raw train DataFrame (samples × features) after any global
        filtering, index=sample_id, columns=feature_id.
    y_train_int :
        Integer labels aligned with X_train_raw rows (0/1).
    samples_train :
        Samples metadata aligned with X_train_raw (same sample_ids);
        used by TBNormalizer inside the preprocessing pipeline.
    feature_ids :
        Sequence of feature_ids defining the feature set.
    config :
        Full configuration, including model_type, C_grid, preprocess,
        val_folds, and random_state.

    Returns
    -------
    best_C :
        The C value with highest mean validation balanced accuracy.
        NaN if no valid model could be evaluated.
    best_mean_val_bacc :
        The corresponding mean validation balanced accuracy.
        NaN if no valid model could be evaluated.
    """
    feature_ids = list(feature_ids)
    if len(feature_ids) == 0:
        return float("nan"), float("nan")

    skf = StratifiedKFold(
        n_splits=config.val_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    # Accumulate per-C validation scores across folds
    C_scores = {float(C): [] for C in config.C_grid}

    for train_idx, val_idx in skf.split(X_train_raw, y_train_int):
        X_tr_raw = X_train_raw.iloc[train_idx]
        X_val_raw = X_train_raw.iloc[val_idx]
        y_tr = y_train_int[train_idx]
        y_val = y_train_int[val_idx]

        # Fresh preprocessing pipeline per fold
        preproc_pipe = make_tb_preprocess_pipeline(
            samples_df=samples_train,
            config=config.preprocess,
        )

        X_tr_proc = preproc_pipe.fit_transform(X_tr_raw)
        X_val_proc = preproc_pipe.transform(X_val_raw)

        # Convert to DataFrames with correct feature_id columns
        X_tr_proc = _to_preprocessed_df(X_tr_proc, X_tr_raw, config)
        X_val_proc = _to_preprocessed_df(X_val_proc, X_val_raw, config)

        # Only keep features present after preprocessing
        cols_in_fold = [fid for fid in feature_ids if fid in X_tr_proc.columns]
        if not cols_in_fold:
            # No surviving features in this fold; treat as chance level
            for C in C_scores:
                C_scores[C].append(0.5)
            continue

        X_tr_sel = X_tr_proc.loc[:, cols_in_fold].to_numpy(dtype=np.float32)
        X_val_sel = X_val_proc.loc[:, cols_in_fold].to_numpy(dtype=np.float32)

        for C in C_scores:
            clf = _make_classifier(C, config)
            clf.fit(X_tr_sel, y_tr)
            y_val_pred = clf.predict(X_val_sel)
            bacc = balanced_accuracy_score(y_val, y_val_pred)
            C_scores[C].append(float(bacc))

    # Compute mean validation scores per C
    mean_scores = {
        C: (float(np.mean(scores)) if len(scores) > 0 else float("nan"))
        for C, scores in C_scores.items()
    }

    # Select best C (max mean val bacc, ignoring NaNs)
    valid_items = [(C, score) for C, score in mean_scores.items() if np.isfinite(score)]
    if not valid_items:
        return float("nan"), float("nan")

    # Sort by score desc, then by C asc for deterministic tie-breaking
    valid_items.sort(key=lambda x: (-x[1], x[0]))
    best_C, best_score = valid_items[0]
    return float(best_C), float(best_score)


# ---------------------------------------------------------------------
# Final fit & evaluation for a fixed feature set
# ---------------------------------------------------------------------


def _fit_and_evaluate_final_model(
    X_train_raw: pd.DataFrame,
    y_train_int: np.ndarray,
    samples_train: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_test_int: np.ndarray,
    feature_ids: Sequence[int],
    best_C: float,
    config: FinalClassificationConfig,
) -> Tuple[float, float, float, float]:
    """
    Fit a final classifier on full train data and evaluate on test.

    Parameters
    ----------
    X_train_raw, y_train_int, samples_train :
        Train split (raw features + labels + metadata).
    X_test_raw, y_test_int :
        Test split (raw features + labels).
    feature_ids :
        Sequence of feature_ids defining the feature set.
    best_C :
        Regularization strength chosen by CV.
    config :
        FinalClassificationConfig controlling preprocessing and model.

    Returns
    -------
    test_bacc, test_precision, test_recall, test_f1 :
        Test metrics for this feature set. NaNs if the model cannot be
        evaluated (e.g., no surviving features).
    """
    feature_ids = list(feature_ids)
    if len(feature_ids) == 0 or not np.isfinite(best_C):
        return (float("nan"),) * 4

    # Fresh preprocessing pipeline on full train
    preproc_pipe = make_tb_preprocess_pipeline(
        samples_df=samples_train,
        config=config.preprocess,
    )

    X_tr_proc = preproc_pipe.fit_transform(X_train_raw)
    X_te_proc = preproc_pipe.transform(X_test_raw)

    X_tr_proc = _to_preprocessed_df(X_tr_proc, X_train_raw, config)
    X_te_proc = _to_preprocessed_df(X_te_proc, X_test_raw, config)

    cols_final = [fid for fid in feature_ids if fid in X_tr_proc.columns]
    if not cols_final:
        return (float("nan"),) * 4

    X_tr_sel = X_tr_proc.loc[:, cols_final].to_numpy(dtype=np.float32)
    X_te_sel = X_te_proc.loc[:, cols_final].to_numpy(dtype=np.float32)

    clf = _make_classifier(best_C, config)
    clf.fit(X_tr_sel, y_train_int)
    y_te_pred = clf.predict(X_te_sel)

    test_bacc = balanced_accuracy_score(y_test_int, y_te_pred)
    test_precision = precision_score(y_test_int, y_te_pred, zero_division=0)
    test_recall = recall_score(y_test_int, y_te_pred, zero_division=0)
    test_f1 = f1_score(y_test_int, y_te_pred, zero_division=0)

    return (
        float(test_bacc),
        float(test_precision),
        float(test_recall),
        float(test_f1),
    )


# ---------------------------------------------------------------------
# Per-pair driver
# ---------------------------------------------------------------------


def _run_for_pair(
    config: FinalClassificationConfig,
    cond_pair: Tuple[str, str],
    selected_all: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run final classification grid and evaluation for a single condition pair.

    Parameters
    ----------
    config :
        FinalClassificationConfig instance.
    cond_pair :
        Tuple (cond_neg, cond_pos) specifying the two label levels.
    selected_all :
        Full selected-features DataFrame loaded from
        `config.selected_features_csv`.

    Returns
    -------
    results_df :
        DataFrame with columns:
          [condition_pair, min_selections, mean_val_bacc, best_C,
           test_bacc, test_precision, test_recall, test_f1]
        for this condition pair.
    """
    cond_neg, cond_pos = cond_pair
    pair_label = f"{cond_neg}_vs_{cond_pos}"

    # Slice selected features for this pair and collapse (in case of duplicates)
    sel_pair = selected_all[selected_all["condition_pair"] == pair_label]
    if sel_pair.empty:
        print(f"[WARN] No selected features found for pair {pair_label}; skipping.")
        return pd.DataFrame(
            columns=[
                "condition_pair",
                "min_selections",
                "mean_val_bacc",
                "best_C",
                "test_bacc",
                "test_precision",
                "test_recall",
                "test_f1",
            ]
        )

    sel_pair = (
        sel_pair.groupby("feature_id", as_index=False)["n_selections"]
        .max()
        .rename(columns={"n_selections": "n_selections"})
    )

    max_sel = int(sel_pair["n_selections"].max())
    if max_sel <= 0:
        print(f"[WARN] Max n_selections <= 0 for pair {pair_label}; skipping.")
        return pd.DataFrame(
            columns=[
                "condition_pair",
                "min_selections",
                "mean_val_bacc",
                "best_C",
                "test_bacc",
                "test_precision",
                "test_recall",
                "test_f1",
            ]
        )

    # Query train and test splits
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

    # Map labels to 0/1 using explicit mapping
    label_to_int = {cond_neg: 0, cond_pos: 1}
    y_train_int = pd.Series(y_train_raw).map(label_to_int).to_numpy()
    y_test_int = pd.Series(y_test_raw).map(label_to_int).to_numpy()

    if np.any(pd.isna(y_train_int)) or np.any(pd.isna(y_test_int)):
        raise ValueError(
            f"Unexpected labels for pair {cond_pair}: "
            f"train labels={set(np.unique(y_train_raw))}, "
            f"test labels={set(np.unique(y_test_raw))}"
        )

    rows: List[dict] = []

    # min_selections runs from 1 to max_sel
    for min_sel in range(1, max_sel + 1):
        feature_ids_k = sel_pair.loc[
            sel_pair["n_selections"] >= min_sel, "feature_id"
        ].tolist()

        if len(feature_ids_k) == 0:
            best_C = float("nan")
            mean_val_bacc = float("nan")
            test_bacc = test_precision = test_recall = test_f1 = float("nan")
        else:
            # CV grid search on train
            best_C, mean_val_bacc = _cv_grid_search_for_feature_set(
                X_train_raw=X_train_raw,
                y_train_int=y_train_int,
                samples_train=samples_train,
                feature_ids=feature_ids_k,
                config=config,
            )

            # Optional final fit on full train + test
            if config.fit_final_classifier and np.isfinite(best_C):
                (
                    test_bacc,
                    test_precision,
                    test_recall,
                    test_f1,
                ) = _fit_and_evaluate_final_model(
                    X_train_raw=X_train_raw,
                    y_train_int=y_train_int,
                    samples_train=samples_all,
                    X_test_raw=X_test_raw,
                    y_test_int=y_test_int,
                    feature_ids=feature_ids_k,
                    best_C=best_C,
                    config=config,
                )
            else:
                test_bacc = test_precision = test_recall = test_f1 = float("nan")

        rows.append(
            {
                "condition_pair": pair_label,
                "min_selections": int(min_sel),
                "mean_val_bacc": float(mean_val_bacc)
                if np.isfinite(mean_val_bacc)
                else float("nan"),
                "best_C": float(best_C) if np.isfinite(best_C) else float("nan"),
                "test_bacc": float(test_bacc)
                if np.isfinite(test_bacc)
                else float("nan"),
                "test_precision": float(test_precision)
                if np.isfinite(test_precision)
                else float("nan"),
                "test_recall": float(test_recall)
                if np.isfinite(test_recall)
                else float("nan"),
                "test_f1": float(test_f1)
                if np.isfinite(test_f1)
                else float("nan"),
            }
        )

    results_df = pd.DataFrame(rows)
    return results_df


# ---------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------


def _plot_min_selections_vs_metrics(
    results_df: pd.DataFrame,
    pair_label: str,
    out_path: Path,
) -> None:
    """
    Plot min_selections vs mean_val_bacc and test_bacc for a condition pair.

    Parameters
    ----------
    results_df :
        DataFrame with columns [min_selections, mean_val_bacc, test_bacc].
    pair_label :
        String label used for the title (e.g. "control_vs_activated").
    out_path :
        Path (PNG) where the figure will be written.
    """
    if results_df.empty:
        return

    ks = results_df["min_selections"].to_numpy(dtype=float)
    val_bacc = results_df["mean_val_bacc"].to_numpy(dtype=float)
    test_bacc = results_df["test_bacc"].to_numpy(dtype=float)

    plt.figure()
    plt.plot(ks, val_bacc, marker="o", label="CV mean val bacc")
    if not np.all(np.isnan(test_bacc)):
        plt.plot(ks, test_bacc, marker="s", linestyle="--", label="Test bacc")
    plt.xlabel("Minimum # bootstrap selections (k)")
    plt.ylabel("Balanced accuracy")
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


def run_final_classification(config: FinalClassificationConfig) -> pd.DataFrame:
    """
    Run final classifier tuning and evaluation for all condition pairs.

    This function:

      - Loads the selected-features CSV produced by
        `bootstrap_stability_selection.py`.
      - For each condition pair in `config.condition_pairs`:
          * Queries the TB star-schema (train & test).
          * For min_selections = 1..max(n_selections):
              - Defines a feature set F_k.
              - Runs CV grid search over C on train.
              - Optionally fits a final classifier on full train and
                evaluates on test.
          * Writes a per-pair plot of min_selections vs
            (mean_val_bacc, test_bacc).
      - Aggregates all results into a single CSV.
      - Writes a JSON sidecar containing the serialized configuration.

    Parameters
    ----------
    config :
        FinalClassificationConfig instance.

    Returns
    -------
    results_all :
        Concatenated results DataFrame across all condition pairs with
        columns:
          [condition_pair, min_selections, mean_val_bacc, best_C,
           test_bacc, test_precision, test_recall, test_f1]
    """
    db_root = Path(config.db_root)
    if not db_root.exists():
        raise FileNotFoundError(f"db_root does not exist: {db_root}")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Assign a run_id if not already set
    if config.run_id is None:
        config.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    selected_path = Path(config.selected_features_csv)
    if not selected_path.exists():
        raise FileNotFoundError(
            f"selected_features_csv does not exist: {selected_path}"
        )

    selected_all = pd.read_csv(selected_path)
    required_cols = {"condition_pair", "feature_id", "n_selections"}
    missing = required_cols - set(selected_all.columns)
    if missing:
        raise ValueError(
            f"selected_features_csv is missing required columns: {missing}"
        )

    all_results: List[pd.DataFrame] = []

    for cond_pair in config.condition_pairs:
        cond_neg, cond_pos = cond_pair
        pair_label = f"{cond_neg}_vs_{cond_pos}"
        print(f"[INFO] Running final classification for {pair_label}")

        results_df = _run_for_pair(config, cond_pair, selected_all)
        all_results.append(results_df)

        # Per-pair plot
        plot_out_dir = Path(f"{out_dir}/plots/run_id__{config.run_id}")
        os.makedirs(plot_out_dir, exist_ok=True)
        plot_path = plot_out_dir / f"final_classification_bacc_vs_min_selections__{pair_label}__{config.run_id}.png"
        _plot_min_selections_vs_metrics(results_df, pair_label, plot_path)

    if all_results:
        results_all = pd.concat(all_results, axis=0, ignore_index=True)
    else:
        results_all = pd.DataFrame(
            columns=[
                "condition_pair",
                "min_selections",
                "mean_val_bacc",
                "best_C",
                "test_bacc",
                "test_precision",
                "test_recall",
                "test_f1",
            ]
        )

    # Write CSV and JSON sidecar
    base_name = f"final_classification__{config.run_id}"
    csv_path = out_dir / f"{base_name}.csv"
    json_path = out_dir / f"{base_name}.json"

    results_all.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(config.to_serializable_dict(), f, indent=2)

    print(f"[INFO] Wrote final classification results to: {csv_path}")
    print(f"[INFO] Wrote config sidecar to:                {json_path}")

    return results_all


__all__ = ["FinalClassificationConfig", "run_final_classification"]


if __name__ == "__main__":
    raise SystemExit(
        "final_classification.py is intended to be imported and used "
        "from a driver script, not executed directly."
    )
