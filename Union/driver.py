#!/usr/bin/env python
"""
run_tb_pipeline

Driver script for the TB omics pipeline:

    1. Define a shared preprocessing configuration (TBPreprocessConfig).
    2. Define a union feature-selection configuration (UnionFeatureSelectorConfig).
    3. Run union-of-sparse-solutions feature selection.
    4. Define a final-classifier configuration (FinalClassifierConfig),
       using the selected-features CSV from step 3.
    5. Run final classifier training / evaluation (train + held-out test).

This script is meant to be edited and run directly by the user, *not*
from the command line with arguments. To change behavior, modify the
"USER SETTINGS" section in `main()`.

With the updated versions of:

    - union_feature_selector.py
    - final_classifier.py

the pipeline now:

  • Uses project → preprocess behavior consistently:
        raw X → project to selected features (+internal standard, if needed)
              → preprocess with make_tb_preprocess_pipeline
              → project preprocessed X to selected features for modeling.

  • Uses a two-stage resampling strategy in both FS and final classifiers:
        1. Optional RandomUnderSampler (RUS) first, enforcing a maximum
           majority:minority ratio `rus_target_majority_ratio`.
        2. Optional BorderlineSMOTE afterwards to balance the classes.

  • In final_classifier, fits:
        - one model per iteration ("iteration {k}"),
        - an ensemble over iterations ("ensemble"),
        - and a union-of-features model ("union").
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from Utilities.query_and_preprocess import TBPreprocessConfig
from union_feature_selector import (
    UnionFeatureSelectorConfig,
    run_union_feature_selection,
)
from final_classifier import (
    FinalClassifierConfig,
    run_final_classifiers,
)


def main() -> None:
    """
    Configure and run the full TB omics pipeline:

        1. Sparse union feature selection
        2. Final classification on held-out test split

    Edit the "USER SETTINGS" section below to point to your TB Parquet
    database and choose hyperparameters.
    """

    # ------------------------------------------------------------------
    # USER SETTINGS
    # ------------------------------------------------------------------

    # Root directory of your TB star-schema Parquet database, produced by
    # tb_peak_to_parquets.py.
    db_root = Path("../Data/Parquets")

    # Output directory for union feature selection (intermediate results)
    fs_out_dir = Path("../Union/results/union_feature_selection")

    # Output directory for final classifiers (summary metrics)
    fc_out_dir = Path("../Union/results/final_classifiers")

    # Condition pairs for binary discrimination.
    condition_pairs: List[Tuple[str, str]] = [
        ("control", "activated"),
        ("control", "incident"),
        ("control", "prevalent"),
        ("activated", "incident"),
        ("activated", "prevalent"),
        ("incident", "prevalent"),
    ]

    # Which column in samples.parquet defines the label?
    label_col = "condition"

    # -------------------------------------
    # Shared preprocessing configuration
    # ----------------------------------
    preprocess_cfg = TBPreprocessConfig(
        # Log2-with-pseudocount config
        fallback_eps=1e-3,

        # Internal standard normalization
        internal_standard_feature_id=9500,
        drop_internal_standard=True,

        # Label-agnostic two-step imputation config
        noise_scale=0.001,
        nan_threshold=0.45,
        k_neighbors=4,
        random_state=42,

        # Optional standardization (after imputation)
        use_standard_scaler=True,
        scaler_with_mean=True,
        scaler_with_std=True,
    )

    # -------------------------------------
    # Union feature-selection configuration
    # -------------------------------------
    fs_config = UnionFeatureSelectorConfig(
        # I/O
        db_root=db_root,
        out_dir=fs_out_dir,
        run_id="5",

        # Query info
        condition_pairs=condition_pairs,
        label_col=label_col,

        # Baseline preprocess
        preprocess=preprocess_cfg,

        # Sparse selector: "l1_prox_svm" or "elastic_net"
        selector_type="l1_prox_svm",

        # L1ProxSVM hyperparameters (if selector_type="l1_prox_svm")
        lambda_= 0.25,
        step_size=None,
        l1_svm_max_iter= 200_000,
        l1_svm_delta_tol= 5e-7,
        l1_svm_fit_intercept=True,

        # Elastic-net hyperparameters (if selector_type="elastic_net")
        elasticnet_C=0.1,
        elasticnet_l1_ratio=0.9,
        elasticnet_max_iter=50000,

        # Evaluation models (inside union FS CV)
        svc_C=1.0,
        lr_C=1.0,

        # Group-wise missingness filter (set to None to skip)
        groupwise_min_prop=0.21,
        groupwise_require_all_groups=True,

        # CV / iteration control
        n_folds=5,
        balanced_accuracy_threshold=0.8,
        max_iterations=25,
        weight_tol=1e-6,
        random_state=42,

        # Optional RUS + BorderlineSMOTE
        use_random_undersampler=True,
        rus_target_majority_ratio=5.0,
        use_borderline_smote=True,
        borderline_smote_k_neighbors=4,

        notes = "Same config as run 4, but with impute after log2."
    )
    # ------------------------------
    # Final classifier configuration
    # ------------------------------
    final_clf_base_config = FinalClassifierConfig(
        # I/O
        db_root=db_root,
        selected_features_csv="PLACEHOLDER_WILL_BE_OVERRIDDEN",
        out_dir=fc_out_dir,
        label_col=label_col,
        preprocess=preprocess_cfg,

        # Classifier family for final models: "svc" or "logreg"
        classifier_type  ="svc",
        cls_class_weight ="balanced",
        cls_max_iter=-1,
        cls_tol=1e-4,

        # Hyperparameter grid for C
        C_grid=[0.01, 0.1, 1.0, 10.0],
        n_folds=5,
        random_state=42,

        # RUS + BorderlineSMOTE
        use_random_undersampler=False,
        rus_target_majority_ratio = 5,
        use_borderline_smote= False,
        borderline_smote_k_neighbors=fs_config.borderline_smote_k_neighbors,

        # Groupwise missingness (mirror union FS)
        groupwise_min_prop= fs_config.groupwise_min_prop,
        groupwise_require_all_groups=fs_config.groupwise_require_all_groups,
    )

    # ------------------------------------------------------------------
    # END USER SETTINGS
    # ------------------------------------------------------------------

    # 1. Run union-of-sparse-solutions feature selection
    # --------------------------------------------------
    print("[DRIVER] Running union feature selection...")

    selected_df = run_union_feature_selection(fs_config)
    selected_csv_path = (
        Path(f"{fs_config.out_dir}/run_id__{fs_config.run_id}") / f"selected_features_union__{fs_config.run_id}.csv"
    )
    if not selected_csv_path.exists():
        raise FileNotFoundError(
            f"Expected selected-features CSV not found:\n  {selected_csv_path}"
        )
    print(f"[DRIVER] Selected-features CSV: {selected_csv_path}")


    # 2. Run final classifiers using the selected features
    # ----------------------------------------------------
    final_clf_config = final_clf_base_config
    final_clf_config.selected_features_csv = selected_csv_path
    final_clf_config.run_id = fs_config.run_id

    print("[DRIVER] Fitting final classifiers...")

    results_df = run_final_classifiers(final_clf_config)


if __name__ == "__main__":
    main()
