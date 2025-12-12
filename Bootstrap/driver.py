#!/usr/bin/env python
"""
run_tb_bootstrap_and_final

Single driver for:
  - bootstrap_feature_selection.py (bootstrap-based stability selection)
  - final_classification.py        (downstream classifier tuning + test eval)

Usage
-----
Edit the "USER SETTINGS" section in `main()` below to set:

  * db_root (location of Parquet TB star-schema)
  * condition_pairs, label_col
  * shared TBPreprocessConfig
  * BootstrapStabilitySelectionConfig (bootstrap stage)
  * FinalClassificationConfig        (final classifier stage)

Then run:

    python run_tb_bootstrap_and_final.py

Pipeline
--------
Stage 1: run_bootstrap_stability_selection
  - Writes selected-features CSV:
        bootstrap_stability_selected_features__{bootstrap_cfg.run_id}.csv
    in bootstrap_cfg.out_dir

Stage 2: run_final_classification
  - Uses the selected-features CSV from Stage 1
  - Writes final results CSV:
        final_classification__{final_cfg.run_id}.csv
    in final_cfg.out_dir
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd 

# ---- Imports from your modules ----
from Utilities.query_and_preprocess import TBPreprocessConfig

from bootstrap_feature_selection import (
    BootstrapStabilitySelectionConfig,
    run_bootstrap_stability_selection,
)

from final_classifier import (
    FinalClassificationConfig,
    run_final_classification,
)


def main() -> None:
    # ------------------------------------------------------------------
    # USER SETTINGS
    # ------------------------------------------------------------------

    # Root directory of your TB Parquet star-schema
    db_root = Path("../Data/Parquet")

    # Output locations
    bootstrap_out_dir = Path("results/bootstrap_stability")   
    final_out_dir = Path("results/final_classification")      

    # Labels / condition pairs
    label_col = "condition"  # or "symptomatic", etc.
    condition_pairs = [
        ("control", "activated"),
        ("control", "incident"),
        ("control", "prevalent"),
        ("activated", "incident"), 
        ("activated", "prevalent"),
        ("incident", "prevalent"),
    ]

    # Shared preprocessing configuration (used by both stages)
    preprocess_cfg = TBPreprocessConfig(
        fallback_eps=1e-3,
        internal_standard_feature_id=9500,
        drop_internal_standard=True,
        noise_scale=0.001,
        nan_threshold=0.5,
        k_neighbors=4,
        random_state=42,
        use_standard_scaler=True,
        scaler_with_mean=True,
        scaler_with_std=True,
    )

    # ------------------------------------------------------------------
    # Bootstrap stability-selection configuration
    # ------------------------------------------------------------------
    bootstrap_cfg = BootstrapStabilitySelectionConfig(

        db_root=db_root,
        out_dir=bootstrap_out_dir,
        condition_pairs=condition_pairs,
        label_col=label_col,
        preprocess=preprocess_cfg,

        # selector choice: "l1_prox_svm" or "elastic_net"
        selector_type="elastic_net",

        # group-wise missingness filter
        groupwise_min_prop=0.21,
        groupwise_require_all_groups=True,

        # L1ProxSVM hyperparameters (if selector_type="l1_prox_svm")
        lambda_=1e-3,
        step_size=None,
        l1_svm_max_iter=2000,
        l1_svm_delta_tol=1e-6,
        l1_svm_fit_intercept=True,

        # Elastic-net hyperparameters (if selector_type="elastic_net")
        elasticnet_C=0.1,
        elasticnet_l1_ratio=0.9,
        elasticnet_max_iter=250000,
        elasticnet_tol=1e-5,

        # Evaluation model hyperparameters (for internal validation curves)
        svc_C=1.0,
        lr_C=1.0,

        # Bootstrapping / CV
        bootstrap = 25,
        val_folds =5,
        weight_tol = 1e-6,
        random_state = 42,

        # Optional SMOTE / RUS inside bootstraps + CV
        use_borderline_smote=True,
        borderline_smote_k_neighbors=4,
        use_random_undersampler=True,
        rus_target_majority_ratio=5.0,

        # Bookkeeping
        run_id= "01",  # None -> timestamp-based run_id
        notes="",
    )

    # ------------------------------------------------------------------
    # Final classification configuration
    # ------------------------------------------------------------------
    # NOTE: `selected_features_csv` will be overwritten after Stage 1 to
    #       point at the CSV just written by bootstrap_cfg. You can put
    #       any placeholder here.
    final_cfg = FinalClassificationConfig(
        db_root=db_root,
        out_dir=final_out_dir,
        selected_features_csv="DUMMY_WILL_BE_OVERWRITTEN.csv",
        condition_pairs=condition_pairs,
        label_col=label_col,
        preprocess=preprocess_cfg,

        # "svc" (linear SVC) or "logistic" (L2 LogisticRegression)
        model_type="logistic",
        C_grid=[0.01, 0.1, 1.0, 10.0, 100.0],
        logistic_max_iter=250000,

        val_folds=5,
        random_state=42,

        # If False: only CV metrics (mean_val_bacc, best_C) are computed.
        #           test_* columns will be NaN.
        fit_final_classifier=True,

        # Bookkeeping
        run_id = bootstrap_cfg.run_id,  # None -> auto; will be derived from bootstrap_cfg.run_id below
        notes  = "",
    )

    # ------------------------------------------------------------------
    # STAGE 1: Bootstrap stability selection
    # ------------------------------------------------------------------
    print("\n[STAGE 1] Bootstrap stability selection")
    selected_df, val_df = run_bootstrap_stability_selection(bootstrap_cfg)

    # Derive selected-features CSV path from bootstrap_cfg.run_id
    selected_csv = (
        Path(bootstrap_cfg.out_dir)
        / f"bootstrap_stability_selected_features__{bootstrap_cfg.run_id}.csv"
    )

    if not selected_csv.exists():
        raise FileNotFoundError(
            f"Expected selected-features CSV not found:\n  {selected_csv}"
        )

    print(f"[INFO] Using selected-features CSV for final stage:\n  {selected_csv}")

    # Wire into final-stage config
    final_cfg.selected_features_csv = selected_csv

    # Optionally tie final run_id to bootstrap run_id for easier bookkeeping
    if final_cfg.run_id is None:
        final_cfg.run_id = f"{bootstrap_cfg.run_id}__final"

    # ------------------------------------------------------------------
    # STAGE 2: Final classifier tuning + test evaluation
    # ------------------------------------------------------------------
    print("\n[STAGE 2] Final classifier tuning and evaluation")
    results_all = run_final_classification(final_cfg)

    final_csv = (
        Path(final_cfg.out_dir)
        / f"final_classification__{final_cfg.run_id}.csv"
    )

    print("\n[DONE] Pipeline complete.")
    print(f"  - Bootstrap outputs directory: {bootstrap_cfg.out_dir}")
    print(f"  - Final classification outputs directory: {final_cfg.out_dir}")
    print(f"  - Final classification CSV: {final_csv}")

    try:
        print("\n[HEAD] final_classification results:")
        print(results_all.head())
    except Exception:
        pass


if __name__ == "__main__":
    main()
