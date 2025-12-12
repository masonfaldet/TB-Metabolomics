# TB-Metabolomics

Analysis pipeline for a tuberculosis (TB) urine metabolomics study.

The repository contains:

- Utilities to convert a raw peak list and metadata into a simple database (“star schema”) stored as Parquet files.
- Preprocessing utilities tailored to TB urine metabolomics (osmolality + internal-standard normalization, log transform, imputation, missingness filters).
- Scripts to perform sparse feature selection and final classification for condition-based comparisons (e.g. control vs activated, incident vs prevalent).
- A separate “Bootstrap” folder for more generic bootstrap-based feature selection / modeling.


---

## Repository structure

Top-level layout:

- **`Utilities/`**
  - `peak_to_parquets.py`  
    Convert a raw peaklist CSV and metadata Excel file into three Parquet tables:

    - `abundances.parquet` – long table of peak areas per sample/feature  
    - `features.parquet` – feature-level information (mz, rt, internal standard flag, counts)  
    - `samples.parquet` – sample-level information (condition, symptomatic status, mean osmolality, train/test split)

  - `query_and_preprocess.py`  
    Query the Parquet “database” and build ML-ready matrices and preprocessing pipelines. Core pieces:

    - `load_tables` – read `samples.parquet`, `features.parquet`, `abundances.parquet`
    - `make_tb_dataset` – return `(X_df, y, samples_sub, features)` for given conditions and split (train/test/all)
    - `TBNormalizer` – sample-wise osmolality and internal-standard normalization
    - `Log2WithPseudocount` – log transform with a small per-feature pseudocount
    - `LabelAgnosticTwoStepImputer` – two-step KNN-based imputation with low-value noise fill
    - `filter_overall_missingness` / `filter_groupwise_missingness` – feature filters based on missingness
    - `TBPreprocessConfig` + `make_tb_preprocess_pipeline` – a configurable preprocessing pipeline, compatible with scikit-learn

- **`Union/`**
  - `union_feature_selector.py`  
    Iterative “union-of-sparse-solutions” feature selection for binary condition pairs. For each pair:

    1. Query **train** split only (using `make_tb_dataset`).
    2. Optionally filter features by group-wise missingness.
    3. Apply a preprocessing pipeline (normalization, imputation, log, scaling).
    4. Optionally apply RandomUnderSampler (RUS) and/or BorderlineSMOTE (train-only).
    5. Fit a sparse model (e.g. L1-penalized SVM or elastic-net logistic regression).
    6. Record non-zero weight features and evaluate them by cross-validated balanced accuracy.
    7. Repeat on the remaining features until performance falls below a threshold or max iterations is reached.

    Outputs (per run):

    - A CSV of selected features with columns such as:
      - `compared_conditions` (e.g. `control_vs_activated`)
      - `feature_id`
      - `weight`
      - `iteration`
      - `balanced_accuracy`
    - Per-pair iteration vs balanced accuracy plots.
    - A JSON sidecar with the full configuration.

  - `final_classifier.py`  
    Fit final classifiers using the selected feature sets and evaluate on a held-out test split. For each condition pair:

    1. Query **train** and **test** splits for that pair.
    2. For each feature set:
       - Perform cross-validated hyperparameter selection (C) on the train split with a “preprocess → project” pattern:
         - Fit preprocessing on raw train-fold data.
         - Transform train/validation.
         - Project onto the selected feature IDs.
         - Optionally apply RUS + BorderlineSMOTE on the train fold only.
         - Fit a linear SVC or logistic regression and compute balanced accuracy.
    3. With the best C, fit a final model on the full train split (preprocess → project, optional RUS/SMOTE) and evaluate:
       - On the train split (no resampling for predictions).
       - On the held-out test split.

    Typical outputs:

    - CSV summarizing per pair and per model (e.g. per iteration / union / ensemble):

      - `condition_pair`
      - `model` (e.g. `iteration 1`, `union`, `ensemble`)
      - `best_C`
      - `n_features`
      - `train_bacc`, `train_precision`, `train_recall`, `train_f1`
      - `test_bacc`, `test_precision`, `test_recall`, `test_f1`

    - JSON sidecar with the configuration used.

  - `driver.py`  
    Example “driver” script wiring together:

    - A shared `TBPreprocessConfig`
    - A `UnionFeatureSelectorConfig`
    - A `FinalClassifierConfig`

    to run the full pipeline:

    1. Union-of-sparse-solutions feature selection.
    2. Final classifier fitting / evaluation on held-out test data.

- **`Bootstrap/`**
  - `bootstrap_feature_selection.py`, `final_classifier.py`, `driver.py`  
    A separate set of scripts for bootstrap-based feature selection / modeling (kept in this repository for reuse and comparison). These are not strictly required for the union-of-sparse-solutions TB pipeline but follow a similar pattern:
    - configuration data classes,
    - reproducible outputs (CSV + JSON),
    - training / evaluation loops using scikit-learn.

- **`.gitignore`**  
  Standard ignore rules for virtual environments, large files (e.g. CSV, Excel, Parquet), and other non-source artifacts.

---

## Data requirements

The utilities assume access to:

1. **Peaklist CSV**  
   A feature-by-sample table with:

   - An `id` column (feature ID).
   - Columns of the form  
     `datafile:...CSU-TB-{sample_id}.mzML:{dtype}`  
     where `{dtype}` is one of `area`, `rt`, or `mz`.

2. **Metadata Excel file**  
   An Excel workbook with at least:

   - A sheet containing sample-level information (e.g. `UrineUnblinded`).
   - A `CSUID` like `CSU-TB-02339`.
   - Columns needed to define:
     - **Condition** (e.g. `SampleType`, `VisitDay`)  
       → mapped internally to `{"control", "activated", "incident", "prevalent", "unknown"}`.
     - **Symptomatic status** (`tbsymptom`)  
       → mapped to `"+", "-", "unknown"`.
     - **Osmolality** (`OsmolReading1`, `OsmolReading2`, `OsmolReading3`)  
       → used to compute `mean_osmoality`.

> **Note:** The `samples.parquet` table includes a `test_split` flag indicating which samples belong to the held-out test set (stratified by condition). All downstream analyses use this split to avoid leakage.

---

## Typical workflow

### 1. Create the Parquet “star schema”

Edit and run:

```bash
python Utilities/peak_to_parquets.py
```

This script will:

- Read the peaklist CSV and metadata Excel file.
- Infer valid sample IDs from both sources.
- Exclude predefined problematic samples.
- Construct and write:

  - `Data/Parquets/abundances.parquet`
  - `Data/Parquets/features.parquet`
  - `Data/Parquets/samples.parquet`

You can adjust the paths, test/train split ratio, and random seed in the `__main__` block.

---

### 2. Run union-of-sparse-solutions feature selection

Edit `Union/driver.py` (or your own driver built on `union_feature_selector.py`) to specify:

- Location of the Parquet database (`db_root`).
- Output directory for feature selection results.
- Condition pairs to compare (e.g. `("control", "activated")`, etc.).
- Preprocessing configuration (`TBPreprocessConfig`).
- Feature-selection configuration (`UnionFeatureSelectorConfig`), including:
  - Sparse selector type (`l1_prox_svm` / elastic-net).
  - Balanced accuracy threshold.
  - Maximum number of iterations.
  - Optional RUS / BorderlineSMOTE behavior.

Then run:

```bash
python Union/driver.py
```

This will write:

- A selected-features CSV with per-condition-pair feature sets and performance.
- Iteration vs balanced-accuracy plots.
- A JSON configuration snapshot for reproducibility.

---

### 3. Fit final classifiers and evaluate on test data

Using the selected-features CSV from step 2, edit the final-classifier driver (e.g. `Union/driver.py` or a dedicated driver script) to create a `FinalClassifierConfig`:

- Point `selected_features_csv` to the union FS output.
- Point `db_root` to the Parquet database.
- Choose a classifier family (`svc` or `logreg`) and a grid of C values.
- Mirror the preprocessing and resampling settings used in feature selection.

Run:

```bash
python Union/driver.py
```

The final classifier script will:

- Perform cross-validated hyperparameter selection on the train split.
- Fit final models per condition pair.
- Evaluate each model on both train and held-out test data.

Outputs:

- A summary CSV in the configured `out_dir` with metrics by:
  - condition pair,
  - model type (`iteration k`, `union`, `ensemble`).
- A JSON sidecar with the configuration used.

---

## Software environment

The code assumes a recent Python 3 environment with:

- `numpy`
- `pandas`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`

A minimal `pip` install could look like:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib
```

If you prefer, you can manage the environment via `conda` or another environment manager.

---

## Reproducibility notes

- All major scripts use configuration data classes (`TBPreprocessConfig`, `UnionFeatureSelectorConfig`, `FinalClassifierConfig`) to make parameters explicit and traceable.
- Key outputs (feature-selection results, final-classifier results) are written as CSVs with matching JSON sidecars capturing the full configuration.
- The train/test split is stored in `samples.parquet` as a `test_split` column and reused consistently across scripts.
