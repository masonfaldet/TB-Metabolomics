#!/usr/bin/env python
"""
tb_query_and_preprocess

Query + preprocessing utilities for the TB urine omics star-schema.

This module assumes the Parquet tables produced by `tb_peak_to_parquets`:

    samples.parquet   : sample_id, condition, symptomatic, mean_osmoality, test_split
    features.parquet  : feature_id, mz, rt, n_obs, internal
    abundances.parquet: sample_id, feature_id, area, rt, mz, peakset_version,
                        internal_standard

Key capabilities
----------------
1. Query the star schema to produce ML-ready matrices:
       X : (n_samples × n_features) feature matrix (areas)
       y : (n_samples,) label array (e.g., condition or symptomatic)

2. Transformer classes that can be chained in scikit-learn Pipelines:
       - Log2WithPseudocount   : log2(x + eps) with train-aware eps per feature.
       - TBNormalizer          : per-sample osmolality + internal-standard
                                 normalization.
       - LabelAgnosticTwoStepImputer : two-step KNN imputer with low-value
                                       noise fill.

3. Standalone helpers for feature filtering by missingness:
       - filter_overall_missingness
       - filter_groupwise_missingness

The design is intentionally light-weight and pandas-friendly:
- `make_tb_dataset` returns X as a DataFrame indexed by sample_id and columns
  given by feature_id.
- Transformers accept and return either numpy arrays or pandas DataFrames.
  When a DataFrame is passed, column names and indices are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# I/O and query layer
# ---------------------------------------------------------------------

def load_tables(root: PathLike) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three TB star-schema tables from a directory.

    Parameters
    ----------
    root:
        Directory containing `samples.parquet`, `features.parquet`,
        and `abundances.parquet`.

    Returns
    -------
    samples, abundances, features : tuple of DataFrames
    """
    root = Path(root)
    samples = pd.read_parquet(root / "samples.parquet")
    abund = pd.read_parquet(root / "abundances.parquet")
    features = pd.read_parquet(root / "features.parquet")
    return samples, abund, features


def query_samples(
    samples: pd.DataFrame,
    *,
    conditions: Optional[Sequence[str]] = None,
    symptomatic: Optional[Sequence[str]] = None,
    split: Optional[str] = "train",
) -> pd.DataFrame:
    """
    Filter the samples table and return a subset.

    Parameters
    ----------
    samples:
        Full samples table.
    conditions:
        Optional list of condition values to keep, e.g. ["control", "activated"].
        If None, no condition-based filtering is applied.
    symptomatic:
        Optional list of symptomatic values to keep, typically ["+", "-"] or a
        subset. If None, no symptomatic-based filtering is applied.
    split:
        Which split to keep:
        - "train" (default): keep samples with test_split == False
        - "test"          : keep samples with test_split == True
        - "all" or None   : ignore test_split and keep all rows.

    Returns
    -------
    pd.DataFrame
        Filtered samples table. Raises ValueError if the result is empty.
    """
    S = samples.copy()

    if conditions is not None:
        S = S[S["condition"].isin(conditions)]

    if symptomatic is not None:
        S = S[S["symptomatic"].isin(symptomatic)]

    if split is None or split == "all":
        pass
    elif split == "train":
        if "test_split" not in S.columns:
            raise KeyError("samples table does not have 'test_split' column.")
        S = S[~S["test_split"]]
    elif split == "test":
        if "test_split" not in S.columns:
            raise KeyError("samples table does not have 'test_split' column.")
        S = S[S["test_split"]]
    else:
        raise ValueError(f"Unrecognized split='{split}', expected 'train', 'test', or 'all'.")

    if S.empty:
        raise ValueError("No samples match the filters provided.")

    return S


def pivot_matrix(
    abund: pd.DataFrame,
    sample_ids: Sequence[str],
    *,
    value_col: str = "area",
) -> pd.DataFrame:
    """
    Pivot abundances into a samples × features matrix.

    Parameters
    ----------
    abund:
        Abundances table.
    sample_ids:
        Ordered list of sample IDs to include as rows.
    value_col:
        Column of `abund` to use as the values (default: "area").

    Returns
    -------
    pd.DataFrame
        DataFrame with index = sample_id, columns = feature_id, values = area.
        The index order matches the order of `sample_ids` (missing rows are
        filled with all-NA).
    """
    A = abund[abund["sample_id"].isin(sample_ids)]
    if A.empty:
        raise ValueError("No abundances for the requested sample_ids.")
    Xw = A.pivot(index="sample_id", columns="feature_id", values=value_col)
    # Ensure row order and include samples that might be entirely missing
    Xw = Xw.reindex(index=pd.Index(sample_ids, name="sample_id")).sort_index()
    # Sort columns (feature_id) for reproducibility
    Xw = Xw.sort_index(axis=1)
    return Xw


def make_tb_dataset(
    root: PathLike,
    *,
    conditions: Optional[Sequence[str]] = None,
    symptomatic: Optional[Sequence[str]] = None,
    split: Optional[str] = "train",
    label_col: str = "condition",
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Query the TB star-schema and return an ML-ready (X, y) pair plus metadata.

    Parameters
    ----------
    root:
        Directory containing the Parquet tables.
    conditions:
        Optional list of conditions to keep, e.g. ["control", "activated"].
    symptomatic:
        Optional list of symptomatic values to keep, e.g. ["+", "-"].
    split:
        "train", "test", "all", or None. See `query_samples`.
    label_col:
        Column in the samples table to use as the target labels, typically
        "condition" or "symptomatic".

    Returns
    -------
    X_df : pd.DataFrame
        samples × features matrix (areas) with index = sample_id and
        columns = feature_id.
    y : np.ndarray
        Array of labels aligned with the rows of X_df.
    samples_sub : pd.DataFrame
        Subset of the samples table used to build X_df.
    features : pd.DataFrame
        Full features table.
    """
    samples, abund, features = load_tables(root)
    S = query_samples(samples, conditions=conditions, symptomatic=symptomatic, split=split)
    sample_ids = S["sample_id"].astype(str).tolist()

    X_df = pivot_matrix(abund, sample_ids=sample_ids, value_col="area")

    if label_col not in S.columns:
        raise KeyError(f"label_col='{label_col}' not found in samples table.")

    # Align labels to rows of X_df
    y = S.set_index("sample_id").loc[X_df.index, label_col].to_numpy()
    return X_df, y, S, features


# ---------------------------------------------------------------------
# Normalization transformers
# ---------------------------------------------------------------------
class TBNormalizer(BaseEstimator, TransformerMixin):
    """
    TB-specific normalization: osmolality + internal standard.

    This transformer expects X to be a DataFrame with:
        - index = sample_id (strings),
        - columns = feature_id (ints).

    Two steps are applied in `transform`:

    1. Osmolality normalization:
       For each sample, divide its feature vector by that sample's
       `mean_osmoality` taken from the samples table.

    2. Internal standard normalization:
       For each sample, divide its (already osmo-normalized) feature vector
       by the *raw* internal-standard feature value for that sample
       (feature_id = `internal_standard_feature_id`). Optionally drop the
       internal-standard column after normalization.

    So for feature j in sample i, the output is approximately:

        X_out[i, j] ≈ A_ij / (mean_osmo_i * IS_i)
    """

    def __init__(
        self,
        samples_df: pd.DataFrame,
        internal_standard_feature_id: int = 9500,
        drop_internal_standard: bool = True,
    ):
        self.samples_df = samples_df.copy()
        self.internal_standard_feature_id = int(internal_standard_feature_id)
        self.drop_internal_standard = bool(drop_internal_standard)

    def fit(self, X, y=None):
        # No parameters to learn; just basic validation.
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "TBNormalizer expects X to be a pandas DataFrame with "
                "index=sample_id and columns=feature_id."
            )
        missing_samples = set(X.index.astype(str)) - set(
            self.samples_df["sample_id"].astype(str)
        )
        if missing_samples:
            raise ValueError(
                f"TBNormalizer: samples_df is missing {len(missing_samples)} "
                "sample_ids present in X."
            )
        self.feature_names_in_ = list(X.columns)
        self.sample_index_in_ = X.index.astype(str)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TBNormalizer.transform expects a pandas DataFrame.")

        # Work on a float copy
        Xn = X.astype(float).copy()

        # Align samples_df to X rows
        S = (
            self.samples_df.copy()
            .assign(sample_id=self.samples_df["sample_id"].astype(str))
            .set_index("sample_id")
            .reindex(Xn.index.astype(str))
        )
        osmo = S["mean_osmoality"].to_numpy(dtype=float)

        # Grab RAW internal-standard values BEFORE any normalization
        std_vals_raw = None
        valid_std_raw = None
        if self.internal_standard_feature_id in Xn.columns:
            std_vals_raw = Xn[self.internal_standard_feature_id].to_numpy(dtype=float)
            valid_std_raw = np.isfinite(std_vals_raw) & (std_vals_raw > 0)

        # Osmolality normalization (row-wise)
        X_arr = Xn.to_numpy(dtype=np.float32)
        valid_osmo = np.isfinite(osmo) & (osmo > 0)

        for i in range(X_arr.shape[0]):
            if valid_osmo[i]:
                X_arr[i, :] /= osmo[i]

        # Internal standard normalization (row-wise) using RAW IS values
        if std_vals_raw is not None:
            for i in range(X_arr.shape[0]):
                if valid_osmo[i] and valid_std_raw[i]:
                    # At this point X_arr[i, :] ~ A_ij / osmo_i
                    # Dividing by std_vals_raw[i] (~ IS_i) gives:
                    #   A_ij / (osmo_i * IS_i)
                    X_arr[i, :] /= std_vals_raw[i]

        Xn.iloc[:, :] = X_arr

        if std_vals_raw is not None and self.drop_internal_standard:
            Xn = Xn.drop(columns=[self.internal_standard_feature_id])

        return Xn


# ---------------------------------------------------------------------
# Log2 transform with train-aware pseudocount
# ---------------------------------------------------------------------


class Log2WithPseudocount(BaseEstimator, TransformerMixin):
    """
    Apply log2(x + eps) with a per-feature pseudocount eps.

    For each feature j, eps_j is defined during `fit` as:

        eps_j = 0.2 * min_positive_j

    where min_positive_j is the smallest strictly positive value observed in
    the training data for that feature. If no positive values are observed,
    a global fallback `fallback_eps` is used.

    This is intended to be used inside a scikit-learn Pipeline so that the
    pseudocounts are estimated on the training split only and then applied
    consistently to both train and test.
    """

    def __init__(self, fallback_eps: float = 1e-3):
        self.fallback_eps = float(fallback_eps)
        self.eps_: Optional[np.ndarray] = None
        self.feature_names_in_: Optional[List] = None

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            return X.to_numpy(dtype=np.float32)
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y=None):
        X_arr = self._to_array(X)
        n, p = X_arr.shape
        eps = np.empty(p, dtype=np.float32)

        for j in range(p):
            col = X_arr[:, j]
            pos = col[col > 0]
            if pos.size > 0:
                eps_j = 0.2 * float(np.nanmin(pos))
            else:
                eps_j = self.fallback_eps
            if not np.isfinite(eps_j) or eps_j <= 0:
                eps_j = self.fallback_eps
            eps[j] = eps_j

        self.eps_ = eps
        return self

    def transform(self, X):
        if self.eps_ is None:
            raise RuntimeError("Log2WithPseudocount must be fitted before transform.")

        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            cols = list(X.columns)
            arr = X.to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)
            cols = None

        if arr.shape[1] != self.eps_.shape[0]:
            raise ValueError(
                f"Expected {self.eps_.shape[0]} features, got {arr.shape[1]}."
            )

        transformed = np.log2(arr + self.eps_)
        if is_df:
            return pd.DataFrame(transformed, index=X.index, columns=cols)
        return transformed


# ---------------------------------------------------------------------
# Label-agnostic two-step imputation
# ---------------------------------------------------------------------


class LabelAgnosticTwoStepImputer(BaseEstimator, TransformerMixin):
    """
    Two-step label-agnostic imputation compatible with scikit-learn.

    Step 1 (fit on train only)
        For each feature whose missingness fraction is >= `nan_threshold`,
        replace missing entries with:

            (min_observed / 5) + U(0, noise_scale * (min_observed / 5))

        where min_observed is the minimum observed (non-NaN) value for that
        feature in the training data.

    Step 2
        Fit a KNNImputer on the step-1-imputed training matrix. During
        `transform`, we apply step 1 using the train-time min values and
        then apply the fitted KNNImputer.

    Notes
    -----
    - This transformer is label-agnostic in that it never uses class labels.
    - All statistics are learned only from the data passed to `fit` (typically
      the training split in a pipeline).
    """

    def __init__(
        self,
        noise_scale: float = 0.01,
        nan_threshold: float = 0.5,
        k: int = 5,
        random_state: Optional[int] = None,
    ):
        self.noise_scale = float(noise_scale)
        self.nan_threshold = float(nan_threshold)
        self.k = int(k)
        self.random_state = random_state

        # Fitted attributes
        self.min_impute_: Optional[dict] = None  # {col_index: min_val}
        self.step1_cols_: Optional[np.ndarray] = None  # boolean mask
        self.imputer_: Optional[KNNImputer] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List] = None

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            return X.to_numpy(dtype=np.float32)
        return np.asarray(X, dtype=np.float32)

    def _rng(self):
        return np.random.default_rng(self.random_state)

    def fit(self, X, y=None):
        X_arr = self._to_array(X)
        n, p = X_arr.shape
        self.n_features_in_ = p

        nan_frac = np.isnan(X_arr).mean(axis=0)
        step1_cols = nan_frac >= self.nan_threshold
        self.step1_cols_ = step1_cols

        min_impute = {}
        for j in range(p):
            if not step1_cols[j]:
                continue
            col = X_arr[:, j]
            obs = col[~np.isnan(col)]
            if obs.size == 0:
                continue
            min_impute[j] = float(np.nanmin(obs))
        self.min_impute_ = min_impute

        # Apply step1 to training data before fitting KNN
        X_step1 = X_arr.copy()
        rng = self._rng()
        for j, min_val in self.min_impute_.items():
            col = X_step1[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                base = float(min_val) / 5.0
                if base < 0:
                    noise = rng.uniform(
                        low = self.noise_scale * base, high = 0.0, size = int(nan_mask.sum())
                    )
                else:
                    noise = rng.uniform(
                        0.0, self.noise_scale * base, size=int(nan_mask.sum())
                    )
                col[nan_mask] = base + noise
                X_step1[:, j] = col

        self.imputer_ = KNNImputer(n_neighbors=self.k)
        self.imputer_.fit(X_step1)
        return self

    def transform(self, X):
        if self.imputer_ is None or self.min_impute_ is None:
            raise RuntimeError("LabelAgnosticTwoStepImputer must be fitted first.")

        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            idx = X.index
            cols = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            idx = None
            cols = None

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_arr.shape[1]}."
            )

        X_step1 = X_arr.copy()
        rng = self._rng()
        for j, min_val in self.min_impute_.items():
            col = X_step1[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                base = float(min_val) / 5.0
                if base < 0:
                    noise = rng.uniform(
                        low = self.noise_scale * base, high = 0.0, size = int(nan_mask.sum())
                    )
                else:
                    noise = rng.uniform(
                        0.0, self.noise_scale * base, size=int(nan_mask.sum())
                    )
                col[nan_mask] = base + noise
                X_step1[:, j] = col

        X_imp = self.imputer_.transform(X_step1)

        if is_df:
            return pd.DataFrame(X_imp, index=idx, columns=cols)
        return X_imp


# ---------------------------------------------------------------------
# Missingness-based feature filtering
# ---------------------------------------------------------------------


def filter_overall_missingness(
    X_df: pd.DataFrame,
    max_na_prop: float = 0.5,
) -> pd.DataFrame:
    """
    Filter features based on overall missingness.

    Parameters
    ----------
    X_df:
        samples × features DataFrame.
    max_na_prop:
        Maximum allowed fraction of missing values per feature. Features with
        NA proportion > max_na_prop are dropped.

    Returns
    -------
    pd.DataFrame
        X_df restricted to features passing the missingness criterion.
    """
    na_prop = X_df.isna().mean(axis=0)
    keep = na_prop <= max_na_prop
    return X_df.loc[:, keep]


def filter_groupwise_missingness(
    X_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("condition",),
    min_prop: float = 0.6,
    min_group_n: int = 3,
    require_all_groups: bool = True,
) -> pd.DataFrame:
    """
    Filter features based on missingness within groups (e.g., conditions).

    A feature is kept if, for each sufficiently large group, at least
    `min_prop` of the values in that group are observed (non-NA).

    Parameters
    ----------
    X_df:
        samples × features DataFrame (index = sample_id).
    samples_df:
        Samples metadata table, with one row per sample_id and at least the
        columns listed in `group_cols`.
    group_cols:
        Column names in `samples_df` to define groups, e.g. ("condition",).
        Groups are formed by unique combinations of these columns.
    min_prop:
        Minimum proportion of observed values required in a group for a
        feature to be considered present in that group.
    min_group_n:
        Minimum group size to enforce the missingness criterion. Groups with
        fewer rows than this are ignored.
    require_all_groups:
        If True (default), a feature must pass the threshold in *all*
        sufficiently large groups. If False, a feature is kept if it passes
        in *at least one* sufficiently large group.

    Returns
    -------
    pd.DataFrame
        X_df restricted to features passing the groupwise missingness criterion.
    """
    # Align samples_df to X_df rows
    S = (
        samples_df.copy()
        .assign(sample_id=samples_df["sample_id"].astype(str))
        .set_index("sample_id")
        .reindex(X_df.index.astype(str))
    )

    if S[list(group_cols)].isna().any().any():
        raise ValueError("NaNs found in group_cols in samples_df after alignment.")

    groups = S.groupby(list(group_cols))

    # Start with all features allowed; refine according to require_all_groups flag
    if require_all_groups:
        keep = pd.Series(True, index=X_df.columns)
    else:
        keep = pd.Series(False, index=X_df.columns)

    for _, idx in groups.groups.items():
        idx = list(idx)
        if len(idx) < min_group_n:
            continue
        sub = X_df.loc[idx]
        obs_prop = 1.0 - sub.isna().mean(axis=0)
        group_keep = obs_prop >= min_prop

        if require_all_groups:
            keep &= group_keep
        else:
            keep |= group_keep

    return X_df.loc[:, keep]


# ---------------------------------------------------------------------
# Convenience: build a standard preprocessing pipeline
# ---------------------------------------------------------------------

@dataclass
class TBPreprocessConfig:
    """
    Configuration for a typical TB preprocessing pipeline.

    This is a convenience container; you can also instantiate the individual
    transformers and pipelines manually.
    """

    # Log transform
    fallback_eps: float = 1e-3

    # Normalization
    internal_standard_feature_id: int = 9500
    drop_internal_standard: bool = True

    # Two-step imputation
    noise_scale: float = 0.001
    nan_threshold: float = 0.5
    k_neighbors: int = 4
    random_state: Optional[int] = None

    # Standardization (optional, appended after imputation)
    use_standard_scaler: bool = True
    scaler_with_mean: bool = True
    scaler_with_std: bool = True

def make_tb_preprocess_pipeline(
    samples_df: pd.DataFrame,
    config: Optional[TBPreprocessConfig] = None,
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline implementing:

        1. Log2WithPseudocount
        2. TBNormalizer (osmolality + internal standard)
        3. LabelAgnosticTwoStepImputer
        4. (Optional) StandardScaler

    The pipeline operates on pandas DataFrames and returns a numpy array
    (output of the imputer or scaler). If you want to preserve DataFrames,
    you can remove the scaler/imputer or wrap the pipeline in additional
    adapters.

    Parameters
    ----------
    samples_df:
        Samples metadata table, used by TBNormalizer for osmolality and
        internal-standard normalization.
    config:
        Optional TBPreprocessConfig instance. If None, defaults are used.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline with steps ("log2", "normalize", "impute", ["scale"]).
    """
    if config is None:
        config = TBPreprocessConfig()

    steps: List[Tuple[str, TransformerMixin]] = [

        (
            "normalize",
            TBNormalizer(
                samples_df=samples_df,
                internal_standard_feature_id=config.internal_standard_feature_id,
                drop_internal_standard=config.drop_internal_standard,
            ),
        ),
        ("log2", Log2WithPseudocount(fallback_eps=config.fallback_eps)),
        (
            "impute",
            LabelAgnosticTwoStepImputer(
                noise_scale=config.noise_scale,
                nan_threshold=config.nan_threshold,
                k=config.k_neighbors,
                random_state=config.random_state,
            ),
        ),
    ]

    if config.use_standard_scaler:
        steps.append(
            (
                "scale",
                StandardScaler(
                    with_mean=config.scaler_with_mean,
                    with_std=config.scaler_with_std,
                ),
            )
        )

    return Pipeline(steps)

