#!/usr/bin/env python
"""tb_peak_to_parquets

Convert a TB urine peaklist CSV and metadata Excel file into a minimal
omics star-schema database with three Parquet tables:

- abundances.parquet
- features.parquet
- samples.parquet
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd


# Samples to exclude from ALL outputs
EXCLUDED_SAMPLE_IDS: Set[str] = {"02347", "02338"}


def _parse_datafile_columns(columns: List[str]) -> pd.DataFrame:
    """Parse peaklist columns of the form
    'datafile:...CSU-TB-{sample_id}.mzML:{dtype}'.

    Returns
    -------
    DataFrame
        Columns: ['column', 'sample_id', 'dtype'].
    """
    records = []
    pattern = re.compile(r"^datafile:.*CSU-TB-(\d+)\.mzML:(area|rt|mz)$")

    for col in columns:
        m = pattern.match(col)
        if m:
            raw_id = m.group(1)
            dtype = m.group(2).lower()
            # Canonical 5-digit sample_id like '02339'
            sample_id = raw_id.zfill(5)
            records.append((col, sample_id, dtype))

    return pd.DataFrame(records, columns=["column", "sample_id", "dtype"])


def _build_abundances(
    peak_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    valid_sample_ids: Set[str],
    peakset_version: str,
) -> pd.DataFrame:
    """Build abundances.parquet table.

    Columns
    -------
    sample_id : str
    feature_id : int
    area : float
    rt : float
    mz : float
    peakset_version : str
    internal_standard : bool
    """
    # Keep only mapping rows for valid samples
    mapping_df = mapping_df[mapping_df["sample_id"].isin(valid_sample_ids)].copy()

    melted_by_dtype: Dict[str, pd.DataFrame] = {}

    for dtype in ("area", "rt", "mz"):
        cols = mapping_df[mapping_df["dtype"] == dtype]
        if cols.empty:
            continue

        use_cols = ["id"] + cols["column"].tolist()
        sub = peak_df[use_cols].copy()

        melted = sub.melt(id_vars=["id"], var_name="column", value_name=dtype)
        melted = melted.merge(cols[["column", "sample_id"]], on="column", how="left")
        melted = melted.rename(columns={"id": "feature_id"})
        melted = melted[["feature_id", "sample_id", dtype]]

        melted_by_dtype[dtype] = melted

    # Outer-merge the three dtypes into a single long table
    abundances: pd.DataFrame | None = None
    for dtype, df in melted_by_dtype.items():
        if abundances is None:
            abundances = df
        else:
            abundances = abundances.merge(
                df, on=["feature_id", "sample_id"], how="outer"
            )

    if abundances is None:
        raise ValueError("No datafile:* columns were found in the peaklist CSV.")

    # Ensure correct dtypes
    abundances["feature_id"] = abundances["feature_id"].astype(int)
    abundances["sample_id"] = abundances["sample_id"].astype(str)

    # Add peakset_version and internal_standard flag
    abundances["peakset_version"] = str(peakset_version)
    # internal_standard is True iff feature_id == 9500
    abundances["internal_standard"] = abundances["feature_id"] == 9500

    # Sort for readability
    abundances = abundances.sort_values(["sample_id", "feature_id"]).reset_index(
        drop=True
    )

    return abundances


def _determine_condition(sample_type: str, visit_day: str) -> str:
    """Determine condition in {"control", "activated", "incident", "prevalent"}.

    Logic (st <- SampleType, vd <- VisitDay)
    ----------------------------------------
    if 'Control'   in st: condition = control
    elif 'Incident' in st and 'Activation' not in vd: condition = incident
    elif 'Prevalent' in st: condition = prevalent
    elif 'Activation' in vd: condition = activated

    If none of the above apply, returns "unknown".
    """
    st = (sample_type or "").upper()
    vd = (visit_day or "").upper()

    if "CONTROL" in st:
        return "control"
    elif "INCIDENT" in st and "ACTIVATION" not in vd:
        return "incident"
    elif "PREVALENT" in st:
        return "prevalent"
    elif "ACTIVATION" in vd:
        return "activated"
    else:
        return "unknown"


def _determine_symptomatic(tbsymptom: str) -> str:
    """Map tbsymptom value to symptomatic in {"+", "-"}.

    NEGATIVE -> "-"
    POSITIVE -> "+"
    anything else -> "unknown"
    """
    ts = (tbsymptom or "").upper().strip()
    if ts == "NEGATIVE":
        return "-"
    if ts == "POSITIVE":
        return "+"
    return "unknown"


def _build_samples(
    meta_df: pd.DataFrame,
    valid_sample_ids: Set[str],
    test_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    """Build samples.parquet table.

    Columns
    -------
    sample_id : str
    condition : {"control", "activated", "incident", "prevalent", "unknown"}
    symptomatic : {"+", "-", "unknown"}
    mean_osmoality : float
    test_split : bool
    """
    # Extract sample_id from CSUID, canonical 5-digit string
    sample_id_series = meta_df["CSUID"].astype(str).str.extract(
        r"CSU-TB-(\d+)"
    )[0].dropna()
    meta_df = meta_df.loc[sample_id_series.index].copy()
    meta_df["sample_id"] = sample_id_series.astype(str).str.zfill(5)

    # Keep only valid samples (intersection with peaklist and not excluded)
    meta_df = meta_df[meta_df["sample_id"].isin(valid_sample_ids)].copy()

    # Condition from SampleType and VisitDay
    meta_df["condition"] = meta_df.apply(
        lambda row: _determine_condition(row.get("SampleType"), row.get("VisitDay")),
        axis=1,
    )

    # Symptomatic from tbsymptom
    meta_df["symptomatic"] = meta_df["tbsymptom"].apply(_determine_symptomatic)

    # Mean osmolality from OsmolReading1/2/3
    osmo_cols = ["OsmolReading1", "OsmolReading2", "OsmolReading3"]
    for c in osmo_cols:
        if c not in meta_df.columns:
            raise KeyError(f"Expected column '{c}' not found in metadata.")
    meta_df[osmo_cols] = meta_df[osmo_cols].apply(pd.to_numeric, errors="coerce")
    meta_df["mean_osmoality"] = meta_df[osmo_cols].mean(axis=1)

    # Start samples table
    samples = meta_df[
        ["sample_id", "condition", "symptomatic", "mean_osmoality"]
    ].copy()

    # Stratified train/test split on condition
    rng = np.random.RandomState(random_state)
    samples["test_split"] = False

    for condition, idx in samples.groupby("condition").groups.items():
        idx = np.array(list(idx))
        n_total = len(idx)
        if n_total == 0:
            continue
        n_test = int(np.floor(n_total * test_ratio))
        # Ensure at least one test sample when possible
        if n_test == 0 and n_total > 1:
            n_test = 1
        if n_test == 0:
            continue

        perm = rng.permutation(idx)
        test_idx = perm[:n_test]
        samples.loc[test_idx, "test_split"] = True

    # Sort for readability
    samples = samples.sort_values("sample_id").reset_index(drop=True)

    return samples


def _build_features(peak_df: pd.DataFrame, abundances: pd.DataFrame) -> pd.DataFrame:
    """Build features.parquet table.

    Columns
    -------
    feature_id : int
    mz : float
    rt : float
    n_obs : int
    internal : bool

    where n_obs is the number of non-NaN area measurements across all samples
    and internal is True iff feature_id == 9500.
    """
    if "id" not in peak_df.columns:
        raise KeyError("Expected column 'id' in peaklist CSV.")
    if "mz" not in peak_df.columns or "rt" not in peak_df.columns:
        raise KeyError("Expected columns 'mz' and 'rt' in peaklist CSV.")

    features = peak_df[["id", "mz", "rt"]].copy()
    features = features.rename(columns={"id": "feature_id"})

    # n_obs: number of non-NaN area measurements per feature across all samples
    n_obs = (
        abundances.groupby("feature_id")["area"]
        .apply(lambda s: s.notna().sum())
        .rename("n_obs")
    )
    features = features.merge(n_obs, on="feature_id", how="left")
    features["n_obs"] = features["n_obs"].fillna(0).astype(int)

    # Internal standard flag: same rule as abundances (feature_id == 9500)
    features["internal"] = features["feature_id"] == 9500

    # Sort
    features = features.sort_values("feature_id").reset_index(drop=True)

    return features


def build_star_schema(
    peaklist_csv: str | Path,
    metadata_xlsx: str | Path,
    out_dir: str | Path,
    test_ratio: float = 0.2,
    random_state: int = 42,
    peakset_version: str | None = None,
) -> None:
    """Build the omics star-schema (abundances, features, samples) and write Parquet files.

    Parameters
    ----------
    peaklist_csv : str or Path
        Path to the peaklist CSV.
    metadata_xlsx : str or Path
        Path to the metadata Excel file.
    out_dir : str or Path
        Directory where the Parquet files will be written.
    test_ratio : float, optional
        Fraction of samples to assign to the test split, by condition.
    random_state : int, optional
        Random seed for the stratified split.
    peakset_version : str, optional
        Label for peakset_version column in abundances; if None, the stem of
        the peaklist CSV filename is used.
    """
    peaklist_csv = Path(peaklist_csv)
    metadata_xlsx = Path(metadata_xlsx)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if peakset_version is None:
        peakset_version = peaklist_csv.stem

    # --- Peaklist ---
    peak_df = pd.read_csv(peaklist_csv)

    if "id" not in peak_df.columns:
        raise KeyError("Peaklist CSV must contain an 'id' column.")

    mapping_df = _parse_datafile_columns(list(peak_df.columns))
    if mapping_df.empty:
        raise ValueError(
            "No 'datafile:*CSU-TB-*.mzML:*' columns found in the peaklist CSV."
        )

    sample_ids_from_peak = set(mapping_df["sample_id"].unique())

    # --- Metadata ---
    meta_df = pd.read_excel(metadata_xlsx, sheet_name="UrineUnblinded")
    sample_id_series = meta_df["CSUID"].astype(str).str.extract(
        r"CSU-TB-(\d+)"
    )[0].dropna()
    sample_ids_from_meta = set(
        sample_id_series.astype(str).str.zfill(5).unique()
    )

    # Valid sample_ids = intersection minus excluded
    valid_sample_ids = sample_ids_from_peak & sample_ids_from_meta
    valid_sample_ids = valid_sample_ids - EXCLUDED_SAMPLE_IDS

    if not valid_sample_ids:
        raise ValueError(
            "No overlapping samples between peaklist and metadata after applying "
            f"exclusions {sorted(EXCLUDED_SAMPLE_IDS)}."
        )

    # --- Build tables ---
    abundances = _build_abundances(
        peak_df=peak_df,
        mapping_df=mapping_df,
        valid_sample_ids=valid_sample_ids,
        peakset_version=peakset_version,
    )

    # Restrict abundances again to valid samples (defensive)
    abundances = abundances[abundances["sample_id"].isin(valid_sample_ids)].copy()

    samples = _build_samples(
        meta_df=meta_df,
        valid_sample_ids=valid_sample_ids,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    # Restrict abundances to samples that made it into samples table
    valid_sample_ids_final = set(samples["sample_id"].unique())
    abundances = abundances[
        abundances["sample_id"].isin(valid_sample_ids_final)
    ].copy()

    features = _build_features(peak_df, abundances)

    # --- Write Parquet files ---
    abundances_path = out_dir / "abundances.parquet"
    features_path = out_dir / "features.parquet"
    samples_path = out_dir / "samples.parquet"

    abundances.to_parquet(abundances_path, index=False)
    features.to_parquet(features_path, index=False)
    samples.to_parquet(samples_path, index=False)


if __name__ == "__main__":

    peaklist_csv = "../Data/Raw/peaklist.csv"
    metadata_xlsx = "../Data/Raw/metadata.xlsx"
    out_dir = "../Data/Parquets/"
    test_ratio = 0.2
    random_state = 42
    peakset_version = None


    build_star_schema(
        peaklist_csv=peaklist_csv,
        metadata_xlsx=metadata_xlsx,
        out_dir=out_dir,
        test_ratio=test_ratio,
        random_state=random_state,
        peakset_version=peakset_version,
    )
