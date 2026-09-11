import importlib
import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


def _log_shape(log_fn: Callable[[str], None] | None, prefix: str, df: pd.DataFrame):
    msg = f"{prefix}: {len(df)} rows × {len(df.columns)} columns"
    if log_fn:
        log_fn(msg)
    logger.info(msg)


def _log_columns(
    log_fn: Callable[[str], None] | None,
    prefix: str,
    columns: list[str],
    detail: str | None = None,
):
    if not columns:
        return
    msg = f"{prefix}: {columns}"
    if detail:
        msg += f" ({detail})"
    if log_fn:
        log_fn(msg)
    logger.info(msg)


def load_dataset(dataset_path: str) -> pd.DataFrame:
    if dataset_path.endswith(".csv"):
        return pd.read_csv(dataset_path)
    elif dataset_path.endswith(".xlsx"):
        return pd.read_excel(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")


def remove_columns(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    existing = [c for c in columns if c in df.columns]
    df = df.drop(columns=existing)
    return df, existing


def drop_null_rows(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, int]:
    n_before = len(df)
    if threshold >= 1.0:
        df = df.dropna()
    else:
        n_cols = len(df.columns)
        max_nulls = int(np.floor(n_cols * (1 - threshold)))
        df = df[df.isnull().sum(axis=1) <= max_nulls]
    return df, n_before - len(df)


def drop_null_columns_by_threshold(
    df: pd.DataFrame, threshold: float, target_column: str | None = None
) -> tuple[pd.DataFrame, list[tuple[str, float]]]:
    if threshold >= 1.0:
        return df, []
    n_rows = len(df)
    if n_rows == 0:
        return df, []
    max_null_ratio = threshold
    null_ratios = df.isnull().sum() / n_rows
    cols_to_keep = null_ratios[null_ratios <= max_null_ratio].index.tolist()
    if target_column and target_column in df.columns and target_column not in cols_to_keep:
        cols_to_keep.append(target_column)
    dropped = [
        (col, float(null_ratios[col]))
        for col in df.columns
        if col not in cols_to_keep
    ]
    return df[cols_to_keep], dropped


def fill_missing_values(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    filled = []
    for col in columns:
        if col not in df.columns:
            continue
        if not df[col].isnull().any():
            continue
        filled.append(col)
        if df[col].dtype in ["float64", "int64", "float32", "int32"]:
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == "object":
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if len(mode) > 0 else "")
        else:
            try:
                df[col] = pd.to_datetime(df[col])
                df[col] = df[col].fillna(pd.Timestamp("1970-01-01"))
            except Exception:
                df[col] = df[col].fillna("")
    return df, filled


def preprocess(
    df: pd.DataFrame,
    metadata: dict,
    training_settings: dict,
    log_fn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    logger.info(f"Starting preprocessing: {len(df)} rows, {len(df.columns)} columns")

    ignore_cols = metadata.get("ignore_columns", [])
    df, removed = remove_columns(df, ignore_cols)
    _log_shape(log_fn, "After removing ignored columns", df)
    _log_columns(log_fn, "Removed columns (ignored in config)", removed)

    row_threshold = training_settings.get("row_acceptance_threshold", 0.75)
    df, n_dropped_rows = drop_null_rows(df, row_threshold)
    _log_shape(log_fn, f"After dropping null rows (threshold={row_threshold})", df)
    if n_dropped_rows:
        _log_columns(log_fn, "Dropped rows", [str(n_dropped_rows)])

    target_column = metadata.get("target_column")
    col_threshold = training_settings.get("column_acceptance_threshold", 0.25)
    df, dropped_ratios = drop_null_columns_by_threshold(df, col_threshold, target_column)
    _log_shape(log_fn, f"After dropping columns above null threshold ({col_threshold})", df)
    if dropped_ratios:
        detail = ", ".join(f"{c} ({r:.1%} null)" for c, r in dropped_ratios)
        _log_columns(log_fn, "Removed columns (above null threshold)", [c for c, _ in dropped_ratios], detail)

    remaining_cols = [c for c in df.columns if c != target_column]
    df, filled_cols = fill_missing_values(df, remaining_cols)
    _log_shape(log_fn, "After filling remaining nulls", df)
    _log_columns(log_fn, "Filled missing values", filled_cols)

    df = df.drop_duplicates()
    _log_shape(log_fn, "After removing duplicate rows", df)

    _log_shape(log_fn, "Preprocessing complete", df)
    return df