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


def load_dataset(dataset_path: str) -> pd.DataFrame:
    if dataset_path.endswith(".csv"):
        return pd.read_csv(dataset_path)
    elif dataset_path.endswith(".xlsx"):
        return pd.read_excel(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")


def remove_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [c for c in columns if c in df.columns]
    return df.drop(columns=existing)


def drop_null_columns(
    df: pd.DataFrame, nullable_columns: list[str], target_column: str | None = None
) -> pd.DataFrame:
    cols_to_check = [
        c for c in df.columns if c not in nullable_columns and c != target_column
    ]
    drop_cols = [c for c in cols_to_check if df[c].isnull().any()]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"Dropping columns with null values: {drop_cols}")
    return df


def drop_null_rows(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if threshold >= 1.0:
        return df.dropna()
    n_cols = len(df.columns)
    max_nulls = int(np.floor(n_cols * (1 - threshold)))
    return df[df.isnull().sum(axis=1) <= max_nulls]


def drop_null_columns_by_threshold(
    df: pd.DataFrame, threshold: float, target_column: str | None = None
) -> pd.DataFrame:
    if threshold >= 1.0:
        return df
    n_rows = len(df)
    if n_rows == 0:
        return df
    max_null_ratio = threshold
    null_ratios = df.isnull().sum() / n_rows
    cols_to_keep = null_ratios[null_ratios <= max_null_ratio].index.tolist()
    if target_column and target_column in df.columns and target_column not in cols_to_keep:
        cols_to_keep.append(target_column)
    return df[cols_to_keep]


def fill_nullable_columns(df: pd.DataFrame, nullable_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in nullable_columns:
        if col not in df.columns:
            continue
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
    return df


def preprocess(
    df: pd.DataFrame,
    metadata: dict,
    training_settings: dict,
    log_fn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    logger.info(f"Starting preprocessing: {len(df)} rows, {len(df.columns)} columns")

    ignore_cols = metadata.get("ignore_columns", [])
    df = remove_columns(df, ignore_cols)
    _log_shape(log_fn, "After removing ignored columns", df)

    nullable_cols = metadata.get("nullable_columns", [])
    target_column = metadata.get("target_column")
    df = fill_nullable_columns(df, nullable_cols)
    _log_shape(log_fn, "After filling nullable columns", df)

    row_threshold = training_settings.get("row_acceptance_threshold", 0.75)
    df = drop_null_rows(df, row_threshold)
    _log_shape(log_fn, f"After dropping null rows (threshold={row_threshold})", df)

    df = drop_null_columns(df, nullable_cols, target_column)
    _log_shape(log_fn, "After dropping columns with nulls (not marked nullable)", df)

    col_threshold = training_settings.get("column_acceptance_threshold", 0.25)
    df = drop_null_columns_by_threshold(df, col_threshold, target_column)
    _log_shape(log_fn, f"After dropping columns above null threshold ({col_threshold})", df)

    df = df.drop_duplicates()
    _log_shape(log_fn, "After removing duplicate rows", df)

    _log_shape(log_fn, "Preprocessing complete", df)
    return df
