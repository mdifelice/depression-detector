import importlib
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


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


def drop_null_columns(df: pd.DataFrame, nullable_columns: list[str]) -> pd.DataFrame:
    cols_to_check = [c for c in df.columns if c not in nullable_columns]
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


def drop_null_columns_by_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if threshold >= 1.0:
        return df
    n_rows = len(df)
    if n_rows == 0:
        return df
    max_null_ratio = threshold
    null_ratios = df.isnull().sum() / n_rows
    cols_to_keep = null_ratios[null_ratios <= max_null_ratio].index.tolist()
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


def preprocess(df: pd.DataFrame, metadata: dict, training_settings: dict) -> pd.DataFrame:
    logger.info(f"Starting preprocessing: {len(df)} rows, {len(df.columns)} columns")

    ignore_cols = metadata.get("ignore_columns", [])
    df = remove_columns(df, ignore_cols)
    logger.info(f"After removing ignored columns: {len(df.columns)} columns")

    nullable_cols = metadata.get("nullable_columns", [])
    df = fill_nullable_columns(df, nullable_cols)

    df = drop_null_columns(df, nullable_cols)
    logger.info(f"After dropping null columns: {len(df.columns)} columns")

    row_threshold = training_settings.get("row_acceptance_threshold", 0.75)
    before = len(df)
    df = drop_null_rows(df, row_threshold)
    logger.info(f"After dropping null rows (threshold={row_threshold}): {before} -> {len(df)} rows")

    col_threshold = training_settings.get("column_acceptance_threshold", 0.25)
    before_cols = len(df.columns)
    df = drop_null_columns_by_threshold(df, col_threshold)
    logger.info(f"After dropping null columns by threshold ({col_threshold}): {before_cols} -> {len(df.columns)} columns")

    df = df.drop_duplicates()
    logger.info(f"After dedup: {len(df)} rows")

    logger.info(f"Preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
    return df
