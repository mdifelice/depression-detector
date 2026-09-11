import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger(__name__)


def _log_shape(log_fn: Callable[[str], None] | None, prefix: str, df: pd.DataFrame):
    msg = f"{prefix}: {len(df)} rows × {len(df.columns)} columns"
    if log_fn:
        log_fn(msg)
    logger.info(msg)


def _normalize_match(value: str) -> str:
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return value


def _random_order(values, random_seed: int) -> list:
    vals = list(values)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(vals)
    return vals


def binarize_target(
    df: pd.DataFrame, target_column: str, positive_values: list[str]
) -> pd.DataFrame:
    df = df.copy()
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    raw_str = df[target_column].astype(str).str.strip()
    is_null = df[target_column].isna()
    if positive_values:
        norm_raw = raw_str.map(_normalize_match)
        norm_pos = {_normalize_match(v) for v in positive_values}
        df[target_column] = norm_raw.isin(norm_pos).astype(int)
    else:
        df[target_column] = ((~is_null) & raw_str.ne("")).astype(int)

    logger.info(f"Binarized target '{target_column}': {df[target_column].value_counts().to_dict()}")
    return df


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:
                    datetime_cols.append(col)
            except Exception:
                pass
    return datetime_cols


def transform_datetime(df: pd.DataFrame, datetime_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") // 10**9
            logger.info(f"Converted datetime column '{col}' to timestamp")
    return df


def ordinal_encode(
    df: pd.DataFrame,
    categorical_columns: dict,
    target_column: str | None = None,
    random_seed: int = 123,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    mappings = {}
    for col, config in categorical_columns.items():
        if col == target_column:
            continue
        if col not in df.columns:
            continue
        if not config.get("ordinal", True):
            continue
        order = config.get("order", [])
        if order:
            mapping = {v: i for i, v in enumerate(order)}
            df[col] = df[col].map(mapping)
            unmapped = df[col].isna().sum()
            if unmapped > 0:
                df[col] = df[col].fillna(-1).astype(int)
            else:
                df[col] = df[col].astype(int)
            mappings[col] = mapping
            logger.info(f"Ordinal encoded '{col}': {mapping}")
        else:
            unique_vals = _random_order(df[col].dropna().unique(), random_seed)
            if unique_vals:
                mapping = {v: i for i, v in enumerate(unique_vals)}
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
                mappings[col] = mapping
                logger.info(
                    f"Ordinal encoded '{col}' (random order, seed={random_seed}): {mapping}"
                )
    return df, mappings


def one_hot_encode(
    df: pd.DataFrame,
    multi_value_columns: list[str],
    categorical_columns: dict,
    max_unique: int,
    fallback_mappings: dict,
    target_column: str | None = None,
    random_seed: int = 123,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    ohe_cols_generated = []

    for col in multi_value_columns:
        if col == target_column:
            continue
        if col not in df.columns:
            continue
        expanded = df[col].astype(str).str.split(",", expand=False)
        all_values = set()
        for vals in expanded:
            for v in vals:
                all_values.add(v.strip())
        if 2 < len(all_values) <= max_unique:
            for val in sorted(all_values):
                new_col = f"{col}__{val}"
                df[new_col] = df[col].astype(str).str.contains(val, regex=False).astype(int)
                ohe_cols_generated.append(new_col)
            df = df.drop(columns=[col])
            logger.info(f"One-hot encoded multi-value '{col}': {len(all_values)} values")

    for col, config in categorical_columns.items():
        if col == target_column:
            continue
        if col not in df.columns:
            continue
        if config.get("ordinal", True):
            continue
        n_unique = df[col].nunique()
        if n_unique <= max_unique:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            ohe_cols_generated.extend(dummies.columns.tolist())
            logger.info(f"One-hot encoded categorical '{col}': {n_unique} values")
        else:
            unique_vals = _random_order(df[col].dropna().unique(), random_seed)
            if unique_vals:
                mapping = {v: i for i, v in enumerate(unique_vals)}
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
                fallback_mappings[col] = mapping
                ohe_cols_generated.append(col)
                logger.info(
                    f"Too many values ({n_unique}) for '{col}', number encoded "
                    f"(random order, seed={random_seed})"
                )

    return df, ohe_cols_generated


def apply_scaling(df: pd.DataFrame, scaling_type: str, exclude_cols: list[str]) -> tuple[pd.DataFrame, Any]:
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    scaler_cls = scalers.get(scaling_type, StandardScaler)
    scaler = scaler_cls()

    if feature_cols:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        logger.info(f"Applied {scaling_type} scaling to {len(feature_cols)} columns")

    return df, scaler


def drop_correlated_columns(
    df: pd.DataFrame, target_column: str, threshold: float
) -> tuple[pd.DataFrame, list[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    if len(numeric_cols) < 2:
        return df, []

    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for i in range(len(upper.columns)):
        for j in range(i):
            col_i = upper.columns[i]
            col_j = upper.index[j]
            if upper.iloc[j, i] > threshold:
                if target_column in df.columns:
                    corr_i = abs(df[col_i].corr(df[target_column]))
                    corr_j = abs(df[col_j].corr(df[target_column]))
                    if corr_i < corr_j:
                        to_drop.add(col_i)
                    else:
                        to_drop.add(col_j)
                else:
                    to_drop.add(col_i)

    if to_drop:
        df = df.drop(columns=list(to_drop))
        logger.info(f"Dropped {len(to_drop)} correlated columns: {to_drop}")

    return df, list(to_drop)


def balance_data(
    df: pd.DataFrame, target_column: str, threshold: int | None, random_seed: int
) -> pd.DataFrame:
    if target_column not in df.columns:
        return df

    X = df.drop(columns=[target_column])
    y = df[target_column]

    n_rows = len(df)
    use_smote = threshold is not None and n_rows > threshold

    if use_smote:
        min_class_count = y.value_counts().min()
        k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
        sampler = SMOTE(random_state=random_seed, k_neighbors=k_neighbors)
        logger.info(f"Applying SMOTE (rows={n_rows}, threshold={threshold})")
    else:
        sampler = RandomUnderSampler(random_state=random_seed)
        logger.info(f"Applying RandomUnderSampler (rows={n_rows})")

    X_res, y_res = sampler.fit_resample(X, y)
    result = pd.concat([X_res, y_res], axis=1)
    logger.info(f"After balancing: {len(result)} rows")
    return result


def engineer(
    df: pd.DataFrame,
    metadata: dict,
    training_settings: dict,
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    target_column = metadata.get("target_column")
    positive_values = metadata.get("positive_values", [])
    categorical_columns = metadata.get("categorical_columns", {})
    multi_value_columns = metadata.get("multi_value_columns", [])

    scaling_type = training_settings.get("scaling_type", "standard")
    max_ohe = training_settings.get("max_ohe_unique_values", 10)
    corr_threshold = training_settings.get("correlation_acceptance_threshold", 0.6)
    oversampling_threshold = training_settings.get("oversampling_threshold")
    random_seed = training_settings.get("random_seed", 123)

    datetime_cols = detect_datetime_columns(df)
    df = transform_datetime(df, datetime_cols)
    _log_shape(log_fn, "After datetime conversion", df)

    if target_column and target_column in df.columns:
        df = binarize_target(df, target_column, positive_values)
        _log_shape(log_fn, f"After binarizing target '{target_column}'", df)

    df, ordinal_mappings = ordinal_encode(df, categorical_columns, target_column, random_seed)
    _log_shape(log_fn, "After categorical (sortable) encoding", df)
    fallback_mappings: dict = {}
    df, ohe_cols = one_hot_encode(
        df, multi_value_columns, categorical_columns, max_ohe, fallback_mappings, target_column, random_seed
    )
    ordinal_mappings.update(fallback_mappings)
    _log_shape(log_fn, "After one-hot / multi-value encoding", df)

    if target_column and target_column in df.columns:
        df, dropped_corr = drop_correlated_columns(df, target_column, corr_threshold)
        _log_shape(log_fn, "After dropping correlated columns", df)

    df = df.drop_duplicates()
    _log_shape(log_fn, "After removing duplicate rows (columns dropped)", df)

    exclude_cols = []
    if target_column and target_column in df.columns:
        exclude_cols.append(target_column)
    df, scaler = apply_scaling(df, scaling_type, exclude_cols)
    _log_shape(log_fn, f"After applying {scaling_type} scaling", df)

    if target_column and target_column in df.columns:
        df = balance_data(df, target_column, oversampling_threshold, random_seed)
        _log_shape(log_fn, "After balancing classes", df)

    feature_names = [c for c in df.columns if c != target_column]

    return {
        "dataframe": df,
        "feature_names": feature_names,
        "scaler": scaler,
        "ordinal_mappings": ordinal_mappings,
        "ohe_columns": ohe_cols,
        "datetime_columns": datetime_cols,
        "target_column": target_column,
    }
