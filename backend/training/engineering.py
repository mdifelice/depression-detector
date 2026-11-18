import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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


def transform_datetime(
    df: pd.DataFrame, datetime_cols: list[str], log_fn: Callable[[str], None] | None = None
) -> pd.DataFrame:
    df = df.copy()
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") // 10**9
            message = f"Converted datetime column '{col}' to timestamp"
            logger.info(message)
            if log_fn:
                log_fn(message)
    return df


def ordinal_encode(
    df: pd.DataFrame,
    categorical_columns: dict,
    target_column: str | None = None,
    random_seed: int = 123,
    log_fn: Callable[[str], None] | None = None,
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
            message = f"Ordinal encoded '{col}': {mapping}"
            logger.info(message)
            if log_fn:
                log_fn(message)
        else:
            unique_vals = _random_order(df[col].dropna().unique(), random_seed)
            if unique_vals:
                mapping = {v: i for i, v in enumerate(unique_vals)}
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
                mappings[col] = mapping
                message = (
                    f"Ordinal encoded '{col}' (random order, seed={random_seed}): {mapping}"
                )
                logger.info(message)
                if log_fn:
                    log_fn(message)
    return df, mappings


def one_hot_encode(
    df: pd.DataFrame,
    multi_value_columns: list[str],
    categorical_columns: dict,
    max_unique: int,
    fallback_mappings: dict,
    target_column: str | None = None,
    random_seed: int = 123,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    ohe_cols_generated = []
    added: list[str] = []

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
                added.append(new_col)
            df = df.drop(columns=[col])
            message = f"One-hot encoded multi-value '{col}' → {sorted(all_values)} ({len(all_values)} values)"
            logger.info(message)
            if log_fn:
                log_fn(message)

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
            added.extend(dummies.columns.tolist())
            message = f"One-hot encoded categorical '{col}' → {list(dummies.columns)} ({n_unique} values)"
            logger.info(message)
            if log_fn:
                log_fn(message)
        else:
            unique_vals = _random_order(df[col].dropna().unique(), random_seed)
            if unique_vals:
                mapping = {v: i for i, v in enumerate(unique_vals)}
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
                fallback_mappings[col] = mapping
                ohe_cols_generated.append(col)
                message = (
                    f"Too many values ({n_unique}) for '{col}' (> {max_unique}), "
                    f"number encoded instead (random order, seed={random_seed})"
                )
                logger.info(message)
                if log_fn:
                    log_fn(message)

    return df, ohe_cols_generated


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float,
    random_seed: int,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split. Everything fitted later (feature selection,
    scaling, sampling, model) only ever sees the training part."""
    can_stratify = y.nunique() > 1 and int(y.value_counts().min()) >= 2
    if not can_stratify:
        message = "Not enough samples in one class to stratify the train/test split"
        logger.warning(message)
        if log_fn:
            log_fn(message)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=y if can_stratify else None,
    )
    message = f"Train/test split: {len(X_train)} train rows, {len(X_test)} test rows (test_ratio={test_ratio})"
    logger.info(message)
    if log_fn:
        log_fn(message)
    return X_train, X_test, y_train, y_test


def drop_correlated_columns(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float,
    log_fn: Callable[[str], None] | None = None,
) -> list[str]:
    """Find highly correlated feature pairs using the training data only.

    For every pair with |r| > threshold the column that is less correlated with
    the target is dropped (ties drop the first column of the pair). Returns the
    columns to drop; the caller removes them from both train and test.
    """
    target = "__target__"
    data = pd.concat([X_train, y_train.rename(target)], axis=1)
    corr = data.corr(numeric_only=True).abs()
    if target not in corr.columns:
        return []

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    features = [c for c in upper.columns if c != target]

    def target_corr(col: str) -> float:
        value = corr.loc[col, target]
        return 0.0 if pd.isna(value) else float(value)

    considered: list[tuple[str, str, float, str]] = []
    to_drop: set[str] = set()
    for col1 in features:
        for col2 in features:
            value = upper.loc[col2, col1]
            if pd.isna(value) or value <= threshold:
                continue
            dropped = col1 if target_corr(col1) <= target_corr(col2) else col2
            to_drop.add(dropped)
            considered.append((col1, col2, float(value), dropped))

    if considered:
        lines = [f"Correlation analysis, train only (pairs with |r| > {threshold}):"]
        for col1, col2, value, dropped in considered:
            lines.append(f"  '{col1}' ↔ '{col2}': |r| = {value:.3f} → dropped '{dropped}'")
        message = "\n".join(lines)
        logger.info(message)
        if log_fn:
            log_fn(message)

    if to_drop:
        message = f"Dropped {len(to_drop)} correlated columns (train-only analysis): {sorted(to_drop)}"
        logger.info(message)
        if log_fn:
            log_fn(message)

    return sorted(to_drop)


def engineer(
    df: pd.DataFrame,
    metadata: dict,
    training_settings: dict,
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    """Encode the dataset into a numeric frame.

    Scaling, class balancing and correlation filtering are deliberately NOT done
    here: they are fitted on the training split only (see split_train_test,
    drop_correlated_columns and the model pipeline in evaluate.py).
    """
    target_column = metadata.get("target_column")
    positive_values = metadata.get("positive_values", [])
    categorical_columns = metadata.get("categorical_columns", {})
    multi_value_columns = metadata.get("multi_value_columns", [])

    max_ohe = training_settings.get("max_ohe_unique_values", 10)
    random_seed = training_settings.get("random_seed", 123)

    datetime_cols = detect_datetime_columns(df)
    df = transform_datetime(df, datetime_cols, log_fn=log_fn)
    _log_shape(log_fn, "After datetime conversion", df)

    if target_column and target_column in df.columns:
        df = binarize_target(df, target_column, positive_values)
        _log_shape(log_fn, f"After binarizing target '{target_column}'", df)

    df, ordinal_mappings = ordinal_encode(
        df, categorical_columns, target_column, random_seed, log_fn=log_fn
    )
    _log_shape(log_fn, "After categorical (sortable) encoding", df)
    fallback_mappings: dict = {}
    df, ohe_cols = one_hot_encode(
        df, multi_value_columns, categorical_columns, max_ohe, fallback_mappings,
        target_column, random_seed, log_fn=log_fn
    )
    ordinal_mappings.update(fallback_mappings)
    _log_shape(log_fn, "After one-hot / multi-value encoding", df)

    # Duplicates are removed before the split so identical rows can't end up on
    # both sides of it.
    df = df.drop_duplicates()
    _log_shape(log_fn, "After removing duplicate rows", df)

    feature_names = [c for c in df.columns if c != target_column]

    return {
        "dataframe": df,
        "feature_names": feature_names,
        "ordinal_mappings": ordinal_mappings,
        "ohe_columns": ohe_cols,
        "datetime_columns": datetime_cols,
        "target_column": target_column,
    }
