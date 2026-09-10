import importlib
import json
import logging
import os
import time
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_val_predict,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from training.defaults import (
    DEFAULT_PARAM_GRIDS,
    DEFAULT_CONSTRUCTOR_PARAMS,
    SEED_ALGORITHMS,
)

logger = logging.getLogger(__name__)


def import_model_class(model_path: str):
    parts = model_path.rsplit(".", 1)
    module_path = parts[0]
    class_name = parts[1]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def instantiate_model(model_path: str, constructor_params: dict):
    cls = import_model_class(model_path)
    filtered_params = {}
    for k, v in constructor_params.items():
        if k in ("use_label_encoder", "eval_metric"):
            continue
        filtered_params[k] = v
    return cls(**filtered_params)


def sanitize_params(params: dict | list[dict]) -> dict | list[dict]:
    if isinstance(params, list):
        return [sanitize_params(p) for p in params]
    return {
        k: v for k, v in params.items()
        if k not in ("use_label_encoder", "eval_metric")
    }


def get_score_for_auc(model, X_val):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_val)
        if scores.shape[1] == 2:
            return scores[:, 1]
        return scores
    elif hasattr(model, "decision_function"):
        return model.decision_function(X_val)
    return None


def generate_roc_chart(y_true, y_score, model_name: str, output_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def generate_confusion_matrix_chart(y_true, y_pred, model_name: str, output_path: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def _effective_n_splits(
    X: pd.DataFrame, y: pd.Series, requested: int
) -> int | None:
    n_samples = len(X)
    if n_samples < 2:
        return None
    if y.nunique() < 2:
        return None
    min_class_count = int(y.value_counts().min())
    max_folds = min(n_samples, min_class_count)
    if max_folds < 2:
        return None
    return max(2, min(requested, max_folds))


def evaluate_model(
    model_path: str,
    constructor_params: dict,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    settings: dict,
    charts_dir: str,
    log_fn=None,
) -> dict:
    model_name = model_path.rsplit(".", 1)[1]
    tune = settings.get("tune", False)
    tune_iterations = settings.get("tune_iterations", 10)
    cv_folds = _effective_n_splits(
        X, y, settings.get("cross_validation_folds", 5)
    )
    cv_tune_folds = _effective_n_splits(
        X, y, settings.get("cross_validation_tune_folds", 5)
    )
    random_seed = settings.get("random_seed", 123)

    effective_constructor = {
        **DEFAULT_CONSTRUCTOR_PARAMS.get(model_path, {}),
        **(constructor_params or {}),
    }
    if model_path in SEED_ALGORITHMS:
        effective_constructor.setdefault("random_state", random_seed)

    def log(msg):
        if log_fn:
            log_fn(msg)
        logger.info(msg)

    log(f"Training {model_name}...")

    effective_grid = sanitize_params(param_grid if param_grid else DEFAULT_PARAM_GRIDS.get(model_path, {}))

    try:
        model = instantiate_model(model_path, effective_constructor)
    except Exception as e:
        log(f"Failed to instantiate {model_name}: {e}")
        return {"model_name": model_name, "model_path": model_path, "error": str(e)}

    if tune and effective_grid and cv_tune_folds is not None:
        log(f"Tuning with {'GridSearchCV' if tune_iterations == 0 else 'RandomizedSearchCV'} (iters={tune_iterations}, folds={cv_tune_folds})")
        try:
            cv_inner = StratifiedKFold(n_splits=cv_tune_folds, shuffle=True, random_state=random_seed)
            if tune_iterations == 0:
                searcher = GridSearchCV(
                    model, effective_grid, cv=cv_inner, scoring="roc_auc", n_jobs=-1, refit=True
                )
            else:
                searcher = RandomizedSearchCV(
                    model, effective_grid, n_iter=min(tune_iterations, 50),
                    cv=cv_inner, scoring="roc_auc", n_jobs=-1, refit=True, random_state=random_seed
                )
            searcher.fit(X, y)
            model = searcher.best_estimator_
            log(f"Best params: {searcher.best_params_}")
        except Exception as e:
            log(f"Tuning failed, using default params: {e}")
            model = instantiate_model(model_path, effective_constructor)
    elif tune and not effective_grid:
        log(f"Skipping tuning for {model_name}: no parameter grid available")
    elif tune:
        log(f"Skipping tuning for {model_name}: too few samples ({len(X)}) for {settings.get('cross_validation_tune_folds', 5)} folds")

    try:
        if cv_folds is not None:
            cv_outer = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
            y_pred = cross_val_predict(model, X, y, cv=cv_outer, method="predict")

            y_score = None
            try:
                temp_model = instantiate_model(model_path, effective_constructor)
                temp_model.fit(X, y)
                y_score = get_score_for_auc(temp_model, X)
            except Exception:
                y_score = None

            if y_score is None:
                try:
                    y_score = cross_val_predict(model, X, y, cv=cv_outer, method="predict_proba")[:, 1]
                except Exception:
                    try:
                        y_score = cross_val_predict(model, X, y, cv=cv_outer, method="decision_function")
                    except Exception:
                        y_score = y_pred.astype(float)
        else:
            log(f"Too few samples ({len(X)}) for cross-validation; evaluating on training data")
            model = instantiate_model(model_path, effective_constructor)
            model.fit(X, y)
            y_pred = model.predict(X)
            y_score = get_score_for_auc(model, X)
            if y_score is None:
                y_score = y_pred.astype(float)

        f1 = f1_score(y, y_pred, zero_division=0)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y, y_score)
        except Exception:
            auc = 0.0

        log(f"{model_name} - F1: {f1:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, AUC: {auc:.4f}")

        os.makedirs(charts_dir, exist_ok=True)
        roc_path = os.path.join(charts_dir, f"{model_name}_roc.png")
        cm_path = os.path.join(charts_dir, f"{model_name}_confusion.png")

        try:
            generate_roc_chart(y, y_score, model_name, roc_path)
        except Exception as e:
            log(f"Failed to generate ROC chart for {model_name}: {e}")

        try:
            generate_confusion_matrix_chart(y, y_pred, model_name, cm_path)
        except Exception as e:
            log(f"Failed to generate confusion matrix for {model_name}: {e}")

        model.fit(X, y)

        return {
            "model_name": model_name,
            "model_path": model_path,
            "f1": float(f1),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "auc": float(auc),
            "roc_chart": f"{model_name}_roc.png" if os.path.exists(roc_path) else None,
            "confusion_matrix_chart": f"{model_name}_confusion.png" if os.path.exists(cm_path) else None,
            "model_object": model,
        }

    except Exception as e:
        log(f"Failed to train {model_name}: {e}")
        return {"model_name": model_name, "model_path": model_path, "error": str(e)}


def select_best_model(results: list[dict]) -> dict | None:
    valid = [r for r in results if "error" not in r and "f1" in r]
    if not valid:
        return None

    qualified = [r for r in valid if r["f1"] >= 0.7]
    if qualified:
        return max(qualified, key=lambda r: r["auc"])

    return max(valid, key=lambda r: r["auc"])


def save_model(best: dict, artifact: dict, models_dir: str, dataset_id: str):
    os.makedirs(models_dir, exist_ok=True)

    model_data = {
        "model": best["model_object"],
        "scaler": artifact["scaler"],
        "feature_names": artifact["feature_names"],
        "ordinal_mappings": artifact["ordinal_mappings"],
        "ohe_columns": artifact["ohe_columns"],
        "datetime_columns": artifact["datetime_columns"],
        "target_column": artifact["target_column"],
        "model_name": best["model_name"],
        "metrics": {
            "f1": best["f1"],
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "auc": best["auc"],
        },
    }

    model_path = os.path.join(models_dir, f"{dataset_id}.joblib")
    joblib.dump(model_data, model_path)
    logger.info(f"Saved best model to {model_path}")
    return model_path
