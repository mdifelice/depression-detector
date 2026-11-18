import importlib
import logging
import multiprocessing
import os
import signal
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import ks_2samp
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from training.defaults import (
    DEFAULT_PARAM_GRIDS,
    DEFAULT_CONSTRUCTOR_PARAMS,
    SEED_ALGORITHMS,
)

logger = logging.getLogger(__name__)

# Saved model artifacts: 1 = bare estimator + external scaler (old),
# 2 = fitted pipeline (scaler/sampler/classifier) + train data for LIME/Anchor.
MODEL_FORMAT_VERSION = 2
MAX_TRAIN_DATA_ROWS = 5000

METRIC_KEYS = (
    "f1", "accuracy", "precision", "recall", "specificity", "auc", "ks", "lift_at_10",
)

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


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


def _prefix_grid(grid: dict | list[dict]) -> dict | list[dict]:
    """Param grids are written for the bare classifier; in the pipeline the
    classifier step is called 'classifier'."""
    if isinstance(grid, list):
        return [_prefix_grid(g) for g in grid]
    return {f"classifier__{k}": v for k, v in grid.items()}


def build_pipeline(classifier, settings: dict, y_train: pd.Series) -> Pipeline:
    """scaler -> (optional sampler) -> classifier, all fitted on the training split only."""
    random_seed = settings.get("random_seed", 123)
    scaler_cls = SCALERS.get(settings.get("scaling_type", "standard"), StandardScaler)
    steps: list[tuple] = [("scaler", scaler_cls())]

    if settings.get("use_sampling", False):
        threshold = settings.get("oversampling_threshold") or 0
        min_class = int(y_train.value_counts().min())
        oversampling = len(y_train) < threshold or min_class < len(y_train) * 0.2
        if oversampling:
            k_neighbors = max(1, min(5, min_class - 1))
            sampler = SMOTE(random_state=random_seed, k_neighbors=k_neighbors)
        else:
            sampler = RandomUnderSampler(random_state=random_seed)
        steps.append(("sampler", sampler))

    steps.append(("classifier", classifier))
    return Pipeline(steps)


def get_score_for_auc(model, X_val):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_val)
        if scores.shape[1] == 2:
            return scores[:, 1]
        return scores
    elif hasattr(model, "decision_function"):
        return model.decision_function(X_val)
    return None


# --- metrics (same definitions as the reference notebook) -------------------

def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov statistic between positive and negative scores."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(ks_2samp(pos, neg).statistic)


def lift_at_10(y_true, y_prob) -> float:
    """Lift in the top decile of scores."""
    y_true = np.asarray(y_true)
    order = np.argsort(np.asarray(y_prob))[::-1]
    y_sorted = y_true[order]
    top = max(1, int(len(y_true) * 0.1))
    top_rate = np.sum(y_sorted[:top]) / top
    overall = np.mean(y_true)
    return float(top_rate / overall) if overall > 0 else 1.0


def _binarized(metric, **fixed):
    """Wrap a label metric so it can score probabilities (threshold 0.5)."""
    def score(y_true, y_pred, **kwargs):
        return metric(y_true, (np.asarray(y_pred) >= 0.5) * 1.0, **fixed)
    return score


def build_scoring(pipeline) -> dict:
    """Scorers for cross_validate.

    Models with predict_proba are scored on probabilities (threshold 0.5), like the
    notebook. Models without it (RidgeClassifier, LinearSVC, ...) use hard labels
    for the threshold metrics and decision_function for the ranking metrics
    (AUC/KS/Lift), so those aren't computed from 0/1 labels.
    """
    has_proba = hasattr(pipeline, "predict_proba")
    has_decision = hasattr(pipeline, "decision_function")
    label_response = "predict_proba" if has_proba else "predict"
    rank_response = (
        "predict_proba" if has_proba else ("decision_function" if has_decision else "predict")
    )

    def thresholded(metric, **fixed):
        return make_scorer(_binarized(metric, **fixed), response_method=label_response)

    def ranking(metric):
        return make_scorer(metric, response_method=rank_response)

    return {
        "f1": thresholded(f1_score, zero_division=0),
        "accuracy": thresholded(accuracy_score),
        "precision": thresholded(precision_score, zero_division=0),
        "recall": thresholded(recall_score, zero_division=0),
        "specificity": thresholded(recall_score, pos_label=0, zero_division=0),
        "auc": ranking(roc_auc_score),
        "ks": ranking(ks_statistic),
        "lift_at_10": ranking(lift_at_10),
    }


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


def compute_metrics(
    pipeline, X_test: pd.DataFrame, y_test: pd.Series, settings: dict, log
) -> tuple[dict, float]:
    """Metrics as computed in the reference notebook: k-fold cross-validation over the
    held-out test split. Returns (metrics, summed CV fit time in seconds)."""
    scoring = build_scoring(pipeline)
    folds = _effective_n_splits(X_test, y_test, settings.get("cross_validation_folds", 5))

    if folds is not None:
        try:
            scores = cross_validate(
                pipeline, X_test, y_test, cv=folds, scoring=scoring, error_score="raise"
            )
            metrics = {key: float(np.mean(scores[f"test_{key}"])) for key in scoring}
            return metrics, float(np.sum(scores["fit_time"]))
        except Exception as e:
            log(f"Cross-validation on the test split failed ({e}); scoring the fitted model directly")
    else:
        log(f"Too few test samples ({len(X_test)}) for cross-validation; scoring the fitted model directly")

    metrics = {}
    for key, scorer in scoring.items():
        try:
            metrics[key] = float(scorer(pipeline, X_test, y_test))
        except Exception as e:
            log(f"Could not compute {key}: {e}")
            metrics[key] = 0.0
    return metrics, 0.0


def generate_roc_chart(y_true, y_score, auc_value: float, model_name: str, output_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
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


def evaluate_model(
    model_path: str,
    constructor_params: dict,
    param_grid: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    settings: dict,
    charts_dir: str,
    log_fn=None,
) -> dict:
    """Train and evaluate one algorithm, enforcing a per-algorithm timeout if set.

    When a timeout is configured the work runs in a child process (fork) so a
    slow algorithm can be genuinely killed once the budget is exceeded. Without
    a timeout it runs inline like normal.
    """
    model_name = model_path.rsplit(".", 1)[1]
    timeout = int(settings.get("timeout", 0) or 0)
    args = (
        model_path, constructor_params, param_grid,
        X_train, X_test, y_train, y_test, settings, charts_dir, log_fn,
    )

    if timeout <= 0:
        return _run_evaluate(*args)

    try:
        ctx = multiprocessing.get_context("fork")
    except (ValueError, AttributeError):
        ctx = multiprocessing.get_context()

    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_train_worker, args=(child_conn, *args))
    proc.start()
    child_conn.close()

    try:
        if parent_conn.poll(timeout):
            result = parent_conn.recv()
            proc.join()
            return result

        msg = f"{model_name} exceeded timeout of {timeout}s - training killed"
        if log_fn:
            log_fn(msg)
        logger.info(msg)

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        proc.join()

        return {
            "model_name": model_name,
            "model_path": model_path,
            "error": f"Timed out after {timeout}s",
        }
    finally:
        parent_conn.close()


def _train_worker(conn, model_path, *args):
    os.setsid()
    try:
        conn.send(_run_evaluate(model_path, *args))
    except Exception as e:
        conn.send({
            "model_name": model_path.rsplit(".", 1)[1],
            "model_path": model_path,
            "error": str(e),
        })
    finally:
        conn.close()


def _run_evaluate(
    model_path: str,
    constructor_params: dict,
    param_grid: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    settings: dict,
    charts_dir: str,
    log_fn=None,
) -> dict:
    model_name = model_path.rsplit(".", 1)[1]
    tune = settings.get("tune", False)
    tune_iterations = settings.get("tune_iterations", 50)
    cv_tune_folds = _effective_n_splits(
        X_train, y_train, settings.get("cross_validation_tune_folds", 5)
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

    def new_pipeline() -> Pipeline:
        return build_pipeline(
            instantiate_model(model_path, effective_constructor), settings, y_train
        )

    log(f"Training {model_name}...")

    effective_grid = sanitize_params(
        param_grid if param_grid else DEFAULT_PARAM_GRIDS.get(model_path, {})
    )

    try:
        pipeline = new_pipeline()
    except Exception as e:
        log(f"Failed to instantiate {model_name}: {e}")
        return {"model_name": model_name, "model_path": model_path, "error": str(e)}

    try:
        start_time = time.time()

        fitted = False
        if tune and effective_grid and cv_tune_folds is not None:
            log(
                f"Tuning with {'GridSearchCV' if tune_iterations == 0 else 'RandomizedSearchCV'} "
                f"(iters={tune_iterations}, folds={cv_tune_folds}, scoring=f1)"
            )
            try:
                cv_inner = StratifiedKFold(
                    n_splits=cv_tune_folds, shuffle=True, random_state=random_seed
                )
                n_jobs = -1 if settings.get("turbo") else None
                pipeline_grid = _prefix_grid(effective_grid)
                if tune_iterations == 0:
                    searcher = GridSearchCV(
                        pipeline, pipeline_grid, cv=cv_inner, scoring="f1",
                        n_jobs=n_jobs, refit=True,
                    )
                else:
                    searcher = RandomizedSearchCV(
                        pipeline, pipeline_grid, n_iter=tune_iterations, cv=cv_inner,
                        scoring="f1", n_jobs=n_jobs, refit=True, random_state=random_seed,
                    )
                searcher.fit(X_train, y_train)
                pipeline = searcher.best_estimator_
                fitted = True
                log(f"Best params: {searcher.best_params_}")
            except Exception as e:
                log(f"Tuning failed, using default params: {e}")
                pipeline = new_pipeline()
        elif tune and not effective_grid:
            log(f"Skipping tuning for {model_name}: no parameter grid available")
        elif tune:
            log(
                f"Skipping tuning for {model_name}: too few training samples ({len(X_train)}) "
                f"for {settings.get('cross_validation_tune_folds', 5)} folds"
            )

        if not fitted:
            pipeline.fit(X_train, y_train)

        elapsed = time.time() - start_time

        metrics, cv_fit_time = compute_metrics(pipeline, X_test, y_test, settings, log)
        train_time = elapsed + cv_fit_time

        log(
            f"{model_name} - F1: {metrics['f1']:.4f}, Acc: {metrics['accuracy']:.4f}, "
            f"Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, "
            f"Spec: {metrics['specificity']:.4f}, AUC: {metrics['auc']:.4f}, "
            f"KS: {metrics['ks']:.4f}, Lift@10: {metrics['lift_at_10']:.3f} "
            f"({train_time:.2f}s)"
        )

        os.makedirs(charts_dir, exist_ok=True)
        roc_path = os.path.join(charts_dir, f"{model_name}_roc.png")
        cm_path = os.path.join(charts_dir, f"{model_name}_confusion.png")

        try:
            y_score = get_score_for_auc(pipeline, X_test)
            if y_score is None:
                y_score = pipeline.predict(X_test).astype(float)
            generate_roc_chart(y_test, y_score, metrics["auc"], model_name, roc_path)
        except Exception as e:
            log(f"Failed to generate ROC chart for {model_name}: {e}")

        try:
            generate_confusion_matrix_chart(
                y_test, pipeline.predict(X_test), model_name, cm_path
            )
        except Exception as e:
            log(f"Failed to generate confusion matrix for {model_name}: {e}")

        return {
            "model_name": model_name,
            "model_path": model_path,
            **{key: metrics[key] for key in METRIC_KEYS},
            "train_time": float(train_time),
            "roc_chart": f"{model_name}_roc.png" if os.path.exists(roc_path) else None,
            "confusion_matrix_chart": f"{model_name}_confusion.png" if os.path.exists(cm_path) else None,
            "model_object": pipeline,
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


def save_model(
    best: dict,
    artifact: dict,
    models_dir: str,
    dataset_id: str,
    X_train: pd.DataFrame,
    settings: dict,
):
    os.makedirs(models_dir, exist_ok=True)

    random_seed = settings.get("random_seed", 123)
    train_data = X_train
    if len(train_data) > MAX_TRAIN_DATA_ROWS:
        train_data = train_data.sample(n=MAX_TRAIN_DATA_ROWS, random_state=random_seed)

    model_data = {
        "format_version": MODEL_FORMAT_VERSION,
        "model": best["model_object"],
        "scaler": None,
        "feature_names": list(X_train.columns),
        "ordinal_mappings": artifact["ordinal_mappings"],
        "ohe_columns": artifact["ohe_columns"],
        "datetime_columns": artifact["datetime_columns"],
        "target_column": artifact["target_column"],
        "model_name": best["model_name"],
        "metrics": {
            **{key: best[key] for key in METRIC_KEYS},
            "train_time": best["train_time"],
        },
        "train_data": train_data.reset_index(drop=True),
        "explain_config": {
            "shap": bool(settings.get("use_shap", True)),
            "lime": bool(settings.get("use_lime", True)),
            "anchor": bool(settings.get("use_anchor", True)),
        },
        "random_seed": random_seed,
    }

    model_path = os.path.join(models_dir, f"{dataset_id}.joblib")
    joblib.dump(model_data, model_path)
    logger.info(f"Saved best model to {model_path}")
    return model_path
