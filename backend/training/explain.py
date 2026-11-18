"""Explainability: SHAP (computed at training time), LIME and Anchor (computed per prediction).

The three methods are implemented as in the reference notebook:

* SHAP: the data goes through the scaler step of the pipeline; the explainer is chosen
  from the classifier (trees -> TreeExplainer, linear models -> LinearExplainer,
  anything else -> KernelExplainer on model.predict with 100 background rows) and
  the held-out test split is explained. Outputs are the summary plot and the
  dependence plot of the most important feature (plus the extra charts/tables the UI uses).
* LIME: LimeTabularExplainer over the raw (unscaled) training data, predict_proba of the
  whole pipeline, 5 features.
* Anchor: alibi AnchorTabular fitted on the raw training data, model.predict of the
  whole pipeline, precision threshold 0.95.
"""
import glob
import importlib.util
import json
import logging
import os
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

# SHAP
MAX_EXPLAINED_ROWS = 2000      # tree / linear explainers are cheap
MAX_KERNEL_ROWS = 50           # KernelExplainer cost grows with rows x background
KERNEL_BACKGROUND_ROWS = 100   # as in the notebook: shap.sample(X_train_transformed, 100)
MAX_LOCAL_ROWS = 10            # rows with their own waterfall chart + table
MAX_CHART_FEATURES = 40
MAX_LABEL_LEN = 40

# LIME / Anchor
LIME_NUM_FEATURES = 5
ANCHOR_THRESHOLD = 0.95
CLASS_NAMES = ["negative", "positive"]

SHAP_DATA_FILENAME = "shap_data.json"


def _log(log_fn, msg: str):
    if log_fn:
        log_fn(msg)
    logger.info(msg)


def _truncate_labels(columns: list[str], max_len: int = MAX_LABEL_LEN) -> list[str]:
    return [c if len(c) <= max_len else c[: max_len - 3] + "..." for c in columns]


# --- shared helpers ---------------------------------------------------------

def transform_features(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply every transforming step of the pipeline except the classifier (the scaler;
    samplers only act at fit time and have no transform)."""
    Xt = X.copy()
    for _, step in pipeline.steps[:-1]:
        if hasattr(step, "transform"):
            Xt = step.transform(Xt)
    return np.asarray(Xt, dtype=float)


def _proba_positive(pipeline, X: pd.DataFrame) -> list[float | None]:
    try:
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X)
            if probs.shape[1] == 2:
                return [float(p) for p in probs[:, 1]]
    except Exception:
        pass
    return [None] * len(X)


def _label_of(index_value):
    try:
        return int(index_value)
    except (TypeError, ValueError):
        return str(index_value)


# --- SHAP -------------------------------------------------------------------

def _explainer_kind(model) -> str:
    """'tree', 'linear' or 'kernel' (the notebook's rule, extended to xgboost/lightgbm
    and to any non-tree estimator that exposes coef_)."""
    module = type(model).__module__
    if "ensemble" in module or "tree" in module or module.startswith(("xgboost", "lightgbm", "catboost")):
        return "tree"
    if "linear_model" in module or hasattr(model, "coef_"):
        return "linear"
    return "kernel"


def _make_explainer(kind: str, model, X_train_t: np.ndarray):
    if kind == "tree":
        return shap.TreeExplainer(model), "TreeExplainer"
    if kind == "linear":
        return shap.LinearExplainer(model, X_train_t), "LinearExplainer"
    background = shap.sample(X_train_t, min(KERNEL_BACKGROUND_ROWS, len(X_train_t)), random_state=0)
    return shap.KernelExplainer(model.predict, background), "KernelExplainer"


def _positive_class_values(raw) -> np.ndarray:
    """SHAP returns a list per class, a (n, p, classes) array or a (n, p) array
    depending on explainer/version; always give (n, p) for the positive class."""
    if isinstance(raw, list):
        raw = raw[1] if len(raw) > 1 else raw[0]
    values = np.asarray(raw)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    return values


def _base_value(explainer) -> float:
    expected = np.asarray(explainer.expected_value).reshape(-1)
    return float(expected[1] if len(expected) > 1 else expected[0])


def _pick_rows(y: pd.Series, n: int, limit: int, seed: int) -> list[int]:
    """Positions of up to `limit` rows, balanced across the two classes when possible."""
    rng = np.random.default_rng(seed)
    limit = min(limit, n)
    if limit >= n:
        return list(range(n))

    labels = np.asarray(y)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    if len(labels) == n and len(pos) and len(neg):
        half = limit // 2
        take_pos = min(max(half, limit - len(neg)), len(pos))
        take_neg = min(limit - take_pos, len(neg))
        idx = list(rng.choice(pos, size=take_pos, replace=False))
        idx += list(rng.choice(neg, size=take_neg, replace=False))
        rng.shuffle(idx)
        return [int(i) for i in idx]
    return [int(i) for i in rng.choice(n, size=limit, replace=False)]


def _compute_shap(pipeline, X_train, X_test, y_test, seed, log_fn):
    """Fit the explainer and compute SHAP values on the test split. Falls back to the
    model-agnostic KernelExplainer when the exact one doesn't support the model."""
    classifier = pipeline.steps[-1][1]
    X_train_t = transform_features(pipeline, X_train)

    kinds = [_explainer_kind(classifier)]
    if kinds[0] != "kernel":
        kinds.append("kernel")

    last_error = None
    for kind in kinds:
        try:
            explainer, name = _make_explainer(kind, classifier, X_train_t)
            limit = MAX_KERNEL_ROWS if kind == "kernel" else MAX_EXPLAINED_ROWS
            rows = _pick_rows(y_test, len(X_test), limit, seed)
            X_exp = X_test.iloc[rows]
            _log(log_fn, f"Computing SHAP values with {name} on {len(X_exp)} test rows...")
            X_exp_t = transform_features(pipeline, X_exp)
            raw_values = (
                explainer.shap_values(X_exp_t, silent=True)
                if kind == "kernel"
                else explainer.shap_values(X_exp_t)
            )
            return name, explainer, rows, X_exp, _positive_class_values(raw_values)
        except Exception as e:
            last_error = e
            _log(log_fn, f"{kind} explainer failed ({e})" + ("; falling back" if kind != kinds[-1] else ""))
    raise last_error


def clear_shap_outputs(output_dir: str):
    """Remove the charts/JSON of a previous training so stale explanations never linger."""
    for old in glob.glob(os.path.join(output_dir, "shap_*.png")):
        os.remove(old)
    json_path = os.path.join(output_dir, SHAP_DATA_FILENAME)
    if os.path.exists(json_path):
        os.remove(json_path)


def _save_plot(plot_fn, path: str):
    plot_fn()
    fig = plt.gcf()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def explain_model(
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
    random_seed: int = 123,
    model_name: str | None = None,
    log_fn=None,
) -> dict:
    """Compute SHAP explanations of the fitted pipeline on the test split and persist
    the charts and a JSON file with every value (all features).

    Returns the explanation data, or {"charts": [], "error": msg} if SHAP fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    clear_shap_outputs(output_dir)
    json_path = os.path.join(output_dir, SHAP_DATA_FILENAME)

    if len(X_test) == 0:
        return {"charts": [], "error": "No rows available to explain"}

    try:
        explainer_name, explainer, rows, X_exp, values = _compute_shap(
            pipeline, X_train, X_test, y_test, random_seed, log_fn
        )
        base_value = _base_value(explainer)

        feature_names = list(X_exp.columns)
        plot_names = _truncate_labels(feature_names)
        raw = X_exp.to_numpy(dtype=float)  # unscaled, so axes/tables are readable
        n_show = min(len(feature_names), MAX_CHART_FEATURES)
        charts: list[str] = []

        mean_abs = np.abs(values).mean(axis=0)
        mean_signed = values.mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        top_index = int(order[0])
        global_importance = [
            {
                "feature": feature_names[int(j)],
                "mean_abs_shap": float(mean_abs[j]),
                "mean_shap": float(mean_signed[j]),
            }
            for j in order
        ]

        chart_height = max(4.0, 0.35 * n_show + 1.5)

        def summary(plot_type=None):
            plt.figure(figsize=(10, chart_height))
            shap.summary_plot(
                values, raw, feature_names=plot_names, plot_type=plot_type,
                max_display=n_show, show=False,
            )

        for filename, plot_type, label in (
            ("shap_beeswarm.png", None, "summary (beeswarm)"),
            ("shap_importance.png", "bar", "feature importance"),
        ):
            try:
                _save_plot(lambda t=plot_type: summary(t), os.path.join(output_dir, filename))
                charts.append(filename)
                _log(log_fn, f"Saved SHAP {label} chart")
            except Exception as e:
                _log(log_fn, f"Failed to save SHAP {label} chart: {e}")

        try:
            def dependence():
                plt.figure(figsize=(8, 6))
                shap.dependence_plot(
                    top_index, values, raw, feature_names=plot_names, show=False
                )
            _save_plot(dependence, os.path.join(output_dir, "shap_dependence.png"))
            charts.append("shap_dependence.png")
            _log(log_fn, f"Saved SHAP dependence chart for '{feature_names[top_index]}'")
        except Exception as e:
            _log(log_fn, f"Failed to save SHAP dependence chart: {e}")

        local_pos = _pick_rows(
            y_test.iloc[rows].reset_index(drop=True), len(X_exp), MAX_LOCAL_ROWS, random_seed
        )
        for k, i in enumerate(local_pos):
            filename = f"shap_waterfall_{k}.png"
            try:
                _save_plot(
                    lambda i=i: shap.plots.waterfall(
                        shap.Explanation(
                            values=values[i],
                            base_values=base_value,
                            data=raw[i],
                            feature_names=plot_names,
                        ),
                        max_display=n_show,
                        show=False,
                    ),
                    os.path.join(output_dir, filename),
                )
                charts.append(filename)
            except Exception as e:
                _log(log_fn, f"Failed to save waterfall {k}: {e}")

        X_local = X_exp.iloc[local_pos]
        data = {
            "model_name": model_name,
            "explainer": explainer_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_explained": int(len(X_exp)),
            "feature_names": feature_names,
            "base_value": base_value,
            "global_importance": global_importance,
            "top_feature": feature_names[top_index],
            "shap_values": values[local_pos].tolist(),
            "feature_values": raw[local_pos].tolist(),
            "sample_indices": [_label_of(X_exp.index[i]) for i in local_pos],
            "predicted_labels": [int(p) for p in pipeline.predict(X_local)],
            "proba_positive": _proba_positive(pipeline, X_local),
            "charts": charts,
        }

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        _log(log_fn, f"SHAP explanation saved ({len(X_exp)} rows, {len(feature_names)} features, {len(charts)} charts)")

        return {**data}

    except Exception as e:
        _log(log_fn, f"SHAP explanation failed: {e}")
        return {"charts": [], "error": str(e)}


# --- LIME / Anchor (single instance) ----------------------------------------

def anchor_available() -> bool:
    return importlib.util.find_spec("alibi") is not None


def instance_methods(model_data: dict) -> list[str]:
    """Per-prediction explanation methods enabled for a saved model."""
    if model_data.get("format_version", 1) < 2 or model_data.get("train_data") is None:
        return []
    config = model_data.get("explain_config", {})
    methods = []
    if config.get("lime", True):
        methods.append("lime")
    if config.get("anchor", True):
        methods.append("anchor")
    return methods


def _to_float(value) -> float:
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _for_label(attribute, label: int):
    """lime stores intercept/score/local_pred per label (dict) in recent versions."""
    return attribute[label] if isinstance(attribute, dict) else attribute


def _lime_explanation(pipeline, X_train: pd.DataFrame, instance: np.ndarray, seed: int) -> dict:
    import lime.lime_tabular

    columns = list(X_train.columns)
    mode = "classification" if hasattr(pipeline, "predict_proba") else "regression"

    def predict_fn(data):
        # LIME sends bare arrays; rebuild the frame so the pipeline sees feature names.
        frame = pd.DataFrame(data, columns=columns)
        return pipeline.predict_proba(frame) if mode == "classification" else pipeline.predict(frame)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.to_numpy(dtype=float),
        feature_names=columns,
        class_names=CLASS_NAMES,
        mode=mode,
        random_state=seed,
    )
    explanation = explainer.explain_instance(
        data_row=instance, predict_fn=predict_fn, num_features=LIME_NUM_FEATURES
    )

    label = 1 if mode == "classification" else 0
    rules = explanation.as_list(label=label)
    feature_indices = [index for index, _ in explanation.local_exp[label]]

    result = {
        "mode": mode,
        "features": [
            {
                "rule": rule,
                "feature": columns[index],
                "value": float(instance[index]),
                "weight": float(weight),
            }
            for (rule, weight), index in zip(rules, feature_indices)
        ],
        "intercept": _to_float(_for_label(explanation.intercept, label)),
        "local_pred": _to_float(_for_label(explanation.local_pred, label)),
        "score": _to_float(_for_label(explanation.score, label)),
    }
    if mode == "classification":
        result["predict_proba"] = [float(p) for p in explanation.predict_proba]
    return result


def _anchor_explanation(pipeline, X_train: pd.DataFrame, instance: np.ndarray, seed: int) -> dict:
    if not anchor_available():
        raise RuntimeError(
            "Anchor needs the 'alibi' package, which only installs on Python 3.12 or older."
        )
    from alibi.explainers import AnchorTabular

    columns = list(X_train.columns)

    def predict_fn(data):
        return pipeline.predict(pd.DataFrame(data, columns=columns))

    explainer = AnchorTabular(predict_fn, feature_names=columns, seed=seed)
    explainer.fit(X_train.to_numpy(dtype=float))
    explanation = explainer.explain(instance, threshold=ANCHOR_THRESHOLD)

    return {
        "prediction": int(predict_fn(instance.reshape(1, -1))[0]),
        "anchor": [str(condition) for condition in explanation.anchor],
        "precision": float(explanation.precision),
        "coverage": float(explanation.coverage),
        "threshold": ANCHOR_THRESHOLD,
    }


def explain_instance(model_data: dict, instance: pd.DataFrame, methods: list[str]) -> dict:
    """LIME and/or Anchor explanation of one encoded (unscaled) instance.

    Each method reports independently: {"lime": {...} | {"error": ...}, "anchor": ...}.
    """
    pipeline = model_data["model"]
    X_train: pd.DataFrame = model_data["train_data"]
    seed = int(model_data.get("random_seed", 123))
    row = instance[list(X_train.columns)].to_numpy(dtype=float)[0]

    builders = {"lime": _lime_explanation, "anchor": _anchor_explanation}
    results = {}
    for method in methods:
        started = time.time()
        try:
            results[method] = {
                **builders[method](pipeline, X_train, row, seed),
                "seconds": round(time.time() - started, 2),
            }
        except Exception as e:
            logger.warning(f"{method} explanation failed: {e}")
            results[method] = {"error": str(e)}
    return results
