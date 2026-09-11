import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

MAX_EXPLAINED_ROWS = 100
MAX_BACKGROUND_ROWS = 50
MAX_WATERFALLS = 8
MAX_LABEL_LEN = 25

SHAP_DATA_FILENAME = "shap_data.json"


def _truncate_labels(columns: list[str], max_len: int = MAX_LABEL_LEN) -> list[str]:
    return [c if len(c) <= max_len else c[: max_len - 3] + "..." for c in columns]


def _log(log_fn, msg: str):
    if log_fn:
        log_fn(msg)
    logger.info(msg)


def _is_tree_model(model) -> bool:
    if hasattr(model, "feature_importances_"):
        return True
    if hasattr(model, "get_booster"):
        return True
    return False


def _is_linear_model(model) -> bool:
    return hasattr(model, "coef_")


def _build_explainer(model, X_sample: pd.DataFrame):
    if _is_tree_model(model):
        try:
            return shap.TreeExplainer(model)
        except Exception as e:
            logger.warning(f"TreeExplainer failed ({e}); falling back")
    elif _is_linear_model(model):
        try:
            return shap.LinearExplainer(model, data=X_sample)
        except Exception as e:
            logger.warning(f"LinearExplainer failed ({e}); falling back")
    return shap.Explainer(
        model.predict,
        masker=shap.maskers.Independent(X_sample),
        feature_names=list(X_sample.columns),
    )


def explain_model(
    model, X: pd.DataFrame, y: pd.Series, output_dir: str, log_fn=None
) -> dict:
    """Compute SHAP explanations for a fitted model and persist charts + JSON.

    Args:
        model: fitted classifier (tree / linear / other).
        X: engineered (scaled) feature frame, same input the model was fitted on.
        y: target series (used to pick an informative explanation sample).
        output_dir: dataset charts directory where "shap_" files are written.
        log_fn: optional training-log callback.

    Returns:
        dict with "charts" (list of chart filenames) and explanation data, or
        {"charts": [], "error": msg} when SHAP cannot be computed.
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(X)

    if n == 0:
        return {"charts": [], "error": "No rows available to explain"}

    try:
        sample_idx = _pick_sample_idx(y, n)
        X_sample = X.iloc[sample_idx]
        X_sample = X_sample.reset_index(drop=True)

        full_names = list(X_sample.columns)
        plot_names = _truncate_labels(full_names)

        background = X.iloc[
            np.random.default_rng(123).choice(n, size=min(MAX_BACKGROUND_ROWS, n), replace=False)
        ]
        background = background.reset_index(drop=True)

        explainer = _build_explainer(model, background)
        _log(log_fn, "Computing SHAP explanations...")

        explanation = explainer(X_sample)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[:, :, 1]
        base_values = float(np.asarray(explanation.base_values).reshape(-1)[0])

        charts = []
        try:
            bar_path = os.path.join(output_dir, "shap_importance.png")
            _save_plot(
                lambda: shap.plots.bar(
                    shap.Explanation(
                        values=values.mean(axis=0),
                        base_values=base_values,
                        data=np.zeros(values.shape[1]),
                        feature_names=plot_names,
                    ),
                    show=False,
                ),
                bar_path,
            )
            charts.append("shap_importance.png")
            _log(log_fn, "Saved SHAP feature importance chart")
        except Exception as e:
            _log(log_fn, f"Failed to save SHAP importance chart: {e}")

        try:
            beeswarm_path = os.path.join(output_dir, "shap_beeswarm.png")
            _save_plot(
                lambda: shap.plots.beeswarm(
                    shap.Explanation(
                        values=values,
                        base_values=np.full(len(sample_idx), base_values),
                        data=X_sample.to_numpy(),
                        feature_names=plot_names,
                    ),
                    show=False,
                ),
                beeswarm_path,
            )
            charts.append("shap_beeswarm.png")
            _log(log_fn, "Saved SHAP beeswarm summary chart")
        except Exception as e:
            _log(log_fn, f"Failed to save SHAP beeswarm chart: {e}")

        n_waterfalls = min(MAX_WATERFALLS, len(X_sample))
        waterfall_rows = []
        for i in range(n_waterfalls):
            try:
                path = os.path.join(output_dir, f"shap_waterfall_{i}.png")
                _save_plot(
                    lambda i=i: shap.plots.waterfall(
                        shap.Explanation(
                            values=values[i],
                            base_values=base_values,
                            data=X_sample.iloc[i].to_numpy(),
                            feature_names=plot_names,
                        ),
                        show=False,
                    ),
                    path,
                )
                charts.append(f"shap_waterfall_{i}.png")
                waterfall_rows.append(int(sample_idx[i]))
            except Exception as e:
                _log(log_fn, f"Failed to save waterfall {i}: {e}")

        side_data = X_sample.to_numpy().tolist()

        data = {
            "feature_names": full_names,
            "base_value": base_values,
            "shap_values": values.tolist(),
            "feature_values": side_data,
            "sample_indices": [int(i) for i in sample_idx[: n_waterfalls]],
            "predicted_labels": [int(p) for p in model.predict(X_sample)],
            "proba_positive": _proba_positive(model, X_sample),
            "charts": charts,
        }

        json_path = os.path.join(output_dir, SHAP_DATA_FILENAME)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        _log(log_fn, f"SHAP explanation saved ({len(sample_idx)} rows, {len(charts)} charts)")

        return {"charts": charts, **data}

    except Exception as e:
        _log(log_fn, f"SHAP explanation failed: {e}")
        return {"charts": [], "error": str(e)}


def _pick_sample_idx(y: pd.Series, n: int) -> list[int]:
    limit = min(MAX_EXPLAINED_ROWS, n)
    rng = np.random.default_rng(seed=123)
    if len(y) == n and y.nunique() > 1:
        pos = list(np.where(y.to_numpy() == 1)[0])
        neg = list(np.where(y.to_numpy() == 0)[0])
        if pos and neg:
            half = max(1, limit // 2)
            idx = rng.choice(pos, size=min(half, len(pos)), replace=False).tolist()
            idx += rng.choice(neg, size=min(limit - len(idx), len(neg)), replace=False).tolist()
            if len(idx) < limit:
                extra = [i for i in rng.choice(n, size=max(limit - len(idx), 0), replace=False) if i not in idx]
                idx += extra
            idx = idx[:limit]
            rng.shuffle(idx)
            return idx
    return rng.choice(n, size=limit, replace=False).tolist()


def _save_plot(plot_fn, path: str):
    plot_fn()
    fig = plt.gcf()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _proba_positive(model, X: pd.DataFrame) -> list[float | None]:
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if probs.shape[1] == 2:
                return [float(p) for p in probs[:, 1]]
        return [None] * len(X)
    except Exception:
        return [None] * len(X)