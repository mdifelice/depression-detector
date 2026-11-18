import json
import logging
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR, MODELS_DIR
from training.preprocessing import preprocess, load_dataset
from training.engineering import engineer, split_train_test, drop_correlated_columns
from training.evaluate import evaluate_model, select_best_model, save_model
from training.explain import explain_model, clear_shap_outputs

logger = logging.getLogger(__name__)

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _dataset_path(dataset_id: str, filename: str) -> Path:
    ext = filename.rsplit(".", 1)[-1].lower()
    return DATA_DIR / f"{dataset_id}.{ext}"


def _job_status_path(dataset_id: str) -> Path:
    return DATA_DIR / f"{dataset_id}.job.json"


def _results_path(dataset_id: str) -> Path:
    return DATA_DIR / dataset_id / "results.json"


def _charts_dir(dataset_id: str) -> str:
    return str(DATA_DIR / dataset_id / "charts")


def _logs_dir(dataset_id: str) -> str:
    return str(DATA_DIR / dataset_id / "logs")


def _log_file(dataset_id: str) -> str:
    return os.path.join(_logs_dir(dataset_id), "training.log")


def _save_job_status(dataset_id: str, status: dict):
    with _jobs_lock:
        _jobs[dataset_id] = status
    path = _job_status_path(dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_save = {k: v for k, v in status.items() if k != "log_lines"}
    with open(path, "w") as f:
        json.dump(to_save, f, indent=2)


def _append_log(dataset_id: str, message: str):
    log_dir = _logs_dir(dataset_id)
    os.makedirs(log_dir, exist_ok=True)
    with open(_log_file(dataset_id), "a") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat()}] {message}\n")
    with _jobs_lock:
        if dataset_id in _jobs:
            _jobs[dataset_id].setdefault("log_lines", [])
            _jobs[dataset_id]["log_lines"].append(message)


def get_job_status(dataset_id: str) -> dict | None:
    with _jobs_lock:
        if dataset_id in _jobs:
            return _jobs[dataset_id].copy()
    path = _job_status_path(dataset_id)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def get_job_logs(dataset_id: str) -> list[str]:
    log_path = _log_file(dataset_id)
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return [line.rstrip() for line in f.readlines()]
    with _jobs_lock:
        if dataset_id in _jobs:
            return _jobs[dataset_id].get("log_lines", []).copy()
    return []


def get_results(dataset_id: str) -> dict | None:
    path = _results_path(dataset_id)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def get_charts(dataset_id: str) -> list[str]:
    charts = _charts_dir(dataset_id)
    if not os.path.isdir(charts):
        return []
    return sorted([f for f in os.listdir(charts) if f.endswith(".png")])


def _run_training(dataset_id: str, metadata: dict, training_settings: dict):
    started_at = datetime.now(timezone.utc).isoformat()

    def set_status(status: str, progress: float, step: str, error: str | None = None):
        done = status in ("completed", "failed")
        _save_job_status(dataset_id, {
            "dataset_id": dataset_id,
            "status": status,
            "progress": round(progress, 3),
            "current_step": step,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat() if done else None,
            "error": error,
        })

    try:
        set_status("running", 0.0, "Starting...")

        def log_fn(msg):
            _append_log(dataset_id, msg)

        filename = metadata.get("filename", "")
        dataset_file = _dataset_path(dataset_id, filename)

        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        _append_log(dataset_id, "=== Training Started ===")
        _append_log(dataset_id, f"Dataset: {filename}")
        clear_shap_outputs(_charts_dir(dataset_id))

        set_status("running", 0.1, "Loading dataset...")
        df = load_dataset(str(dataset_file))
        _append_log(dataset_id, f"Loaded dataset: {len(df)} rows × {len(df.columns)} columns")

        set_status("running", 0.2, "Preprocessing...")
        df = preprocess(df, metadata, training_settings, log_fn=log_fn)

        set_status("running", 0.3, "Engineering features...")
        artifact = engineer(df, metadata, training_settings, log_fn=log_fn)
        engineered_df = artifact["dataframe"]

        target_column = artifact["target_column"]
        feature_names = artifact["feature_names"]

        if not target_column or target_column not in engineered_df.columns:
            raise ValueError("No target column found after engineering")

        X = engineered_df[feature_names]
        y = engineered_df[target_column]

        _append_log(dataset_id, f"Features: {len(feature_names)}, Samples: {len(X)}")
        _append_log(dataset_id, f"Class distribution: {y.value_counts().to_dict()}")

        random_seed = training_settings.get("random_seed", 123)
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, training_settings.get("test_ratio", 0.2), random_seed, log_fn=log_fn
        )

        correlated = drop_correlated_columns(
            X_train, y_train,
            training_settings.get("correlation_acceptance_threshold", 0.8),
            log_fn=log_fn,
        )
        if correlated:
            X_train = X_train.drop(columns=correlated)
            X_test = X_test.drop(columns=correlated)
        _append_log(dataset_id, f"Model features: {len(X_train.columns)}")

        selected_models = training_settings.get("selected_models", [])
        models_config = training_settings.get("models", {})
        charts = _charts_dir(dataset_id)

        results = []
        total = len(selected_models)

        for i, model_path in enumerate(selected_models):
            model_name = model_path.rsplit(".", 1)[1]
            set_status(
                "running", 0.4 + (0.5 * (i / total)),
                f"Training {model_name} ({i+1}/{total})...",
            )

            cfg = models_config.get(model_path, {})
            constructor_params = cfg.get("constructor_params", {})
            param_grid = cfg.get("param_grid", {})

            result = evaluate_model(
                model_path, constructor_params, param_grid,
                X_train, X_test, y_train, y_test,
                training_settings, charts, log_fn,
            )
            results.append(result)

        set_status("running", 0.9, "Selecting best model...")
        best = select_best_model(results)

        results_summary = []
        for r in results:
            entry = {k: v for k, v in r.items() if k != "model_object"}
            entry["is_best"] = best is not None and r.get("model_name") == best.get("model_name")
            results_summary.append(entry)

        results_path = _results_path(dataset_id)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results_summary, f, indent=2)

        if best:
            save_model(best, artifact, MODELS_DIR, dataset_id, X_train, training_settings)
            _append_log(dataset_id, f"Best model: {best['model_name']} (AUC={best['auc']:.4f}, F1={best['f1']:.4f})")

            if training_settings.get("use_shap", True):
                set_status("running", 0.92, "Computing model explanations (SHAP)...")
                explain_model(
                    best["model_object"], X_train, X_test, y_test, charts,
                    random_seed=random_seed, model_name=best["model_name"], log_fn=log_fn,
                )
            else:
                _append_log(dataset_id, "SHAP explanations disabled in the training settings")
        else:
            _append_log(dataset_id, "No valid model found")

        _append_log(dataset_id, "=== Training Complete ===")
        set_status("completed", 1.0, "Done")

    except Exception as e:
        tb = traceback.format_exc()
        _append_log(dataset_id, f"ERROR: {e}\n{tb}")
        set_status("failed", 0.0, f"Failed: {e}", error=str(e))


def start_training(dataset_id: str, metadata: dict, training_settings: dict):
    existing = get_job_status(dataset_id)
    if existing and existing.get("status") == "running":
        raise ValueError("Training already in progress for this dataset")

    log_path = _log_file(dataset_id)
    if os.path.exists(log_path):
        os.remove(log_path)

    thread = threading.Thread(
        target=_run_training, args=(dataset_id, metadata, training_settings), daemon=True
    )
    thread.start()

    return get_job_status(dataset_id)
