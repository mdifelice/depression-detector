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
from training.engineering import engineer
from training.evaluate import evaluate_model, select_best_model, save_model

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
    try:
        _save_job_status(dataset_id, {
            "dataset_id": dataset_id,
            "status": "running",
            "progress": 0.0,
            "current_step": "Starting...",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
        })

        def log_fn(msg):
            _append_log(dataset_id, msg)

        filename = metadata.get("filename", "")
        dataset_file = _dataset_path(dataset_id, filename)

        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        _append_log(dataset_id, "=== Training Started ===")
        _append_log(dataset_id, f"Dataset: {filename}")

        _save_job_status(dataset_id, {
            "dataset_id": dataset_id,
            "status": "running",
            "progress": 0.1,
            "current_step": "Loading dataset...",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
        })

        df = load_dataset(str(dataset_file))
        _append_log(dataset_id, f"Loaded dataset: {len(df)} rows × {len(df.columns)} columns")

        _save_job_status(dataset_id, {
            "dataset_id": dataset_id,
            "status": "running",
            "progress": 0.2,
            "current_step": "Preprocessing...",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
        })

        df = preprocess(df, metadata, training_settings, log_fn=log_fn)

        _save_job_status(dataset_id, {
            "dataset_id": dataset_id,
            "status": "running",
            "progress": 0.4,
            "current_step": "Engineering features...",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
        })

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

        selected_models = training_settings.get("selected_models", [])
        models_config = training_settings.get("models", {})
        charts = _charts_dir(dataset_id)

        results = []
        total = len(selected_models)

        for i, model_path in enumerate(selected_models):
            progress = 0.4 + (0.5 * (i / total))
            model_name = model_path.rsplit(".", 1)[1]

            _save_job_status(dataset_id, {
                "dataset_id": dataset_id,
                "status": "running",
                "progress": round(progress, 3),
                "current_step": f"Training {model_name} ({i+1}/{total})...",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "error": None,
            })

            cfg = models_config.get(model_path, {})
            constructor_params = cfg.get("constructor_params", {})
            param_grid = cfg.get("param_grid", {})

            result = evaluate_model(
                model_path, constructor_params, param_grid,
                X, y, training_settings, charts, log_fn
            )
            results.append(result)

        _save_job_status(dataset_id, {
            "dataset_id": dataset_id,
            "status": "running",
            "progress": 0.9,
            "current_step": "Selecting best model...",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
        })

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
            save_model(best, artifact, MODELS_DIR, dataset_id)
            _append_log(dataset_id, f"Best model: {best['model_name']} (AUC={best['auc']:.4f}, F1={best['f1']:.4f})")
        else:
            _append_log(dataset_id, "No valid model found")

        _append_log(dataset_id, "=== Training Complete ===")

        _save_job_status(dataset_id, {
            "dataset_id": dataset_id,
            "status": "completed",
            "progress": 1.0,
            "current_step": "Done",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
        })

    except Exception as e:
        tb = traceback.format_exc()
        _append_log(dataset_id, f"ERROR: {e}\n{tb}")
        _save_job_status(dataset_id, {
            "dataset_id": dataset_id,
            "status": "failed",
            "progress": 0.0,
            "current_step": f"Failed: {e}",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        })


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
