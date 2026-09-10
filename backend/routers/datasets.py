import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException, Request

from config import DATA_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE
from auth import get_current_user
from models import (
    DatasetMetadata,
    DatasetMetadataUpdate,
    DatasetInfo,
    DatasetColumnsResponse,
    ColumnValuesResponse,
    CategoricalColumnConfig,
    TrainingSettings,
    TrainingSettingsUpdate,
    AVAILABLE_MODELS,
    JobStatus,
)
from training import runner
from training.defaults import build_default_models

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _metadata_path(dataset_id: str) -> Path:
    return DATA_DIR / f"{dataset_id}.metadata.json"


def _training_path(dataset_id: str) -> Path:
    return DATA_DIR / f"{dataset_id}.training.json"


def _dataset_path(dataset_id: str, filename: str) -> Path:
    ext = filename.rsplit(".", 1)[-1].lower()
    return DATA_DIR / f"{dataset_id}.{ext}"


def _load_metadata(dataset_id: str) -> DatasetMetadata:
    path = _metadata_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    with open(path, "r") as f:
        return DatasetMetadata(**json.load(f))


def _save_metadata(metadata: DatasetMetadata):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_metadata_path(metadata.id), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)


def _load_training(dataset_id: str) -> TrainingSettings:
    path = _training_path(dataset_id)
    if not path.exists():
        return TrainingSettings()
    with open(path, "r") as f:
        return TrainingSettings(**json.load(f))


def _save_training(dataset_id: str, settings: TrainingSettings):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_training_path(dataset_id), "w") as f:
        json.dump(settings.model_dump(), f, indent=2)


@router.post("", response_model=DatasetInfo)
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
):
    user = get_current_user(request)

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    dataset_id = str(uuid.uuid4())
    save_path = _dataset_path(dataset_id, file.filename)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(content)

    metadata = DatasetMetadata(
        id=dataset_id,
        filename=file.filename,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        uploaded_by=user.email,
    )
    _save_metadata(metadata)
    _save_training(dataset_id, TrainingSettings(models=build_default_models()))

    return DatasetInfo(
        id=metadata.id,
        filename=metadata.filename,
        uploaded_at=metadata.uploaded_at,
        uploaded_by=metadata.uploaded_by,
    )


@router.get("", response_model=list[DatasetInfo])
async def list_datasets(request: Request):
    user = get_current_user(request)
    datasets = []
    for path in sorted(DATA_DIR.glob("*.metadata.json")):
        with open(path, "r") as f:
            meta = DatasetMetadata(**json.load(f))
        if meta.uploaded_by == user.email:
            datasets.append(DatasetInfo(
                id=meta.id,
                filename=meta.filename,
                uploaded_at=meta.uploaded_at,
                uploaded_by=meta.uploaded_by,
            ))
    return datasets


@router.get("/{dataset_id}", response_model=DatasetMetadata)
async def get_dataset(dataset_id: str, request: Request):
    get_current_user(request)
    return _load_metadata(dataset_id)


@router.get("/{dataset_id}/columns", response_model=DatasetColumnsResponse)
async def get_dataset_columns(dataset_id: str, request: Request):
    get_current_user(request)
    meta = _load_metadata(dataset_id)

    dataset_file = _dataset_path(dataset_id, meta.filename)
    if not dataset_file.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    ext = meta.filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        df = pd.read_csv(dataset_file)
    elif ext == "xlsx":
        df = pd.read_excel(dataset_file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    sample_rows = []
    value_counts = []
    null_counts = []
    for col in df.columns:
        unique_vals = df[col].dropna().astype(str).unique()
        sample_rows.append(list(unique_vals[:3]))
        value_counts.append(len(unique_vals))
        null_counts.append(int(df[col].isna().sum()))

    return DatasetColumnsResponse(
        columns=list(df.columns),
        sample=sample_rows,
        unique_value_counts=value_counts,
        null_counts=null_counts,
        row_count=len(df),
    )


@router.get("/{dataset_id}/columns/{column}/values", response_model=ColumnValuesResponse)
async def get_column_values(dataset_id: str, column: str, request: Request):
    get_current_user(request)
    meta = _load_metadata(dataset_id)

    dataset_file = _dataset_path(dataset_id, meta.filename)
    if not dataset_file.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    ext = meta.filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        df = pd.read_csv(dataset_file)
    elif ext == "xlsx":
        df = pd.read_excel(dataset_file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found")

    unique_vals = sorted(df[column].dropna().astype(str).unique().tolist())
    return ColumnValuesResponse(column=column, unique_values=unique_vals)


@router.patch("/{dataset_id}", response_model=DatasetMetadata)
async def update_dataset(dataset_id: str, update: DatasetMetadataUpdate, request: Request):
    get_current_user(request)
    metadata = _load_metadata(dataset_id)

    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(metadata, field, value)

    _save_metadata(metadata)
    return metadata


@router.get("/{dataset_id}/training", response_model=TrainingSettings)
async def get_training_settings(dataset_id: str, request: Request):
    get_current_user(request)
    _load_metadata(dataset_id)
    return _load_training(dataset_id)


@router.get("/{dataset_id}/training/models", response_model=list[str])
async def get_available_models(dataset_id: str, request: Request):
    get_current_user(request)
    _load_metadata(dataset_id)
    return AVAILABLE_MODELS


@router.patch("/{dataset_id}/training", response_model=TrainingSettings)
async def update_training_settings(
    dataset_id: str, update: TrainingSettingsUpdate, request: Request
):
    get_current_user(request)
    _load_metadata(dataset_id)
    settings = _load_training(dataset_id)

    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(settings, field, value)

    _save_training(dataset_id, settings)
    return settings


@router.post("/{dataset_id}/train", response_model=JobStatus)
async def start_training(dataset_id: str, request: Request):
    get_current_user(request)
    metadata = _load_metadata(dataset_id)
    if not metadata.target_column:
        raise HTTPException(status_code=400, detail="Target column not configured")
    training = _load_training(dataset_id)
    try:
        return runner.start_training(dataset_id, metadata.model_dump(), training.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dataset_id}/train/status", response_model=JobStatus)
async def train_status(dataset_id: str, request: Request):
    get_current_user(request)
    status = runner.get_job_status(dataset_id)
    if not status:
        raise HTTPException(status_code=404, detail="No training job found")
    return JobStatus(**status)


@router.get("/{dataset_id}/train/logs")
async def train_logs(dataset_id: str, request: Request):
    get_current_user(request)
    return {"logs": runner.get_job_logs(dataset_id)}


@router.get("/{dataset_id}/train/results")
async def train_results(dataset_id: str, request: Request):
    get_current_user(request)
    results = runner.get_results(dataset_id)
    if results is None:
        raise HTTPException(status_code=404, detail="No results found")
    return {"results": results}


@router.get("/{dataset_id}/train/charts")
async def train_charts(dataset_id: str, request: Request):
    get_current_user(request)
    charts = runner.get_charts(dataset_id)
    urls = [
        f"/static/{dataset_id}/charts/{name}"
        for name in charts
    ]
    return {"charts": urls}


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str, request: Request):
    get_current_user(request)
    metadata = _load_metadata(dataset_id)

    dataset_file = _dataset_path(dataset_id, metadata.filename)
    if dataset_file.exists():
        dataset_file.unlink()

    meta_file = _metadata_path(dataset_id)
    if meta_file.exists():
        meta_file.unlink()

    train_file = _training_path(dataset_id)
    if train_file.exists():
        train_file.unlink()

    job_file = DATA_DIR / f"{dataset_id}.job.json"
    if job_file.exists():
        job_file.unlink()

    results_dir = DATA_DIR / dataset_id
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)

    return {"message": "Dataset deleted"}