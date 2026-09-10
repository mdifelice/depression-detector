import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Request, HTTPException

from config import DATA_DIR, MODELS_DIR
from auth import get_current_user
from models import (
    DatasetMetadata,
    PredictionRequest,
    PredictionResponse,
    SavedModelInfo,
)

router = APIRouter(prefix="/models", tags=["models"])


def _metadata_path(dataset_id: str) -> Path:
    return DATA_DIR / f"{dataset_id}.metadata.json"


def _load_metadata(dataset_id: str) -> DatasetMetadata:
    path = _metadata_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    with open(path, "r") as f:
        return DatasetMetadata(**json.load(f))


def _get_model_data(dataset_id: str) -> dict:
    model_file = MODELS_DIR / f"{dataset_id}.joblib"
    if not model_file.exists():
        raise HTTPException(status_code=404, detail="No trained model found for this dataset")
    return joblib.load(model_file)


def _prediction_log_path(dataset_id: str) -> Path:
    return DATA_DIR / dataset_id / "logs" / "predictions.log"


def _log_prediction(dataset_id: str, values: dict, result: dict):
    path = _prediction_log_path(dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "input": values,
        "prediction": result["prediction"],
        "probability": result.get("probability"),
        "label": result["label"],
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@router.get("", response_model=list[SavedModelInfo])
async def list_models(request: Request):
    user = get_current_user(request)
    models = []
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        dataset_id = path.stem
        try:
            metadata = _load_metadata(dataset_id)
        except HTTPException:
            continue
        if metadata.uploaded_by == user.email:
            model_data = joblib.load(path)
            models.append(SavedModelInfo(
                dataset_id=dataset_id,
                filename=metadata.filename,
                model_name=model_data["model_name"],
                metrics=model_data["metrics"],
            ))
    return models


@router.get("/{dataset_id}", response_model=SavedModelInfo)
async def get_model(dataset_id: str, request: Request):
    get_current_user(request)
    metadata = _load_metadata(dataset_id)
    model_data = _get_model_data(dataset_id)
    return SavedModelInfo(
        dataset_id=dataset_id,
        filename=metadata.filename,
        model_name=model_data["model_name"],
        metrics=model_data["metrics"],
    )


@router.get("/{dataset_id}/schema")
async def get_model_schema(dataset_id: str, request: Request):
    get_current_user(request)
    metadata = _load_metadata(dataset_id)
    model_data = _get_model_data(dataset_id)

    target = metadata.target_column
    ignore_cols = metadata.ignore_columns
    nullable_cols = metadata.nullable_columns
    categorical = metadata.categorical_columns
    multi_value = metadata.multi_value_columns

    dataset_file = DATA_DIR / f"{dataset_id}.{metadata.filename.rsplit('.', 1)[-1].lower()}"
    if dataset_file.exists():
        if metadata.filename.endswith(".csv"):
            df = pd.read_csv(dataset_file)
        else:
            df = pd.read_excel(dataset_file)
    else:
        df = pd.DataFrame()

    fields = []
    for col in df.columns:
        if col == target or col in ignore_cols:
            continue
        if col not in model_data.get("feature_names", []) and col not in categorical and col not in multi_value:
            continue

        col_type = "text"
        if col in categorical:
            cfg = categorical[col]
            if cfg.order:
                col_type = "select"
                options = cfg.order
            else:
                col_type = "select"
                options = sorted(df[col].dropna().unique().tolist()) if len(df) > 0 else []
        elif col in multi_value:
            col_type = "multi"
            options = []
        elif pd.api.types.is_numeric_dtype(df[col].dtype) if len(df) > 0 else False:
            col_type = "number"
            options = []
        else:
            try:
                pd.to_datetime(df[col], errors="raise")
                col_type = "date"
                options = []
            except Exception:
                col_type = "text"
                options = []

        fields.append({
            "name": col,
            "type": col_type,
            "options": options,
            "required": col not in nullable_cols,
        })

    return {"fields": fields, "target_column": target}


def _encode_input(values: dict, model_data: dict, metadata: DatasetMetadata) -> pd.DataFrame:
    df = pd.DataFrame([values])

    ordinal_mappings = model_data.get("ordinal_mappings", {})
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)

    multi_value_cols = metadata.multi_value_columns
    for col in multi_value_cols:
        if col in df.columns:
            val = str(df[col].iloc[0]) if df[col].iloc[0] is not None else ""
            parts = [p.strip() for p in val.split(",") if p.strip()]
            for ohe_prefix in [c for c in model_data.get("ohe_columns", []) if c.startswith(f"{col}__")]:
                total_prefix = ohe_prefix.split("__")[1]
                df[ohe_prefix] = 1 if total_prefix in parts else 0
            df = df.drop(columns=[col])

    ohe_columns = model_data.get("ohe_columns", [])
    categorical_cols = metadata.categorical_columns
    for col, config in categorical_cols.items():
        if col in df.columns and not getattr(config, "ordinal", True):
            ohe_cols_for_col = [c for c in ohe_columns if c.startswith(f"{col}_")]
            if ohe_cols_for_col:
                val = str(df[col].iloc[0])
                for ohe_col in ohe_cols_for_col:
                    suffix = ohe_col[len(col) + 1:]
                    df[ohe_col] = 1 if val == suffix else 0
                df = df.drop(columns=[col])

    datetime_cols = model_data.get("datetime_columns", [])
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") // 10**9

    feature_names = model_data.get("feature_names", [])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]
    return df


@router.post("/{dataset_id}/predict", response_model=PredictionResponse)
async def predict(dataset_id: str, request: Request, payload: PredictionRequest):
    get_current_user(request)
    metadata = _load_metadata(dataset_id)
    model_data = _get_model_data(dataset_id)

    df = _encode_input(payload.values, model_data, metadata)
    scaler = model_data["scaler"]
    if scaler is not None:
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    model = model_data["model"]
    prediction = int(model.predict(df)[0])

    probability = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
            if probs.shape[1] == 2:
                probability = float(probs[0][1])
    except Exception:
        pass

    label = "positive" if prediction == 1 else "negative"
    if probability is not None:
        label = f"positive ({probability:.1%})" if probability >= 0.5 else f"negative ({1 - probability:.1%})"

    result = {
        "prediction": prediction,
        "probability": probability,
        "label": label,
    }
    _log_prediction(dataset_id, payload.values, result)

    return PredictionResponse(**result)


@router.get("/{dataset_id}/prediction-logs")
async def get_prediction_logs(dataset_id: str, request: Request):
    get_current_user(request)
    path = _prediction_log_path(dataset_id)
    if not path.exists():
        return {"logs": []}
    logs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return {"logs": logs}


@router.delete("/{dataset_id}")
async def delete_model(dataset_id: str, request: Request):
    get_current_user(request)
    model_file = MODELS_DIR / f"{dataset_id}.joblib"
    if model_file.exists():
        model_file.unlink()
        return {"message": "Model deleted"}
    raise HTTPException(status_code=404, detail="Model not found")