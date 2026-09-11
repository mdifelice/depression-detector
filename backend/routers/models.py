import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Request, HTTPException, Query

from config import DATA_DIR, MODELS_DIR
from auth import get_current_user
from models import (
    DatasetMetadata,
    PredictionRequest,
    PredictionResponse,
    SavedModelInfo,
    SampleRowsResponse,
)
from training.explain import SHAP_DATA_FILENAME

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


def _has_explanation(dataset_id: str) -> bool:
    return (DATA_DIR / dataset_id / "charts" / SHAP_DATA_FILENAME).exists()


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
                has_explanation=_has_explanation(dataset_id),
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
        has_explanation=_has_explanation(dataset_id),
    )


@router.get("/{dataset_id}/schema")
async def get_model_schema(dataset_id: str, request: Request):
    get_current_user(request)
    metadata = _load_metadata(dataset_id)
    model_data = _get_model_data(dataset_id)
    fields, _ = _build_fields(metadata, model_data)
    return {"fields": fields, "target_column": metadata.target_column}


def _build_fields(metadata: DatasetMetadata, model_data: dict) -> tuple[list[dict], list[str]]:
    target = metadata.target_column
    ignore_cols = metadata.ignore_columns
    nullable_cols = metadata.nullable_columns
    categorical = metadata.categorical_columns
    multi_value = metadata.multi_value_columns

    dataset_file = DATA_DIR / f"{metadata.id}.{metadata.filename.rsplit('.', 1)[-1].lower()}"
    if dataset_file.exists():
        if metadata.filename.endswith(".csv"):
            df = pd.read_csv(dataset_file)
        else:
            df = pd.read_excel(dataset_file)
    else:
        df = pd.DataFrame()

    fields = []
    field_cols: list[str] = []
    for col in df.columns:
        if col == target or col in ignore_cols:
            continue
        if col not in model_data.get("feature_names", []) and col not in categorical and col not in multi_value:
            continue
        field_cols.append(col)

        col_type = "text"
        options: list[str] = []
        col_min: float | None = None
        col_max: float | None = None
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
            expanded = df[col].astype(str).str.split(",", expand=False)
            counts: dict[str, int] = {}
            for vals in expanded:
                for v in vals:
                    token = v.strip()
                    if token and token.lower() not in {"none", "nan", "null"}:
                        counts[token] = counts.get(token, 0) + 1
            options = [v for v, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)][:30]
        elif pd.api.types.is_numeric_dtype(df[col].dtype) if len(df) > 0 else False:
            col_type = "number"
            if len(df) > 0:
                nums = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(nums) > 0:
                    col_min = float(nums.min())
                    col_max = float(nums.max())
        else:
            try:
                pd.to_datetime(df[col], errors="raise")
                col_type = "date"
            except Exception:
                col_type = "text"
                if len(df) > 0:
                    vc = df[col].dropna().astype(str).value_counts()
                    vc = vc[vc.index.astype(str).str.strip().ne("")]
                    options = [str(v) for v in vc.index[:20]]

        fields.append({
            "name": col,
            "type": col_type,
            "options": options,
            "required": col not in nullable_cols,
            "min": col_min,
            "max": col_max,
        })

    return fields, field_cols


@router.get("/{dataset_id}/sample-rows", response_model=SampleRowsResponse)
async def get_sample_rows(
    dataset_id: str,
    request: Request,
    n: int = Query(default=50, ge=1, le=200),
    random: bool = False,
):
    get_current_user(request)
    metadata = _load_metadata(dataset_id)
    model_data = _get_model_data(dataset_id)

    fields, field_cols = _build_fields(metadata, model_data)
    if not field_cols:
        return SampleRowsResponse(columns=[], rows=[])

    dataset_file = DATA_DIR / f"{metadata.id}.{metadata.filename.rsplit('.', 1)[-1].lower()}"
    if not dataset_file.exists():
        return SampleRowsResponse(columns=[], rows=[])
    if metadata.filename.endswith(".csv"):
        df = pd.read_csv(dataset_file)
    else:
        df = pd.read_excel(dataset_file)

    df = df[field_cols]
    if len(df) == 0:
        return SampleRowsResponse(columns=field_cols, rows=[])
    if random:
        sample = df.sample(n=min(n, len(df)))
    else:
        sample = df.head(n)

    type_by_col = {f["name"]: f["type"] for f in fields}
    rows = []
    for _, row in sample.iterrows():
        record: dict = {}
        for col in field_cols:
            col_type = type_by_col[col]
            v = row[col]
            if pd.isna(v):
                record[col] = ""
            elif col_type == "number":
                if isinstance(v, (float, np.floating)):
                    record[col] = int(v) if float(v).is_integer() else float(v)
                else:
                    record[col] = float(v)
            elif col_type == "date":
                ts = pd.to_datetime(v, errors="coerce")
                if not pd.isna(ts):
                    ts = ts.tz_localize(None) if ts.tz is not None else ts
                    record[col] = ts.strftime("%Y-%m-%d")
                else:
                    record[col] = ""
            else:
                record[col] = str(v).strip()
        rows.append(record)

    return SampleRowsResponse(columns=field_cols, rows=rows)


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


@router.get("/{dataset_id}/explanation")
async def get_explanation(dataset_id: str, request: Request):
    get_current_user(request)
    _get_model_data(dataset_id)
    json_path = DATA_DIR / dataset_id / "charts" / SHAP_DATA_FILENAME
    if not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No explanation available for this dataset (retrain to generate SHAP explanations)",
        )
    with open(json_path, "r") as f:
        return json.load(f)


@router.delete("/{dataset_id}")
async def delete_model(dataset_id: str, request: Request):
    get_current_user(request)
    model_file = MODELS_DIR / f"{dataset_id}.joblib"
    if model_file.exists():
        model_file.unlink()
        return {"message": "Model deleted"}
    raise HTTPException(status_code=404, detail="Model not found")