"""Local FastAPI inference service for the AADS ML pipeline.

This is a development/portfolio API. Add authentication, rate limiting, deployment
monitoring, and artifact integrity controls before exposing it outside a trusted network.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import json
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data import MODEL_FEATURE_COLUMNS, MODEL_INPUT_COLUMNS, validate_frame
from src.monitoring import get_alert_history, log_prediction, log_prediction_batch, monitoring_summary, save_feedback
from src.registry import ACTIVE_META, ACTIVE_MODEL
from src.features import TEMPORAL_CONTEXT_ROWS

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="AADS · Automotive AI Security API", version="4.2.0",
              description="CAN telemetry ML inference, local alert history, and feedback. Research use only.")


class TelemetryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Time: float = 0.0
    ID: str = Field(min_length=1, max_length=64)
    Signal1_of_ID: float | None = None
    Signal2_of_ID: float | None = None
    Signal3_of_ID: float | None = None
    Signal4_of_ID: float | None = None

    @model_validator(mode="after")
    def at_least_one_signal(self):
        if all(getattr(self, c) is None for c in ["Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID"]):
            raise ValueError("At least one signal field must contain a numeric value.")
        if not math.isfinite(self.Time):
            raise ValueError("Time must be finite.")
        for col in ["Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID"]:
            value = getattr(self, col)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{col} must be finite or null.")
        return self


class TelemetryRecord(TelemetryPoint):
    """Single event plus optional preceding context (up to 100 earlier bus messages, matching training features)."""
    history: list[TelemetryPoint] = Field(default_factory=list, max_length=TEMPORAL_CONTEXT_ROWS)


class Prediction(BaseModel):
    prediction: int = Field(description="Predicted class at active threshold: 0=normal, 1=attack")
    anomaly_probability: float | None = Field(default=None, description="Uncalibrated classifier score for class 1")
    threshold: float = Field(default=0.5, description="Threshold used for the class decision")
    model_version: str = Field(default="unknown")
    event_id: int | None = None
    temporal_context_rows: int = Field(default=0, description="Number of preceding messages supplied for temporal features")
    warning: str = "Research result, not a safety guarantee."


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[TelemetryRecord] = Field(min_length=1, max_length=10000)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewed_label: int = Field(ge=0, le=1, description="0=normal, 1=attack")
    reviewer: str = Field(default="analyst", min_length=1, max_length=100)
    comment: str = Field(default="", max_length=2000)


@lru_cache(maxsize=4)
def get_model(model_path: str, mtime_ns: int):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError("No promoted model artifact found. Promote a model in Model Registry or run scripts/train.py.")
    return joblib.load(path)


def _active_meta() -> dict[str, Any]:
    if ACTIVE_META.exists():
        try:
            return json.loads(ACTIVE_META.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"model_version": "artifact-without-metadata", "threshold": 0.5, "model_name": "unknown"}


def _predict_one(record: TelemetryRecord, source: str = "api") -> dict[str, Any]:
    if not ACTIVE_MODEL.is_file():
        raise HTTPException(status_code=503, detail="No promoted model artifact. Promote one in Model Registry or run `python -m scripts.train`.")
    try:
        meta = _active_meta()
        model = get_model(str(ACTIVE_MODEL.resolve()), ACTIVE_MODEL.stat().st_mtime_ns)
        current_payload = record.model_dump(exclude={"history"})
        frame, _ = validate_frame(pd.DataFrame([current_payload]))
        x = frame[MODEL_INPUT_COLUMNS]
        history_frame = None
        if record.history:
            history_raw = pd.DataFrame([item.model_dump() for item in record.history])
            history_frame, _ = validate_frame(history_raw)
            history_frame = history_frame[MODEL_INPUT_COLUMNS]
            if history_frame["Time"].max() > float(current_payload["Time"]):
                raise ValueError("history must contain only messages at or before the current message time; future records are not allowed.")
        probability = None
        if hasattr(model, "predict_proba"):
            try:
                probability = float(model.predict_proba(x, history=history_frame)[0, 1])
            except TypeError:
                # Backward compatibility for older promoted scikit-learn artifacts.
                probability = float(model.predict_proba(x)[0, 1])
        threshold = float(meta.get("threshold", 0.5))
        try:
            prediction = int(probability >= threshold) if probability is not None else int(model.predict(x)[0])
        except TypeError:
            prediction = int(model.predict(x)[0])
        event_id = None
        try:
            event_id = log_prediction(prediction, probability, str(meta.get("model_version", "unknown")),
                                      source, can_id=str(record.ID),
                                      record=current_payload)
        except Exception:
            event_id = None
        return {"prediction": prediction, "anomaly_probability": probability, "threshold": threshold,
                "model_version": str(meta.get("model_version", "unknown")), "event_id": event_id,
                "temporal_context_rows": len(record.history),
                "warning": "Research result, not a safety guarantee. Provide recent chronological history for sequence-aware scoring."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.get("/health")
def health() -> dict[str, Any]:
    configured = ACTIVE_MODEL.is_file()
    meta = _active_meta() if configured else {}
    model_state = "not_configured"
    status = "ok"
    if configured:
        try:
            get_model(str(ACTIVE_MODEL.resolve()), ACTIVE_MODEL.stat().st_mtime_ns)
            model_state = "loaded"
        except Exception as exc:
            model_state = "load_error"
            status = "degraded"
            meta["load_error"] = str(exc)
    return {"status": status, "model": model_state,
            "model_version": meta.get("model_version"), "load_error": meta.get("load_error"),
            "model_path": str(ACTIVE_MODEL) if configured else "run python -m scripts.train or promote a registry candidate"}


@app.post("/predict", response_model=Prediction)
def predict(record: TelemetryRecord) -> dict[str, Any]:
    return _predict_one(record)


@app.post("/predict/batch")
def predict_batch(request: BatchRequest) -> dict[str, Any]:
    """Score a batch in one model call and log events in one SQLite transaction."""
    if not ACTIVE_MODEL.is_file():
        raise HTTPException(status_code=503, detail="No promoted model artifact. Promote one in Model Registry or run `python -m scripts.train`.")
    try:
        meta = _active_meta()
        model = get_model(str(ACTIVE_MODEL.resolve()), ACTIVE_MODEL.stat().st_mtime_ns)
        raw = pd.DataFrame([record.model_dump(exclude={"history"}) for record in request.records])
        frame, _ = validate_frame(raw)
        x = frame[MODEL_INPUT_COLUMNS]
        if hasattr(model, "predict_proba"):
            try:
                probability = model.predict_proba(x)[:, 1].astype(float).tolist()
            except TypeError:
                probability = model.predict_proba(x)[:, 1].astype(float).tolist()
        else:
            probability = [None] * len(frame)
        threshold = float(meta.get("threshold", 0.5))
        predictions = [int(p >= threshold) if p is not None else int(y) for p, y in
                       zip(probability, model.predict(x).tolist() if hasattr(model, "predict_proba") else model.predict(x).tolist())]
        version = str(meta.get("model_version", "unknown"))
        events = [{"prediction": pred, "probability": prob, "model_version": version, "source": "api_batch",
                   "can_id": str(record.ID), "record": record.model_dump(exclude={"history"})}
                  for record, pred, prob in zip(request.records, predictions, probability)]
        try:
            event_ids = log_prediction_batch(events)
        except Exception:
            event_ids = [None] * len(events)
        results = [{"prediction": pred, "anomaly_probability": prob, "threshold": threshold,
                    "model_version": version, "event_id": event_id,
                    "warning": "Research result, not a safety guarantee."}
                   for pred, prob, event_id in zip(predictions, probability, event_ids)]
        return {"count": len(results), "model_version": version, "results": results}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {exc}") from exc


@app.get("/alerts")
def alerts(limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
    frame = get_alert_history(limit=limit)
    if frame.empty:
        return {"count": 0, "alerts": []}
    frame = frame.astype(object).where(pd.notna(frame), None)
    return {"count": len(frame), "alerts": frame.to_dict(orient="records")}


@app.post("/alerts/{prediction_id}/feedback")
def alert_feedback(prediction_id: int, request: FeedbackRequest) -> dict[str, Any]:
    try:
        save_feedback(prediction_id, request.reviewed_label, request.comment, request.reviewer)
        return {"status": "saved", "prediction_id": prediction_id, "reviewed_label": request.reviewed_label}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return monitoring_summary()
