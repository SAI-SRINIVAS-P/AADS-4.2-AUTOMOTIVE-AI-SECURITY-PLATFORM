import json
import numpy as np
import pandas as pd
import joblib
import pytest
from fastapi import HTTPException

import api
from src.data import MODEL_FEATURE_COLUMNS, SIGNAL_COLUMNS
from src.models import make_supervised_models


def test_predict_endpoint_loads_artifact_and_returns_class_score(tmp_path, monkeypatch):
    rng = np.random.default_rng(8)
    frame = pd.DataFrame({"ID": [f"id{i % 3}" for i in range(120)]})
    for i, column in enumerate(SIGNAL_COLUMNS):
        frame[column] = rng.normal(loc=i, scale=1.0, size=len(frame))
    labels = np.array([0] * 60 + [1] * 60)
    model = make_supervised_models()["Random Forest"]
    model.fit(frame[MODEL_FEATURE_COLUMNS], labels)
    artifact = tmp_path / "model.joblib"
    joblib.dump(model, artifact)
    metadata = tmp_path / "active.json"
    metadata.write_text(json.dumps({"threshold":.5,"model_version":"test-model"}),encoding="utf-8")
    monkeypatch.setattr(api, "ACTIVE_MODEL", artifact)
    monkeypatch.setattr(api, "ACTIVE_META", metadata)
    monkeypatch.setattr(api, "log_prediction", lambda *a, **kw: 123)
    api.get_model.cache_clear()
    record = api.TelemetryRecord(ID="id1", Time=12.0, Signal1_of_ID=0.1, Signal2_of_ID=1.0,
                                 Signal3_of_ID=None, Signal4_of_ID=None)
    result = api.predict(record)
    assert result["prediction"] in (0, 1)
    assert result["anomaly_probability"] is not None
    assert 0.0 <= result["anomaly_probability"] <= 1.0
    assert result["event_id"] == 123
    api.get_model.cache_clear()


def test_predict_returns_service_unavailable_without_artifact(tmp_path, monkeypatch):
    missing = tmp_path / "missing.joblib"
    monkeypatch.setattr(api, "ACTIVE_MODEL", missing)
    with pytest.raises(HTTPException) as error:
        api.predict(api.TelemetryRecord(ID="id1", Signal1_of_ID=1.0))
    assert error.value.status_code == 503


def test_record_requires_signal():
    with pytest.raises(Exception):
        api.TelemetryRecord(ID="id1")


def test_http_health_and_prediction(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    rng = np.random.default_rng(32)
    frame = pd.DataFrame({"ID": [f"id{i % 2}" for i in range(80)]})
    for i, col in enumerate(SIGNAL_COLUMNS):
        frame[col] = rng.normal(i, 1, size=80)
    y = np.array([0] * 40 + [1] * 40)
    model = make_supervised_models()["Extra Trees"].fit(frame[MODEL_FEATURE_COLUMNS], y)
    artifact=tmp_path/"model.joblib"; joblib.dump(model,artifact)
    metadata=tmp_path/"active.json"; metadata.write_text(json.dumps({"threshold":.4,"model_version":"http-test"}),encoding="utf-8")
    monkeypatch.setattr(api,"ACTIVE_MODEL",artifact)
    monkeypatch.setattr(api,"ACTIVE_META",metadata)
    monkeypatch.setattr(api,"log_prediction",lambda *a,**kw: 77)
    monkeypatch.setattr(api,"log_prediction_batch",lambda events: [78 for _ in events])
    api.get_model.cache_clear()
    client=TestClient(api.app)
    assert client.get("/health").status_code==200
    response=client.post("/predict",json={"ID":"id1","Signal1_of_ID":1.0,"Signal2_of_ID":2.0})
    assert response.status_code==200, response.text
    data=response.json()
    assert data["event_id"]==77
    assert data["model_version"]=="http-test"
    batch=client.post("/predict/batch",json={"records":[{"ID":"id1","Signal1_of_ID":1.0}]})
    assert batch.status_code==200
    assert batch.json()["count"]==1
    assert batch.json()["results"][0]["event_id"]==78
    api.get_model.cache_clear()


def test_single_prediction_accepts_prior_message_context(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    rng = np.random.default_rng(123)
    n = 90
    frame = pd.DataFrame({"Time": np.arange(n, dtype=float), "ID": [f"id{i % 3}" for i in range(n)]})
    for i, col in enumerate(SIGNAL_COLUMNS):
        frame[col] = rng.normal(i, 1.0, n)
    y = np.array([0] * 45 + [1] * 45)
    model = make_supervised_models().get("Extra Trees")
    model.fit(frame[MODEL_FEATURE_COLUMNS], y)
    artifact = tmp_path / "model.joblib"
    joblib.dump(model, artifact)
    metadata = tmp_path / "active.json"
    metadata.write_text(json.dumps({"threshold": 0.5, "model_version": "temporal-test"}), encoding="utf-8")
    monkeypatch.setattr(api, "ACTIVE_MODEL", artifact)
    monkeypatch.setattr(api, "ACTIVE_META", metadata)
    monkeypatch.setattr(api, "log_prediction", lambda *a, **kw: 234)
    api.get_model.cache_clear()
    client = TestClient(api.app)
    payload = {
        "Time": 100.0, "ID": "id1", "Signal1_of_ID": 0.1, "Signal2_of_ID": 0.2,
        "history": [
            {"Time": 97.0, "ID": "id1", "Signal1_of_ID": 0.0},
            {"Time": 98.0, "ID": "id2", "Signal1_of_ID": 0.3},
            {"Time": 99.0, "ID": "id1", "Signal1_of_ID": 0.2},
        ],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["temporal_context_rows"] == 3
    api.get_model.cache_clear()
