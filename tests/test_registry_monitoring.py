import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.monitoring import get_alert_history, log_prediction, save_feedback, monitoring_summary
from src.registry import promote_candidate, save_experiment
from src.models import evaluate_models
from src.data import SIGNAL_COLUMNS


def test_persist_feedback_and_summary(tmp_path):
    db=tmp_path/"monitor.sqlite3"
    event_id=log_prediction(1,.9,"model-v1","test",can_id="0x101",ground_truth=1,db_path=db)
    save_feedback(event_id,1,"confirmed test record","tester",db_path=db)
    history=get_alert_history(db_path=db)
    assert len(history)==1
    assert int(history.iloc[0]["reviewed_label"])==1
    summary=monitoring_summary(db_path=db)
    assert summary["predictions_logged"]==1
    assert summary["reviewed_alerts"]==1


def test_registry_promotion(tmp_path, monkeypatch):
    import src.registry as registry
    root=tmp_path
    monkeypatch.setattr(registry,"ROOT",root)
    monkeypatch.setattr(registry,"ARTIFACTS",root/"artifacts")
    monkeypatch.setattr(registry,"RUNS_DIR",root/"artifacts"/"experiments")
    monkeypatch.setattr(registry,"REGISTRY_FILE",root/"artifacts"/"model_registry.json")
    monkeypatch.setattr(registry,"ACTIVE_MODEL",root/"artifacts"/"supervised_model.joblib")
    monkeypatch.setattr(registry,"ACTIVE_META",root/"artifacts"/"active_model.json")
    monkeypatch.setattr(registry,"ACTIVE_BASELINE",root/"artifacts"/"reference_baseline.csv.gz")
    model={"demo":"model"}
    import datetime
    registry.ARTIFACTS.mkdir(parents=True,exist_ok=True)
    run_id="unit-run"
    run_dir=registry.RUNS_DIR/run_id
    run_dir.mkdir(parents=True)
    joblib.dump(model,run_dir/"rf.joblib")
    data={"run_id":run_id,"candidates":[{"model_name":"Random Forest","artifact":"artifacts/experiments/unit-run/rf.joblib","threshold":.42,"metrics":{"F1":.7}}]}
    registry.REGISTRY_FILE.write_text(json.dumps([data]),encoding="utf-8")
    active=registry.promote_candidate(run_id,"Random Forest")
    assert registry.ACTIVE_MODEL.exists()
    assert active["threshold"]==.42
    assert joblib.load(registry.ACTIVE_MODEL)==model


def test_save_experiment_persists_metrics_and_candidate_artifacts(tmp_path, monkeypatch):
    import src.registry as registry
    from src.registry import save_experiment, list_registry
    from src.models import evaluate_models
    from src.data import validate_frame
    rng=np.random.default_rng(91)
    frame=pd.DataFrame({"ID":[f"id{i%3}" for i in range(100)],"Time":np.arange(100,dtype=float)})
    for i,col in enumerate(SIGNAL_COLUMNS):
        frame[col]=rng.normal(i,1,size=100)
    frame["Label"]=[0]*50+[1]*50
    frame,_=validate_frame(frame)
    results=evaluate_models(frame,test_size=.2,max_train_rows=40,max_test_rows=20)
    monkeypatch.setattr(registry,"ROOT",tmp_path)
    monkeypatch.setattr(registry,"ARTIFACTS",tmp_path/"artifacts")
    monkeypatch.setattr(registry,"RUNS_DIR",tmp_path/"artifacts"/"experiments")
    monkeypatch.setattr(registry,"REGISTRY_FILE",tmp_path/"artifacts"/"model_registry.json")
    monkeypatch.setattr(registry,"ACTIVE_MODEL",tmp_path/"artifacts"/"supervised_model.joblib")
    monkeypatch.setattr(registry,"ACTIVE_META",tmp_path/"artifacts"/"active_model.json")
    monkeypatch.setattr(registry,"ACTIVE_BASELINE",tmp_path/"artifacts"/"reference_baseline.csv.gz")
    run_id=save_experiment(results,"unit.csv",len(frame))
    assert (tmp_path/"artifacts"/"experiments"/run_id/"metadata.json").exists()
    assert len(list_registry())==1
    assert len(list_registry()[0]["candidates"])==3
    active=promote_candidate(run_id,"Random Forest")
    assert active["baseline_artifact"]=="artifacts/reference_baseline.csv.gz"
    assert registry.ACTIVE_BASELINE.exists()
