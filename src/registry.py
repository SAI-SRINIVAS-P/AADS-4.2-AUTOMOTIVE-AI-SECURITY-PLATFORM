"""Local experiment/model registry. Keeps artifacts and metadata reproducible without a service dependency."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RUNS_DIR = ARTIFACTS / "experiments"
REGISTRY_FILE = ARTIFACTS / "model_registry.json"
ACTIVE_MODEL = ARTIFACTS / "supervised_model.joblib"
ACTIVE_META = ARTIFACTS / "active_model.json"
ACTIVE_BASELINE = ARTIFACTS / "reference_baseline.csv.gz"


def _read_registry() -> list[dict[str, Any]]:
    if not REGISTRY_FILE.exists():
        return []
    try:
        data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def list_registry() -> list[dict[str, Any]]:
    return sorted(_read_registry(), key=lambda x: x.get("created_at", ""), reverse=True)


def active_model_metadata() -> dict[str, Any] | None:
    if not ACTIVE_META.exists():
        return None
    try:
        return json.loads(ACTIVE_META.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_experiment(results: dict, dataset_name: str, dataset_rows: int) -> str:
    """Persist trained candidates, their held-out metrics, and thresholds as a new immutable run."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:7]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics = results["metrics"]
    candidates = []
    for model_name, model in results["fitted"].items():
        safe_name = model_name.lower().replace(" ", "_")
        model_path = run_dir / f"{safe_name}.joblib"
        joblib.dump(model, model_path, compress=3)
        row = metrics.loc[model_name].to_dict()
        row = {k: (float(v) if pd.notna(v) else None) for k, v in row.items()}
        candidates.append({
            "model_name": model_name,
            "artifact": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            "threshold": float(results.get("thresholds", {}).get(model_name, 0.5)),
            "metrics": row,
            "validation": results.get("validation_metrics", {}).get(model_name, {}),
            "promoted": False,
        })
    reference = results.get("training_reference")
    baseline_relative = None
    if isinstance(reference, pd.DataFrame) and not reference.empty:
        reference_path = run_dir / "reference_baseline.csv.gz"
        reference.to_csv(reference_path, index=False, compression="gzip")
        baseline_relative = str(reference_path.relative_to(ROOT)).replace("\\", "/")
    metadata = {
        "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name, "dataset_rows": int(dataset_rows),
        "split_name": results.get("split_name"), "validation_split_name": results.get("validation_split_name"),
        "train_rows": int(results.get("train_rows", 0)), "test_rows": int(results.get("test_rows", 0)),
        "validation_rows": int(results.get("validation_rows", 0)),
        "train_attack_rate": float(results.get("train_attack_rate", 0)),
        "test_attack_rate": float(results.get("test_attack_rate", 0)),
        "baseline_artifact": baseline_relative,
        "notes": list(results.get("notes", [])), "candidates": candidates,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    all_runs = _read_registry()
    all_runs.append(metadata)
    REGISTRY_FILE.write_text(json.dumps(all_runs, indent=2), encoding="utf-8")
    return run_id


def promote_candidate(run_id: str, model_name: str) -> dict[str, Any]:
    """Explicitly promote a candidate; does not imply that deployment safety is certified."""
    registry = _read_registry()
    run = next((r for r in registry if r.get("run_id") == run_id), None)
    if run is None:
        raise ValueError(f"Experiment run {run_id!r} was not found.")
    candidate = next((c for c in run.get("candidates", []) if c.get("model_name") == model_name), None)
    if candidate is None:
        raise ValueError(f"Model {model_name!r} was not found in run {run_id!r}.")
    source = ROOT / candidate["artifact"]
    if not source.is_file():
        raise FileNotFoundError(f"Model artifact is missing: {source}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    temp = ARTIFACTS / "supervised_model.joblib.tmp"
    shutil.copy2(source, temp)
    temp.replace(ACTIVE_MODEL)
    baseline_relative = run.get("baseline_artifact")
    if baseline_relative:
        baseline_source = ROOT / baseline_relative
        if baseline_source.is_file():
            shutil.copy2(baseline_source, ACTIVE_BASELINE.with_suffix(ACTIVE_BASELINE.suffix + ".tmp"))
            ACTIVE_BASELINE.with_suffix(ACTIVE_BASELINE.suffix + ".tmp").replace(ACTIVE_BASELINE)
        elif ACTIVE_BASELINE.exists():
            ACTIVE_BASELINE.unlink()
    elif ACTIVE_BASELINE.exists():
        ACTIVE_BASELINE.unlink()
    active = {
        "run_id": run_id, "model_name": model_name,
        "model_version": f"{run_id}:{model_name}",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "threshold": float(candidate.get("threshold", 0.5)),
        "artifact": str(ACTIVE_MODEL.relative_to(ROOT)).replace("\\", "/"),
        "baseline_artifact": str(ACTIVE_BASELINE.relative_to(ROOT)).replace("\\", "/") if ACTIVE_BASELINE.is_file() else None,
        "metrics": candidate.get("metrics", {}),
    }
    ACTIVE_META.write_text(json.dumps(active, indent=2), encoding="utf-8")
    for r in registry:
        for c in r.get("candidates", []):
            c["promoted"] = (r.get("run_id") == run_id and c.get("model_name") == model_name)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return active


def import_trained_model(model, model_name: str, threshold: float, dataset_name: str, row_count: int,
                         reference_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Register a full-data trained model with training metadata when no evaluation run exists."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:7]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "trained_model.joblib"
    joblib.dump(model, model_path, compress=3)
    baseline_relative = None
    if isinstance(reference_frame, pd.DataFrame) and not reference_frame.empty:
        cols = [c for c in ["Time", *MODEL_FEATURE_COLUMNS] if c in reference_frame.columns]
        reference = reference_frame[cols].copy()
        if len(reference) > 5000:
            reference = reference.iloc[np.linspace(0, len(reference) - 1, 5000, dtype=int)]
        baseline_path = run_dir / "reference_baseline.csv.gz"
        reference.to_csv(baseline_path, index=False, compression="gzip")
        baseline_relative = str(baseline_path.relative_to(ROOT)).replace("\\", "/")
    run = {
        "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name, "dataset_rows": int(row_count),
        "split_name": "Full labelled dataset (training artifact; no independent score claimed)",
        "train_rows": int(row_count), "test_rows": 0, "baseline_artifact": baseline_relative,
        "notes": ["This artifact was trained for inference. It has no independent held-out metrics from this training run."],
        "candidates": [{"model_name": model_name, "artifact": str(model_path.relative_to(ROOT)).replace("\\", "/"),
                        "threshold": float(threshold), "metrics": {}, "validation": {}, "promoted": False}],
    }
    REGISTRY_FILE.write_text(json.dumps(_read_registry() + [run], indent=2), encoding="utf-8")
    return {"run_id": run_id, "model_name": model_name}
