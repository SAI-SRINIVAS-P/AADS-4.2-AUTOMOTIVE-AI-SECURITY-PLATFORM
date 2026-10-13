"""SQLite-backed prediction history and analyst feedback for local monitoring."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "artifacts" / "aads_monitoring.sqlite3"


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    return con


@contextmanager
def _connection(db_path: str | Path | None = None):
    con = _connect(db_path)
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db(db_path: str | Path | None = None) -> None:
    with _connection(db_path) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model_version TEXT NOT NULL,
            source TEXT NOT NULL,
            can_id TEXT,
            prediction INTEGER,
            probability REAL,
            anomaly_score REAL,
            ground_truth INTEGER,
            record_json TEXT
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS feedback (
            prediction_id INTEGER PRIMARY KEY,
            reviewed_label INTEGER NOT NULL CHECK (reviewed_label IN (0, 1)),
            reviewer TEXT NOT NULL DEFAULT 'analyst',
            comment TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id)
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at)")
        con.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_prediction(prediction: int, probability: float | None, model_version: str,
                   source: str, can_id: str | None = None, anomaly_score: float | None = None,
                   ground_truth: int | None = None, record: dict[str, Any] | None = None,
                   db_path: str | Path | None = None) -> int:
    init_db(db_path)
    payload = json.dumps(record or {}, allow_nan=False, default=str)
    with _connection(db_path) as con:
        cur = con.execute("""INSERT INTO predictions
            (created_at, model_version, source, can_id, prediction, probability, anomaly_score, ground_truth, record_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (_now(), str(model_version), str(source), can_id, int(prediction),
             None if probability is None else float(probability),
             None if anomaly_score is None else float(anomaly_score),
             None if ground_truth is None else int(ground_truth), payload))
        con.commit()
        return int(cur.lastrowid)



def log_prediction_batch(events: list[dict[str, Any]], db_path: str | Path | None = None) -> list[int]:
    """Insert a batch of inference events in one SQLite transaction."""
    if not events:
        return []
    init_db(db_path)
    ids: list[int] = []
    with _connection(db_path) as con:
        for event in events:
            record = json.dumps(event.get("record") or {}, allow_nan=False, default=str)
            cur = con.execute("""INSERT INTO predictions
                (created_at, model_version, source, can_id, prediction, probability, anomaly_score, ground_truth, record_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (_now(), str(event.get("model_version", "unknown")), str(event.get("source", "api_batch")),
                 event.get("can_id"), int(event["prediction"]),
                 None if event.get("probability") is None else float(event["probability"]),
                 None if event.get("anomaly_score") is None else float(event["anomaly_score"]),
                 None if event.get("ground_truth") is None else int(event["ground_truth"]), record))
            ids.append(int(cur.lastrowid))
        con.commit()
    return ids

def log_alert_batch(scored: pd.DataFrame, model_version: str, max_alerts: int = 500,
                    db_path: str | Path | None = None) -> int:
    """Persist top-scoring alerts only, avoiding a million-row monitoring database write."""
    init_db(db_path)
    if scored.empty or "anomaly_score" not in scored:
        return 0
    top = scored.nlargest(min(max_alerts, len(scored)), "anomaly_score")
    records = []
    for _, row in top.iterrows():
        label = row.get("Label")
        try:
            label_value = int(label) if pd.notna(label) and int(label) in (0, 1) else None
        except (ValueError, TypeError):
            label_value = None
        record_payload = {k: (None if pd.isna(v) else v) for k, v in row.items()
                          if k in {"Time", "ID", "Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID"}}
        records.append((_now(), model_version, "isolation_forest", str(row.get("ID", "UNKNOWN")),
                        int(row.get("anomaly_prediction", 0)), None, float(row["anomaly_score"]),
                        label_value, json.dumps(record_payload, allow_nan=False, default=str)))
    with _connection(db_path) as con:
        con.executemany("""INSERT INTO predictions
            (created_at, model_version, source, can_id, prediction, probability, anomaly_score, ground_truth, record_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", records)
        con.commit()
    return len(records)


def get_alert_history(limit: int = 500, db_path: str | Path | None = None) -> pd.DataFrame:
    init_db(db_path)
    with _connection(db_path) as con:
        rows = con.execute("""SELECT p.*, f.reviewed_label, f.reviewer, f.comment, f.created_at AS reviewed_at
            FROM predictions p LEFT JOIN feedback f ON f.prediction_id=p.id
            ORDER BY p.id DESC LIMIT ?""", (int(limit),)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def save_feedback(prediction_id: int, reviewed_label: int, comment: str = "",
                  reviewer: str = "analyst", db_path: str | Path | None = None) -> None:
    if reviewed_label not in (0, 1):
        raise ValueError("Reviewed label must be 0 (normal) or 1 (attack).")
    init_db(db_path)
    with _connection(db_path) as con:
        exists = con.execute("SELECT id FROM predictions WHERE id=?", (int(prediction_id),)).fetchone()
        if exists is None:
            raise ValueError(f"Alert {prediction_id} does not exist.")
        con.execute("""INSERT INTO feedback (prediction_id, reviewed_label, reviewer, comment, created_at)
            VALUES (?, ?, ?, ?, ?) ON CONFLICT(prediction_id) DO UPDATE SET
            reviewed_label=excluded.reviewed_label, reviewer=excluded.reviewer,
            comment=excluded.comment, created_at=excluded.created_at""",
            (int(prediction_id), int(reviewed_label), reviewer[:100], comment[:2000], _now()))
        con.commit()


def monitoring_summary(db_path: str | Path | None = None) -> dict[str, Any]:
    init_db(db_path)
    with _connection(db_path) as con:
        total = con.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
        flagged = con.execute("SELECT COUNT(*) AS n FROM predictions WHERE prediction=1").fetchone()["n"]
        reviewed = con.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]
        with_truth = con.execute("SELECT COUNT(*) AS n FROM predictions WHERE ground_truth IS NOT NULL").fetchone()["n"]
        agreements = con.execute("""SELECT COUNT(*) AS n FROM predictions p JOIN feedback f ON p.id=f.prediction_id
            WHERE p.ground_truth IS NOT NULL AND p.ground_truth=f.reviewed_label""").fetchone()["n"]
        matched = con.execute("""SELECT COUNT(*) AS n FROM predictions p JOIN feedback f ON p.id=f.prediction_id
            WHERE p.ground_truth IS NOT NULL""").fetchone()["n"]
    return {"predictions_logged": int(total), "flagged_predictions": int(flagged),
            "reviewed_alerts": int(reviewed), "predictions_with_ground_truth": int(with_truth),
            "review_agreement_with_ground_truth": (float(agreements) / matched if matched else None)}
