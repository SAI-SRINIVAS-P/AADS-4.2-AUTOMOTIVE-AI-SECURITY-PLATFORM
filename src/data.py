from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TARGET_COLUMN = "Label"
TIME_COLUMN = "Time"
ID_COLUMN = "ID"
SIGNAL_COLUMNS = ["Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID"]
# Kept for display/backwards compatibility. The model intentionally excludes absolute Time.
FEATURE_COLUMNS = [TIME_COLUMN, ID_COLUMN, *SIGNAL_COLUMNS]
MODEL_FEATURE_COLUMNS = [ID_COLUMN, *SIGNAL_COLUMNS]
# Raw prediction inputs include Time only to derive relative temporal features; absolute Time is not a predictor.
MODEL_INPUT_COLUMNS = [TIME_COLUMN, *MODEL_FEATURE_COLUMNS]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load CSV or gzip CSV. Low-memory parsing is disabled because mixed CAN IDs are common."""
    return pd.read_csv(path, compression="infer", low_memory=False)


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _find_column(columns: Iterable[object], aliases: set[str]) -> object | None:
    mapping = {_key(col): col for col in columns}
    for alias in aliases:
        if alias in mapping:
            return mapping[alias]
    return None


def _parse_payload(value: object) -> list[float]:
    """Convert a common CAN hex payload (e.g. '00 FF 1A' or '00FF1A') to byte values."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = re.sub(r"(?i)0x", "", text)
    # Prefer space/comma/colon-separated byte tokens.
    tokens = [t for t in re.split(r"[\s,:;|-]+", text) if t]
    if len(tokens) > 1:
        try:
            return [float(int(token, 16)) for token in tokens]
        except ValueError:
            pass
    compact = re.sub(r"[^0-9a-fA-F]", "", text)
    if compact and len(compact) % 2 == 0:
        try:
            return [float(int(compact[i:i + 2], 16)) for i in range(0, len(compact), 2)]
        except ValueError:
            pass
    # Fail explicitly rather than silently inventing byte values.
    raise ValueError(f"Could not parse CAN payload value {text!r}. Expected hexadecimal bytes.")


def _normalize_labels(series: pd.Series) -> tuple[pd.Series, str | None]:
    """Normalize recognizable binary labels; leave unknown labels missing, never silently call them attacks."""
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = series.notna()
    numeric_ratio = (numeric.notna() & non_null).sum() / max(int(non_null.sum()), 1)
    if numeric_ratio >= 0.98:
        unique = set(numeric.dropna().unique())
        if unique.issubset({0, 1}):
            return numeric, None
        if unique.issubset({-1, 1}):
            return numeric.map({-1: 0, 1: 1}), "Mapped labels -1/1 to 0/1."
        return numeric, "Numeric target is not binary 0/1; only rows explicitly labelled 0 or 1 are used for supervised evaluation."

    normal_keys = {"0", "normal", "benign", "benigntraffic", "noattack", "noattacktraffic", "nonattack", "nonattacktraffic", "regular", "regulartraffic", "normaltraffic", "false", "clean", "legitimate", "nonmalicious", "nomaliciousactivity"}
    attack_keys = {"1", "attack", "anomaly", "malicious", "attacktraffic", "abnormal", "true", "intrusion", "dos", "ddos", "fuzzy", "spoofing", "spoof", "injection", "impersonation", "rpmspoofing", "geartspoofing", "rands", "attackfree"}
    def convert(value: object) -> float:
        if pd.isna(value):
            return np.nan
        key = _key(value)
        if key in normal_keys or any(token in key for token in ("normaltraffic", "benigntraffic", "noattack")):
            return 0.0
        if key in attack_keys or any(token in key for token in ("attack", "malicious", "intrusion", "spoof", "injection", "fuzzy", "denialofservice")):
            # Explicitly avoid a likely dataset sentinel such as "attack_free" being
            # interpreted as attack when it semantically means no attack.
            if "free" in key and ("attack" in key or "malicious" in key):
                return 0.0
            return 1.0
        return np.nan

    mapped = series.map(convert).astype(float)
    unknown_count = int((series.notna() & mapped.isna()).sum())
    note = "Mapped recognizable text labels to binary classes (normal=0; attack=1)."
    if unknown_count:
        note += f" Left {unknown_count:,} unrecognized labels missing instead of silently treating them as attacks. Review source-label semantics."
    return mapped, note


def normalize_schema(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map common CAN CSV schemas onto AADS canonical columns.

    Supports the project's canonical columns, Timestamp/CAN ID/Label schemas, per-byte
    DATA[0] columns, and a single hexadecimal DATA/Payload column. The original columns
    are retained to make the normalized export auditable.
    """
    if frame is None or len(frame.columns) == 0:
        raise ValueError("The CSV has no columns.")
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    notes: list[str] = []
    cols = list(df.columns)

    # Only rename where a canonical column is absent.
    aliases = {
        TIME_COLUMN: {"time", "timestamp", "ts", "timestampseconds", "timeinseconds"},
        ID_COLUMN: {"id", "canid", "canidentifier", "canbusid", "arbitrationid", "arbitrationidentifier", "messageid", "frameid"},
        TARGET_COLUMN: {"label", "class", "target", "attack", "attacklabel", "attacktype", "isattack", "anomaly", "groundtruth"},
    }
    for canonical, choices in aliases.items():
        if canonical not in df.columns:
            source = _find_column(cols, choices)
            if source is not None:
                df[canonical] = df[source]
                notes.append(f"Mapped source column '{source}' to '{canonical}' while preserving the original column.")

    # Normalize per-byte signal columns like DATA[0], data0, byte_0, payload[0].
    for idx, canonical in enumerate(SIGNAL_COLUMNS):
        if canonical in df.columns:
            continue
        candidates = {
            f"data{idx}", f"byte{idx}", f"payload{idx}", f"datafield{idx}",
            f"signal{idx + 1}", f"signal{idx + 1}ofid", f"signal{idx + 1}ofcanid"
        }
        source = _find_column(cols, candidates)
        if source is not None:
            df[canonical] = df[source]
            notes.append(f"Mapped source column '{source}' to '{canonical}' while preserving the original column.")

    missing_signals = [c for c in SIGNAL_COLUMNS if c not in df.columns]
    if missing_signals:
        payload_col = _find_column(df.columns, {"data", "payload", "canpayload", "messagebytes", "canmessage", "datafield"})
        if payload_col is not None:
            parsed = df[payload_col].map(_parse_payload)
            for idx, canonical in enumerate(SIGNAL_COLUMNS):
                if canonical not in df.columns:
                    df[canonical] = parsed.map(lambda items, i=idx: items[i] if len(items) > i else np.nan)
            notes.append(f"Expanded hexadecimal payload column '{payload_col}' into four byte-signal features.")

    # Required columns must now exist before numeric conversion.
    required = [ID_COLUMN, *SIGNAL_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        detected = ", ".join(map(str, cols[:20]))
        raise ValueError(
            "This CSV schema is not supported yet. Missing canonical fields: " + ", ".join(missing) +
            f". Detected columns: {detected}. Expected CAN ID plus four signal columns, per-byte DATA[0..3], or a hexadecimal DATA/Payload column."
        )

    if TIME_COLUMN not in df.columns:
        df[TIME_COLUMN] = np.arange(len(df), dtype=float)
        notes.append("No timestamp column found; used row order for the timeline. This is not a real timestamp.")

    if TARGET_COLUMN in df.columns:
        before = df[TARGET_COLUMN].copy()
        df[TARGET_COLUMN], label_note = _normalize_labels(before)
        if label_note:
            notes.append(label_note)

    # Normalize only model input types; retain original columns otherwise.
    df[TIME_COLUMN] = pd.to_numeric(df[TIME_COLUMN], errors="coerce")
    for column in SIGNAL_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[ID_COLUMN] = df[ID_COLUMN].astype("string").fillna("UNKNOWN").astype(str)

    if df.empty:
        raise ValueError("The uploaded dataset has no rows.")
    all_missing = [column for column in SIGNAL_COLUMNS if df[column].isna().all()]
    if all_missing:
        notes.append("Entirely missing signal columns will be retained and imputed: " + ", ".join(all_missing))
    if all(df[column].isna().all() for column in SIGNAL_COLUMNS):
        raise ValueError("No usable numeric payload/signal values were found in any signal column.")
    return df, notes


def validate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compatibility wrapper used by the UI/API and unit tests."""
    return normalize_schema(frame)



def has_binary_classes(frame: pd.DataFrame, target_column: str = TARGET_COLUMN) -> bool:
    """Return True only when both numeric classes 0 and 1 are actually present.

    Do not call `.isin([0, 1]).nunique()` here: that counts the boolean membership
    mask (usually just `True`) instead of the label values, hiding valid datasets.
    """
    if target_column not in frame.columns:
        return False
    labels = pd.to_numeric(frame[target_column], errors="coerce")
    binary_labels = labels[labels.isin([0, 1])].dropna()
    return binary_labels.nunique() == 2

def model_features(frame: pd.DataFrame) -> pd.DataFrame:
    columns = MODEL_INPUT_COLUMNS if TIME_COLUMN in frame.columns else MODEL_FEATURE_COLUMNS
    return frame[columns].copy()
