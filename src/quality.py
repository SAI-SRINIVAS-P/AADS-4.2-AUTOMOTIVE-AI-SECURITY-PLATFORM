"""Dataset profiling, quality checks, and drift diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ID_COLUMN, SIGNAL_COLUMNS, TARGET_COLUMN, TIME_COLUMN


def data_quality_report(frame: pd.DataFrame) -> dict:
    columns = [c for c in [TIME_COLUMN, ID_COLUMN, *SIGNAL_COLUMNS, TARGET_COLUMN] if c in frame]
    usable = frame[columns] if columns else frame
    duplicate_count = int(usable.duplicated().sum())
    signal_missing = {c: float(frame[c].isna().mean()) for c in SIGNAL_COLUMNS if c in frame}
    label_counts = {}
    if TARGET_COLUMN in frame:
        label_counts = {str(k): int(v) for k, v in frame[TARGET_COLUMN].value_counts(dropna=False).items()}
    time_valid = pd.to_numeric(frame[TIME_COLUMN], errors="coerce").dropna() if TIME_COLUMN in frame else pd.Series(dtype=float)
    time_ordered = bool(time_valid.is_monotonic_increasing) if len(time_valid) > 1 else None
    constant_signals = [c for c in SIGNAL_COLUMNS if c in frame and frame[c].nunique(dropna=True) <= 1]
    findings = []
    if duplicate_count:
        findings.append(f"{duplicate_count:,} exact duplicate rows found in canonical columns.")
    if time_ordered is False:
        findings.append("Timestamps are not monotonically increasing; sort by time before temporal analysis.")
    if any(v > 0.8 for v in signal_missing.values()):
        findings.append("At least one signal is mostly missing. Confirm that this is expected for the source format.")
    if constant_signals:
        findings.append("Constant or entirely missing signals: " + ", ".join(constant_signals) + ".")
    if TARGET_COLUMN not in frame:
        findings.append("No target labels detected: supervised evaluation is unavailable; unsupervised scoring remains possible.")
    elif frame[TARGET_COLUMN].dropna().nunique() < 2:
        findings.append("The target contains fewer than two classes; supervised classification cannot be evaluated.")
    return {
        "rows": int(len(frame)), "columns": int(len(frame.columns)),
        "missing_cells": int(frame.isna().sum().sum()),
        "missing_cell_fraction": float(frame.isna().to_numpy().mean()) if frame.size else 0.0,
        "duplicate_rows": duplicate_count, "unique_ids": int(frame[ID_COLUMN].nunique()) if ID_COLUMN in frame else 0,
        "signal_missing_fraction": signal_missing, "label_counts": label_counts,
        "time_ordered": time_ordered, "constant_signals": constant_signals,
        "findings": findings,
    }


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) < 20 or len(cur) < 20 or np.nanstd(ref) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Simple Population Stability Index (PSI) plus missing-rate changes.

    This is a screening indicator, not a statistical hypothesis test or an alert by itself.
    """
    rows = []
    for col in SIGNAL_COLUMNS:
        if col not in reference or col not in current:
            continue
        psi = _psi(reference[col], current[col])
        ref_missing = float(reference[col].isna().mean())
        cur_missing = float(current[col].isna().mean())
        if np.isnan(psi):
            status = "Insufficient variation/data"
        elif psi < 0.1:
            status = "Low"
        elif psi < 0.25:
            status = "Moderate"
        else:
            status = "High"
        rows.append({"Feature": col, "PSI": psi, "Reference missing %": ref_missing * 100,
                     "Current missing %": cur_missing * 100,
                     "Missing-rate change (pp)": (cur_missing - ref_missing) * 100, "Drift signal": status})
    if ID_COLUMN in reference and ID_COLUMN in current:
        known = set(reference[ID_COLUMN].astype(str).unique())
        unseen = float((~current[ID_COLUMN].astype(str).isin(known)).mean())
        rows.append({"Feature": "CAN ID (new categories)", "PSI": np.nan,
                     "Reference missing %": 0.0, "Current missing %": 0.0,
                     "Missing-rate change (pp)": 0.0, "Drift signal": f"{unseen:.1%} unseen IDs"})
    return pd.DataFrame(rows)
