"""Local model sensitivity explanations via one-feature ablation (not causal explanations)."""
from __future__ import annotations

import pandas as pd

from .data import ID_COLUMN, MODEL_INPUT_COLUMNS, SIGNAL_COLUMNS, TIME_COLUMN
from .features import TEMPORAL_CONTEXT_ROWS


def _score(model, record: pd.DataFrame, history: pd.DataFrame | None) -> float:
    try:
        return float(model.predict_proba(record[MODEL_INPUT_COLUMNS], history=history)[0, 1])
    except TypeError:
        return float(model.predict_proba(record[[c for c in MODEL_INPUT_COLUMNS if c in record.columns]])[0, 1])


def explain_record(model, record: pd.DataFrame, reference: pd.DataFrame, max_features: int = 12) -> pd.DataFrame:
    """Measure one-field replacement sensitivity using preceding telemetry as context.

    These attributions are diagnostics, not SHAP values or causal explanations.
    """
    if len(record) != 1:
        raise ValueError("explain_record expects exactly one record.")
    raw_record = record.copy()
    if TIME_COLUMN not in raw_record:
        raw_record[TIME_COLUMN] = 0.0
    history = None
    if TIME_COLUMN in reference and len(reference):
        t = pd.to_numeric(raw_record[TIME_COLUMN], errors="coerce").iloc[0]
        prior = reference[pd.to_numeric(reference[TIME_COLUMN], errors="coerce") < t]
        if len(prior):
            history = prior.sort_values(TIME_COLUMN, kind="stable").tail(TEMPORAL_CONTEXT_ROWS)[MODEL_INPUT_COLUMNS]
    base = _score(model, raw_record, history)
    rows = []
    reference = reference.copy()
    for col in SIGNAL_COLUMNS:
        values = pd.to_numeric(reference[col], errors="coerce").dropna()
        if values.empty or pd.isna(raw_record[col].iloc[0]):
            continue
        value = float(values.median())
        changed = raw_record.copy()
        changed.loc[changed.index[0], col] = value
        perturbed = _score(model, changed, history)
        rows.append({"Feature": col, "Observed value": float(raw_record[col].iloc[0]),
                     "Reference value (median)": value, "Score change after replacement": perturbed - base,
                     "Interpretation": "Score decreases after replacement" if perturbed < base else "Score increases after replacement" if perturbed > base else "No score change"})
    if ID_COLUMN in raw_record and ID_COLUMN in reference and len(reference):
        mode = reference[ID_COLUMN].astype(str).mode()
        if not mode.empty:
            changed = raw_record.copy()
            changed.loc[changed.index[0], ID_COLUMN] = mode.iloc[0]
            perturbed = _score(model, changed, history)
            rows.append({"Feature": "ID", "Observed value": str(raw_record[ID_COLUMN].iloc[0]),
                         "Reference value (median)": str(mode.iloc[0]), "Score change after replacement": perturbed - base,
                         "Interpretation": "Score decreases after replacement" if perturbed < base else "Score increases after replacement" if perturbed > base else "No score change"})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Absolute effect"] = out["Score change after replacement"].abs()
    return out.sort_values("Absolute effect", ascending=False).head(max_features).drop(columns="Absolute effect")
