"""Causal, sequence-aware feature engineering shared by training and inference.

Temporal features only use the current row and earlier CAN messages. The absolute
clock timestamp is never included as a predictor. Input rows are returned in their
original order after chronological calculations are performed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data import ID_COLUMN, SIGNAL_COLUMNS, TIME_COLUMN

PAYLOAD_SUMMARY_COLUMNS = [
    "payload_non_null_count",
    "payload_missing_fraction",
    "payload_zero_count",
    "payload_zero_fraction",
    "payload_mean",
    "payload_std",
    "payload_min",
    "payload_max",
    "payload_range",
]
TEMPORAL_GAP_COLUMNS = [
    "log_inter_message_gap",
    "log_same_id_message_gap",
    "has_previous_message",
    "has_previous_same_id",
    "same_id_prior_count_100",
]
TEMPORAL_DELTA_COLUMNS = [
    name
    for signal in SIGNAL_COLUMNS
    for name in (f"{signal}_delta_prev_same_id", f"{signal}_abs_delta_prev_same_id")
]
TEMPORAL_ROLLING_COLUMNS = [
    name
    for signal in SIGNAL_COLUMNS
    for name in (f"{signal}_mean_prev5_same_id", f"{signal}_std_prev5_same_id")
]
TEMPORAL_FEATURE_COLUMNS = [*TEMPORAL_GAP_COLUMNS, *TEMPORAL_DELTA_COLUMNS, *TEMPORAL_ROLLING_COLUMNS]
BASE_ENGINEERED_NUMERIC_COLUMNS = [*SIGNAL_COLUMNS, *PAYLOAD_SUMMARY_COLUMNS]
ENGINEERED_NUMERIC_COLUMNS = [*BASE_ENGINEERED_NUMERIC_COLUMNS, *TEMPORAL_FEATURE_COLUMNS]
BASE_FEATURED_COLUMNS = [ID_COLUMN, *BASE_ENGINEERED_NUMERIC_COLUMNS]
FEATURED_COLUMNS = [ID_COLUMN, *ENGINEERED_NUMERIC_COLUMNS]
TEMPORAL_CONTEXT_ROWS = 100


def prepare_raw_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean model input frame while retaining Time only for relative features.

    Missing Time is replaced with row order, making the sequence feature contract usable
    for legacy callers. For deployed inference, real timestamps are strongly recommended.
    """
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    out = frame.copy().reset_index(drop=True)
    required = [ID_COLUMN, *SIGNAL_COLUMNS]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError("Missing model input columns: " + ", ".join(missing))
    if TIME_COLUMN not in out.columns:
        out[TIME_COLUMN] = np.arange(len(out), dtype=float)
    times = pd.to_numeric(out[TIME_COLUMN], errors="coerce").astype(float)
    if times.notna().sum() == 0:
        times = pd.Series(np.arange(len(out), dtype=float))
    else:
        times = times.interpolate(limit_direction="both").ffill().bfill()
    out[TIME_COLUMN] = times.to_numpy(dtype=float)
    out[ID_COLUMN] = out[ID_COLUMN].astype("string").fillna("UNKNOWN").astype(str)
    for column in SIGNAL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    return out[[TIME_COLUMN, ID_COLUMN, *SIGNAL_COLUMNS]]


def make_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create row-local and causal temporal features for a sequence of CAN records.

    Uses previous bus messages for message timing, previous values for the same CAN ID,
    and rolling summaries over up to five previous messages with that same ID. Current
    values are never included in their own rolling baseline. No target/label column is
    read. The returned rows preserve input order.
    """
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=FEATURED_COLUMNS, index=getattr(frame, "index", None))
    original_index = frame.index
    raw = prepare_raw_inputs(frame)
    raw["__position"] = np.arange(len(raw), dtype=np.int64)
    ordered = raw.sort_values([TIME_COLUMN, "__position"], kind="stable").reset_index(drop=True)
    ordered["__sequence_position"] = np.arange(len(ordered), dtype=np.int64)

    signals = ordered[SIGNAL_COLUMNS]
    non_null = signals.notna().sum(axis=1)
    zero_count = signals.eq(0).sum(axis=1)
    features = pd.DataFrame(index=ordered.index)
    features[ID_COLUMN] = ordered[ID_COLUMN]
    features[SIGNAL_COLUMNS] = signals
    features["payload_non_null_count"] = non_null.astype(float)
    features["payload_missing_fraction"] = 1.0 - non_null / float(len(SIGNAL_COLUMNS))
    features["payload_zero_count"] = zero_count.astype(float)
    features["payload_zero_fraction"] = (zero_count / non_null.replace(0, np.nan)).fillna(0.0)
    features["payload_mean"] = signals.mean(axis=1).fillna(0.0)
    features["payload_std"] = signals.std(axis=1, ddof=0).fillna(0.0)
    features["payload_min"] = signals.min(axis=1).fillna(0.0)
    features["payload_max"] = signals.max(axis=1).fillna(0.0)
    features["payload_range"] = (signals.max(axis=1) - signals.min(axis=1)).fillna(0.0)

    # Temporal context is capped at 100 preceding bus messages. This makes training
    # features identical to online inference when the API receives the same 100-row window.
    sequence_position = ordered["__sequence_position"].to_numpy(dtype=np.int64)
    time_values = ordered[TIME_COLUMN].to_numpy(dtype=float)
    id_values = ordered[ID_COLUMN].to_numpy(dtype=object)
    signal_values = {signal: ordered[signal].to_numpy(dtype=float) for signal in SIGNAL_COLUMNS}

    inter_gap = ordered[TIME_COLUMN].diff().clip(lower=0).to_numpy(dtype=float)
    same_id_gap = np.zeros(len(ordered), dtype=float)
    recent_same_id_count = np.zeros(len(ordered), dtype=np.int16)
    has_previous_same_id = np.zeros(len(ordered), dtype=float)
    temporal_delta = {signal: np.zeros(len(ordered), dtype=float) for signal in SIGNAL_COLUMNS}
    temporal_abs_delta = {signal: np.zeros(len(ordered), dtype=float) for signal in SIGNAL_COLUMNS}
    temporal_mean = {signal: np.zeros(len(ordered), dtype=float) for signal in SIGNAL_COLUMNS}
    temporal_std = {signal: np.zeros(len(ordered), dtype=float) for signal in SIGNAL_COLUMNS}

    # Vectorized within each CAN ID. Prefix sums give exact rolling summaries over the
    # previous five same-ID events, excluding any event outside the previous 100 bus rows.
    groups = ordered.groupby(ID_COLUMN, sort=False)
    for positions in groups.indices.values():
        positions = np.asarray(positions, dtype=np.int64)
        local_index = np.arange(len(positions), dtype=np.int64)
        lower = np.searchsorted(positions, positions - TEMPORAL_CONTEXT_ROWS, side="left")
        prior_count = local_index - lower
        recent_same_id_count[positions] = prior_count.astype(np.int16)
        previous_valid = local_index > lower
        has_previous_same_id[positions] = previous_valid.astype(float)

        previous_local = np.maximum(local_index - 1, 0)
        previous_global_positions = positions[previous_local]
        previous_time = time_values[previous_global_positions]
        gap = time_values[positions] - previous_time
        gap = np.where(previous_valid, np.maximum(gap, 0.0), 0.0)
        same_id_gap[positions] = gap

        for signal in SIGNAL_COLUMNS:
            values = signal_values[signal][positions]
            previous = values[previous_local]
            previous = np.where(previous_valid, previous, np.nan)
            delta = values - previous
            temporal_delta[signal][positions] = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            temporal_abs_delta[signal][positions] = np.abs(temporal_delta[signal][positions])

            # For current group position j, use values in [max(lower[j], j-5), j).
            start = np.maximum(lower, local_index - 5)
            valid = np.isfinite(values)
            clean = np.where(valid, values, 0.0)
            count_prefix = np.concatenate(([0], np.cumsum(valid.astype(np.int32))))
            sum_prefix = np.concatenate(([0.0], np.cumsum(clean, dtype=np.float64)))
            square_prefix = np.concatenate(([0.0], np.cumsum(clean * clean, dtype=np.float64)))
            counts = count_prefix[local_index] - count_prefix[start]
            sums = sum_prefix[local_index] - sum_prefix[start]
            square_sums = square_prefix[local_index] - square_prefix[start]
            means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            variance = np.divide(square_sums, counts, out=np.zeros_like(square_sums), where=counts > 0) - means * means
            stds = np.sqrt(np.maximum(variance, 0.0))
            stds[counts < 2] = 0.0
            temporal_mean[signal][positions] = means
            temporal_std[signal][positions] = stds

    features["log_inter_message_gap"] = np.log1p(np.maximum(inter_gap, 0.0)).astype(float)
    features["log_inter_message_gap"] = features["log_inter_message_gap"].fillna(0.0)
    features["log_same_id_message_gap"] = np.log1p(same_id_gap).astype(float)
    features["has_previous_message"] = (sequence_position > 0).astype(float)
    features["has_previous_same_id"] = has_previous_same_id
    features["same_id_prior_count_100"] = recent_same_id_count.astype(float)
    for signal in SIGNAL_COLUMNS:
        features[f"{signal}_delta_prev_same_id"] = temporal_delta[signal]
        features[f"{signal}_abs_delta_prev_same_id"] = temporal_abs_delta[signal]
        features[f"{signal}_mean_prev5_same_id"] = temporal_mean[signal]
        features[f"{signal}_std_prev5_same_id"] = temporal_std[signal]

    # Restore original row order; callers can safely align features with labels by position.
    features["__position"] = ordered["__position"].to_numpy()
    features = features.sort_values("__position", kind="stable").drop(columns="__position")
    features = features[FEATURED_COLUMNS].reset_index(drop=True)
    if len(original_index) == len(features):
        features.index = original_index
    return features


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer that mirrors the same feature contract at inference."""

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))
        return make_engineered_features(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURED_COLUMNS, dtype=object)
