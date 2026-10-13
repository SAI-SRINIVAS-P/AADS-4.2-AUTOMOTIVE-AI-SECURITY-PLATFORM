"""Sequence-aware supervised and unsupervised models for AADS.

Temporal features are engineered only from a row and its preceding CAN messages. For
batch inference, preserve chronological order. For one-record API inference, pass a
preceding history window when available.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import ID_COLUMN, MODEL_FEATURE_COLUMNS, MODEL_INPUT_COLUMNS, SIGNAL_COLUMNS, TARGET_COLUMN, TIME_COLUMN
from .features import (BASE_ENGINEERED_NUMERIC_COLUMNS, BASE_FEATURED_COLUMNS, ENGINEERED_NUMERIC_COLUMNS,
                       TEMPORAL_CONTEXT_ROWS, TEMPORAL_FEATURE_COLUMNS, make_engineered_features, prepare_raw_inputs)

DEFAULT_MAX_TRAIN_ROWS = 50_000
DEFAULT_MAX_TEST_ROWS = 25_000


def make_preprocessor(include_temporal: bool = True) -> ColumnTransformer:
    numeric = ENGINEERED_NUMERIC_COLUMNS if include_temporal else BASE_ENGINEERED_NUMERIC_COLUMNS
    categorical = [ID_COLUMN]
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([("numeric", num_pipe, numeric), ("categorical", cat_pipe, categorical)],
                             remainder="drop", sparse_threshold=0.3)


def _sample_indices(y: pd.Series | np.ndarray, max_rows: int, random_state: int = 42) -> np.ndarray:
    labels = np.asarray(y, dtype=int)
    positions = np.arange(len(labels))
    if max_rows < 1:
        raise ValueError("max_rows must be positive.")
    if len(positions) <= max_rows:
        return positions
    if np.unique(labels).size == 2 and np.bincount(labels, minlength=2).min() >= 2:
        selected, _ = train_test_split(positions, train_size=max_rows, random_state=random_state, stratify=labels)
        return np.sort(selected)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(positions, size=max_rows, replace=False))


def _binary_probability(classifier, transformed) -> np.ndarray:
    probabilities = classifier.predict_proba(transformed)
    classes = list(classifier.classes_)
    if 1 not in classes:
        return np.zeros(len(transformed), dtype=float)
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


class TemporalClassifier(ClassifierMixin, BaseEstimator):
    """Classifier that engineers causal temporal features then fits a tree model.

    `fit_engineered` lets a model comparison share one feature-engineering pass across
    candidates. Ordinary `.fit(X, y)` still works for scripts and API tests.
    """

    def __init__(self, estimator, max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS, include_temporal: bool = True):
        self.estimator = estimator
        self.max_train_rows = max_train_rows
        self.include_temporal = include_temporal

    def fit_engineered(self, engineered: pd.DataFrame, y: pd.Series | np.ndarray):
        target = np.asarray(y, dtype=int)
        if len(engineered) != len(target):
            raise ValueError("Feature rows and labels have different lengths.")
        if np.unique(target).size != 2:
            raise ValueError("Supervised training requires both normal (0) and attack (1) labels.")
        fit_features = engineered if self.include_temporal else engineered[BASE_FEATURED_COLUMNS]
        self.preprocessor_ = make_preprocessor(include_temporal=self.include_temporal)
        transformed = self.preprocessor_.fit_transform(fit_features)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(transformed, target)
        self.classes_ = np.asarray(self.estimator_.classes_)
        self.n_features_in_ = len(MODEL_INPUT_COLUMNS)
        self.feature_names_in_ = np.asarray(MODEL_INPUT_COLUMNS, dtype=object)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        raw = prepare_raw_inputs(X)
        labels = np.asarray(y, dtype=int)
        if len(raw) != len(labels):
            raise ValueError("Input rows and labels have different lengths.")
        # Engineer features using the full training sequence before row sampling, so a
        # sampled row's temporal features still use its real preceding messages.
        engineered = make_engineered_features(raw)
        if not self.include_temporal:
            engineered = engineered[BASE_FEATURED_COLUMNS]
        idx = _sample_indices(labels, self.max_train_rows)
        return self.fit_engineered(engineered.iloc[idx].reset_index(drop=True), labels[idx])

    @property
    def named_steps(self) -> dict[str, Any]:
        if not hasattr(self, "preprocessor_"):
            return {}
        return {"preprocess": self.preprocessor_, "model": self.estimator_}

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_

    def _transform(self, X: pd.DataFrame, history: pd.DataFrame | None = None):
        current = prepare_raw_inputs(X)
        if history is None or len(history) == 0:
            featured = make_engineered_features(current)
        else:
            context = prepare_raw_inputs(history)
            # The history window must be earlier than the records being scored. Rejecting
            # future context prevents silent temporal leakage in serving and evaluation.
            if context[TIME_COLUMN].max() > current[TIME_COLUMN].min():
                raise ValueError("History contains a timestamp later than the records being scored.")
            combined = pd.concat([context, current], ignore_index=True)
            featured_all = make_engineered_features(combined)
            featured = featured_all.iloc[len(context):].reset_index(drop=True)
        if not self.include_temporal:
            featured = featured[BASE_FEATURED_COLUMNS]
        return self.preprocessor_.transform(featured)

    def predict_proba(self, X: pd.DataFrame, history: pd.DataFrame | None = None) -> np.ndarray:
        transformed = self._transform(X, history=history)
        p_attack = _binary_probability(self.estimator_, transformed)
        return np.column_stack([1.0 - p_attack, p_attack])

    def predict(self, X: pd.DataFrame, history: pd.DataFrame | None = None) -> np.ndarray:
        return (self.predict_proba(X, history=history)[:, 1] >= 0.5).astype(int)


def make_supervised_models(max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS) -> dict[str, TemporalClassifier]:
    estimators = {
        "Random Forest": (RandomForestClassifier(n_estimators=140, max_depth=24, min_samples_leaf=2,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=42), True),
        "Extra Trees": (ExtraTreesClassifier(n_estimators=160, max_depth=24, min_samples_leaf=2,
                                            class_weight="balanced", n_jobs=-1, random_state=42), True),
        "Extra Trees (payload-only baseline)": (ExtraTreesClassifier(n_estimators=160, max_depth=24, min_samples_leaf=2,
                                                                     class_weight="balanced", n_jobs=-1, random_state=42), False),
    }
    return {name: TemporalClassifier(model, max_train_rows=max_train_rows, include_temporal=use_temporal)
            for name, (model, use_temporal) in estimators.items()}


def _chronological_split(labelled: pd.DataFrame, test_size: float, max_test_rows: int):
    notes: list[str] = []
    sort_frame = labelled.sort_values(TIME_COLUMN, kind="stable") if TIME_COLUMN in labelled else labelled
    split_name = "Chronological holdout (latest records)"
    test_count = min(max(2, int(np.ceil(len(sort_frame) * test_size))), max_test_rows)
    train, test = sort_frame.iloc[:-test_count], sort_frame.iloc[-test_count:]
    if (train.empty or train[TARGET_COLUMN].nunique() < 2 or test[TARGET_COLUMN].nunique() < 2):
        notes.append("Latest time window did not contain both classes; fell back to stratified random holdout. This can overestimate temporal generalization.")
        test_fraction = min(test_size, max_test_rows / len(labelled))
        train, test = train_test_split(labelled, test_size=test_fraction, random_state=42,
                                       stratify=labelled[TARGET_COLUMN].astype(int))
        split_name = "Stratified random holdout (fallback)"
    return train, test, split_name, notes


def _inner_validation_split(train: pd.DataFrame):
    ordered = train.sort_values(TIME_COLUMN, kind="stable") if TIME_COLUMN in train else train
    count = min(max(2, int(np.ceil(len(ordered) * 0.15))), 25_000)
    fit_part, validation = ordered.iloc[:-count], ordered.iloc[-count:]
    if (len(fit_part) >= 20 and fit_part[TARGET_COLUMN].nunique() == 2
            and validation[TARGET_COLUMN].nunique() == 2):
        return fit_part, validation, "Time-ordered validation slice"
    fit_part, validation = train_test_split(train, test_size=0.2, random_state=23,
                                            stratify=train[TARGET_COLUMN].astype(int))
    return fit_part, validation, "Stratified validation fallback"


def _best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5, 0.0
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    index = int(np.nanargmax(f1))
    return float(np.clip(thresholds[index], 0.01, 0.99)), float(f1[index])


def evaluate_models(frame: pd.DataFrame, test_size: float = 0.2,
                    contamination: float = 0.1, max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS,
                    max_test_rows: int = DEFAULT_MAX_TEST_ROWS,
                    split_strategy: str = "chronological") -> dict:
    if TARGET_COLUMN not in frame or frame[TARGET_COLUMN].dropna().nunique() < 2:
        raise ValueError("Supervised evaluation requires a Label column containing both classes 0 and 1.")
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("Holdout fraction must be between 0.05 and 0.50.")
    labelled = frame[frame[TARGET_COLUMN].isin([0, 1])].copy()
    if len(labelled) < 50:
        raise ValueError("At least 50 labelled rows are recommended for a meaningful model evaluation.")
    class_counts = labelled[TARGET_COLUMN].value_counts()
    if len(class_counts) < 2 or int(class_counts.min()) < 5:
        raise ValueError("At least five records from each class (Label=0 and Label=1) are required for a stable holdout/validation split.")
    if split_strategy == "chronological":
        train_full, test, split_name, notes = _chronological_split(labelled, test_size, max_test_rows)
    else:
        test_fraction = min(test_size, max_test_rows / len(labelled))
        train_full, test = train_test_split(labelled, test_size=test_fraction, random_state=42,
                                            stratify=labelled[TARGET_COLUMN].astype(int))
        split_name = "Stratified random holdout"
        notes = ["Random splitting can overestimate performance on sequential telemetry; use as a diagnostic only."]
    if train_full[TARGET_COLUMN].nunique() < 2 or test[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Train and test partitions must each include normal (0) and attack (1) rows.")

    train_rate = float(train_full[TARGET_COLUMN].mean())
    test_rate = float(test[TARGET_COLUMN].mean())
    if abs(train_rate - test_rate) > 0.05:
        notes.append(f"Attack prevalence differs: train {train_rate:.2%}, test {test_rate:.2%}. Interpret precision/F1 under this shift.")

    fit_base, validation, validation_name = _inner_validation_split(train_full)
    fit_base = fit_base.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    validation = validation.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    test = test.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    fit_y = fit_base[TARGET_COLUMN].astype(int).to_numpy()
    fit_indices = _sample_indices(fit_y, max_train_rows)
    fit_sample = fit_base.iloc[fit_indices].reset_index(drop=True)

    # Compute causal temporal features ONCE over the complete training segment, before
    # stratified row sampling. This preserves proper prior-message context for sampled rows.
    training_features_all = make_engineered_features(fit_base[MODEL_INPUT_COLUMNS].reset_index(drop=True))
    fit_features = training_features_all.iloc[fit_indices].reset_index(drop=True)
    y_test = test[TARGET_COLUMN].astype(int).to_numpy()
    metrics_rows, confusions, fitted, thresholds, validation_metrics = [], {}, {}, {}, {}

    # Context must be strictly earlier than each scored partition, including fallback splits.
    validation_cutoff = pd.to_numeric(validation[TIME_COLUMN], errors="coerce").min()
    validation_history = fit_base[pd.to_numeric(fit_base[TIME_COLUMN], errors="coerce") < validation_cutoff].tail(TEMPORAL_CONTEXT_ROWS)[MODEL_INPUT_COLUMNS].copy()
    test_cutoff = pd.to_numeric(test[TIME_COLUMN], errors="coerce").min()
    training_context = train_full.sort_values(TIME_COLUMN, kind="stable")
    preceding_test_context = training_context[pd.to_numeric(training_context[TIME_COLUMN], errors="coerce") < test_cutoff].tail(TEMPORAL_CONTEXT_ROWS)[MODEL_INPUT_COLUMNS].copy()

    for name, candidate in make_supervised_models(max_train_rows=max_train_rows).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidate.fit_engineered(fit_features, fit_sample[TARGET_COLUMN].astype(int).to_numpy())
        val_probability = candidate.predict_proba(validation[MODEL_INPUT_COLUMNS], history=validation_history)[:, 1]
        threshold, val_f1 = _best_f1_threshold(validation[TARGET_COLUMN].astype(int).to_numpy(), val_probability)
        test_probability = candidate.predict_proba(test[MODEL_INPUT_COLUMNS], history=preceding_test_context)[:, 1]
        pred = (test_probability >= threshold).astype(int)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics_rows.append({
            "Model": name, "Accuracy": accuracy_score(y_test, pred),
            "Balanced accuracy": balanced_accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, test_probability),
            "PR-AUC": average_precision_score(y_test, test_probability),
            "False-positive rate": fp / max(fp + tn, 1),
            "Predicted attack rate": float(np.mean(pred)), "Threshold": threshold,
            "Validation F1": val_f1,
        })
        confusions[name] = cm
        fitted[name] = candidate
        thresholds[name] = threshold
        validation_metrics[name] = {"threshold": threshold, "f1": val_f1, "validation_rows": len(validation)}

    metrics = pd.DataFrame(metrics_rows).set_index("Model")
    best_model_name = str(metrics["F1"].idxmax())
    importance = _feature_importance(fitted.get(best_model_name))
    reference_columns = [c for c in MODEL_INPUT_COLUMNS if c in fit_sample.columns]
    training_reference = fit_sample[reference_columns].copy()
    if len(training_reference) > 5000:
        training_reference = training_reference.iloc[np.linspace(0, len(training_reference) - 1, 5000, dtype=int)].copy()
    notes.extend([
        f"Causal temporal features added: {len(TEMPORAL_FEATURE_COLUMNS)} features from inter-message timing, per-CAN-ID signal changes, recent ID frequency, and previous-five-message rolling summaries. A payload-only Extra Trees baseline is included to measure the temporal feature uplift.",
        "Temporal features use preceding records only; absolute timestamps and labels are excluded from classifier features.",
        f"Decision thresholds are tuned on a separate validation slice ({validation_name}); the final chronological test window is not used for threshold selection.",
        "Candidate ranking on this test set is descriptive. Keep a separate untouched benchmark for formal reporting.",
        "Temporal signal importance can be dataset-specific. Validate on separate vehicles/drives and per-attack scenarios before claiming real-world generalization.",
    ])
    if len(labelled) > len(fit_sample) + len(validation) + len(test):
        notes.append(f"Feature context used the full fitting window; model fitting capped at {len(fit_sample):,} rows, validation at {len(validation):,}, and testing at {len(test):,} rows.")
    return {
        "metrics": metrics, "confusion": confusions, "importance": importance,
        "fitted": fitted, "thresholds": thresholds, "validation_metrics": validation_metrics,
        "train_rows": len(fit_sample), "test_rows": len(test), "validation_rows": len(validation),
        "split_name": split_name, "validation_split_name": validation_name, "notes": notes,
        "train_attack_rate": train_rate, "test_attack_rate": test_rate,
        "best_model_name": best_model_name, "training_reference": training_reference,
        "temporal_feature_count": len(TEMPORAL_FEATURE_COLUMNS),
    }


def _feature_importance(model: TemporalClassifier | None) -> pd.DataFrame:
    if model is None or not hasattr(model, "preprocessor_"):
        return pd.DataFrame(columns=["feature", "importance"])
    try:
        names = model.preprocessor_.get_feature_names_out()
        values = model.estimator_.feature_importances_
        clean = [n.replace("numeric__", "").replace("categorical__", "ID=") for n in names]
        return pd.DataFrame({"feature": clean, "importance": values}).sort_values("importance", ascending=False)
    except (AttributeError, ValueError):
        return pd.DataFrame(columns=["feature", "importance"])


def get_feature_importance(model: TemporalClassifier) -> pd.DataFrame:
    return _feature_importance(model)


class TemporalIsolationForest(BaseEstimator):
    """Isolation Forest trained on normal-only feature rows with temporal context intact."""

    def __init__(self, contamination: float = 0.1, max_fit_rows: int = 80_000):
        self.contamination = contamination
        self.max_fit_rows = max_fit_rows

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None):
        if not 0.01 <= self.contamination <= 0.4:
            raise ValueError("Contamination must be between 0.01 and 0.40.")
        raw = prepare_raw_inputs(X)
        features = make_engineered_features(raw)
        if y is not None:
            labels = np.asarray(y)
            has_labels = np.isin(labels, [0, 1]).any()
        else:
            labels = None
            has_labels = False
        if has_labels:
            indices = np.flatnonzero(labels == 0)
            if len(indices) < 20:
                raise ValueError("At least 20 normal-labelled records are required for label-aware Isolation Forest training.")
        else:
            indices = np.arange(len(raw))
        if len(indices) > self.max_fit_rows:
            indices = np.sort(np.random.default_rng(42).choice(indices, size=self.max_fit_rows, replace=False))
        if len(indices) < 20:
            raise ValueError("At least 20 records are needed to fit Isolation Forest.")
        self.preprocessor_ = make_preprocessor()
        transformed = self.preprocessor_.fit_transform(features.iloc[indices].reset_index(drop=True))
        self.model_ = IsolationForest(n_estimators=150, contamination=self.contamination,
                                      random_state=42, n_jobs=-1).fit(transformed)
        return self

    def _transform(self, X: pd.DataFrame, history: pd.DataFrame | None = None):
        current = prepare_raw_inputs(X)
        if history is None or len(history) == 0:
            features = make_engineered_features(current)
        else:
            context = prepare_raw_inputs(history)
            if context[TIME_COLUMN].max() > current[TIME_COLUMN].min():
                raise ValueError("History contains a timestamp later than the records being scored.")
            combined = pd.concat([context, current], ignore_index=True)
            features_all = make_engineered_features(combined)
            features = features_all.iloc[len(context):].reset_index(drop=True)
        return self.preprocessor_.transform(features)

    def predict(self, X: pd.DataFrame, history: pd.DataFrame | None = None) -> np.ndarray:
        transformed = self._transform(X, history=history)
        return self.model_.predict(transformed)

    def decision_function(self, X: pd.DataFrame, history: pd.DataFrame | None = None) -> np.ndarray:
        transformed = self._transform(X, history=history)
        return self.model_.decision_function(transformed)


def fit_detector(frame: pd.DataFrame, contamination: float = 0.1, max_fit_rows: int = 80_000) -> TemporalIsolationForest:
    detector = TemporalIsolationForest(contamination=contamination, max_fit_rows=max_fit_rows)
    y = frame[TARGET_COLUMN] if TARGET_COLUMN in frame else None
    return detector.fit(frame[MODEL_INPUT_COLUMNS], y)


def predict_frame(detector, frame: pd.DataFrame, batch_size: int = 25_000,
                  context_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score in batches while carrying a preceding history window across batch boundaries."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    ordered = frame.sort_values(TIME_COLUMN, kind="stable") if TIME_COLUMN in frame else frame.copy()
    flags: list[int] = []
    scores: list[float] = []
    source_context = None
    if context_frame is not None:
        source_context = context_frame.sort_values(TIME_COLUMN, kind="stable") if TIME_COLUMN in context_frame else context_frame
    for start in range(0, len(ordered), batch_size):
        block = ordered.iloc[start:start + batch_size]
        history = None
        if source_context is not None and TIME_COLUMN in block.columns:
            cutoff = pd.to_numeric(block[TIME_COLUMN], errors="coerce").min()
            history = source_context[pd.to_numeric(source_context[TIME_COLUMN], errors="coerce") < cutoff].tail(TEMPORAL_CONTEXT_ROWS)[MODEL_INPUT_COLUMNS]
        elif start > 0:
            history = ordered.iloc[max(0, start - TEMPORAL_CONTEXT_ROWS):start][MODEL_INPUT_COLUMNS]
        x = block[MODEL_INPUT_COLUMNS]
        predicted = detector.predict(x, history=history)
        decision = detector.decision_function(x, history=history)
        flags.extend((predicted == -1).astype(int).tolist())
        scores.extend((-decision).astype(float).tolist())
    result = ordered.copy()
    result["anomaly_prediction"] = flags
    result["anomaly_score"] = scores
    return result
