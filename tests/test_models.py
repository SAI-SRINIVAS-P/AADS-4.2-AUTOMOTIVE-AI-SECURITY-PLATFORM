import numpy as np
import pandas as pd

from src.data import SIGNAL_COLUMNS, validate_frame
from src.features import make_engineered_features
from src.models import fit_detector, make_supervised_models, predict_frame


def make_frame(n=160):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"ID": [f"id{i % 3}" for i in range(n)], "Time": np.arange(n, dtype=float)})
    for i, column in enumerate(SIGNAL_COLUMNS):
        frame[column] = rng.normal(loc=i, scale=.5, size=n)
    frame["Label"] = [0] * (n // 2) + [1] * (n - n // 2)
    return validate_frame(frame)[0]


def test_feature_engineering_is_finite_and_shared():
    frame = make_frame(80)
    featured = make_engineered_features(frame)
    assert "payload_mean" in featured
    assert "payload_missing_fraction" in featured
    assert len(featured) == len(frame)
    assert np.isfinite(featured.select_dtypes(include=np.number).to_numpy()).all()


def test_isolation_forest_fit_predict_and_batches():
    frame = make_frame(120)
    frame.loc[5, "Signal1_of_ID"] = np.nan
    model = fit_detector(frame, contamination=.1, max_fit_rows=80)
    result = predict_frame(model, frame, batch_size=17)
    assert len(result) == len(frame)
    assert result["anomaly_prediction"].isin([0, 1]).all()
    assert np.isfinite(result["anomaly_score"]).all()


def test_supervised_pipeline_accepts_raw_records():
    frame=make_frame(160)
    X=frame[["ID",*SIGNAL_COLUMNS]]
    y=frame["Label"].astype(int)
    model=make_supervised_models()["Extra Trees"]
    model.fit(X,y)
    preds=model.predict(X.iloc[:10])
    probs=model.predict_proba(X.iloc[:10])[:,1]
    assert len(preds)==10
    assert ((probs>=0)&(probs<=1)).all()


def test_label_aware_detector_refuses_to_fit_without_normal_baseline():
    import pytest
    frame = make_frame(80)
    frame["Label"] = 1
    with pytest.raises(ValueError, match="normal-labelled"):
        fit_detector(frame, contamination=.1, max_fit_rows=50)


def test_temporal_features_only_use_current_and_prior_messages():
    from src.features import TEMPORAL_FEATURE_COLUMNS
    frame = pd.DataFrame({
        "Time": [1.0, 2.0, 3.0, 4.0],
        "ID": ["a", "a", "b", "a"],
        "Signal1_of_ID": [10.0, 12.0, 7.0, 15.0],
        "Signal2_of_ID": [0.0, 1.0, 0.0, 4.0],
        "Signal3_of_ID": [2.0, 3.0, 1.0, 8.0],
        "Signal4_of_ID": [5.0, 6.0, 1.0, 9.0],
    })
    first = make_engineered_features(frame)
    changed = frame.copy()
    changed.loc[3, "Signal1_of_ID"] = 999.0
    second = make_engineered_features(changed)
    # Changing a later message must not change features for any earlier record.
    pd.testing.assert_frame_equal(first.iloc[:3], second.iloc[:3])
    assert "log_same_id_message_gap" in TEMPORAL_FEATURE_COLUMNS
    assert first.loc[3, "Signal1_of_ID_delta_prev_same_id"] == 3.0
    assert first.loc[3, "Signal1_of_ID_mean_prev5_same_id"] == 11.0


def test_temporal_classifier_uses_history_and_rejects_future_context():
    import pytest
    frame = make_frame(100)
    x = frame[["Time", "ID", *SIGNAL_COLUMNS]]
    y = frame["Label"].astype(int)
    model = make_supervised_models()["Extra Trees"].fit(x, y)
    current = x.iloc[[80]].copy()
    context = x.iloc[70:80].copy()
    without_context = model.predict_proba(current)[:, 1]
    with_context = model.predict_proba(current, history=context)[:, 1]
    assert without_context.shape == with_context.shape == (1,)
    future = x.iloc[90:95].copy()
    with pytest.raises(ValueError, match="later than"):
        model.predict_proba(current, history=future)


def test_payload_only_baseline_is_available_for_temporal_feature_ablation():
    frame = make_frame(120)
    models = make_supervised_models()
    assert "Extra Trees (payload-only baseline)" in models
    baseline = models["Extra Trees (payload-only baseline)"].fit(
        frame[["Time", "ID", *SIGNAL_COLUMNS]], frame["Label"].astype(int)
    )
    assert baseline.include_temporal is False
    assert len(baseline.preprocessor_.transformers_[0][2]) == len(SIGNAL_COLUMNS) + 9


def test_full_sequence_features_match_api_history_window():
    from src.features import TEMPORAL_CONTEXT_ROWS, make_engineered_features
    frame = make_frame(140)
    x = frame[["Time", "ID", *SIGNAL_COLUMNS]]
    y = frame["Label"].astype(int)
    model = make_supervised_models()["Random Forest"].fit(x, y)
    target_index = 125
    target = x.iloc[[target_index]]
    history = x.iloc[max(0, target_index - TEMPORAL_CONTEXT_ROWS):target_index]
    expected_features = make_engineered_features(x).iloc[[target_index]]
    actual_transformed = model._transform(target, history=history)
    expected_transformed = model.preprocessor_.transform(expected_features)
    if hasattr(actual_transformed, "toarray"):
        actual_transformed = actual_transformed.toarray()
    if hasattr(expected_transformed, "toarray"):
        expected_transformed = expected_transformed.toarray()
    assert np.allclose(actual_transformed, expected_transformed)
