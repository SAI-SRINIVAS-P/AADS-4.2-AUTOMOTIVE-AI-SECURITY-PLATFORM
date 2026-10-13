import pandas as pd
import pytest

from src.data import SIGNAL_COLUMNS, validate_frame, has_binary_classes
from src.quality import data_quality_report, drift_report


def test_validate_frame_adds_time_and_preserves_missing_values():
    frame = pd.DataFrame({
        "ID": ["id1", "id2"], "Signal1_of_ID": [1.0, 2.0],
        "Signal2_of_ID": [None, 3.0], "Signal3_of_ID": [0.0, 0.1],
        "Signal4_of_ID": [None, None],
    })
    result, notes = validate_frame(frame)
    assert "Time" in result
    assert result["Signal2_of_ID"].isna().sum() == 1
    assert len(result) == 2


def test_missing_required_columns_raise():
    with pytest.raises(ValueError, match="schema is not supported"):
        validate_frame(pd.DataFrame({"ID": ["id1"]}))


def test_maps_can_id_and_hex_payload_and_text_labels():
    frame = pd.DataFrame({
        "Timestamp": [1.0, 2.0, 3.0, 4.0], "CAN ID": ["0x100", "0x101", "0x100", "0x101"],
        "DATA": ["00 FF 10 20", "0A 0B", "01 02 03 04", "FF EE DD CC"],
        "Class": ["Normal", "DoS", "Normal", "Fuzzy"],
    })
    result, notes = validate_frame(frame)
    assert result["ID"].tolist() == ["0x100", "0x101", "0x100", "0x101"]
    assert "CAN ID" in result.columns and "Timestamp" in result.columns and "Class" in result.columns
    assert result["Label"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert result.loc[0, SIGNAL_COLUMNS[0]] == 0
    assert result.loc[0, SIGNAL_COLUMNS[1]] == 255
    assert any("payload" in note.lower() for note in notes)


def test_maps_per_byte_columns():
    frame = pd.DataFrame({
        "timestamp": [1, 2, 3, 4], "Arbitration_ID": [1, 2, 1, 2],
        "DATA[0]": [1, 2, 3, 4], "DATA[1]": [5, 6, 7, 8],
        "DATA[2]": [9, 10, 11, 12], "DATA[3]": [13, 14, 15, 16],
        "Label": [0, 1, 0, 1],
    })
    result, _ = validate_frame(frame)
    assert result["Signal4_of_ID"].tolist() == [13, 14, 15, 16]


def test_unknown_text_labels_are_not_silently_marked_as_attacks():
    frame = pd.DataFrame({"ID": ["x", "x", "x"], "Signal1_of_ID": [0, 1, 2],
                          "Signal2_of_ID": [1, 2, 3], "Signal3_of_ID": [0, 0, 0],
                          "Signal4_of_ID": [0, 0, 0], "Label": ["Normal", "DoS", "unclassified"]})
    result, notes = validate_frame(frame)
    assert result["Label"].iloc[0] == 0
    assert result["Label"].iloc[1] == 1
    assert pd.isna(result["Label"].iloc[2])
    assert any("unrecognized labels" in n for n in notes)


def test_quality_report_and_drift_report():
    ref = pd.DataFrame({"ID": ["a"] * 100, **{c: range(100) for c in SIGNAL_COLUMNS}, "Time": range(100)})
    cur = pd.DataFrame({"ID": ["new"] * 100, **{c: range(100, 200) for c in SIGNAL_COLUMNS}, "Time": range(100, 200)})
    quality = data_quality_report(ref)
    assert quality["rows"] == 100
    assert quality["unique_ids"] == 1
    report = drift_report(ref, cur)
    assert "CAN ID (new categories)" in report["Feature"].tolist()


def test_has_binary_classes_checks_label_values_not_boolean_membership_mask():
    # Both 0 and 1 are present and should enable supervised training.
    assert has_binary_classes(pd.DataFrame({"Label": [0, 1, 0, 1]})) is True
    # A single class or missing/unknown labels must not enable supervised training.
    assert has_binary_classes(pd.DataFrame({"Label": [0, 0, 0]})) is False
    assert has_binary_classes(pd.DataFrame({"Label": [None, "unknown"]})) is False
    assert has_binary_classes(pd.DataFrame({"Other": [0, 1]})) is False
