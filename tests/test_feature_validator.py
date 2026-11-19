import numpy as np
import pandas as pd
import pytest

from detection.config import FEATURE_COLUMNS, PROCESSED_DIR
from detection.feature_validator import SchemaError, validate_features


def make_frame(rows=3):
    data = {name: np.arange(rows, dtype=float) + i for i, name in enumerate(FEATURE_COLUMNS)}
    return pd.DataFrame(data)


def test_valid_frame_passes_unchanged():
    frame = make_frame()
    result = validate_features(frame)
    assert result.rejected == {}
    pd.testing.assert_frame_equal(result.features, frame)


def test_columns_are_returned_in_model_order():
    frame = make_frame()
    shuffled = frame[list(reversed(FEATURE_COLUMNS))]
    result = validate_features(shuffled)
    assert list(result.features.columns) == list(FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(result.features, frame)


def test_label_column_never_reaches_features():
    frame = make_frame().assign(label="Benign")
    assert "label" not in validate_features(frame).features.columns


def test_missing_column_is_refused():
    with pytest.raises(SchemaError, match="IAT"):
        validate_features(make_frame().drop(columns="IAT"))


def test_unexpected_column_is_refused():
    with pytest.raises(SchemaError, match="src_ip"):
        validate_features(make_frame().assign(src_ip="10.0.0.1"))


def test_duplicate_column_is_refused():
    frame = make_frame()
    frame.insert(0, "IAT", 1.0, allow_duplicates=True)
    with pytest.raises(SchemaError, match="Duplicate"):
        validate_features(frame)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "not-a-number", None, 1e308, -1e39])
def test_bad_value_rejects_only_its_own_row(bad):
    # Values past float32's range become infinity inside scikit-learn, so they count as bad too.
    frame = make_frame().astype({"Rate": object})
    frame.loc[1, "Rate"] = bad
    result = validate_features(frame)
    assert list(result.features.index) == [0, 2]
    assert list(result.rejected) == [1]
    assert "Rate" in result.rejected[1]


def test_numeric_strings_are_accepted_as_numbers():
    frame = make_frame().astype({"Rate": object})
    frame.loc[0, "Rate"] = "12.5"
    result = validate_features(frame)
    assert result.rejected == {}
    assert result.features.loc[0, "Rate"] == 12.5


def test_input_frame_is_not_modified():
    frame = make_frame()
    original = frame.copy()
    validate_features(frame.assign(label="Benign"))
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("split", ["train", "test"])
def test_configured_schema_matches_processed_files(split):
    path = PROCESSED_DIR / f"{split}_processed.csv"
    if not path.is_file():
        pytest.skip("Processed dataset is not available.")
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    assert columns == [*FEATURE_COLUMNS, "label"]
