import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, RAW_COLUMNS, TARGET_COLUMN
from src.data import load_data
from src.data.clean_data import clean_data
from src.data.validate_data import DataValidationError, validate_data


def test_clean_data_binarises_target_and_drops_num(raw_df):
    clean = clean_data(raw_df)
    assert "num" not in clean.columns
    assert list(clean.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert set(clean[TARGET_COLUMN].unique()) <= {0, 1}
    assert (clean[TARGET_COLUMN] == (raw_df["num"].astype(int) > 0).astype(int).values).all()


def test_clean_data_converts_question_marks_to_nan(raw_df):
    clean = clean_data(raw_df)
    assert pd.isna(clean.loc[0, "ca"])
    assert pd.isna(clean.loc[3, "thal"])
    assert all(pd.api.types.is_numeric_dtype(clean[c]) for c in clean.columns)


def test_clean_data_drops_duplicates(raw_df):
    duplicated = pd.concat([raw_df, raw_df.iloc[[10, 11]]], ignore_index=True)
    assert len(clean_data(duplicated)) == len(raw_df)


def test_validate_reports_missing_and_out_of_range(clean_df):
    df = clean_df.copy()
    df.loc[1, "age"] = 200
    report = validate_data(df, FEATURE_COLUMNS + [TARGET_COLUMN], TARGET_COLUMN, [0, 1])
    assert report["out_of_range"] == {"age": 1}
    assert report["missing_values"] == {"ca": 2, "thal": 1}


def test_validate_raises_on_missing_columns(clean_df):
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_data(clean_df.drop(columns=["chol"]), FEATURE_COLUMNS)


def test_validate_raises_on_bad_target(clean_df):
    df = clean_df.copy()
    df.loc[0, TARGET_COLUMN] = 3
    with pytest.raises(DataValidationError, match="Unexpected target"):
        validate_data(df, FEATURE_COLUMNS + [TARGET_COLUMN], TARGET_COLUMN, [0, 1])


def test_validate_raises_on_empty():
    with pytest.raises(DataValidationError):
        validate_data(pd.DataFrame(columns=RAW_COLUMNS), RAW_COLUMNS)


def test_load_raw_data_reads_headerless_file_with_question_marks(tmp_path):
    path = tmp_path / "heart.data"
    path.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n"
                    "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2\n")
    data = load_data.load_raw_data(path, download_if_missing=False)
    assert list(data.columns) == RAW_COLUMNS
    assert data.shape == (2, 14)
    assert pd.isna(data.loc[0, "ca"])


def test_load_raw_data_missing_file_without_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data.load_raw_data(tmp_path / "nope.data", download_if_missing=False)


def test_download_dataset_writes_file(tmp_path, monkeypatch):
    class FakeResponse:
        content = b"1,2,3\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(load_data.requests, "get", lambda url, timeout: FakeResponse())
    out = load_data.download_dataset(url="http://example", output_path=tmp_path / "sub" / "d.csv")
    assert out.read_bytes() == b"1,2,3\n"
