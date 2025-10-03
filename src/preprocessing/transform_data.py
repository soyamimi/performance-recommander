import math
import os
import re
from pathlib import Path
from typing import Any, Optional

from scipy.sparse import save_npz

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from conf.config import settings
from utils.helpers import read_csv_to_dataframe

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def clean_text(text: str) -> str:
    if pd.isna(text):
        return text

    text = re.sub(r"[(\[{].*?[)\]}]", "", text).strip()
    text = re.sub(r"(등+|\*+)$", "", text.strip())
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_months(row) -> list:
    all_days = pd.date_range(row["start_date"], row["end_date"], freq="D")
    return all_days.month.unique().tolist()


def expand_months(
    data: pd.DataFrame, start_date_col: str, end_date_col: str
) -> pd.DataFrame:
    data["start_date"] = pd.to_datetime(data[start_date_col], format="%Y.%m.%d")
    data["end_date"] = pd.to_datetime(data[end_date_col], format="%Y.%m.%d")
    data["months"] = data.apply(get_months, axis=1)
    return data.explode("months").reset_index(drop=True)


def extract_min_age(value: str) -> int | None:
    value = str(value).strip()

    if "전체" in value:
        return 0

    match_year = re.search(r"만\s*(\d+)\s*세", value)
    if match_year:
        return int(match_year.group(1))

    match_month = re.search(r"(\d+)\s*개월", value)
    if match_month:
        months = int(match_month.group(1))
        return int(max(1, math.ceil(months / 12)))

    return None


def create_preprocessing_pipeline(
    categorical_cols: Optional[list[Any]] = None,
    numeric_cols: Optional[list[Any]] = None,
    text_cols: Optional[list[Any]] = None,
) -> Pipeline:

    categorical_cols = categorical_cols or []
    numeric_cols = numeric_cols or []
    text_cols = text_cols or []

    transformers = []

    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )

    if numeric_cols:
        transformers.append(("num", MinMaxScaler(), numeric_cols))

    if text_cols:
        for text_col in text_cols:
            transformers.append(
                (f"text_{text_col}", TfidfVectorizer(max_features=5000), text_col)
            )

    column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline([("preprocessor", column_transformer)])


if __name__ == "__main__":
    df = read_csv_to_dataframe(
        BASE_DIR / Path(settings.RAW_DATA_FOLDER) / "performances_details_2025.csv"
    )

    cols_to_clean = ["prfnm", "fcltynm", "prfcast"]
    df[cols_to_clean] = df[cols_to_clean].fillna("")

    for col in cols_to_clean:
        df[col] = df[col].apply(clean_text)

    df = expand_months(df, "prfpdfrom", "prfpdto")

    df["age"] = df["prfage"].apply(extract_min_age)

    pipeline = create_preprocessing_pipeline(
        categorical_cols=["area", "genrenm", "months", "age"],
        numeric_cols=[],
        text_cols=["prfnm", "fcltynm", "prfcast"],
    )

    X_processed = pipeline.fit_transform(df)
    save_npz(
        BASE_DIR / Path(settings.PREPROCESSED_DATA_FOLDER) / "preprocessed_data.npz",
        X_processed,
    )
