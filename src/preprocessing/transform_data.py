"""
This module handles preprocessing of performance-related data,
including text cleaning, age extraction, and feature transformation
into a sparse matrix for recommendation systems.
"""

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

# Define base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def clean_text(text: str) -> str:
    """
    Clean a text string by removing unwanted characters and patterns.

    Steps:
        - Return original if value is NaN
        - Remove substrings inside parentheses/brackets/braces
        - Clean trailing symbols like '*' or '등'
        - Remove punctuation and special characters
        - Normalize extra whitespace

    Args:
        text (str): The raw text input.

    Returns:
        str: The cleaned text string.
    """
    if pd.isna(text):
        return text

    # Remove bracketed expressions
    text = re.sub(r"[(\[{].*?[)\]}]", "", text).strip()

    # Remove trailing asterisks or '등'
    text = re.sub(r"(등+|\*+)$", "", text.strip())

    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r"[^\w\s]", "", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_min_age(value: str) -> int | None:
    """
    Extract the minimum age in years from a textual age description.

    Handles formats like:
        - "전체 관람가" (returns 0)
        - "만 7세"      (returns 7)
        - "24개월"      (returns 2)
        - Otherwise, returns None

    Args:
        value (str): The age string found in the data.

    Returns:
        int | None: The extracted age in years, or None if unrecognized.
    """
    value = str(value).strip()

    # If age is described as 'all ages'
    if "전체" in value:
        return 0

    # Match "만 X세" format
    match_year = re.search(r"만\s*(\d+)\s*세", value)
    if match_year:
        return int(match_year.group(1))

    # Match age specified in months, e.g. "24개월"
    match_month = re.search(r"(\d+)\s*개월", value)
    if match_month:
        months = int(match_month.group(1))
        # Convert months to at least 1 year if over 12 months
        return int(max(1, math.ceil(months / 12)))

    return None


def create_preprocessing_pipeline(
    categorical_cols: Optional[list[Any]] = None,
    numeric_cols: Optional[list[Any]] = None,
    text_cols: Optional[list[Any]] = None,
) -> Pipeline:
    """
    Create a preprocessing pipeline for categorical, numeric, and text columns.

    Args:
        categorical_cols (list[Any], optional): List of categorical column names.
        numeric_cols (list[Any], optional): List of numeric column names.
        text_cols (list[Any], optional): List of text column names.

    Returns:
        Pipeline: A scikit-learn Pipeline with a ColumnTransformer.
    """
    # Ensure lists are not None
    categorical_cols = categorical_cols or []
    numeric_cols = numeric_cols or []
    text_cols = text_cols or []

    transformers = []

    # One-Hot Encoding for categorical features
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )

    # Normalization for numeric features
    if numeric_cols:
        transformers.append(("num", MinMaxScaler(), numeric_cols))

    # Add TF-IDF vectorization for each text column independently
    if text_cols:
        for text_col in text_cols:
            transformers.append(
                (f"text_{text_col}", TfidfVectorizer(max_features=5000), text_col)
            )

    # Combine transformations, drop remaining unlisted columns
    column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline([("preprocessor", column_transformer)])


if __name__ == "__main__":
    # Load the raw data into a DataFrame
    df = read_csv_to_dataframe(
        BASE_DIR / Path(settings.RAW_DATA_FOLDER) / "performances_details_2025.csv"
    )

    # Columns to clean
    cols_to_clean = ["prfnm", "fcltynm", "sty"]
    df[cols_to_clean] = df[cols_to_clean].fillna("")

    # Apply text cleaning to relevant columns
    for col in cols_to_clean:
        df[col] = df[col].apply(clean_text)

    # Extract numeric age values from textual age fields
    df["age"] = df["prfage"].apply(extract_min_age)

    # Create the pipeline with specified column types
    pipeline = create_preprocessing_pipeline(
        categorical_cols=["genrenm", "age"],
        numeric_cols=[],
        text_cols=["prfnm", "fcltynm", "sty"],
    )

    # Fit and transform the dataset into a sparse feature matrix
    X_processed = pipeline.fit_transform(df)

    # Save the feature matrix to a .npz file
    save_npz(
        BASE_DIR / Path(settings.PREPROCESSED_DATA_FOLDER) / "preprocessed_data.npz",
        X_processed,
    )
