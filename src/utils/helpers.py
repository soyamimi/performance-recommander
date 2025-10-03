import os
from pathlib import Path

import pandas as pd


def save_to_csv(df: pd.DataFrame, path: Path) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {path}")


def read_csv_to_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")
