import os
from pathlib import Path

import pandas as pd

from conf.config import settings
from services.kopis_client import KopisClient
from utils.helpers import save_to_csv, read_csv_to_dataframe

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def get_performances(start_date: str, end_date: str, file_to_create: str) -> None:
    df = client.fetch_performances(start_date, end_date)
    save_to_csv(df, path=BASE_DIR / Path(settings.RAW_DATA_FOLDER) / file_to_create)


def get_performances_details(from_file_name: str, to_file_name: str) -> None:
    df = read_csv_to_dataframe(
        BASE_DIR / Path(settings.RAW_DATA_FOLDER) / from_file_name
    )
    mt20ids = df["mt20id"].dropna().unique()

    all_details = []

    for idx, mt20id in enumerate(mt20ids, 1):
        print(f"Fetching {idx}/{len(mt20ids)}: {mt20id}")
        detail = client.fetch_performance_detail(mt20id)
        if detail:
            all_details.append(detail)

    details_df = pd.DataFrame(all_details)
    save_to_csv(
        details_df, path=BASE_DIR / Path(settings.RAW_DATA_FOLDER) / to_file_name
    )


if __name__ == "__main__":
    client = KopisClient()
    get_performances(
        start_date="01012025",
        end_date="31122025",
        file_to_create="performances_2025.csv",
    )
    get_performances_details(
        from_file_name="performances_2025.csv",
        to_file_name="performances_details_2025.csv",
    )
