import os
from pathlib import Path

import numpy as np
import pandas as pd

from conf.config import settings
from utils.helpers import save_to_csv, read_csv_to_dataframe

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def generate_user_data(
    performance_file: str, nb_users: int, user_file_to_create: str
) -> None:
    df = read_csv_to_dataframe(
        BASE_DIR / Path(settings.RAW_DATA_FOLDER) / performance_file
    )

    ratings = np.random.choice(
        [0, 1, 2, 3, 4, 5], size=(nb_users, len(df)), p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
    )

    user_ratings = pd.DataFrame(ratings, columns=df["mt20id"])
    # user_ratings["UserID"] = [f"User_{i+1}" for i in range(nb_users)]

    save_to_csv(
        user_ratings,
        path=BASE_DIR / Path(settings.RAW_DATA_FOLDER) / user_file_to_create,
    )


if __name__ == "__main__":
    generate_user_data(
        performance_file="performances_details_2025.csv",
        nb_users=200,
        user_file_to_create="users_2025.csv",
    )
