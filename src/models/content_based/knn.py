import os
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from conf.config import settings

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


X = load_npz(
    BASE_DIR / Path(settings.PREPROCESSED_DATA_FOLDER) / "preprocessed_data.npz"
)
