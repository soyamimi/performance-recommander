import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score

from conf.config import settings
from utils.helpers import read_csv_to_dataframe

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

class LightFMRecommender:
    def __init__(self, interactions, item_features, loss, epochs, num_threads):
        self.interactions = interactions
        self.item_features = item_features
        self.model = LightFM(loss=loss)
        self.epochs = epochs
        self.num_threads = num_threads

    def train(self):
        self.model.fit(self.interactions, item_features=self.item_features,
                       epochs=self.epochs, num_threads=self.num_threads)

    def evaluate(self, k):
        precision = precision_at_k(self.model, self.interactions,
                                   item_features=self.item_features, k=k).mean()
        auc = auc_score(self.model, self.interactions,
                        item_features=self.item_features).mean()
        return precision, auc

    def recommend(self, user_id):
        n_items = self.interactions.shape[1]
        scores = self.model.predict(user_id, np.arange(n_items), item_features=self.item_features)
        top_items = np.argsort(-scores)
        return top_items, scores[top_items]


if __name__ == "__main__":
    user_data= read_csv_to_dataframe(
        BASE_DIR / Path(settings.RAW_DATA_FOLDER) / "users_2025.csv"
    ).to_numpy()
    interactions = csr_matrix(user_data)

    item_features = load_npz(
        BASE_DIR / Path(settings.PREPROCESSED_DATA_FOLDER) / "preprocessed_data.npz"
    )

    recommender = LightFMRecommender(interactions, item_features=item_features, epochs=30,loss='warp', num_threads=2)

    print("train...")
    recommender.train()
    print("train finished")

    precision, auc = recommender.evaluate(k=3)
    print(f"Precision@3: {precision:.3f}, AUC: {auc:.3f}\n")
