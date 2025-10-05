"""
Module containing classic and modified KNN-based recommender system classes.
It loads and processes data, computes similarity matrices, and generates item recommendations.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from conf.config import settings
from utils.helpers import read_csv_to_dataframe

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


class ClassicKNNRecommender:
    """
    A classic K-Nearest Neighbors recommender using cosine similarity.

    Attributes:
        X (scipy.sparse.csr_matrix): Feature matrix of items.
        df_items (pd.DataFrame): DataFrame of item metadata.
        sim_matrix (ndarray): Precomputed cosine similarity matrix.
    """

    def __init__(self, X: csr_matrix, df_items: pd.DataFrame):
        """
        Initialize the recommender with item vectors and metadata.

        Args:
            X: The feature matrix representing items.
            df_items: A DataFrame containing item metadata.
        """
        self.X = X
        self.df_items = df_items
        # Precompute cosine similarity across all items
        self.sim_matrix = cosine_similarity(X)

    def recommend(self, item_index: int, top_k: int = 10) -> pd.DataFrame:
        """
        Recommend top_k similar items to the given item index.

        Args:
            item_index (int): Index of the item to find recommendations for.
            top_k (int): Number of recommendations to return.

        Returns:
            pd.DataFrame: DataFrame of recommended items and their similarity scores.
        """
        # Get similarity scores for the target item
        scores = self.sim_matrix[item_index]
        # Sort indices in descending order of similarity
        indices = scores.argsort()[::-1]
        # Exclude the item itself
        indices = indices[indices != item_index]

        # Create a DataFrame for the top_k similar items
        return pd.DataFrame(
            [{"item": idx, "score": scores[idx]} for idx in indices[:top_k]]
        ).merge(self.df_items, left_on="item", right_index=True)


class ModifiedKNNRecommender:
    """
    A modified KNN recommender combining trunk-based and percentile-based strategies.

    Attributes:
        X (scipy.sparse matrix or ndarray): Feature matrix of items.
        df_items (pd.DataFrame): DataFrame of item metadata.
        sim_matrix (ndarray): Precomputed cosine similarity matrix.
        knn_model (NearestNeighbors or None): Lazy-initialized NearestNeighbors model.
    """

    def __init__(self, X: csr_matrix, df_items: pd.DataFrame):
        """
        Initialize the modified recommender with item vectors and metadata.

        Args:
            X: The feature matrix representing items.
            df_items: A DataFrame containing item metadata.
        """
        self.X = X
        self.df_items = df_items
        self.sim_matrix = cosine_similarity(X)
        self.knn_model = None  # Will be initialized on first use

    def trunk_list(self, item_index: int, top_k=5) -> list[tuple[Any, Any]]:
        """
        Get top_k most similar items.

        Args:
            item_index (int): Index of the item to find similar items for.
            top_k (int): Number of top similar items to retrieve.

        Returns:
            list of tuples: Each tuple contains (item_index, similarity_score).
        """
        scores = self.sim_matrix[item_index]
        indices = scores.argsort()[::-1]
        indices = indices[indices != item_index]  # Exclude the reference item
        return [(idx, scores[idx]) for idx in indices[:top_k]]

    def knn_on_trunk(self, trunk_indices: list[int], k: int = 3) -> list[Any]:
        """
        Apply KNN on the top trunk_indices to expand recommendations.

        Args:
            trunk_indices (list[int]): Indices of trunk items.
            k (int): Number of neighbors to find for each trunk item.

        Returns:
            list of tuples: Each tuple is (source_idx, neighbor_idx, distance).
        """
        # Lazy-load the KNN model
        if self.knn_model is None:
            self.knn_model = NearestNeighbors(
                metric="cosine", algorithm="brute", n_neighbors=k + 1
            )
            self.knn_model.fit(self.X)

        results = []
        for idx in trunk_indices:
            # For each trunk item, find nearest neighbors
            distances, neighbors = self.knn_model.kneighbors(self.X[idx])
            # The first neighbor is the item itself, so we skip it
            nbrs = neighbors.flatten()[1:]
            dists = distances.flatten()[1:]
            results.extend(list(zip([idx] * len(nbrs), nbrs, dists)))
        return results

    def percentile_range(
        self, item_index: int, low: float = 0.6, high: float = 0.8
    ) -> list[tuple[Any, Any]]:
        """
        Get items whose similarity scores fall within the percentile range.

        Args:
            item_index (int): Index of the item to find similar items for.
            low (float): Lower similarity percentile (0 to 1).
            high (float): Upper similarity percentile (0 to 1).

        Returns:
            list of tuples: Each tuple is (item_index, similarity_score).
        """
        scores = self.sim_matrix[item_index]
        # Compute actual score values corresponding to the given percentiles
        p_low = np.percentile(scores, low * 100)
        p_high = np.percentile(scores, high * 100)

        mask = (scores >= p_low) & (scores <= p_high)
        indices = np.where(mask)[0]
        # Exclude the item itself
        indices = [i for i in indices if i != item_index]
        return [(idx, scores[idx]) for idx in indices]

    def recommend(
        self,
        item_index: int,
        trunk_k: int = 5,
        knn_k: int = 3,
        low_pct: int = 0.7,
        high_pct: int = 0.9,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Combine trunk-based and percentile-based recommendations.

        Args:
            item_index (int): Index of the item to recommend for.
            trunk_k (int): Number of trunk items.
            knn_k (int): Neighbors for KNN on trunk.
            low_pct (float): Lower percentile bound.
            high_pct (float): Upper percentile bound.

        Returns:
            tuple of pd.DataFrames: (df_trunk, df_knn, df_percentile).
        """
        # Get trunk list of items
        trunk = self.trunk_list(item_index, top_k=trunk_k)
        trunk_indices = [i for i, _ in trunk]

        # Expand trunk items by KNN
        knn_recs = self.knn_on_trunk(trunk_indices, k=knn_k)
        # Get items in the specified percentile range
        percentile_recs = self.percentile_range(item_index, low=low_pct, high=high_pct)

        # Check trunk recommendations DataFrame
        df_trunk = pd.DataFrame(
            [{"method": "trunk", "item": idx, "score": score} for idx, score in trunk]
        )
        df_trunk = df_trunk.merge(self.df_items, left_on="item", right_index=True)

        # Check KNN-based recommendations DataFrame
        df_knn = pd.DataFrame(
            [
                {"method": "trunk_knn", "source": src, "item": nbr, "distance": dist}
                for src, nbr, dist in knn_recs
            ]
        )
        df_knn = df_knn.merge(self.df_items, left_on="item", right_index=True)

        # Check percentile-based recommendations DataFrame
        df_percentile = pd.DataFrame(
            [
                {"method": "percentile", "item": idx, "score": score}
                for idx, score in percentile_recs
            ]
        )
        df_percentile = df_percentile.merge(
            self.df_items, left_on="item", right_index=True
        )

        return df_trunk, df_knn, df_percentile


if __name__ == "__main__":
    # Load preprocessed item vectors
    X = load_npz(
        BASE_DIR / Path(settings.PREPROCESSED_DATA_FOLDER) / "preprocessed_data.npz"
    )
    # Load raw item
    df = read_csv_to_dataframe(
        BASE_DIR / Path(settings.RAW_DATA_FOLDER) / "performances_details_2025.csv"
    )

    # Reset index for item lookup convenience
    df_items = df.reset_index(drop=True)

    # Instantiate recommenders
    classicKNN = ClassicKNNRecommender(X, df_items)
    modifiedKNN = ModifiedKNNRecommender(X, df_items)

    # Example: get recommendations for item at index 500
    classic_result = classicKNN.recommend(500, top_k=10)
    _, _, modified_results = modifiedKNN.recommend(500)

    # Display results
    print("-----------------------------------")
    print("Performance :")
    print(df_items.loc[500])
    print("-----------------------------------")
    print("Recommended Items:")
    print("-----------------------------------")
    print(classic_result[["prfnm", "genrenm", "score"]])
    print("-----------------------------------")
    print("Modified KNN:")
    print("-----------------------------------")
    print(
        modified_results[["prfnm", "genrenm", "score"]]
        .sort_values(by="score", ascending=False)
        .head(10)
    )
