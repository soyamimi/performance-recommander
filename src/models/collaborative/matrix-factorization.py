import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import Tensor

from conf.config import settings
from utils.helpers import read_csv_to_dataframe

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def prepare_data(ratings_array: np.ndarray) -> tuple[Tensor, Tensor]:
    ratings_tensor: Tensor = torch.tensor(ratings_array, dtype=torch.float32)
    observed_indices: Tensor = torch.nonzero(ratings_tensor)
    return ratings_tensor, observed_indices


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=10):
        super().__init__()
        self.user_factors: nn.Embedding = nn.Embedding(num_users, latent_dim)
        self.item_factors: nn.Embedding = nn.Embedding(num_items, latent_dim)
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self) -> Tensor:
        return torch.matmul(self.user_factors.weight, self.item_factors.weight.t())


def train_model(
    model: nn.Module,
    ratings_tensor: Tensor,
    observed_indices: Tensor,
    lr: float,
    epochs: int,
) -> nn.Module:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="sum")

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model()

        user_idx = observed_indices[:, 0]
        item_idx = observed_indices[:, 1]

        pred_values = pred[user_idx, item_idx]
        true_values = ratings_tensor[user_idx, item_idx]

        loss = loss_fn(pred_values, true_values)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model


def recommend(
    model: nn.Module,
    ratings_array: Tensor,
    user_id: int,
    top_k: int,
    performance_ids: list,
) -> list[list[Any] | Any]:

    predicted_ratings = model().detach()

    user_ratings = predicted_ratings[user_id].clone()
    user_ratings[ratings_array[user_id] != 0] = float("-inf")

    top_k_indices = torch.topk(user_ratings, top_k).indices.numpy()

    return [performance_ids[i] for i in top_k_indices]


if __name__ == "__main__":
    df = read_csv_to_dataframe(
        BASE_DIR / Path(settings.RAW_DATA_FOLDER) / "users_2025.csv"
    )
    performance_ids = df.columns.tolist()

    ratings_tensor, observed_indices = prepare_data(df.to_numpy())

    num_users, num_performances = ratings_tensor.shape
    model = MatrixFactorization(num_users, num_performances, latent_dim=10)

    trained_model = train_model(
        model, ratings_tensor, observed_indices, lr=0.01, epochs=3000
    )

    recommended_performances = recommend(
        trained_model,
        ratings_tensor,
        user_id=0,
        top_k=5,
        performance_ids=performance_ids,
    )
    print(f"Recommended performances for user 0: {recommended_performances}")
