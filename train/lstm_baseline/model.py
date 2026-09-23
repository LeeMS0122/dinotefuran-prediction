from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def embedding_dimension(cardinality: int, maximum: int = 16) -> int:
    return int(min(maximum, max(2, round(cardinality ** 0.25 * 2))))


class HybridSequenceLSTM(nn.Module):
    """LSTM history encoder plus leakage-safe static categorical/numeric context."""

    def __init__(
        self,
        categorical_cardinalities: Sequence[int],
        numeric_size: int,
        sequence_channels: int = 2,
        hidden_size: int = 32,
        num_layers: int = 1,
        bidirectional: bool = False,
        embedding_max_dim: int = 16,
        mlp_hidden_size: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=sequence_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.embeddings = nn.ModuleList()
        embedding_total = 0
        for cardinality in categorical_cardinalities:
            dimension = embedding_dimension(cardinality, embedding_max_dim)
            self.embeddings.append(nn.Embedding(cardinality, dimension, padding_idx=0))
            embedding_total += dimension
        direction_factor = 2 if bidirectional else 1
        combined_size = hidden_size * direction_factor + embedding_total + numeric_size
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        if self.lstm.bidirectional:
            sequence_vector = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            sequence_vector = hidden[-1]
        parts = [sequence_vector]
        if len(self.embeddings):
            parts.extend(
                embedding(categorical[:, column])
                for column, embedding in enumerate(self.embeddings)
            )
        if numeric.shape[1]:
            parts.append(numeric)
        return self.classifier(torch.cat(parts, dim=1)).squeeze(1)

