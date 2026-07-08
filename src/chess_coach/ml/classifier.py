# classifier.py
# 3-layer feedforward network: board features → chess concept probabilities.
#
# Architecture
# ------------
#   Input  768  (piece-placement encoding)
#   Layer1 512  Linear → BatchNorm → ReLU → Dropout(0.3)
#   Layer2 256  Linear → BatchNorm → ReLU → Dropout(0.3)
#   Output  57  Linear  (raw logits — sigmoid applied at inference)
#
# BatchNorm makes training more stable and lets you use higher learning rates.
# Dropout prevents overfitting on the ~67k labeled examples.

from __future__ import annotations

import torch
import torch.nn as nn

from .board_encoder import INPUT_SIZE
from .concept_vocab import NUM_CONCEPTS, CONCEPTS


class ChessConceptClassifier(nn.Module):
    def __init__(
        self,
        input_size:  int   = INPUT_SIZE,
        hidden1:     int   = 512,
        hidden2:     int   = 256,
        num_concepts: int  = NUM_CONCEPTS,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, num_concepts),
            # No activation — BCEWithLogitsLoss includes sigmoid internally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_concepts(
        self, fen: str, threshold: float = 0.4
    ) -> list[tuple[str, float]]:
        """
        Given a FEN string, return (concept_name, probability) pairs
        for all concepts above the threshold, sorted by probability.
        """
        from .board_encoder import fen_to_tensor
        self.eval()
        x      = fen_to_tensor(fen).unsqueeze(0)
        logits = self.forward(x).squeeze(0)
        probs  = torch.sigmoid(logits)
        return sorted(
            [(CONCEPTS[i], probs[i].item())
             for i in range(NUM_CONCEPTS) if probs[i] > threshold],
            key=lambda t: -t[1],
        )