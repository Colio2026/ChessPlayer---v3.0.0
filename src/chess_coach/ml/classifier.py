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

from .board_encoder import COMBINED_SIZE
from .concept_vocab import NUM_CONCEPTS, CONCEPTS


class ChessConceptClassifier(nn.Module):
    def __init__(
        self,
        input_size:  int   = COMBINED_SIZE,
        hidden1:     int   = 1024,
        hidden2:     int   = 512,
        num_concepts: int  = NUM_CONCEPTS,
        dropout:     float = 0.4,
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
        self,
        fen: str,
        threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return (concept_name, probability) pairs above threshold, sorted by prob.

        threshold:
          - None  → load calibrated per-class thresholds from data/thresholds.json
                    if that file exists, otherwise fall back to 0.4 globally.
          - float → use that value for every class (overrides the calibrated file).
        """
        from .board_encoder import fen_to_tensor, move_to_tensor
        from .evaluate      import load_thresholds

        self.eval()
        device = next(self.parameters()).device

        if threshold is None:
            t_vec = load_thresholds()                        # [NUM_CONCEPTS]
        else:
            t_vec = torch.full((NUM_CONCEPTS,), threshold)

        board_t = fen_to_tensor(fen)
        move_t  = move_to_tensor("")   # no move context at inference time
        x       = torch.cat([board_t, move_t]).unsqueeze(0).to(device)
        logits = self.forward(x).squeeze(0).cpu()
        probs  = torch.sigmoid(logits)

        return sorted(
            [(CONCEPTS[i], probs[i].item())
             for i in range(NUM_CONCEPTS) if probs[i] >= t_vec[i]],
            key=lambda t: -t[1],
        )