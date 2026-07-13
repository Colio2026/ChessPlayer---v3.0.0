# classifier.py
# 3-layer feedforward network: board features → chess concept probabilities.
#
# Architecture (Phase 2 + concept bottleneck)
# -------------------------------------------
#   Input  1188  (1001 board+attack+pawn+mobility + 128 move + 59 algo bits)
#   Layer1 1536  Linear → BatchNorm → ReLU → Dropout(0.4)
#   Layer2  768  Linear → BatchNorm → ReLU → Dropout(0.2)
#   Output   50  Linear  (raw logits — sigmoid applied at inference)
#
# The 59 algo bits are pre-computed structural concept flags from label_positions.py
# (26 per-color concepts × [white, black] + 7 global bits). They give the model
# verified structural priors so it can bootstrap complex/rare concepts from them.
#
# L2 dropout is lower (0.2) because Phase 1 showed 62.7% of L2 neurons
# were suppressed with 0.4 — the bottleneck was losing effective capacity.

from __future__ import annotations

import torch
import torch.nn as nn

from .board_encoder import COMBINED_SIZE
from .concept_vocab import NUM_CONCEPTS, CONCEPTS


class ChessConceptClassifier(nn.Module):
    def __init__(
        self,
        input_size:   int   = COMBINED_SIZE,
        hidden1:      int   = 1536,
        hidden2:      int   = 768,
        num_concepts: int   = NUM_CONCEPTS,
        dropout:      float = 0.4,
        dropout2:     float = 0.2,
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
            nn.Dropout(dropout2),   # lower — L2 BN had 62.7% suppressed neurons

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
        import numpy as np
        from .board_encoder import fen_to_tensor, move_to_tensor
        from .evaluate      import load_thresholds
        from tools.label_positions import algo_feature_vector

        self.eval()
        device = next(self.parameters()).device

        if threshold is None:
            t_vec = load_thresholds()                        # [NUM_CONCEPTS]
        else:
            t_vec = torch.full((NUM_CONCEPTS,), threshold)

        board_t = fen_to_tensor(fen)
        move_t  = move_to_tensor("")   # no move context at inference time
        algo_t  = torch.from_numpy(algo_feature_vector(fen))
        x       = torch.cat([board_t, move_t, algo_t]).unsqueeze(0).to(device)
        logits = self.forward(x).squeeze(0).cpu()
        probs  = torch.sigmoid(logits)

        return sorted(
            [(CONCEPTS[i], probs[i].item())
             for i in range(NUM_CONCEPTS) if probs[i] >= t_vec[i]],
            key=lambda t: -t[1],
        )