# classifier.py
# GRU + MLP network: board features + move history → chess concept probabilities.
#
# Architecture (Phase 3 — current)
# ---------------------------------
#   Static input  1188  (1001 board + 128 move + 59 algo bits)
#   GRU input      128  one-hot from-sq + to-sq per history step
#   Combined      1444  = 1188 + 256 GRU hidden
#   Layer1        1536  → BatchNorm → ReLU → Dropout(0.4)
#   Layer2         768  → BatchNorm → ReLU → Dropout(0.2)
#   Output          50  (raw logits)
#
# Architecture (Phase 4-B — activated by passing phase4=True to __init__)
# ------------------------------------------------------------------
#   Raw static    2851  (1001 board + 128 move + 1663 spatial + 59 v3 summary)
#   Spatial proj  1663 → 256  Linear → ReLU → Dropout(0.3) inside model
#   v3 summary      59  bypasses bottleneck — direct actualized concept bits
#   GRU input      144  history_rich per-step (piece, capture, check, color)
#   Combined      1700  = 1001 + 128 + 256 proj + 59 v3 + 256 GRU hidden
#   Layer1        1024  → BatchNorm → ReLU → Dropout(0.4)
#   Layer2         512  → BatchNorm → ReLU → Dropout(0.2)
#   Output          53  (raw logits)
#
# Puzzles (no game history) receive seq_len=0 → GRU output zeroed out.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .board_encoder import (
    COMBINED_SIZE, MOVE_SIZE, GRU_HIDDEN, MAX_SEQ_LEN,
    COMBINED_SIZE_V4, MOVE_SIZE_V4,
    ALGO_SIZE_V4, PROJ_SIZE_V4, COMBINED_SIZE_V4B,
    INPUT_SIZE, ALGO_SIZE, STATIC_SIZE_V4,
)
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
        phase4:       bool  = False,    # True → Phase 4 sizes (COMBINED_SIZE_V4, MOVE_SIZE_V4)
    ) -> None:
        super().__init__()
        if phase4:
            input_size = COMBINED_SIZE_V4B   # 1700 = 1129 board+move + 256 proj + 59 v3 + 256 GRU
            hidden1    = 1024
            hidden2    = 512
        gru_in = MOVE_SIZE_V4 if phase4 else MOVE_SIZE
        self.spatial_proj = nn.Sequential(
            nn.Linear(ALGO_SIZE_V4, PROJ_SIZE_V4),
            nn.ReLU(),
            nn.Dropout(0.30),
        ) if phase4 else None
        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=GRU_HIDDEN,
            num_layers=1,
            batch_first=True,
        )
        self.gru_dropout = nn.Dropout(0.3)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),

            nn.Linear(hidden2, num_concepts),
            # No activation — BCEWithLogitsLoss includes sigmoid internally
        )

    def forward(
        self,
        x:       torch.Tensor,   # (B, 1188) static board features
        hist:    torch.Tensor,   # (B, MAX_SEQ_LEN, 128) padded move history
        seq_len: torch.Tensor,   # (B,) actual history lengths
    ) -> torch.Tensor:
        # Run GRU over padded move history
        packed    = pack_padded_sequence(
            hist, seq_len.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)           # hidden: (1, B, GRU_HIDDEN)
        gru_out   = hidden.squeeze(0)          # (B, GRU_HIDDEN)
        gru_out   = self.gru_dropout(gru_out)  # regularise history branch

        # Mask out GRU output for puzzle examples with no history (seq_len == 0)
        no_hist = (seq_len == 0).float().unsqueeze(1).to(gru_out.device)
        gru_out = gru_out * (1.0 - no_hist)

        # Phase 4-B: compress 1663-dim spatial features to 256; v3 summary bypasses bottleneck
        # x layout: [board(1001), move(128), spatial(1663), v3_summary(59)] = 2851
        # v3_summary encodes actualized concepts (piece IS on outpost, backward pawn EXISTS)
        # which directly match what labels measure — spatial maps only encode structural potential
        if self.spatial_proj is not None:
            board_move = x[:, :INPUT_SIZE + MOVE_SIZE]            # (B, 1129)
            spatial    = x[:, INPUT_SIZE + MOVE_SIZE:STATIC_SIZE_V4]  # (B, 1663)
            v3_summary = x[:, STATIC_SIZE_V4:]                    # (B,  59)
            x = torch.cat([board_move, self.spatial_proj(spatial), v3_summary], dim=1)  # (B, 1444)

        combined = torch.cat([x, gru_out], dim=1)   # (B, 1700) phase4 or (B, 1444) phase3
        return self.net(combined)

    @torch.no_grad()
    def predict_concepts(
        self,
        fen:            str,
        history_uci:    list[str] | None = None,
        threshold:      float | None     = None,
    ) -> list[tuple[str, float]]:
        """
        Return (concept_name, probability) pairs above threshold, sorted by prob.

        fen          : FEN of the position to analyse
        history_uci  : list of UCI move strings leading to this position (empty = no history)
        threshold    : None → load calibrated per-class thresholds from data/thresholds.json
                       float → use that value for every class
        """
        from .board_encoder import fen_to_tensor, move_to_tensor, history_to_tensor
        from .evaluate      import load_thresholds
        from tools.label_positions import algo_feature_vector

        self.eval()
        device = next(self.parameters()).device

        if threshold is None:
            t_vec = load_thresholds()
        else:
            t_vec = torch.full((NUM_CONCEPTS,), threshold)

        board_t = fen_to_tensor(fen)
        move_t  = move_to_tensor("")
        algo_t  = torch.from_numpy(algo_feature_vector(fen))
        x       = torch.cat([board_t, move_t, algo_t]).unsqueeze(0).to(device)

        hist_t, seq_len = history_to_tensor(history_uci or [])
        hist_t   = hist_t.unsqueeze(0).to(device)
        seq_len_t = torch.tensor([seq_len])

        logits = self.forward(x, hist_t, seq_len_t).squeeze(0).cpu()
        probs  = torch.sigmoid(logits)

        return sorted(
            [(CONCEPTS[i], probs[i].item())
             for i in range(NUM_CONCEPTS) if probs[i] >= t_vec[i]],
            key=lambda t: -t[1],
        )
