# classifier.py
# GRU + MLP network: board features + move history → chess concept probabilities.
#
# Architecture (Phase 3)
# ---------------------------------
#   Static input  1188  (1001 board + 128 move + 59 algo bits)
#   GRU input      128  one-hot from-sq + to-sq per history step
#   Combined      1444  = 1188 + 256 GRU hidden
#   Layer1        1536  → BatchNorm → ReLU → Dropout(0.4)
#   Layer2         768  → BatchNorm → ReLU → Dropout(0.2)
#   Output          49  (raw logits)
#
# Architecture (Phase 4-B — phase4=True)
# ------------------------------------------------------------------
#   x layout     3013  [board(1001), move(128), algo_v4(1811), v3(59), sf(14)]
#   Spatial proj 1811 → 256  Linear → ReLU → Dropout(0.3) compresses algo_v4
#   v3 summary     59  bypasses bottleneck — direct actualized concept bits
#   sf features    14  Stockfish classical eval per side; bypasses bottleneck
#   GRU input     144  history_rich per-step (piece, capture, check, color)
#   Combined     1714  = 1001 + 128 + 256 proj + 59 v3 + 14 sf + 256 GRU hidden
#   Layer1       1024  → BatchNorm → ReLU → Dropout(0.4)
#   Layer2        512  → BatchNorm → ReLU → Dropout(0.2)
#   Output         49  (raw logits)
#
# Architecture (Phase 5D — phase5=True)
# ------------------------------------------------------------------
#   x layout     5061  [nnue(2048), board(1001), move(128), algo_v4(1811), sf(14), v3(59)]
#   NNUE proj   2048 → 256  Linear → ReLU → Dropout(0.3)  (SF evaluation signal)
#   Algo proj   1811 → 256  Linear → ReLU → Dropout(0.3)  (explicit concept features)
#   After proj   1714  [nnue_proj(256), board(1001), move(128), algo_proj(256), sf(14), v3(59)]
#   GRU input     144  history_rich per-step (same as Phase 4)
#   Combined     1970  = 1714 static + 256 GRU hidden
#   Layer1       1024  → BatchNorm → ReLU → Dropout(0.4)
#   Layer2        512  → BatchNorm → ReLU → Dropout(0.2)
#   Output         49  (raw logits)
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
    INPUT_SIZE, ALGO_SIZE, STATIC_SIZE_V4, SF_SIZE, SF_BREAK,
    NNUE_SIZE,
    NNUE_PROJ_SIZE, COMBINED_SIZE_V5D,
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
        phase4:       bool  = False,
        phase5:       bool  = False,    # True → Phase 5: frozen NNUE perception layer
    ) -> None:
        super().__init__()
        self._phase5 = phase5
        if phase5:
            input_size = COMBINED_SIZE_V5D  # 1970 = 1714 static + 256 GRU
            hidden1    = 1024
            hidden2    = 512
        elif phase4:
            input_size = COMBINED_SIZE_V4B  # 1714 = 1458 projected + 256 GRU
            hidden1    = 1024
            hidden2    = 512
        gru_in = MOVE_SIZE_V4 if (phase4 or phase5) else MOVE_SIZE
        # Phase 4 + Phase 5D: compress algo_v4(1811) → 256 before the head.
        self.spatial_proj = nn.Sequential(
            nn.Linear(ALGO_SIZE_V4, PROJ_SIZE_V4),
            nn.ReLU(),
            nn.Dropout(0.30),
        ) if phase4 or phase5 else None
        # Phase 5D: additionally compress NNUE(2048) → 256 (SF evaluation signal).
        # Runs alongside spatial_proj — both bottlenecks feed the head in parallel.
        self.nnue_proj = nn.Sequential(
            nn.Linear(NNUE_SIZE, NNUE_PROJ_SIZE),
            nn.ReLU(),
            nn.Dropout(0.30),
        ) if phase5 else None
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

        # Phase 5D: two parallel bottlenecks — nnue_proj(2048→256) + spatial_proj(1811→256)
        # x layout raw: [nnue(2048), board(1001), move(128), algo_v4(1811), sf(14), v3(59)]
        # Phase 4:  spatial_proj only on algo_v4(1811) → 256
        # Phase 3:  x passes through unchanged
        if self._phase5:
            _bm  = NNUE_SIZE + INPUT_SIZE + MOVE_SIZE              # board+move end = 3177
            _ae  = _bm + ALGO_SIZE_V4                              # algo end       = 4988
            nnue_proj  = self.nnue_proj(x[:, :NNUE_SIZE])         # (B, 256)
            board_move = x[:, NNUE_SIZE:_bm]                       # (B, 1129)
            algo_proj  = self.spatial_proj(x[:, _bm:_ae])         # (B, 256)
            sf_v3      = x[:, _ae:]                                # (B,  73) sf+v3
            x = torch.cat([nnue_proj, board_move, algo_proj, sf_v3], dim=1)  # (B, 1714)
        elif self.spatial_proj is not None:
            board_move = x[:, :INPUT_SIZE + MOVE_SIZE]                     # (B, 1129)
            spatial    = x[:, INPUT_SIZE + MOVE_SIZE:STATIC_SIZE_V4]       # (B, 1811)
            v3_summary = x[:, STATIC_SIZE_V4:SF_BREAK]                     # (B,   59)
            sf_t       = x[:, SF_BREAK:]                                    # (B,   14)
            x = torch.cat([board_move, self.spatial_proj(spatial), v3_summary, sf_t], dim=1)

        combined = torch.cat([x, gru_out], dim=1)
        return self.net(combined)

    @torch.no_grad()
    def predict_concepts(
        self,
        fen:            str,
        history_uci:    list[str] | None = None,
        history_rich:   list[dict] | None = None,
        threshold:      float | None     = None,
    ) -> list[tuple[str, float]]:
        """
        Return (concept_name, probability) pairs above threshold, sorted by prob.

        fen          : FEN of the position to analyse
        history_uci  : UCI move strings (Phase 3) — ignored when model is Phase 4
        history_rich : rich move dicts (Phase 4) — used when model has spatial_proj
        threshold    : None → load calibrated per-class thresholds from data/thresholds.json
                       float → use that value for every class
        """
        from .board_encoder import (
            fen_to_tensor, move_to_tensor,
            history_to_tensor, history_rich_to_tensor,
        )
        from .evaluate import load_thresholds
        from tools.label_positions import algo_feature_vector, algo_feature_vector_v4

        self.eval()
        device = next(self.parameters()).device

        if threshold is None:
            t_vec = load_thresholds()
        else:
            t_vec = torch.full((NUM_CONCEPTS,), threshold)

        move_t = move_to_tensor("")

        if self._phase5:
            from tools.nnue_reader import compute_activations, load_feature_transformer
            from .paths import NNUE_WEIGHTS as nnue_path
            if nnue_path.exists():
                biases, weights = load_feature_transformer(str(nnue_path))
                nnue_t = torch.from_numpy(compute_activations(fen, biases, weights))
            else:
                nnue_t = torch.zeros(NNUE_SIZE, dtype=torch.float32)
            board_t  = fen_to_tensor(fen)
            algo_v4  = torch.from_numpy(algo_feature_vector_v4(fen))
            v3_t     = torch.from_numpy(algo_feature_vector(fen))
            sf_t     = torch.zeros(SF_SIZE, dtype=torch.float32)
            x = torch.cat([nnue_t, board_t, move_t, algo_v4, sf_t, v3_t]).unsqueeze(0).to(device)
            hist_t, seq_len = history_rich_to_tensor(history_rich or [])
        elif self.spatial_proj is not None:
            board_t = fen_to_tensor(fen)
            algo_v4 = torch.from_numpy(algo_feature_vector_v4(fen))
            v3_t    = torch.from_numpy(algo_feature_vector(fen))
            sf_t    = torch.zeros(SF_SIZE, dtype=torch.float32)
            x = torch.cat([board_t, move_t, algo_v4, v3_t, sf_t]).unsqueeze(0).to(device)
            hist_t, seq_len = history_rich_to_tensor(history_rich or [])
        else:
            board_t = fen_to_tensor(fen)
            algo_t  = torch.from_numpy(algo_feature_vector(fen))
            x = torch.cat([board_t, move_t, algo_t]).unsqueeze(0).to(device)
            hist_t, seq_len = history_to_tensor(history_uci or [])

        hist_t    = hist_t.unsqueeze(0).to(device)
        seq_len_t = torch.tensor([seq_len])

        logits = self.forward(x, hist_t, seq_len_t).squeeze(0).cpu()
        probs  = torch.sigmoid(logits)

        return sorted(
            [(CONCEPTS[i], probs[i].item())
             for i in range(NUM_CONCEPTS) if probs[i] >= t_vec[i]],
            key=lambda t: -t[1],
        )
