# classifier.py
# GRU + MLP network: board features + move history → chess concept probabilities.
#
# Architecture (Phase 3)
# ---------------------------------
#   Static input  1197  (1001 board + 128 move + 68 algo bits)
#   GRU input      128  one-hot from-sq + to-sq per history step
#   Combined      1453  = 1197 + 256 GRU hidden
#   Layer1        1536  → BatchNorm → ReLU → Dropout(0.4)
#   Layer2         768  → BatchNorm → ReLU → Dropout(0.2)
#   Output          49  (raw logits)
#
# Architecture (Phase 4-B — phase4=True)
# ------------------------------------------------------------------
#   x layout     3702  [board(1001), move(128), algo_v4(2491), v3(68), sf(14)]
#   Spatial proj 2491 → 256  Linear → ReLU → Dropout(0.3) compresses algo_v4
#   v3 summary     68  bypasses bottleneck — direct actualized concept bits
#   sf features    14  Stockfish classical eval per side; bypasses bottleneck
#   GRU input     144  history_rich per-step (piece, capture, check, color)
#   Combined     1723  = 1001 + 128 + 256 proj + 68 v3 + 14 sf + 256 GRU hidden
#   Layer1       1024  → BatchNorm → ReLU → Dropout(0.4)
#   Layer2        512  → BatchNorm → ReLU → Dropout(0.2)
#   Output         49  (raw logits)
#
# Architecture (Phase 5D — phase5=True)
# ------------------------------------------------------------------
#   x layout     5750  [nnue(2048), board(1001), move(128), algo_v4(2491), sf(14), v3(68)]
#   NNUE proj   2048 → 256  Linear → ReLU → Dropout(0.3)  (SF evaluation signal)
#   Algo proj   2491 → 256  Linear → ReLU → Dropout(0.3)  (explicit concept features)
#   After proj   1723  [nnue_proj(256), board(1001), move(128), algo_proj(256), sf(14), v3(68)]
#   GRU input     144  history_rich per-step (same as Phase 4)
#   Combined     1979  = 1723 static + 256 GRU hidden
#   Layer1       1024  → BatchNorm → ReLU → Dropout(0.4)
#   Layer2        512  → BatchNorm → ReLU → Dropout(0.2)
#   Output         49  (raw logits)
#
# Puzzles (no game history) receive seq_len=0 → GRU output zeroed out.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

import torch.nn.functional as F

from .board_encoder import (
    COMBINED_SIZE, MOVE_SIZE, GRU_HIDDEN, MAX_SEQ_LEN,
    COMBINED_SIZE_V4, MOVE_SIZE_V4,
    ALGO_SIZE_V4, PROJ_SIZE_V4, COMBINED_SIZE_V4B,
    INPUT_SIZE, ALGO_SIZE, STATIC_SIZE_V4, SF_SIZE, SF_BREAK,
    NNUE_SIZE,
    NNUE_PROJ_SIZE, COMBINED_SIZE_V5D,
    TB_SIZE, COMBINED_SIZE_V6, ECO_CLASSES, ECO_DIM, eco_to_idx,
)
from .concept_vocab import NUM_CONCEPTS, CONCEPTS

# ── Phase 6B expert concept partitioning ──────────────────────────────────────
# Indices into the 49-concept CONCEPTS list.
_EXPERT_SIZES   = [15, 8, 9, 11, 6]    # Tactical, Structural, Pawn, Endgame, Strategic
_ENDGAME_EXPERT = 3                     # index of the endgame expert head
_LOAD_BALANCE_W = 0.01                  # weight for load-balance auxiliary loss


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
            input_size = COMBINED_SIZE_V4B  # 1723 = 1467 projected + 256 GRU
            hidden1    = 1024
            hidden2    = 512
        gru_in = MOVE_SIZE_V4 if (phase4 or phase5) else MOVE_SIZE
        # Phase 4 + Phase 5D: compress algo_v4(2491) → 256 before the head.
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

        # Phase 5D: two parallel bottlenecks — nnue_proj(2048→256) + spatial_proj(2491→256)
        # x layout raw: [nnue(2048), board(1001), move(128), algo_v4(2491), sf(14), v3(68)]
        # Phase 4:  spatial_proj only on algo_v4(2491) → 256
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
            spatial    = x[:, INPUT_SIZE + MOVE_SIZE:STATIC_SIZE_V4]       # (B, 2491)
            v3_summary = x[:, STATIC_SIZE_V4:SF_BREAK]                     # (B,   68)
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
        eco_code:       str | None       = None,   # ignored for Phase 3/4/5; used by MoE
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


# ── Phase 6B: Mixture of Experts ──────────────────────────────────────────────

class GatingNetwork(nn.Module):
    """
    Learns which expert to trust per position.

    Input:  combined post-GRU vector (COMBINED_SIZE_V6) + ECO embedding (ECO_DIM)
    Output: raw gate logits (B, 5) — softmax applied in MoEConceptClassifier.forward()
            after tablebase conditioning so the prior is applied before normalisation.
    """

    def __init__(
        self,
        input_dim:   int = COMBINED_SIZE_V6,
        n_experts:   int = 5,
        eco_classes: int = ECO_CLASSES,
        eco_dim:     int = ECO_DIM,
    ) -> None:
        super().__init__()
        self.eco_proj = nn.Embedding(eco_classes, eco_dim)
        self.gate     = nn.Sequential(
            nn.Linear(input_dim + eco_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_experts),
        )

    def forward(self, combined: torch.Tensor, eco_idx: torch.Tensor) -> torch.Tensor:
        eco_emb  = self.eco_proj(eco_idx)               # (B, ECO_DIM)
        gate_in  = torch.cat([combined, eco_emb], dim=1)  # (B, input_dim + ECO_DIM)
        return self.gate(gate_in)                        # (B, 5) raw logits


class ExpertHead(nn.Module):
    """
    Lightweight specialist MLP for one concept domain.

    Input:  combined post-GRU vector (COMBINED_SIZE_V6)
    Output: raw logits for the N concepts in this expert's domain
    """

    def __init__(self, input_dim: int, n_concepts: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_concepts),
        )

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        return self.net(combined)


class MoEConceptClassifier(nn.Module):
    """
    Phase 6B: Mixture of Experts classifier with ECO + tablebase gating.

    Architecture
    ------------
    Shared encoder (from Phase 4C):
        spatial_proj  : algo_v4(3779) → 256
        gru           : history_rich(144) → 256 hidden
    Five expert heads (Tactical / Structural / Pawn / Endgame / Strategic):
        each : combined(1740) → 512 → BN → ReLU → 256 → BN → ReLU → N_concepts
    Gating network:
        eco_proj      : ECO_IDX → ECO_DIM(64)
        gate          : combined(1740) + eco(64) → 256 → ReLU → 5 logits
    Tablebase conditioning:
        gate_logits[:, ENDGAME_EXPERT] += 2.0 * is_tb   (applied before softmax)

    Input x layout (raw, 5007-dim):
        [board(1001), move(128), algo_v4(3779), v3(82), sf(14), tb(3)]

    Returns: (logits [B, 49], gate_weights [B, 5])
    """

    def __init__(self) -> None:
        super().__init__()
        gru_in = MOVE_SIZE_V4   # 144

        # Shared encoder — same topology as Phase 4C
        self.spatial_proj = nn.Sequential(
            nn.Linear(ALGO_SIZE_V4, PROJ_SIZE_V4),
            nn.ReLU(),
            nn.Dropout(0.30),
        )
        self.gru = nn.GRU(
            input_size  = gru_in,
            hidden_size = GRU_HIDDEN,
            num_layers  = 1,
            batch_first = True,
        )
        self.gru_dropout = nn.Dropout(0.3)

        combined_dim = COMBINED_SIZE_V6  # 1740

        # Gating network
        self.gate_network = GatingNetwork(input_dim=combined_dim)

        # Expert heads
        self.experts = nn.ModuleList([
            ExpertHead(combined_dim, n) for n in _EXPERT_SIZES
        ])

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:       torch.Tensor,              # (B, 5007)
        hist:    torch.Tensor,              # (B, MAX_SEQ_LEN, 144)
        seq_len: torch.Tensor,              # (B,)
        eco_idx: torch.Tensor | None = None,  # (B,) long — ECO class index
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits [B, 49], gate_weights [B, 5])."""
        # GRU over move history
        packed    = pack_padded_sequence(
            hist, seq_len.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)          # (1, B, GRU_HIDDEN)
        gru_out   = hidden.squeeze(0)         # (B, 256)
        gru_out   = self.gru_dropout(gru_out)
        no_hist   = (seq_len == 0).float().unsqueeze(1).to(gru_out.device)
        gru_out   = gru_out * (1.0 - no_hist)

        # Spatial projection: algo_v4 (3779) → proj (256)
        # x layout: [board(1001), move(128), algo_v4(3779), v3(82), sf(14), tb(3)]
        board_move = x[:, :INPUT_SIZE + MOVE_SIZE]          # (B, 1129)
        algo       = x[:, INPUT_SIZE + MOVE_SIZE:STATIC_SIZE_V4]  # (B, 3779)
        v3_sf_tb   = x[:, STATIC_SIZE_V4:]                  # (B, 82+14+3=99)
        algo_proj  = self.spatial_proj(algo)                 # (B, 256)
        static     = torch.cat([board_move, algo_proj, v3_sf_tb], dim=1)  # (B, 1484)

        combined = torch.cat([static, gru_out], dim=1)      # (B, 1740)

        # Gate: raw logits + TB endgame prior
        if eco_idx is None:
            eco_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        gate_logits = self.gate_network(combined, eco_idx)   # (B, 5)

        is_tb   = x[:, -1]                                   # (B,) — tb[2]=is_tb
        tb_boost = torch.zeros_like(gate_logits)
        tb_boost[:, _ENDGAME_EXPERT] = is_tb * 2.0
        gate_weights = F.softmax(gate_logits + tb_boost, dim=-1)  # (B, 5)

        # Expert outputs assembled into 49-dim logit vector
        logits = torch.zeros(x.shape[0], NUM_CONCEPTS, device=x.device)
        offset = 0
        for j, (expert, n) in enumerate(zip(self.experts, _EXPERT_SIZES)):
            expert_out = expert(combined)                     # (B, n)
            w          = gate_weights[:, j:j+1]              # (B, 1)
            logits[:, offset:offset + n] += w * expert_out
            offset += n

        return logits, gate_weights

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_concepts(
        self,
        fen:          str,
        history_uci:  list[str] | None  = None,
        history_rich: list[dict] | None = None,
        threshold:    float | None      = None,
        eco_code:     str | None        = None,
    ) -> list[tuple[str, float]]:
        from .board_encoder import fen_to_tensor, move_to_tensor, history_rich_to_tensor
        from .evaluate import load_thresholds
        from tools.label_positions import algo_feature_vector, algo_feature_vector_v4

        self.eval()
        device = next(self.parameters()).device

        t_vec = load_thresholds() if threshold is None else torch.full((NUM_CONCEPTS,), threshold)

        board_t  = fen_to_tensor(fen)
        move_t   = move_to_tensor("")
        algo_v4  = torch.from_numpy(algo_feature_vector_v4(fen))
        v3_t     = torch.from_numpy(algo_feature_vector(fen))
        sf_t     = torch.zeros(SF_SIZE,  dtype=torch.float32)
        tb_t     = torch.zeros(TB_SIZE,  dtype=torch.float32)   # zeros at inference
        x        = torch.cat([board_t, move_t, algo_v4, v3_t, sf_t, tb_t]).unsqueeze(0).to(device)

        hist_t, seq_len = history_rich_to_tensor(history_rich or [])
        hist_t          = hist_t.unsqueeze(0).to(device)
        seq_len_t       = torch.tensor([seq_len])
        eco_idx_t       = torch.tensor([eco_to_idx(eco_code)], dtype=torch.long, device=device)

        logits, _ = self.forward(x, hist_t, seq_len_t, eco_idx=eco_idx_t)
        probs     = torch.sigmoid(logits.squeeze(0)).cpu()

        return sorted(
            [(CONCEPTS[i], probs[i].item()) for i in range(NUM_CONCEPTS) if probs[i] >= t_vec[i]],
            key=lambda t: -t[1],
        )


def load_balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """Penalise gate collapse: encourage even expert utilisation across the batch."""
    mean_gate = gate_weights.mean(dim=0)                        # (5,)
    target    = torch.ones_like(mean_gate) / gate_weights.shape[1]
    return F.mse_loss(mean_gate, target)
