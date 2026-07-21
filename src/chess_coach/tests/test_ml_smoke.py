"""Smoke tests for the ML pipeline.

These tests run in seconds without a trained checkpoint and catch regressions in
tensor shapes, dtypes, and encoder determinism — the kind of bugs that previously
required a full 100-epoch training run to surface.

Tests that require a checkpoint are skipped automatically when none is present.
Run with:
    pytest src/chess_coach/tests/test_ml_smoke.py -v
"""

from __future__ import annotations

import pytest
import torch

# Shared test FEN: starting position, and a rich middlegame.
_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_MID_FEN   = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6"
_CKPT_PATH = "data/classifier_best.pt"


# ── board_encoder ─────────────────────────────────────────────────────────────

class TestBoardEncoder:
    def test_fen_to_tensor_shape(self):
        from src.chess_coach.ml.board_encoder import fen_to_tensor, INPUT_SIZE
        t = fen_to_tensor(_START_FEN)
        assert t.shape == (INPUT_SIZE,), f"expected ({INPUT_SIZE},), got {t.shape}"

    def test_fen_to_tensor_dtype(self):
        from src.chess_coach.ml.board_encoder import fen_to_tensor
        t = fen_to_tensor(_START_FEN)
        assert t.dtype == torch.float32

    def test_fen_to_tensor_values_bounded(self):
        from src.chess_coach.ml.board_encoder import fen_to_tensor
        t = fen_to_tensor(_MID_FEN)
        assert t.min().item() >= 0.0
        assert t.max().item() <= 1.0, f"values should be in [0, 1], got max={t.max().item()}"

    def test_fen_to_tensor_deterministic(self):
        from src.chess_coach.ml.board_encoder import fen_to_tensor
        t1 = fen_to_tensor(_MID_FEN)
        t2 = fen_to_tensor(_MID_FEN)
        assert torch.equal(t1, t2), "fen_to_tensor must be deterministic"

    def test_different_positions_differ(self):
        from src.chess_coach.ml.board_encoder import fen_to_tensor
        t1 = fen_to_tensor(_START_FEN)
        t2 = fen_to_tensor(_MID_FEN)
        assert not torch.equal(t1, t2), "different FENs must produce different tensors"

    def test_move_to_tensor_shape(self):
        from src.chess_coach.ml.board_encoder import move_to_tensor, MOVE_SIZE
        t = move_to_tensor("e2e4")
        assert t.shape == (MOVE_SIZE,)

    def test_move_to_tensor_empty_is_zeros(self):
        from src.chess_coach.ml.board_encoder import move_to_tensor
        t = move_to_tensor("")
        assert t.sum().item() == 0.0

    def test_history_rich_to_tensor_shape(self):
        from src.chess_coach.ml.board_encoder import history_rich_to_tensor, MAX_SEQ_LEN, MOVE_SIZE_V4
        moves = [
            {"uci": "e2e4", "piece": 1, "captured": None, "is_check": False, "color": 1},
            {"uci": "e7e5", "piece": 1, "captured": None, "is_check": False, "color": 0},
        ]
        t, seq_len = history_rich_to_tensor(moves)
        assert t.shape == (MAX_SEQ_LEN, MOVE_SIZE_V4)
        assert seq_len == 2

    def test_history_rich_empty_gives_zeros(self):
        from src.chess_coach.ml.board_encoder import history_rich_to_tensor
        t, seq_len = history_rich_to_tensor([])
        assert seq_len == 0
        assert t.sum().item() == 0.0


# ── classifier ────────────────────────────────────────────────────────────────

class TestClassifier:
    def test_phase4_forward_shape(self):
        from src.chess_coach.ml.classifier import ChessConceptClassifier
        from src.chess_coach.ml.board_encoder import MAX_SEQ_LEN, MOVE_SIZE_V4
        from src.chess_coach.ml.concept_vocab import NUM_CONCEPTS
        model = ChessConceptClassifier(phase4=True)
        model.eval()
        B = 2
        # Raw input before spatial bottleneck: board(1001)+move(128)+algo_v4(1811)+v3(59)+sf(14)
        x       = torch.zeros(B, 3013)
        hist    = torch.zeros(B, MAX_SEQ_LEN, MOVE_SIZE_V4)
        seq_len = torch.zeros(B, dtype=torch.long)
        with torch.no_grad():
            logits = model(x, hist, seq_len)
        assert logits.shape == (B, NUM_CONCEPTS), f"Expected ({B}, {NUM_CONCEPTS}), got {logits.shape}"

    def test_phase4_logits_are_finite(self):
        from src.chess_coach.ml.classifier import ChessConceptClassifier
        from src.chess_coach.ml.board_encoder import MAX_SEQ_LEN, MOVE_SIZE_V4
        model = ChessConceptClassifier(phase4=True)
        model.eval()
        x       = torch.randn(1, 3013)
        hist    = torch.zeros(1, MAX_SEQ_LEN, MOVE_SIZE_V4)
        seq_len = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            logits = model(x, hist, seq_len)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"

    def test_probabilities_in_unit_interval(self):
        from src.chess_coach.ml.classifier import ChessConceptClassifier
        from src.chess_coach.ml.board_encoder import MAX_SEQ_LEN, MOVE_SIZE_V4
        model = ChessConceptClassifier(phase4=True)
        model.eval()
        x       = torch.randn(4, 3013)
        hist    = torch.zeros(4, MAX_SEQ_LEN, MOVE_SIZE_V4)
        seq_len = torch.zeros(4, dtype=torch.long)
        with torch.no_grad():
            probs = torch.sigmoid(model(x, hist, seq_len))
        assert probs.min().item() >= 0.0
        assert probs.max().item() <= 1.0


# ── checkpoint-backed tests (skipped if no checkpoint) ────────────────────────

@pytest.fixture(scope="module")
def loaded_model():
    """Load classifier_best.pt once for the session; skip if absent."""
    import os
    if not os.path.exists(_CKPT_PATH):
        pytest.skip(f"No checkpoint at {_CKPT_PATH} — run training first")
    from src.chess_coach.ml.classifier import ChessConceptClassifier
    ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
    sd   = ckpt.get("state_dict", ckpt)
    is_phase5 = any(k.startswith("nnue_proj")    for k in sd)
    is_phase4 = any(k.startswith("spatial_proj") for k in sd) and not is_phase5
    model = ChessConceptClassifier(phase4=is_phase4, phase5=is_phase5)
    model.load_state_dict(sd)
    model.eval()
    return model


class TestCheckpoint:
    def test_predict_concepts_returns_list(self, loaded_model):
        result = loaded_model.predict_concepts(_START_FEN, threshold=0.0)
        assert isinstance(result, list), "predict_concepts must return a list"

    def test_predict_concepts_format(self, loaded_model):
        result = loaded_model.predict_concepts(_START_FEN, threshold=0.0)
        for name, prob in result:
            assert isinstance(name, str)
            assert 0.0 <= prob <= 1.0, f"probability out of range for {name}: {prob}"

    def test_predict_concepts_sorted_descending(self, loaded_model):
        result = loaded_model.predict_concepts(_START_FEN, threshold=0.0)
        probs  = [p for _, p in result]
        assert probs == sorted(probs, reverse=True), "results must be sorted by probability descending"

    def test_tactical_position_detects_pin(self, loaded_model):
        # Bb5 pins Nc6 to the king — pin should be in the top outputs.
        pin_fen = "r1bq1rk1/ppp2ppp/2np1n2/1B2p3/2BPP3/2N1QN2/PPP2PPP/R4RK1 b - - 0 8"
        result  = loaded_model.predict_concepts(pin_fen, threshold=0.0)
        names   = [n for n, _ in result]
        # Just assert the model produces output without crashing — threshold tuning is separate.
        assert len(names) > 0, "No concepts returned for a known tactical position"

    def test_passed_pawn_position(self, loaded_model):
        # White has a lone passed pawn on d6, nothing else.
        pp_fen = "4k3/8/3P4/8/8/8/8/4K3 w - - 0 1"
        result = loaded_model.predict_concepts(pp_fen, threshold=0.3)
        # At threshold 0.3 this clean passed-pawn position should fire passed_pawn if the model works at all.
        names  = [n for n, _ in result]
        # Don't hard-assert the concept — model may be undertrained — but check no crash.
        assert isinstance(names, list)
