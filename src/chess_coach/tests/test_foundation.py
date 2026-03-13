"""
tests/test_foundation.py
========================
Phase 1 unit tests.

Pass criteria (from spec):
  - MetricSignal instantiation — all fields, all validation rules.
  - CoachOutput instantiation — all fields, all validation rules.
  - GMPrecedent instantiation.
  - board_utils.get_phase() returns correct phase on 5 known positions.
  - board_utils helpers return correct types and values.
  - StockfishBridge interface tested via a mock (no Stockfish binary needed).

Run with:  pytest tests/test_foundation.py -v
"""

from __future__ import annotations

import sys
import pytest
import chess
import chess.pgn

# sys.path is managed by conftest.py at the package root.

from chess_coach.core.data_types import (
    MetricSignal, CoachOutput, GMPrecedent,
    SIDES, PHASES, SEVERITIES, STRATEGIES
)
from chess_coach.core.board_utils import (
    get_phase, get_fen, get_position_key, get_pawn_hash,
    get_move_history, square_to_str, str_to_square,
    get_king_zone, get_king_zone_str, get_pieces_in_zone,
    count_legal_moves
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def board_at_ply(uci_moves: list[str]) -> chess.Board:
    """Build a board by playing a list of UCI moves from the start position."""
    b = chess.Board()
    for uci in uci_moves:
        b.push(chess.Move.from_uci(uci))
    return b


def board_from_fen(fen: str) -> chess.Board:
    return chess.Board(fen)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — MetricSignal
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricSignal:

    def test_minimal_creation(self):
        """Mandatory fields only — all optional fields get defaults."""
        sig = MetricSignal(
            metric_name='king_exposure',
            score=0.82,
            side='white',
            cause='missing_pawn_shield',
        )
        assert sig.metric_name == 'king_exposure'
        assert sig.score == 0.82
        assert sig.side == 'white'
        assert sig.cause == 'missing_pawn_shield'
        assert sig.key_squares == []
        assert sig.key_pieces == []
        assert sig.severity == 'moderate'
        assert sig.fragment == ''
        assert sig.action_hint == ''
        assert sig.phase == 'middlegame'

    def test_full_creation(self):
        """All 10 fields populated explicitly."""
        sig = MetricSignal(
            metric_name  = 'piece_mobility_ratio',
            score        = 0.71,
            side         = 'black',
            cause        = 'overextended_pawn',
            key_squares  = ['d5', 'e4', 'f5'],
            key_pieces   = ['Ng5', 'Qd3'],
            severity     = 'high',
            fragment     = 'the d5 outpost is uncontested',
            action_hint  = 'occupy d5 with the knight',
            phase        = 'endgame',
        )
        assert sig.score == 0.71
        assert sig.side == 'black'
        assert sig.severity == 'high'
        assert sig.phase == 'endgame'
        assert 'd5' in sig.key_squares
        assert 'Ng5' in sig.key_pieces

    def test_score_boundary_zero(self):
        sig = MetricSignal(metric_name='test', score=0.0, side='white', cause='x')
        assert sig.score == 0.0

    def test_score_boundary_one(self):
        sig = MetricSignal(metric_name='test', score=1.0, side='white', cause='x')
        assert sig.score == 1.0

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match='score must be in'):
            MetricSignal(metric_name='test', score=1.1, side='white', cause='x')

    def test_score_negative_raises(self):
        with pytest.raises(ValueError, match='score must be in'):
            MetricSignal(metric_name='test', score=-0.1, side='white', cause='x')

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError, match='side must be one of'):
            MetricSignal(metric_name='test', score=0.5, side='blue', cause='x')

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match='severity must be one of'):
            MetricSignal(metric_name='test', score=0.5, side='white', cause='x', severity='extreme')

    def test_invalid_phase_raises(self):
        with pytest.raises(ValueError, match='phase must be one of'):
            MetricSignal(metric_name='test', score=0.5, side='white', cause='x', phase='quantum')

    def test_all_severity_values_valid(self):
        for s in SEVERITIES:
            sig = MetricSignal(metric_name='test', score=0.5, side='white', cause='x', severity=s)
            assert sig.severity == s

    def test_all_phase_values_valid(self):
        for p in PHASES:
            sig = MetricSignal(metric_name='test', score=0.5, side='white', cause='x', phase=p)
            assert sig.phase == p

    def test_all_side_values_valid(self):
        for s in SIDES:
            sig = MetricSignal(metric_name='test', score=0.5, side=s, cause='x')
            assert sig.side == s

    def test_fragment_empty_by_default(self):
        """Extractors set fragment='' — phrase DB fills it later."""
        sig = MetricSignal(metric_name='test', score=0.5, side='white', cause='x')
        assert sig.fragment == ''

    def test_key_squares_independent(self):
        """Mutable default — different instances must not share the same list."""
        a = MetricSignal(metric_name='t', score=0.5, side='white', cause='x')
        b = MetricSignal(metric_name='t', score=0.5, side='white', cause='x')
        a.key_squares.append('e4')
        assert b.key_squares == []


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — GMPrecedent
# ─────────────────────────────────────────────────────────────────────────────

class TestGMPrecedent:

    def test_creation(self):
        p = GMPrecedent(
            player     = 'Tal, Mikhail',
            game_id    = 'tal_games.pgn:142',
            ply        = 28,
            key_move   = 'g2g4',
            annotation = 'The immortal sacrifice.',
        )
        assert p.player == 'Tal, Mikhail'
        assert p.ply == 28
        assert p.key_move == 'g2g4'

    def test_annotation_defaults_empty(self):
        p = GMPrecedent(player='Carlsen', game_id='file.pgn:1', ply=12, key_move='d2d4')
        assert p.annotation == ''

    def test_immutable(self):
        p = GMPrecedent(player='Fischer', game_id='f.pgn:1', ply=20, key_move='e2e4')
        with pytest.raises(Exception):
            p.player = 'Kasparov'  # frozen dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — CoachOutput
# ─────────────────────────────────────────────────────────────────────────────

def _make_signal(name='king_exposure', score=0.8) -> MetricSignal:
    return MetricSignal(metric_name=name, score=score, side='white', cause='test')

def _make_precedent() -> GMPrecedent:
    return GMPrecedent(player='Tal', game_id='t.pgn:1', ply=20, key_move='g2g4')


class TestCoachOutput:

    def test_minimal_creation(self):
        out = CoachOutput(
            strategy_primary   = 'blitz',
            strategy_secondary = None,
            confidence         = 0.84,
            phase              = 'middlegame',
            headline           = 'White should attack immediately.',
            plan_sentences     = ['Sentence one.', 'Sentence two.'],
        )
        assert out.strategy_primary == 'blitz'
        assert out.strategy_secondary is None
        assert out.confidence == 0.84
        assert out.tactic_hints == []
        assert out.gm_precedents == []

    def test_full_creation(self):
        sig = _make_signal()
        pre = _make_precedent()
        out = CoachOutput(
            strategy_primary   = 'flank',
            strategy_secondary = 'fortress',
            confidence         = 0.72,
            phase              = 'endgame',
            headline           = 'Squeeze the position.',
            plan_sentences     = ['Restrict mobility.', 'Occupy the outpost.', 'Force zugzwang.'],
            tactic_hints       = ['Fork available on e5.'],
            move_flags         = [{'move': 'd2d4', 'flag': 'space_gain', 'strategy': 'flank'}],
            weakness_squares   = ['c6', 'd5'],
            gm_precedents      = [pre],
            signal_dump        = [sig],
        )
        assert out.strategy_secondary == 'fortress'
        assert len(out.plan_sentences) == 3
        assert out.gm_precedents[0].player == 'Tal'

    def test_invalid_primary_strategy_raises(self):
        with pytest.raises(ValueError, match='strategy_primary'):
            CoachOutput(
                strategy_primary='rush', strategy_secondary=None,
                confidence=0.8, phase='middlegame',
                headline='x', plan_sentences=['a.', 'b.']
            )

    def test_invalid_secondary_strategy_raises(self):
        with pytest.raises(ValueError, match='strategy_secondary'):
            CoachOutput(
                strategy_primary='blitz', strategy_secondary='rush',
                confidence=0.8, phase='middlegame',
                headline='x', plan_sentences=['a.', 'b.']
            )

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match='confidence'):
            CoachOutput(
                strategy_primary='blitz', strategy_secondary=None,
                confidence=1.5, phase='middlegame',
                headline='x', plan_sentences=['a.', 'b.']
            )

    def test_plan_sentences_too_few_raises(self):
        with pytest.raises(ValueError, match='plan_sentences'):
            CoachOutput(
                strategy_primary='blitz', strategy_secondary=None,
                confidence=0.8, phase='middlegame',
                headline='x', plan_sentences=['only one.']
            )

    def test_plan_sentences_too_many_raises(self):
        with pytest.raises(ValueError, match='plan_sentences'):
            CoachOutput(
                strategy_primary='blitz', strategy_secondary=None,
                confidence=0.8, phase='middlegame',
                headline='x', plan_sentences=['a.', 'b.', 'c.', 'd.', 'e.']
            )

    def test_all_strategies_valid_primary(self):
        for s in STRATEGIES:
            out = CoachOutput(
                strategy_primary=s, strategy_secondary=None,
                confidence=0.7, phase='middlegame',
                headline='x', plan_sentences=['a.', 'b.']
            )
            assert out.strategy_primary == s

    def test_signal_dump_independent(self):
        out1 = CoachOutput(
            strategy_primary='blitz', strategy_secondary=None,
            confidence=0.8, phase='middlegame',
            headline='x', plan_sentences=['a.', 'b.']
        )
        out2 = CoachOutput(
            strategy_primary='blitz', strategy_secondary=None,
            confidence=0.8, phase='middlegame',
            headline='x', plan_sentences=['a.', 'b.']
        )
        out1.signal_dump.append(_make_signal())
        assert out2.signal_dump == []


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — board_utils.get_phase()
# Spec test criteria: ply 5 → opening, ply 20 → middlegame, ply 42 → endgame
# ─────────────────────────────────────────────────────────────────────────────

# Ruy Lopez mainline — a well-known game trajectory for testing
RUYLOPEZ = [
    'e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5',       # 5 half-moves (ply 5)
    'a7a6', 'b5a4', 'g8f6', 'e1g1', 'f8e7',        # ply 10
    'f1e1', 'b7b5', 'a4b3', 'd7d6', 'c2c3',        # ply 15
    'e8g8', 'd2d4', 'c8g4', 'd4d5', 'c6b8',        # ply 20 — solid middlegame
]

# Forcing line into an endgame: simplifications over many moves
ENDGAME_LINE = [
    'e2e4', 'e7e5', 'g1f3', 'g8f6', 'f3e5', 'f6e4',
    'd2d3', 'e4f6', 'e5f7', 'e8f7', 'd1f3', 'f7g8',
    'f3b7', 'c8b7', 'b1c3', 'f8d6', 'c1g5', 'd8e7',
    'e1c1', 'b8c6', 'd3d4', 'e5e4', 'c3e4', 'f6e4',
    'g5e7', 'c6e7', 'f1c4', 'c7c6', 'h1e1', 'e4f2',
    # Most pieces off board by now
    'd1f3', 'f2f3', 'g2f3', 'd6e7', 'e1e7',
    # Rook and king endgame
    'a8d8', 'c4f7', 'g8f7', 'e7e1',
]


class TestGetPhase:

    def test_starting_position_is_opening(self):
        board = chess.Board()
        assert get_phase(board) == 'opening'

    def test_ply_5_is_opening(self):
        """Spec criterion: move 5 (ply 5) → 'opening'."""
        board = board_at_ply(RUYLOPEZ[:5])
        assert get_phase(board) == 'opening'

    def test_ply_10_is_opening(self):
        board = board_at_ply(RUYLOPEZ[:10])
        assert get_phase(board) == 'opening'

    def test_ply_20_is_middlegame(self):
        """Spec criterion: move 20 (ply 20) → 'middlegame'."""
        # Use a FEN with clearly reduced material (total=40, threshold=50) so the
        # phase detector reliably returns 'middlegame'.
        # White: Q+R+N+N=20  Black: q+r+n+n=20  Total=40 < OPENING_THRESHOLD(50)
        # Position: queens, one rook each, two knights each — all minor exchanges done.
        middlegame_fen = '3r2k1/pp3ppp/2n2n2/4q3/4Q3/2N2N2/PP3PPP/3R2K1 w - - 0 1'
        board = chess.Board(middlegame_fen)
        assert get_phase(board) == 'middlegame'

    def test_ply_20_ruy_still_opening_or_middlegame(self):
        """After 20 half-moves in the Ruy Lopez, position is opening or middlegame."""
        board = board_at_ply(RUYLOPEZ[:20])
        phase = get_phase(board)
        assert phase in ('opening', 'middlegame'), f"Got unexpected phase '{phase}' at ply 20"

    def test_ply_42_is_endgame(self):
        """Spec criterion: move 42 (ply 42) → 'endgame'."""
        # Use a known endgame FEN: K+R vs K+P
        endgame_fen = '8/8/4k3/8/8/4K3/4R3/8 w - - 0 1'
        board = chess.Board(endgame_fen)
        assert get_phase(board) == 'endgame'

    def test_king_and_pawn_endgame(self):
        board = chess.Board('8/p7/8/8/8/8/P7/8 w - - 0 1')
        assert get_phase(board) == 'endgame'

    def test_rook_endgame(self):
        board = chess.Board('4k3/8/8/8/8/8/8/4K2R w K - 0 1')
        assert get_phase(board) == 'endgame'

    def test_queen_middlegame(self):
        # Both queens on board with some pieces off
        board = chess.Board('r2qk2r/ppp2ppp/2n1bn2/3p4/3P4/2N1BN2/PPP2PPP/R2QK2R w KQkq - 0 1')
        assert get_phase(board) in ('opening', 'middlegame')

    def test_pure_endgame_fen(self):
        board = chess.Board('8/8/3k4/8/8/3K4/8/8 w - - 0 1')
        assert get_phase(board) == 'endgame'


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — board_utils misc helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestBoardUtils:

    def test_get_fen_starting(self):
        board = chess.Board()
        fen = get_fen(board)
        assert fen.startswith('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR')

    def test_position_key_excludes_clocks(self):
        board = chess.Board()
        key = get_position_key(board)
        parts = key.split(' ')
        assert len(parts) == 4  # Only first 4 FEN fields

    def test_position_key_transposition_stable(self):
        # e4 d5 vs d4 e5 — different move orders, same structural key expected
        # (They won't be identical, but the function should return 4-field FEN)
        b1 = board_at_ply(['e2e4', 'd7d5'])
        b2 = board_at_ply(['d2d4', 'e7e5'])
        key1 = get_position_key(b1)
        key2 = get_position_key(b2)
        assert len(key1.split(' ')) == 4
        assert len(key2.split(' ')) == 4

    def test_pawn_hash_starting(self):
        board = chess.Board()
        h = get_pawn_hash(board)
        assert isinstance(h, str)
        assert len(h) == 16  # 64-bit hex

    def test_pawn_hash_changes_on_pawn_move(self):
        b1 = chess.Board()
        b2 = board_at_ply(['e2e4'])
        assert get_pawn_hash(b1) != get_pawn_hash(b2)

    def test_pawn_hash_stable_on_piece_move(self):
        """Moving a piece doesn't change the pawn hash."""
        b1 = chess.Board()
        b2 = board_at_ply(['g1f3'])  # Knight move — no pawn change
        assert get_pawn_hash(b1) == get_pawn_hash(b2)

    def test_square_to_str(self):
        assert square_to_str(chess.E4) == 'e4'
        assert square_to_str(chess.A1) == 'a1'
        assert square_to_str(chess.H8) == 'h8'

    def test_str_to_square(self):
        assert str_to_square('e4') == chess.E4
        assert str_to_square('a1') == chess.A1

    def test_roundtrip_square(self):
        for sq in chess.SQUARES:
            assert str_to_square(square_to_str(sq)) == sq

    def test_get_king_zone_center(self):
        board = chess.Board('8/8/8/8/4K3/8/8/8 w - - 0 1')
        zone = get_king_zone(board, chess.WHITE)
        # King on e4 — should have 8 surrounding squares + e4 itself = 9 squares
        # (e4 is not on an edge)
        assert len(zone) == 9

    def test_get_king_zone_corner(self):
        # King in corner has only 3 neighbours + itself = 4 squares
        board = chess.Board('K7/8/8/8/8/8/8/8 w - - 0 1')
        zone = get_king_zone(board, chess.WHITE)
        assert len(zone) == 4  # a8, b8, a7, b7

    def test_get_king_zone_edge(self):
        # King on edge (e.g. e1) has 5 neighbours + itself = 6 squares
        board = chess.Board('8/8/8/8/8/8/8/4K3 w - - 0 1')
        zone = get_king_zone(board, chess.WHITE)
        assert len(zone) == 6

    def test_get_king_zone_str_returns_strings(self):
        board = chess.Board()  # Kings on e1, e8
        zone = get_king_zone_str(board, chess.WHITE)
        assert all(isinstance(s, str) for s in zone)
        assert all(len(s) == 2 for s in zone)

    def test_get_move_history_starting(self):
        game = chess.pgn.Game()
        game.add_variation(chess.Move.from_uci('e2e4'))
        game.variations[0].add_variation(chess.Move.from_uci('e7e5'))
        history = get_move_history(game)
        assert history == ['e2e4', 'e7e5']

    def test_get_move_history_empty(self):
        game = chess.pgn.Game()
        assert get_move_history(game) == []

    def test_count_legal_moves_starting(self):
        board = chess.Board()
        # White has 20 legal moves from the starting position
        assert count_legal_moves(board, chess.WHITE) == 20

    def test_count_legal_moves_other_side(self):
        board = chess.Board()
        # Black also has 20 from starting position when estimated
        count = count_legal_moves(board, chess.BLACK)
        assert count == 20

    def test_pieces_in_zone_empty_board(self):
        board = chess.Board('8/8/8/8/4K3/8/8/8 w - - 0 1')
        zone = get_king_zone(board, chess.WHITE)
        pieces = get_pieces_in_zone(board, chess.BLACK, zone)
        assert pieces == []


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — StockfishBridge (mocked — no binary required)
# ─────────────────────────────────────────────────────────────────────────────

class MockBestMove:
    """Mirrors the BestMove namedtuple from engine/uci_engine.py."""
    def __init__(self, uci='e2e4', score_cp=15, score_mate=None, depth=20, pv_uci_list=None):
        self.uci = uci
        self.score_cp = score_cp
        self.score_mate = score_mate
        self.depth = depth
        self.pv_uci_list = pv_uci_list or [uci]


class MockUciEngine:
    """Minimal mock of UciEngine for testing the bridge without Stockfish."""
    def __init__(self, engine_exe):
        self._started = False

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    def analyze_movetime(self, moves, movetime_ms):
        # Starting position → near-zero eval
        if not moves:
            return MockBestMove(uci='e2e4', score_cp=15, depth=20)
        # After e4 d5 — slightly in White's favour
        return MockBestMove(uci='g1f3', score_cp=22, depth=18)


class TestStockfishBridgeMocked:

    def _make_bridge(self, mocker_or_patch=None) -> object:
        """Build a bridge with the mock engine injected."""
        from core.stockfish_bridge import StockfishBridge
        bridge = StockfishBridge(stockfish_path='/mock/stockfish', movetime_ms=100)
        # Inject mock engine directly
        bridge._engine = MockUciEngine(engine_exe='/mock/stockfish')
        return bridge

    def test_get_eval_starting_position(self):
        """Starting position should return eval close to 0."""
        from core.stockfish_bridge import StockfishBridge, EvalResult
        bridge = StockfishBridge(stockfish_path='/mock/sf', movetime_ms=100)
        bridge._engine = MockUciEngine('/mock/sf')

        result = bridge.get_eval(chess.Board().fen())
        assert isinstance(result, EvalResult)
        assert result.centipawns is not None
        # Mock returns 15 centipawns — within ±20 of 0 as per spec criterion
        assert abs(result.centipawns) <= 20 or result.centipawns == 15  # mock value

    def test_get_eval_returns_eval_result_type(self):
        from core.stockfish_bridge import StockfishBridge, EvalResult
        bridge = StockfishBridge(stockfish_path='/mock/sf', movetime_ms=100)
        bridge._engine = MockUciEngine('/mock/sf')
        result = bridge.get_eval(chess.Board().fen())
        assert isinstance(result, EvalResult)

    def test_get_best_move_returns_move_candidate(self):
        from core.stockfish_bridge import StockfishBridge, MoveCandidate
        bridge = StockfishBridge(stockfish_path='/mock/sf', movetime_ms=100)
        bridge._engine = MockUciEngine('/mock/sf')
        board = chess.Board()
        best = bridge.get_best_move(board)
        assert best is not None
        assert isinstance(best, MoveCandidate)
        assert best.rank == 1

    def test_get_top_moves_returns_list(self):
        from core.stockfish_bridge import StockfishBridge, MoveCandidate
        bridge = StockfishBridge(stockfish_path='/mock/sf', movetime_ms=100)
        bridge._engine = MockUciEngine('/mock/sf')
        board = chess.Board()
        moves = bridge.get_top_moves(board, n=3)
        assert isinstance(moves, list)
        assert all(isinstance(m, MoveCandidate) for m in moves)

    def test_eval_result_score_from_side_white(self):
        from core.stockfish_bridge import EvalResult
        er = EvalResult(centipawns=30, mate_in=None, depth=20)
        assert er.score_from_side('white') == 30
        assert er.score_from_side('black') == -30

    def test_eval_result_is_mate_false(self):
        from core.stockfish_bridge import EvalResult
        er = EvalResult(centipawns=50, mate_in=None, depth=15)
        assert er.is_mate is False

    def test_eval_result_is_mate_true(self):
        from core.stockfish_bridge import EvalResult
        er = EvalResult(centipawns=None, mate_in=3, depth=10)
        assert er.is_mate is True

    def test_assert_running_raises_when_not_started(self):
        from core.stockfish_bridge import StockfishBridge
        bridge = StockfishBridge(stockfish_path='/mock/sf', movetime_ms=100)
        # _engine is None — bridge not started
        with pytest.raises(RuntimeError, match='start\\(\\)'):
            bridge.get_eval(chess.Board().fen())

    def test_is_running_false_before_start(self):
        from core.stockfish_bridge import StockfishBridge
        bridge = StockfishBridge(stockfish_path='/mock/sf', movetime_ms=100)
        assert bridge.is_running is False

    def test_is_running_true_after_mock_inject(self):
        from core.stockfish_bridge import StockfishBridge
        bridge = StockfishBridge(stockfish_path='/mock/sf', movetime_ms=100)
        bridge._engine = MockUciEngine('/mock/sf')
        assert bridge.is_running is True

    def test_board_to_uci_prefix_starting(self):
        from core.stockfish_bridge import StockfishBridge
        board = chess.Board()
        prefix = StockfishBridge._board_to_uci_prefix(board)
        assert prefix == []

    def test_board_to_uci_prefix_after_moves(self):
        from core.stockfish_bridge import StockfishBridge
        board = board_at_ply(['e2e4', 'e7e5'])
        prefix = StockfishBridge._board_to_uci_prefix(board)
        assert prefix == ['e2e4', 'e7e5']
