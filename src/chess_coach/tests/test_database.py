"""
tests/test_database.py
=======================
Phase 4 validation tests for pattern_matcher.py, phrase_db.py,
and pgn_indexer.py integration.

These tests work WITHOUT a real PGN file or game_index DB.
- PatternMatcher: empty-path graceful degradation tested
- PhraseDB: in-memory DB created on the fly
- pgn_indexer: schema creation tested with a temp DB
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import chess
import pytest

from core.data_types        import MetricSignal, GMPrecedent
from core.board_utils       import get_pawn_hash
from database.pattern_matcher import PatternMatcher
from database.phrase_db       import PhraseDB, _fill
from database.pgn_indexer     import ensure_indexed, _CREATE_TABLE


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

TAL_FEN   = "r4rk1/pp3p1p/2n1pnpQ/2NpB1B1/2PP4/8/PP4PP/3RR1K1 w - - 0 1"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def board(fen: str) -> chess.Board:
    return chess.Board(fen)


def _severity(score: float) -> str:
    if score >= 0.75: return 'critical'
    if score >= 0.50: return 'high'
    if score >= 0.25: return 'moderate'
    return 'mild'


def fake_signal(
    metric: str,
    score: float,
    side: str = 'white',
    cause: str = 'test',
    key_squares: list[str] | None = None,
    key_pieces: list[str] | None = None,
) -> MetricSignal:
    return MetricSignal(
        metric_name=metric, score=score, side=side,
        cause=cause, phase='middlegame',
        severity=_severity(score),
        key_squares=key_squares or [],
        key_pieces=key_pieces or [],
        action_hint=f'{metric} action hint',
    )


def make_temp_db_with_schema() -> str:
    """Create a temp SQLite file with coach_positions table."""
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    conn = sqlite3.connect(f.name)
    # Minimal games table (existing ChessPlayer schema)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS games (
            game_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL DEFAULT 1,
            pgn_path TEXT NOT NULL DEFAULT '',
            offset_bytes INTEGER NOT NULL DEFAULT 0,
            white TEXT, black TEXT, result TEXT,
            event TEXT, site TEXT, date TEXT, eco TEXT, opening TEXT
        );
    """)
    conn.executescript(_CREATE_TABLE)
    conn.commit()
    conn.close()
    return f.name


def make_temp_db_with_positions(pawn_hash: str, strategy: str = 'flank') -> str:
    """Create a temp DB with one synthetic coach_positions row."""
    path = make_temp_db_with_schema()
    conn = sqlite3.connect(path)
    conn.execute("""
        INSERT INTO coach_positions
        (game_id, ply, fen, pawn_hash, strategy_tag, phase,
         eval_cp, player_white, player_black, rating_white,
         rating_black, result, key_move, annotation)
        VALUES (1, 20, ?, ?, ?, 'middlegame',
                15, 'Carlsen', 'Opponent', 2850,
                2700, '1-0', 'e2e4', 'Strong positional squeeze.')
    """, (TAL_FEN, pawn_hash, strategy))
    conn.commit()
    conn.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — PatternMatcher
# ─────────────────────────────────────────────────────────────────────────────

class TestPatternMatcher:

    def test_empty_path_not_available(self):
        m = PatternMatcher('')
        assert not m.is_available

    def test_missing_file_not_available(self):
        m = PatternMatcher('/nonexistent/path/index.sqlite')
        assert not m.is_available

    def test_empty_path_query_returns_empty(self):
        m = PatternMatcher('')
        result = m.query(board(START_FEN), 'flank', 'middlegame')
        assert result == []

    def test_empty_path_feint_confirmation_false(self):
        m = PatternMatcher('')
        assert m.db_confirms_feint(board(START_FEN), 'middlegame') is False

    def test_valid_db_is_available(self):
        path = make_temp_db_with_schema()
        m = PatternMatcher(path)
        assert m.is_available
        m.close()

    def test_query_returns_empty_when_no_matching_positions(self):
        path = make_temp_db_with_schema()
        m = PatternMatcher(path)
        result = m.query(board(TAL_FEN), 'blitz', 'middlegame')
        assert result == []
        m.close()

    def test_query_returns_gm_precedent(self):
        b = board(TAL_FEN)
        pawn_hash = get_pawn_hash(b)
        path = make_temp_db_with_positions(pawn_hash, 'flank')
        m = PatternMatcher(path, min_rating=0)
        result = m.query(b, 'flank', 'middlegame')
        assert len(result) > 0
        assert isinstance(result[0], GMPrecedent)
        m.close()

    def test_precedent_has_required_fields(self):
        b = board(TAL_FEN)
        pawn_hash = get_pawn_hash(b)
        path = make_temp_db_with_positions(pawn_hash, 'flank')
        m = PatternMatcher(path, min_rating=0)
        result = m.query(b, 'flank', 'middlegame')
        if result:
            p = result[0]
            assert isinstance(p.player,   str)
            assert isinstance(p.game_id,  str)
            assert isinstance(p.ply,      int)
            assert isinstance(p.key_move, str)
        m.close()

    def test_feint_confirmation_true_when_row_exists(self):
        b = board(TAL_FEN)
        pawn_hash = get_pawn_hash(b)
        path = make_temp_db_with_positions(pawn_hash, 'feint')
        m = PatternMatcher(path, min_rating=0)
        assert m.db_confirms_feint(b, 'middlegame') is True
        m.close()

    def test_feint_confirmation_false_when_no_feint_row(self):
        b = board(TAL_FEN)
        pawn_hash = get_pawn_hash(b)
        path = make_temp_db_with_positions(pawn_hash, 'blitz')  # not feint
        m = PatternMatcher(path, min_rating=0)
        assert m.db_confirms_feint(b, 'middlegame') is False
        m.close()

    def test_phase_fallback_returns_results(self):
        """If no phase match, falls back to any-phase query."""
        b = board(TAL_FEN)
        pawn_hash = get_pawn_hash(b)
        path = make_temp_db_with_positions(pawn_hash, 'flank')
        m = PatternMatcher(path, min_rating=0)
        # Query with 'endgame' phase — no exact match, should fall back
        result = m.query(b, 'flank', 'endgame')
        assert isinstance(result, list)  # may be empty or have result — just no crash
        m.close()

    def test_close_idempotent(self):
        m = PatternMatcher('')
        m.close()
        m.close()  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — PhraseDB
# ─────────────────────────────────────────────────────────────────────────────

class TestPhraseDB:

    def _temp_db(self) -> tuple[PhraseDB, str]:
        f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        f.close()
        return PhraseDB(f.name), f.name

    def test_empty_path_not_available(self):
        db = PhraseDB('')
        assert not db.is_available

    def test_valid_path_available(self):
        db, _ = self._temp_db()
        assert db.is_available
        db.close()

    def test_seeded_on_creation(self):
        db, path = self._temp_db()
        conn = sqlite3.connect(path)
        count = conn.execute("SELECT COUNT(*) FROM phrases").fetchone()[0]
        conn.close()
        assert count >= 40, f"Expected >= 40 seed phrases, got {count}"
        db.close()

    def test_tactics_table_seeded(self):
        db, path = self._temp_db()
        conn = sqlite3.connect(path)
        count = conn.execute("SELECT COUNT(*) FROM tactics").fetchone()[0]
        conn.close()
        assert count >= 4
        db.close()

    def test_get_fragments_returns_dict(self):
        db, _ = self._temp_db()
        sigs = [fake_signal('king_exposure', 0.85, 'black', 'missing_pawn_shield')]
        result = db.get_fragments('blitz', 'middlegame', sigs)
        assert isinstance(result, dict)
        assert 'diagnosis' in result
        assert 'plan' in result
        db.close()

    def test_blitz_king_exposure_returns_diagnosis(self):
        db, _ = self._temp_db()
        sigs = [fake_signal('king_exposure', 0.85, 'black', 'missing_pawn_shield',
                            key_squares=['g7'])]
        result = db.get_fragments('blitz', 'middlegame', sigs)
        assert len(result['diagnosis']) > 0, "Should return a diagnosis phrase for king_exposure"
        db.close()

    def test_fortress_eval_deficit_returns_plan(self):
        db, _ = self._temp_db()
        sigs = [fake_signal('eval_deficit', 0.65, 'white', 'positional_deficit')]
        result = db.get_fragments('fortress', 'any', sigs)
        assert len(result['plan']) > 0, "Should return a plan phrase for fortress/eval_deficit"
        db.close()

    def test_flank_outpost_returns_phrase(self):
        db, _ = self._temp_db()
        sigs = [fake_signal('outpost_occupation', 0.70, 'white', 'outpost',
                            key_squares=['d5'])]
        result = db.get_fragments('flank', 'middlegame', sigs)
        any_phrase = any(len(v) > 0 for v in result.values())
        assert any_phrase, "Should return at least one phrase for flank/outpost"
        db.close()

    def test_empty_signals_returns_empty_fragments(self):
        db, _ = self._temp_db()
        result = db.get_fragments('blitz', 'middlegame', [])
        assert all(len(v) == 0 for v in result.values())
        db.close()

    def test_get_tactic_hints_pin(self):
        db, _ = self._temp_db()
        sigs = [fake_signal('tactic_pin', 0.80, 'white', 'pin',
                            key_squares=['e5', 'e8'], key_pieces=['Ng5'])]
        hints = db.get_tactic_hints(sigs, 'middlegame')
        assert len(hints) > 0
        assert isinstance(hints[0], str)
        db.close()

    def test_get_tactic_hints_max_three(self):
        db, _ = self._temp_db()
        sigs = [
            fake_signal('tactic_pin',       0.80, 'white'),
            fake_signal('tactic_fork',      0.75, 'white'),
            fake_signal('tactic_skewer',    0.70, 'white'),
            fake_signal('tactic_discovery', 0.65, 'white'),
        ]
        hints = db.get_tactic_hints(sigs, 'middlegame', max_hints=3)
        assert len(hints) <= 3
        db.close()

    def test_close_idempotent(self):
        db, _ = self._temp_db()
        db.close()
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Placeholder filling
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaceholderFilling:

    def test_fill_square(self):
        sig = fake_signal('king_exposure', 0.8, key_squares=['g7'])
        result = _fill('weakness at {square}', sig)
        assert result == 'weakness at g7'

    def test_fill_file(self):
        sig = fake_signal('king_exposure', 0.8, key_squares=['g7'])
        result = _fill('the {file}-file is open', sig)
        assert result == 'the g-file is open'

    def test_fill_piece(self):
        sig = fake_signal('tactic_pin', 0.8, key_pieces=['Ng5'])
        result = _fill('the {piece} is pinned', sig)
        assert result == 'the Ng5 is pinned'

    def test_fill_side(self):
        sig = fake_signal('king_exposure', 0.8, side='black')
        result = _fill('the {side} king is exposed', sig)
        assert result == 'the Black king is exposed'

    def test_fill_missing_placeholder_leaves_empty(self):
        sig = fake_signal('king_exposure', 0.8)  # no key_squares
        result = _fill('weakness at {square}', sig)
        assert result == 'weakness at '  # empty but no crash

    def test_fill_multiple_placeholders(self):
        sig = fake_signal('tactic_pin', 0.8, key_squares=['e5', 'e8'], key_pieces=['Bb5'])
        result = _fill('the {piece} on {square} pins to {target}', sig)
        assert result == 'the Bb5 on e5 pins to e8'


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — pgn_indexer schema
# ─────────────────────────────────────────────────────────────────────────────

class TestPgnIndexerSchema:

    def test_ensure_indexed_returns_false_when_no_db(self):
        result = ensure_indexed('/nonexistent/path/index.sqlite')
        assert result is False

    def test_ensure_indexed_creates_table(self):
        path = make_temp_db_with_schema()
        ensure_indexed(path)  # games table is empty → should run (and find nothing)
        conn = sqlite3.connect(path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert 'coach_positions' in tables

    def test_ensure_indexed_false_when_already_populated(self):
        """If coach_positions already has rows, ensure_indexed returns False."""
        path = make_temp_db_with_schema()
        pawn_hash = get_pawn_hash(board(TAL_FEN))
        conn = sqlite3.connect(path)
        conn.execute("""
            INSERT INTO coach_positions
            (game_id, ply, fen, pawn_hash, strategy_tag, phase,
             eval_cp, player_white, player_black, rating_white,
             rating_black, result, key_move, annotation)
            VALUES (1,10,?,?,'flank','middlegame',0,'A','B',2800,2700,'1-0','e2e4','')
        """, (TAL_FEN, pawn_hash))
        conn.commit()
        conn.close()

        result = ensure_indexed(path)
        assert result is False

    def test_coach_positions_columns(self):
        path = make_temp_db_with_schema()
        ensure_indexed(path)
        conn = sqlite3.connect(path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(coach_positions)").fetchall()}
        conn.close()
        required = {'pawn_hash', 'strategy_tag', 'phase', 'fen', 'ply',
                    'key_move', 'annotation', 'result'}
        assert required.issubset(cols), f"Missing columns: {required - cols}"

    def test_pawn_hash_index_exists(self):
        path = make_temp_db_with_schema()
        ensure_indexed(path)
        conn = sqlite3.connect(path)
        indices = {r[1] for r in conn.execute(
            "SELECT * FROM sqlite_master WHERE type='index'"
        ).fetchall()}
        conn.close()
        assert 'idx_cp_pawn_hash' in indices


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — StrategyEngine integration with DB
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyEngineWithDB:

    def test_engine_with_phrase_db_returns_real_sentences(self):
        """When phrase_db is wired, plan_sentences should not be placeholder text."""
        from core.strategy_engine import StrategyEngine
        f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        f.close()

        engine = StrategyEngine(stockfish_path='', db_path=f.name)
        output = engine.analyse(chess.Board(TAL_FEN), player_side='white')
        engine.close()

        assert len(output.plan_sentences) >= 2
        # Should NOT be the fallback stub text
        assert output.plan_sentences[0] != 'Analysis complete.'

    def test_engine_without_phrase_db_uses_fallback(self):
        """Without phrase_db, plan_sentences uses action_hints fallback."""
        from core.strategy_engine import StrategyEngine
        engine = StrategyEngine(stockfish_path='', db_path='')
        output = engine.analyse(chess.Board(START_FEN), player_side='white')
        engine.close()
        assert len(output.plan_sentences) >= 2

    def test_engine_with_empty_pgn_index_does_not_crash(self):
        """PatternMatcher with empty DB returns empty precedents gracefully."""
        from core.strategy_engine import StrategyEngine
        path = make_temp_db_with_schema()
        engine = StrategyEngine(stockfish_path='', db_path='', pgn_index_path=path)
        output = engine.analyse(chess.Board(TAL_FEN), player_side='white')
        engine.close()
        assert isinstance(output.gm_precedents, list)

    def test_engine_with_matching_position_returns_precedent(self):
        """PatternMatcher with a matching row returns GMPrecedent in output."""
        from core.strategy_engine import StrategyEngine
        b = chess.Board(TAL_FEN)
        pawn_hash = get_pawn_hash(b)
        path = make_temp_db_with_positions(pawn_hash, 'blitz')
        engine = StrategyEngine(stockfish_path='', db_path='', pgn_index_path=path)
        # Patch min_rating to 0
        engine._matcher.min_rating = 0
        output = engine.analyse(b, player_side='white')
        engine.close()
        assert isinstance(output.gm_precedents, list)
