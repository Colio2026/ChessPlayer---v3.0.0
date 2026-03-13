"""
tests/test_strategies.py
=========================
Phase 3 validation tests. Covers strategy detectors, phase_filter,
conflict_resolver, and strategy_engine integration.

Validation positions
--------------------
TAL_FEN      : Tal-style attack. Black king exposed > 0.80.
               Blitz must score > 0.70 for White.

SQUEEZE_FEN  : Carlsen-style squeeze. White advanced pawns, space grip.
               Flank score for White must exceed Flank score for Black.

FORTRESS_FEN : Locked pawn structure. High fixedness.
               Fortress score elevated when eval_deficit injected.

START_FEN    : Starting position. All strategies near-neutral.

Phase positions
---------------
OPENING_FEN     : mat=78  → 'opening'
MIDDLEGAME_FEN  : mat=34  → 'middlegame'
ENDGAME_FEN     : mat=5   → 'endgame'

Conflict resolver test
----------------------
blitz=0.71, flank=0.68, king_exposure=0.82 → Blitz wins (Rule 2).
blitz=0.70, flank=0.71 in endgame          → Flank wins (Rule 3).
Both within 0.08                           → Tie band fires (Rule 4).
Feint highest without DB                   → Demoted to secondary (Rule 5).
"""

from __future__ import annotations

import sys
import chess
import pytest

from chess_coach.core.data_types       import MetricSignal, CoachOutput
from chess_coach.core.phase_filter     import apply_phase_filter
from chess_coach.core.conflict_resolver import resolve, ResolverResult
from chess_coach.core.strategy_engine  import StrategyEngine
from chess_coach.extractors.king_safety    import extract_king_safety
from chess_coach.extractors.space_control  import extract_space_control
from chess_coach.extractors.piece_mobility import extract_piece_mobility
from chess_coach.extractors.pawn_structure import extract_pawn_structure
from chess_coach.extractors.material_balance import extract_material_balance
from chess_coach.extractors.tactic_scanner import extract_tactics
from chess_coach.strategies.blitz_detector    import score_blitz
from chess_coach.strategies.flank_detector    import score_flank
from chess_coach.strategies.fortress_detector import score_fortress
from chess_coach.strategies.feint_detector    import score_feint


# ─────────────────────────────────────────────────────────────────────────────
# Validation positions
# ─────────────────────────────────────────────────────────────────────────────

TAL_FEN      = "r4rk1/pp3p1p/2n1pnpQ/2NpB1B1/2PP4/8/PP4PP/3RR1K1 w - - 0 1"
SQUEEZE_FEN  = "r2qkb1r/1b3ppp/p1nppn2/Pp2P3/3P1P2/2N2N2/1PP1B1PP/R1BQK2R w KQkq - 0 1"
FORTRESS_FEN = "r1bqk2r/6pp/4pn2/ppbpP3/PPPP4/3B1NN1/6PP/R1BQK2R w KQkq - 0 1"
START_FEN    = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

OPENING_FEN    = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MIDDLEGAME_FEN = "r1bqk3/p1p1pppp/2n5/8/8/2N5/P1P1PPPP/R1BQK3 w KQkq - 0 1"
ENDGAME_FEN    = "4k3/8/8/8/8/8/8/4K2R w - - 0 1"


def board(fen: str) -> chess.Board:
    return chess.Board(fen)


def all_signals(fen: str, phase: str = 'middlegame') -> list[MetricSignal]:
    b = chess.Board(fen)
    sigs = []
    sigs.extend(extract_king_safety(b, phase))
    sigs.extend(extract_space_control(b, phase=phase))
    sigs.extend(extract_piece_mobility(b, phase=phase))
    sigs.extend(extract_pawn_structure(b, phase))
    sigs.extend(extract_material_balance(b, phase=phase))
    sigs.extend(extract_tactics(b, phase))
    return sigs


def fake_signal(metric: str, score: float, side: str = 'white') -> MetricSignal:
    return MetricSignal(
        metric_name=metric, score=score, side=side,
        cause='test', phase='middlegame',
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — phase_filter
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseFilter:

    def test_opening_classification(self):
        phase, _ = apply_phase_filter([], board(OPENING_FEN))
        assert phase == 'opening'

    def test_middlegame_classification(self):
        phase, _ = apply_phase_filter([], board(MIDDLEGAME_FEN))
        assert phase == 'middlegame'

    def test_endgame_classification(self):
        phase, _ = apply_phase_filter([], board(ENDGAME_FEN))
        assert phase == 'endgame'

    def test_phase_injected_into_signals(self):
        sig = fake_signal('king_exposure', 0.5, 'black')
        phase, weighted = apply_phase_filter([sig], board(MIDDLEGAME_FEN))
        assert all(s.phase == 'middlegame' for s in weighted)

    def test_passed_pawn_boosted_in_endgame(self):
        sig = fake_signal('passed_pawn', 0.5, 'white')
        _, end_sigs = apply_phase_filter([sig], board(ENDGAME_FEN))
        _, mid_sigs = apply_phase_filter([sig], board(MIDDLEGAME_FEN))
        end_score = end_sigs[0].score
        mid_score = mid_sigs[0].score
        assert end_score > mid_score, "Passed pawn should score higher in endgame"

    def test_king_exposure_reduced_in_endgame(self):
        sig = fake_signal('king_exposure', 0.8, 'black')
        _, end_sigs = apply_phase_filter([sig], board(ENDGAME_FEN))
        _, mid_sigs = apply_phase_filter([sig], board(MIDDLEGAME_FEN))
        assert end_sigs[0].score < mid_sigs[0].score, \
            "King exposure should be less penalised in endgame"

    def test_scores_clamped_to_one(self):
        sig = fake_signal('passed_pawn', 0.9, 'white')
        _, end_sigs = apply_phase_filter([sig], board(ENDGAME_FEN))
        assert end_sigs[0].score <= 1.0

    def test_unknown_metric_unchanged(self):
        sig = fake_signal('custom_metric_xyz', 0.6, 'white')
        _, mid_sigs = apply_phase_filter([sig], board(MIDDLEGAME_FEN))
        assert mid_sigs[0].score == 0.6, "Unknown metrics should have 1.0 multiplier"

    def test_returns_list_of_metric_signals(self):
        sigs = all_signals(TAL_FEN)
        phase, weighted = apply_phase_filter(sigs, board(TAL_FEN))
        assert all(isinstance(s, MetricSignal) for s in weighted)

    def test_signal_count_preserved(self):
        sigs = all_signals(START_FEN)
        _, weighted = apply_phase_filter(sigs, board(START_FEN))
        assert len(weighted) == len(sigs)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — blitz_detector
# ─────────────────────────────────────────────────────────────────────────────

class TestBlitzDetector:

    def test_returns_float_in_range(self):
        sigs = all_signals(START_FEN)
        score = score_blitz(sigs, 'white')
        assert 0.0 <= score <= 1.0

    def test_tal_position_blitz_fires_for_white(self):
        """TAL_FEN: White attacking, Black king exposed > 0.80 → blitz > 0.70."""
        sigs = all_signals(TAL_FEN)
        score = score_blitz(sigs, 'white')
        assert score > 0.70, (
            f"Blitz should fire for White in Tal position. Got {score:.4f}"
        )

    def test_tal_blitz_higher_from_white_than_black(self):
        """Blitz score is relative — White is attacking, not Black."""
        sigs = all_signals(TAL_FEN)
        white_score = score_blitz(sigs, 'white')
        black_score = score_blitz(sigs, 'black')
        assert white_score > black_score, (
            f"White blitz ({white_score:.3f}) should exceed Black blitz ({black_score:.3f})"
        )

    def test_starting_position_blitz_low(self):
        """Starting position: no attack available → blitz < 0.40."""
        sigs = all_signals(START_FEN)
        score = score_blitz(sigs, 'white')
        assert score < 0.40, f"Blitz should be low at start: {score:.4f}"

    def test_king_emergency_floor(self):
        """If king_exposure(opp) > 0.80, blitz score must be >= 0.70."""
        # Inject a high king_exposure signal for black
        sigs = [
            fake_signal('king_exposure', 0.85, 'black'),
            fake_signal('piece_mobility_ratio', 0.55, 'white'),
        ]
        score = score_blitz(sigs, 'white')
        assert score >= 0.70, f"King emergency floor should give >= 0.70, got {score:.4f}"

    def test_tempo_chain_bonus_applied(self):
        """3+ position history with White mobility > 0.55 adds tempo bonus."""
        base_sigs = [fake_signal('king_exposure', 0.85, 'black')]
        history = [
            [fake_signal('piece_mobility_ratio', 0.60, 'white')],
            [fake_signal('piece_mobility_ratio', 0.62, 'white')],
            [fake_signal('piece_mobility_ratio', 0.58, 'white')],
        ]
        with_history    = score_blitz(base_sigs, 'white', history_signals=history)
        without_history = score_blitz(base_sigs, 'white', history_signals=None)
        assert with_history >= without_history, "Tempo chain should not decrease score"

    def test_no_history_still_returns_valid_score(self):
        sigs = all_signals(TAL_FEN)
        score = score_blitz(sigs, 'white', history_signals=None)
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — flank_detector
# ─────────────────────────────────────────────────────────────────────────────

class TestFlankDetector:

    def test_returns_float_in_range(self):
        sigs = all_signals(START_FEN)
        score = score_flank(sigs, 'white')
        assert 0.0 <= score <= 1.0

    def test_squeeze_position_white_leads_flank(self):
        """SQUEEZE_FEN: White has more space → White flank > Black flank."""
        sigs = all_signals(SQUEEZE_FEN)
        white_score = score_flank(sigs, 'white')
        black_score = score_flank(sigs, 'black')
        assert white_score > black_score, (
            f"White flank ({white_score:.3f}) should exceed Black ({black_score:.3f})"
        )

    def test_starting_position_near_equal(self):
        """Starting position: space equal → both flank scores similar."""
        sigs = all_signals(START_FEN)
        w = score_flank(sigs, 'white')
        b = score_flank(sigs, 'black')
        assert abs(w - b) < 0.25, f"Flank scores should be near equal at start: w={w:.3f} b={b:.3f}"

    def test_trend_multiplier_raises_score(self):
        """Active space_trend signal for White lifts the flank score."""
        base_sigs = [
            fake_signal('space_delta_queenside', 0.65, 'white'),
            fake_signal('space_delta_kingside',  0.60, 'white'),
            fake_signal('piece_mobility_ratio',  0.58, 'white'),
        ]
        # Without trend signal
        score_no_trend = score_flank(base_sigs, 'white', history_signals=[[]])

        # With trend signal active
        trend_sigs = base_sigs + [fake_signal('space_trend', 0.70, 'white')]
        score_trend = score_flank(trend_sigs, 'white', history_signals=[[]])

        assert score_trend >= score_no_trend, "Active space_trend should not lower flank score"

    def test_no_history_snapshot_fallback(self):
        """No history → snapshot fallback at reduced weight (×0.80)."""
        sigs = [
            fake_signal('space_delta_queenside', 0.70, 'white'),
            fake_signal('piece_mobility_ratio',  0.60, 'white'),
        ]
        with_hist    = score_flank(sigs, 'white', history_signals=[[]])
        without_hist = score_flank(sigs, 'white', history_signals=None)
        assert without_hist <= with_hist + 0.01, "No-history should give reduced or equal score"

    def test_bad_piece_count_raises_flank_score(self):
        """Opponent bad pieces contribute to flank score."""
        no_bad = [fake_signal('space_delta_queenside', 0.60, 'white')]
        with_bad = no_bad + [
            fake_signal('bad_piece', 0.70, 'black'),
            fake_signal('bad_piece', 0.65, 'black'),
        ]
        s_no  = score_flank(no_bad,  'white')
        s_yes = score_flank(with_bad, 'white')
        assert s_yes > s_no, "Bad opponent pieces should raise flank score"


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — fortress_detector
# ─────────────────────────────────────────────────────────────────────────────

class TestFortressDetector:

    def test_returns_float_in_range(self):
        sigs = all_signals(START_FEN)
        score = score_fortress(sigs, 'white')
        assert 0.0 <= score <= 1.0

    def test_no_deficit_caps_fortress(self):
        """Without eval_deficit, fortress is capped at 0.30."""
        sigs = [fake_signal('pawn_fixedness', 0.8, 'white')]
        score = score_fortress(sigs, 'white')
        assert score <= 0.30, f"Fortress without deficit must be <= 0.30, got {score:.4f}"

    def test_eval_deficit_raises_fortress(self):
        """With eval_deficit, fortress score exceeds the no-deficit cap."""
        no_def = [fake_signal('pawn_fixedness', 0.8, 'white')]
        with_def = no_def + [fake_signal('eval_deficit', 0.6, 'white')]
        s_no  = score_fortress(no_def,  'white')
        s_yes = score_fortress(with_def, 'white')
        assert s_yes > s_no, "eval_deficit should raise fortress score"

    def test_fortress_fen_with_deficit_above_threshold(self):
        """FORTRESS_FEN with injected deficit should score > 0.40."""
        sigs = all_signals(FORTRESS_FEN)
        # Inject deficit signal
        sigs.append(fake_signal('eval_deficit', 0.55, 'black'))
        score = score_fortress(sigs, 'black')
        assert score > 0.40, f"Fortress with deficit should exceed 0.40, got {score:.4f}"

    def test_locked_structure_contributes(self):
        """High fixedness raises fortress score when deficit present."""
        base = [fake_signal('eval_deficit', 0.5, 'white')]
        with_fix = base + [fake_signal('pawn_fixedness', 0.80, 'white')]
        s_base = score_fortress(base, 'white')
        s_fix  = score_fortress(with_fix, 'white')
        assert s_fix > s_base, "High fixedness should raise fortress score"

    def test_fortress_lower_than_blitz_in_attacking_position(self):
        """In the Tal attack position, blitz should score higher than fortress."""
        sigs = all_signals(TAL_FEN)
        blitz_s   = score_blitz(sigs, 'white')
        fortress_s = score_fortress(sigs, 'white')
        assert blitz_s > fortress_s, (
            f"Blitz ({blitz_s:.3f}) should dominate fortress ({fortress_s:.3f}) in Tal position"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — feint_detector
# ─────────────────────────────────────────────────────────────────────────────

class TestFeintDetector:

    def test_returns_float_in_range(self):
        sigs = all_signals(START_FEN)
        score = score_feint(sigs, 'white')
        assert 0.0 <= score <= 1.0

    def test_capped_below_fire_threshold_without_db(self):
        """Without DB confirmation, feint must be < 0.65 (cannot fire as primary)."""
        # Use signals that would maximise feint score
        sigs = [
            fake_signal('piece_mobility_ratio', 0.50, 'white'),  # equal mobility
            fake_signal('outpost_occupation',   0.70, 'white'),  # latent threat
            fake_signal('pawn_fixedness',       0.80, 'white'),  # tension
            fake_signal('space_delta_queenside', 0.80, 'black'), # opp overcommit
        ]
        score = score_feint(sigs, 'white', db_confirmation=False)
        assert score < 0.65, f"Feint without DB must be < 0.65, got {score:.4f}"

    def test_db_confirmation_lifts_cap(self):
        """With DB confirmation, feint score can exceed 0.64."""
        sigs = [
            fake_signal('piece_mobility_ratio',  0.50, 'white'),
            fake_signal('outpost_occupation',    0.70, 'white'),
            fake_signal('pawn_fixedness',        0.80, 'white'),
            fake_signal('space_delta_queenside', 0.80, 'black'),
            fake_signal('space_delta_kingside',  0.20, 'black'),
        ]
        score = score_feint(sigs, 'white', db_confirmation=True)
        # With DB, the cap is lifted — score can go higher
        # It may still be < 0.65 if the raw score itself is low, but the cap is gone
        score_no_db = score_feint(sigs, 'white', db_confirmation=False)
        assert score >= score_no_db, "DB confirmation should not reduce feint score"

    def test_tactical_position_feint_low(self):
        """When many tactics fire, quiet/feint score should be low."""
        sigs = all_signals(TAL_FEN)  # lots of tactics in attack position
        score = score_feint(sigs, 'white')
        # Not asserting a hard threshold — just that it's modest
        assert score < 0.65, f"Feint should be low in tactical position: {score:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — conflict_resolver
# ─────────────────────────────────────────────────────────────────────────────

class TestConflictResolver:

    def _resolve(self, scores, king_exp=0.0, eval_deficit=0.0,
                 phase='middlegame', player_side='white', db=False):
        context = {'king_exposure': king_exp, 'eval_deficit': eval_deficit}
        return resolve(scores, context, phase, player_side, db_confirmation=db)

    def test_returns_resolver_result(self):
        r = self._resolve({'blitz': 0.71, 'flank': 0.50, 'fortress': 0.30, 'feint': 0.20})
        assert isinstance(r, ResolverResult)

    def test_highest_score_wins_normally(self):
        """No special rules triggered — highest score wins."""
        r = self._resolve({'blitz': 0.80, 'flank': 0.60, 'fortress': 0.30, 'feint': 0.20})
        assert r.primary == 'blitz'
        assert r.confidence == 0.80

    def test_rule2_king_emergency_blitz_overrides_flank(self):
        """Rule 2: king_exposure > 0.80 → blitz wins over flank."""
        r = self._resolve(
            {'blitz': 0.71, 'flank': 0.68, 'fortress': 0.20, 'feint': 0.10},
            king_exp=0.82
        )
        assert r.primary == 'blitz', (
            f"Blitz should override Flank with king emergency. Got: {r.primary}"
        )

    def test_rule1_eval_deficit_boosts_fortress(self):
        """Rule 1: eval_deficit > 0.50 gives fortress +0.25 bonus."""
        r = self._resolve(
            {'blitz': 0.50, 'flank': 0.55, 'fortress': 0.45, 'feint': 0.20},
            eval_deficit=0.60
        )
        # fortress gets 0.45 + 0.25 = 0.70, should win
        assert r.primary == 'fortress', (
            f"Fortress should win with eval deficit bonus. Got: {r.primary}"
        )

    def test_rule3_endgame_flank_overrides_blitz(self):
        """Rule 3: In endgame, flank beats blitz when both above 0.65."""
        r = self._resolve(
            {'blitz': 0.70, 'flank': 0.71, 'fortress': 0.30, 'feint': 0.10},
            phase='endgame'
        )
        assert r.primary == 'flank', (
            f"Flank should override Blitz in endgame. Got: {r.primary}"
        )

    def test_rule4_tie_band_fires_both_strategies(self):
        """Rule 4: scores within 0.08 → tie_band=True, secondary returned."""
        r = self._resolve(
            {'blitz': 0.74, 'flank': 0.70, 'fortress': 0.30, 'feint': 0.20}
        )
        assert r.tie_band is True, "Scores within 0.08 should trigger tie band"
        assert r.secondary is not None, "Secondary strategy should be set in tie band"
        assert r.secondary != r.primary

    def test_rule4_no_tie_band_when_gap_large(self):
        """No tie band when gap > 0.08."""
        r = self._resolve(
            {'blitz': 0.80, 'flank': 0.60, 'fortress': 0.30, 'feint': 0.20}
        )
        assert r.tie_band is False

    def test_rule5_feint_demoted_without_db(self):
        """Rule 5: Feint cannot be primary without DB confirmation."""
        r = self._resolve(
            {'blitz': 0.50, 'flank': 0.55, 'fortress': 0.40, 'feint': 0.80},
            db=False
        )
        assert r.primary != 'feint', "Feint should not be primary without DB"

    def test_rule5_feint_allowed_with_db(self):
        """Rule 5: Feint CAN be primary with DB confirmation."""
        r = self._resolve(
            {'blitz': 0.50, 'flank': 0.55, 'fortress': 0.40, 'feint': 0.80},
            db=True
        )
        assert r.primary == 'feint', "Feint should be primary when DB confirms"

    def test_confidence_matches_primary_score(self):
        r = self._resolve({'blitz': 0.75, 'flank': 0.60, 'fortress': 0.30, 'feint': 0.20})
        assert r.confidence == 0.75

    def test_rules_applied_in_priority_order(self):
        """Rule 2 (king emergency) fires before Rule 3 (endgame phase)."""
        # endgame + king emergency: Rule 2 fires first → blitz wins
        r = self._resolve(
            {'blitz': 0.68, 'flank': 0.70, 'fortress': 0.20, 'feint': 0.10},
            king_exp=0.85, phase='endgame'
        )
        # Rule 2 (king emergency) should override Rule 3 (endgame → flank)
        # because the king is in immediate danger
        assert r.primary == 'blitz', (
            f"Rule 2 king emergency should override Rule 3 endgame. Got: {r.primary}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — strategy_engine integration
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyEngine:

    def test_engine_instantiates_without_stockfish(self):
        """Engine should work without a Stockfish path."""
        engine = StrategyEngine(stockfish_path='')
        assert engine is not None

    def test_analyse_returns_coach_output(self):
        engine = StrategyEngine(stockfish_path='')
        output = engine.analyse(board(START_FEN), player_side='white')
        assert isinstance(output, CoachOutput)

    def test_analyse_strategy_primary_is_valid(self):
        engine = StrategyEngine(stockfish_path='')
        output = engine.analyse(board(START_FEN), player_side='white')
        assert output.strategy_primary in ('blitz', 'flank', 'fortress', 'feint', 'general')

    def test_analyse_phase_is_valid(self):
        engine = StrategyEngine(stockfish_path='')
        output = engine.analyse(board(START_FEN), player_side='white')
        assert output.phase in ('opening', 'middlegame', 'endgame')

    def test_analyse_confidence_in_range(self):
        engine = StrategyEngine(stockfish_path='')
        output = engine.analyse(board(TAL_FEN), player_side='white')
        assert 0.0 <= output.confidence <= 1.0

    def test_analyse_signal_dump_populated(self):
        engine = StrategyEngine(stockfish_path='')
        output = engine.analyse(board(TAL_FEN), player_side='white')
        assert len(output.signal_dump) > 0, "signal_dump should contain MetricSignals"
        assert all(isinstance(s, MetricSignal) for s in output.signal_dump)

    def test_analyse_headline_is_string(self):
        engine = StrategyEngine(stockfish_path='')
        output = engine.analyse(board(TAL_FEN), player_side='white')
        assert isinstance(output.headline, str)
        assert len(output.headline) > 10

    def test_analyse_plan_sentences_count(self):
        engine = StrategyEngine(stockfish_path='')
        output = engine.analyse(board(TAL_FEN), player_side='white')
        assert 2 <= len(output.plan_sentences) <= 4

    def test_tal_position_blitz_primary_for_white(self):
        """TAL_FEN: White attacking → strategy_primary should be blitz."""
        engine = StrategyEngine(stockfish_path='')
        output = engine.analyse(board(TAL_FEN), player_side='white')
        assert output.strategy_primary == 'blitz', (
            f"Expected blitz as primary for Tal attack. Got: {output.strategy_primary} "
            f"(confidence={output.confidence:.3f})"
        )

    def test_weakness_squares_are_valid_algebraic(self):
        engine = StrategyEngine(stockfish_path='')
        output = engine.analyse(board(TAL_FEN), player_side='white')
        for sq in output.weakness_squares:
            assert len(sq) == 2
            assert sq[0] in 'abcdefgh'
            assert sq[1] in '12345678'

    def test_endgame_phase_classified_correctly(self):
        engine = StrategyEngine(stockfish_path='')
        output = engine.analyse(board(ENDGAME_FEN), player_side='white')
        assert output.phase == 'endgame'

    def test_analyse_with_history_boards(self):
        """Passing history_boards should not crash the engine."""
        engine = StrategyEngine(stockfish_path='')
        b = chess.Board()
        history = [chess.Board()]
        for m in ['e2e4', 'e7e5', 'd2d4']:
            b.push(chess.Move.from_uci(m))
            history.append(b.copy())
        output = engine.analyse(b, player_side='white', history_boards=history)
        assert isinstance(output, CoachOutput)

    def test_close_does_not_raise(self):
        engine = StrategyEngine(stockfish_path='')
        engine.close()  # should not raise

    def test_both_sides_analysable(self):
        """analyse() works for both 'white' and 'black'."""
        engine = StrategyEngine(stockfish_path='')
        w = engine.analyse(board(SQUEEZE_FEN), player_side='white')
        b = engine.analyse(board(SQUEEZE_FEN), player_side='black')
        assert isinstance(w, CoachOutput)
        assert isinstance(b, CoachOutput)

    def test_relative_scoring_squeeze_white_leads(self):
        """In the squeeze position, White should have higher confidence than Black."""
        engine = StrategyEngine(stockfish_path='')
        w = engine.analyse(board(SQUEEZE_FEN), player_side='white')
        b = engine.analyse(board(SQUEEZE_FEN), player_side='black')
        assert w.confidence >= b.confidence, (
            f"White confidence ({w.confidence:.3f}) should be >= Black ({b.confidence:.3f}) "
            f"in squeeze position"
        )
