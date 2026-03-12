"""
extractors/pawn_structure.py
Pawn fixedness, chain stability, outpost detection, passed pawns,
isolated/doubled pawns, and Zobrist pawn hash.

Relative scoring:
  fixedness: (fixed_w + fixed_b) / total_pawns  [both sides contribute — 
              a locked structure affects everyone equally. Signal side = 
              whoever benefits from the lock — the stronger piece player.]
  outpost:   per-side signals at outpost squares in opponent half.
  passed:    per-side signals with the passed pawn squares.
  weak_pawns: isolated/doubled — per side.
"""
from __future__ import annotations
import chess
from core.data_types import MetricSignal
from core.board_utils import square_to_str, get_pawn_hash


def extract_pawn_structure(
    board: chess.Board,
    phase: str = "middlegame",
) -> list[MetricSignal]:
    """Return pawn structure MetricSignals for both sides."""
    signals: list[MetricSignal] = []
    signals.extend(_fixedness_signal(board, phase))
    signals.extend(_outpost_signals(board, phase))
    signals.extend(_passed_pawn_signals(board, phase))
    signals.extend(_weak_pawn_signals(board, chess.WHITE, phase))
    signals.extend(_weak_pawn_signals(board, chess.BLACK, phase))
    return signals


# ── Fixedness ─────────────────────────────────────────────────────────────────

def _is_fixed(sq: chess.Square, color: chess.Color, board: chess.Board) -> bool:
    """A pawn is fixed if the square directly in front of it is occupied."""
    rank = chess.square_rank(sq)
    direction = 1 if color == chess.WHITE else -1
    front_rank = rank + direction
    if not (0 <= front_rank <= 7):
        return False
    front_sq = chess.square(chess.square_file(sq), front_rank)
    return board.piece_at(front_sq) is not None


def _fixedness_signal(board: chess.Board, phase: str) -> list[MetricSignal]:
    white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
    total = len(white_pawns) + len(black_pawns)
    if total == 0:
        return []

    fixed_w = [sq for sq in white_pawns if _is_fixed(sq, chess.WHITE, board)]
    fixed_b = [sq for sq in black_pawns if _is_fixed(sq, chess.BLACK, board)]
    fixed_total = len(fixed_w) + len(fixed_b)
    score = fixed_total / total

    # Fixedness benefits whoever has better pieces for a static position
    # Emit as a shared structural signal (side=white by convention)
    fixed_squares = [square_to_str(sq) for sq in fixed_w + fixed_b]
    sev = "high" if score >= 0.65 else "moderate" if score >= 0.35 else "mild"
    return [MetricSignal(
        metric_name="pawn_fixedness", score=round(score, 4), side="white",
        cause="locked_pawn_structure" if score >= 0.5 else "fluid_pawn_structure",
        key_squares=fixed_squares[:6], severity=sev, fragment="",
        action_hint=("position is locked — manoeuvre pieces for long-term outposts"
                     if score >= 0.5 else
                     "fluid structure — pawn breaks available on both sides"),
        phase=phase,
    )]


# ── Outposts ──────────────────────────────────────────────────────────────────

def _outpost_squares(color: chess.Color, board: chess.Board) -> list[chess.Square]:
    """
    Squares in opponent half that cannot be attacked by any opponent pawn
    and are reachable by own pieces.
    """
    opp = not color
    opp_pawns = board.pieces(chess.PAWN, opp)
    outposts: list[chess.Square] = []

    # Candidate ranks: 4-7 for White (ranks 5-8), 0-3 for Black (ranks 1-4)
    if color == chess.WHITE:
        candidate_ranks = range(4, 8)
    else:
        candidate_ranks = range(0, 4)

    for f in range(8):
        for r in candidate_ranks:
            sq = chess.square(f, r)
            # Check if any opponent pawn can attack this square
            attacked_by_opp_pawn = False
            for psq in opp_pawns:
                pf, pr = chess.square_file(psq), chess.square_rank(psq)
                attack_dir = -1 if opp == chess.WHITE else 1
                if pr + attack_dir == r and abs(pf - f) == 1:
                    attacked_by_opp_pawn = True
                    break
            if not attacked_by_opp_pawn:
                # Confirm we have a piece that can reach here and is not already there
                if board.is_attacked_by(color, sq):
                    occupant = board.piece_at(sq)
                    if occupant is None or occupant.color != color:
                        outposts.append(sq)

    return outposts


def _outpost_signals(board: chess.Board, phase: str) -> list[MetricSignal]:
    signals: list[MetricSignal] = []
    for color in (chess.WHITE, chess.BLACK):
        side_str = "white" if color == chess.WHITE else "black"
        outposts = _outpost_squares(color, board)
        if not outposts:
            continue
        # Check which outposts are actually occupied by own pieces
        occupied = [sq for sq in outposts if
                    board.piece_at(sq) is not None and
                    board.piece_at(sq).color == color and  # type: ignore
                    board.piece_at(sq).piece_type not in (chess.PAWN, chess.KING)]  # type: ignore
        score = min(len(outposts) / 8.0, 1.0)
        if not outposts:
            continue
        sev = "high" if occupied else ("moderate" if len(outposts) >= 3 else "mild")
        key_sqs = [square_to_str(sq) for sq in (occupied or outposts)[:4]]
        hint = (f"occupy outpost on {key_sqs[0]} with a knight or bishop — cannot be challenged by pawns"
                if key_sqs else "outpost squares available — manoeuvre pieces there")
        signals.append(MetricSignal(
            metric_name="outpost_occupation", score=round(score, 4), side=side_str,
            cause="outpost_occupied" if occupied else "outpost_available",
            key_squares=key_sqs, severity=sev, fragment="",
            action_hint=hint, phase=phase,
        ))
    return signals


# ── Passed pawns ──────────────────────────────────────────────────────────────

def _is_passed(sq: chess.Square, color: chess.Color, board: chess.Board) -> bool:
    """No opponent pawn in front on same or adjacent file."""
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    opp = not color
    opp_pawns = board.pieces(chess.PAWN, opp)
    for psq in opp_pawns:
        pf = chess.square_file(psq)
        pr = chess.square_rank(psq)
        if abs(pf - f) <= 1:
            if (color == chess.WHITE and pr > r) or (color == chess.BLACK and pr < r):
                return False
    return True


def _passed_pawn_signals(board: chess.Board, phase: str) -> list[MetricSignal]:
    signals: list[MetricSignal] = []
    for color in (chess.WHITE, chess.BLACK):
        side_str = "white" if color == chess.WHITE else "black"
        passed = [sq for sq in board.pieces(chess.PAWN, color) if _is_passed(sq, color, board)]
        if not passed:
            continue
        # Score: number of passed pawns, weighted by advancement
        raw = sum(
            chess.square_rank(sq) / 7.0 if color == chess.WHITE else (7 - chess.square_rank(sq)) / 7.0
            for sq in passed
        )
        score = min(raw / 3.0, 1.0)
        key_sqs = [square_to_str(sq) for sq in sorted(
            passed,
            key=lambda s: chess.square_rank(s) if color == chess.WHITE else -chess.square_rank(s),
            reverse=True,
        )[:4]]
        sev = "critical" if score > 0.7 else "high" if score > 0.4 else "moderate"
        signals.append(MetricSignal(
            metric_name="passed_pawn", score=round(score, 4), side=side_str,
            cause="advanced_passed_pawn" if score > 0.5 else "passed_pawn",
            key_squares=key_sqs, severity=sev, fragment="",
            action_hint=f"advance and support passed pawn on {key_sqs[0]}",
            phase=phase,
        ))
    return signals


# ── Weak pawns ────────────────────────────────────────────────────────────────

def _weak_pawn_signals(
    board: chess.Board, color: chess.Color, phase: str
) -> list[MetricSignal]:
    side_str = "white" if color == chess.WHITE else "black"
    pawns = list(board.pieces(chess.PAWN, color))
    if not pawns:
        return []

    pawn_files = [chess.square_file(sq) for sq in pawns]
    isolated, doubled, weak_sqs = [], [], []

    for sq in pawns:
        f = chess.square_file(sq)
        # Isolated: no own pawn on adjacent files
        if (f - 1) not in pawn_files and (f + 1) not in pawn_files:
            isolated.append(sq)
            weak_sqs.append(square_to_str(sq))
        # Doubled: another own pawn on same file
        if pawn_files.count(f) > 1 and sq not in doubled:
            doubled.extend([s for s in pawns if chess.square_file(s) == f])
            weak_sqs.append(square_to_str(sq))

    total_weak = len(set(isolated + doubled))
    if total_weak == 0:
        return []

    score = min(total_weak / len(pawns), 1.0)
    sev = "high" if score > 0.5 else "moderate" if score > 0.25 else "mild"
    causes = []
    if isolated:
        causes.append("isolated_pawn")
    if doubled:
        causes.append("doubled_pawn")

    return [MetricSignal(
        metric_name="weak_pawns", score=round(score, 4), side=side_str,
        cause=causes[0] if causes else "weak_pawn",
        key_squares=list(dict.fromkeys(weak_sqs))[:6], severity=sev, fragment="",
        action_hint=("exchange isolated pawns or protect them with pieces"
                     if "isolated_pawn" in causes else
                     "resolve doubled pawns by trading or advancing"),
        phase=phase,
    )]
