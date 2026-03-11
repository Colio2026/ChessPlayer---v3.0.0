from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import chess.pgn

from pgn.store import PgnStore

if TYPE_CHECKING:
    from pgn.move_tree import MoveTree


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContinuationStat:
    san: str
    count: int
    white_wins: int
    draws: int
    black_wins: int

    @property
    def white_score_pct(self) -> float:
        if self.count <= 0:
            return 0.0
        return ((self.white_wins + 0.5 * self.draws) / self.count) * 100.0


# ── Tree query (fast path — used by VariationsPanel) ─────────────────────────

def query_continuations(
    tree: "MoveTree | None",
    prefix_uci: list[str],
    max_out: int = 30,
) -> list[ContinuationStat]:
    """
    Query the in-memory MoveTree for continuations from prefix_uci.
    Returns [] if the tree is not built or the position was never seen.
    """
    if tree is None or not tree.is_built():
        return []
    return tree.query(prefix_uci, max_out=max_out)


# ── Store-based slow path (used by browser_ops for the small in-view set) ────

@dataclass
class _MutableStat:
    count: int = 0
    white_wins: int = 0
    draws: int = 0
    black_wins: int = 0


def _pos_key(board: chess.Board) -> str:
    parts = board.fen().split()
    return " ".join(parts[:4])


def _tally(bucket: _MutableStat, result: str) -> None:
    bucket.count += 1
    if result == "1-0":
        bucket.white_wins += 1
    elif result == "0-1":
        bucket.black_wins += 1
    else:
        bucket.draws += 1


def _build_stats(raw: dict[str, _MutableStat], max_out: int) -> list[ContinuationStat]:
    out = [
        ContinuationStat(
            san=san, count=s.count,
            white_wins=s.white_wins, draws=s.draws, black_wins=s.black_wins,
        )
        for san, s in raw.items()
    ]
    out.sort(key=lambda x: (-x.count, -x.white_score_pct))
    return out[:max_out]


def common_continuations_from_store(
    store: PgnStore,
    game_ids: list[int],
    prefix_uci: list[str],
    max_games: int = 200,
    max_out: int = 30,
) -> list[ContinuationStat]:
    """
    Position-aware continuation query directly from the store.
    Used by the game-browser panel for the small filtered subset visible
    in the table — not the full library.
    """
    target_board = chess.Board()
    for uci in prefix_uci:
        try:
            target_board.push(chess.Move.from_uci(uci))
        except Exception:
            return []

    target_key = _pos_key(target_board)
    stats: dict[str, _MutableStat] = defaultdict(_MutableStat)

    for gid in game_ids[:max_games]:
        try:
            pgn_text = store.open_game_pgn_text(gid)
            game = chess.pgn.read_game(io.StringIO(pgn_text))
            if game is None:
                continue
            result = str(game.headers.get("Result", "")).strip()
            board  = chess.Board()
            node   = game
            while True:
                if _pos_key(board) == target_key:
                    if node.variations:
                        mv = node.variations[0].move
                        if mv in board.legal_moves:
                            _tally(stats[board.san(mv)], result)
                    break
                if not node.variations:
                    break
                main = node.variations[0]
                board.push(main.move)
                node = main
        except Exception:
            continue

    return _build_stats(stats, max_out)


def root_continuation_stats_from_store(
    store: PgnStore,
    game_ids: list[int],
    max_out: int = 50,
) -> list[ContinuationStat]:
    """Legacy wrapper: continuations from starting position."""
    return common_continuations_from_store(
        store, game_ids, prefix_uci=[], max_games=len(game_ids), max_out=max_out
    )
