from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass

import chess
import chess.pgn

from pgn.store import PgnStore


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


@dataclass
class _MutableStat:
    count: int = 0
    white_wins: int = 0
    draws: int = 0
    black_wins: int = 0


def _read_first_game(pgn_text: str) -> chess.pgn.Game | None:
    return chess.pgn.read_game(io.StringIO(pgn_text))


def _first_mainline_move(game: chess.pgn.Game) -> chess.Move | None:
    if not game.variations:
        return None
    return game.variations[0].move


def root_continuation_stats_from_store(
    store: PgnStore,
    game_ids: list[int],
    max_out: int = 50,
) -> list[ContinuationStat]:
    stats: dict[str, _MutableStat] = defaultdict(_MutableStat)
    board = chess.Board()

    for gid in game_ids:
        try:
            pgn_text = store.open_game_pgn_text(gid)
            game = _read_first_game(pgn_text)
            if game is None:
                continue

            move = _first_mainline_move(game)
            if move is None or move not in board.legal_moves:
                continue

            san = board.san(move)
            result = str(game.headers.get("Result", "")).strip()

            bucket = stats[san]
            bucket.count += 1
            if result == "1-0":
                bucket.white_wins += 1
            elif result == "0-1":
                bucket.black_wins += 1
            else:
                bucket.draws += 1
        except Exception:
            continue

    out = [
        ContinuationStat(
            san=san,
            count=entry.count,
            white_wins=entry.white_wins,
            draws=entry.draws,
            black_wins=entry.black_wins,
        )
        for san, entry in stats.items()
    ]
    out.sort(key=lambda item: (-item.count, item.san))
    return out[:max_out]
