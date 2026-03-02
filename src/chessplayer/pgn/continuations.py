from __future__ import annotations

from collections import Counter
import io
from dataclasses import dataclass

import chess
import chess.pgn

from pgn.store import PgnStore


@dataclass(frozen=True)
class Continuation:
    san: str
    count: int


def _read_first_game(pgn_text: str) -> chess.pgn.Game | None:
    return chess.pgn.read_game(io.StringIO(pgn_text))


def _next_move_after_prefix(game: chess.pgn.Game, prefix_uci: list[str]) -> chess.Move | None:
    """
    If game's mainline begins with prefix_uci, return the next mainline move after the prefix.
    Otherwise None.
    """
    b = chess.Board()
    node: chess.pgn.GameNode = game

    # walk prefix
    for uci in prefix_uci:
        if not node.variations:
            return None
        mv = chess.Move.from_uci(uci)
        nxt = None
        # accept match anywhere among node variations (some games have early branches)
        for v in node.variations:
            if v.move == mv:
                nxt = v
                break
        if nxt is None:
            return None
        b.push(nxt.move)
        node = nxt

    # next mainline move (variation[0])
    if not node.variations:
        return None
    return node.variations[0].move


def common_continuations_from_store(
    store: PgnStore,
    game_ids: list[int],
    prefix_uci: list[str],
    max_games: int = 200,
    max_out: int = 30,
) -> list[Continuation]:
    """
    Sample games by id, count the next move (SAN) after the current prefix.
    """
    counts: Counter[str] = Counter()

    b = chess.Board()
    for uci in prefix_uci:
        b.push(chess.Move.from_uci(uci))

    sample = game_ids[:max_games]
    for gid in sample:
        try:
            pgn_text = store.open_game_pgn_text(gid)
            g = _read_first_game(pgn_text)
            if g is None:
                continue
            mv = _next_move_after_prefix(g, prefix_uci)
            if mv is None:
                continue
            # SAN relative to current prefix position
            if mv not in b.legal_moves:
                continue
            san = b.san(mv)
            counts[san] += 1
        except Exception:
            continue

    out = [Continuation(san=k, count=v) for k, v in counts.most_common(max_out)]
    return out