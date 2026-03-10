from __future__ import annotations

from dataclasses import dataclass


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


def query_continuations(
    tree,                        # pgn.move_tree.MoveTree
    prefix_uci: list[str],
    max_out: int = 50,
) -> list[ContinuationStat]:
    """
    Return continuation stats one ply beyond prefix_uci using the
    pre-built MoveTree. Returns an empty list if the tree is not built.
    """
    if tree is None or not tree.is_built():
        return []
    return tree.query(prefix_uci, max_out=max_out)
