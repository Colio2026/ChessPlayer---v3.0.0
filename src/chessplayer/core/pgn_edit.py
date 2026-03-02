from __future__ import annotations

import io
from dataclasses import dataclass
import chess
import chess.pgn

from core.game_session import GameSession, MoveResult


@dataclass
class LoadedGame:
    game: chess.pgn.Game


class PgnEditor:
    """
    PGN tree editor.
    - Divergence always creates a new variation and makes it active.
    - Supports header edits.
    - Supports applying engine moves by UCI.
    """

    def __init__(self) -> None:
        self.session = GameSession()
        self.loaded: LoadedGame | None = None
        self.current_node: chess.pgn.GameNode | None = None
        self.dirty: bool = False
        self._pending_promo_prefix: str | None = None

    # ---------- lifecycle ----------

    def new_freeplay(self) -> None:
        self.session.reset()
        g = chess.pgn.Game()
        self.loaded = LoadedGame(game=g)
        self.current_node = g
        self.dirty = False
        self._pending_promo_prefix = None

    def load_pgn_text(self, pgn_text: str) -> None:
        pgn_io = io.StringIO(pgn_text)
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            raise ValueError("No game found in PGN text")

        self.loaded = LoadedGame(game=game)
        self.session.reset()

        # walk mainline and sync board to end? no: start at root
        self.current_node = game
        self.dirty = False
        self._pending_promo_prefix = None

    # ---------- headers ----------

    def headers(self) -> dict[str, str]:
        if not self.loaded:
            return {}
        return dict(self.loaded.game.headers)

    def set_header(self, key: str, value: str) -> None:
        if not self.loaded:
            self.new_freeplay()
        assert self.loaded is not None
        self.loaded.game.headers[key] = value
        self.dirty = True

    def add_header(self, key: str, value: str) -> None:
        self.set_header(key, value)

    # ---------- navigation ----------

    def step_back(self) -> bool:
        ok = self.session.undo()
        if not ok:
            return False
        if self.current_node and self.current_node.parent is not None:
            self.current_node = self.current_node.parent
        return True

    def step_forward_mainline(self) -> bool:
        if not self.current_node or not self.current_node.variations:
            return False
        nxt = self.current_node.variations[0]
        mv = nxt.move
        self.session.board.push(mv)
        self.session._undo.append(mv)  # keep undo consistent
        self.session._redo.clear()
        self.current_node = nxt
        return True

    # ---------- moves / variations ----------

    def _activate_or_create(self, mv: chess.Move) -> chess.pgn.GameNode:
        assert self.current_node is not None

        # Existing variation? promote to active.
        for i, v in enumerate(self.current_node.variations):
            if v.move == mv:
                if i != 0:
                    self.current_node.variations.insert(0, self.current_node.variations.pop(i))
                return self.current_node.variations[0]

        # Create new variation and make active.
        new = self.current_node.add_variation(mv)
        self.current_node.variations.remove(new)
        self.current_node.variations.insert(0, new)
        self.dirty = True
        return new

    def try_user_move(self, from_sq: chess.Square, to_sq: chess.Square) -> MoveResult:
        if self.current_node is None:
            self.new_freeplay()

        res = self.session.try_move(from_sq, to_sq)
        if res.promotion_required and res.promotion_uci_prefix:
            self._pending_promo_prefix = res.promotion_uci_prefix
            return res
        if not res.ok:
            return res

        mv = chess.Move.from_uci(res.uci)  # type: ignore[arg-type]
        self.current_node = self._activate_or_create(mv)
        return res

    def resolve_promotion(self, promo: str) -> MoveResult:
        if not self._pending_promo_prefix:
            return MoveResult(ok=False, reason="no_pending_promotion")
        res = self.session.try_promotion(self._pending_promo_prefix, promo)
        if not res.ok:
            return res
        mv = chess.Move.from_uci(res.uci)  # type: ignore[arg-type]
        assert self.current_node is not None
        self.current_node = self._activate_or_create(mv)
        self._pending_promo_prefix = None
        return res

    def apply_uci_move(self, uci: str) -> MoveResult:
        """
        Apply a move already chosen elsewhere (engine / click-to-step).
        Always treats it as a played move that can create a variation.
        """
        if self.current_node is None:
            self.new_freeplay()

        mv = chess.Move.from_uci(uci.strip())
        if mv not in self.session.board.legal_moves:
            return MoveResult(ok=False, reason="illegal")

        san = self.session.board.san(mv)
        self.session.board.push(mv)
        self.session._undo.append(mv)
        self.session._redo.clear()

        self.current_node = self._activate_or_create(mv)
        return MoveResult(ok=True, uci=mv.uci(), san=san)

    # ---------- export / helpers ----------

    def export_pgn(self) -> str:
        if not self.loaded:
            return str(chess.pgn.Game())
        return str(self.loaded.game)

    def current_san(self) -> str | None:
        if not self.current_node or self.current_node.move is None:
            return None
        try:
            return self.current_node.san()
        except Exception:
            try:
                return self.current_node.move.uci()
            except Exception:
                return None

    def played_prefix_uci(self) -> list[str]:
        return [m.uci() for m in self.session.board.move_stack]