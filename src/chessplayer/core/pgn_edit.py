from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
    - Supports header edits, annotations, save to file, save to library.
    - Supports applying engine moves by UCI.
    """

    def __init__(self) -> None:
        self.session  = GameSession()
        self.loaded:  LoadedGame | None = None
        self.current_node: chess.pgn.GameNode | None = None
        self.dirty:   bool = False
        self._pending_promo_prefix: str | None = None

        # Tracks which library file + byte offset this game was loaded from.
        # Set by window.py after opening a game from the store.
        self.source_pgn_path:   Path | None = None
        self.source_offset:     int  | None = None

    # lifecycle

    def new_freeplay(self) -> None:
        self.session.reset()
        g = chess.pgn.Game()
        self.loaded       = LoadedGame(game=g)
        self.current_node = g
        self.dirty        = False
        self.source_pgn_path  = None
        self.source_offset    = None
        self._pending_promo_prefix = None

    def load_pgn_text(self, pgn_text: str) -> None:
        pgn_io = io.StringIO(pgn_text)
        game   = chess.pgn.read_game(pgn_io)
        if game is None:
            raise ValueError("No game found in PGN text")
        self.loaded       = LoadedGame(game=game)
        self.session.reset()
        self.current_node = game
        self.dirty        = False
        self._pending_promo_prefix = None

    # headers

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

    # navigation

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
        mv  = nxt.move
        self.session.board.push(mv)
        self.session._undo.append(mv)
        self.session._redo.clear()
        self.current_node = nxt
        return True

    def navigate_to_ply(self, ply: int) -> bool:
        """Jump to a specific ply (1-based) on the mainline. ply=0 = start."""
        if not self.loaded:
            return False
        self.session.reset()
        self.current_node = self.loaded.game
        for _ in range(ply):
            if not self.current_node.variations:
                break
            nxt = self.current_node.variations[0]
            self.session.board.push(nxt.move)
            self.session._undo.append(nxt.move)
            self.session._redo.clear()
            self.current_node = nxt
        return True

    # annotations

    def get_comment_at_ply(self, ply: int) -> str:
        """Return existing comment at ply (1-based), or empty string."""
        if not self.loaded:
            return ""
        node = self.loaded.game
        for _ in range(ply):
            if not node.variations:
                return ""
            node = node.variations[0]
        return node.comment or ""

    def insert_comment(self, comment_text: str) -> None:
        """Insert or replace comment on current_node. Called by coach narrator."""
        if self.current_node is None:
            return
        self.current_node.comment = comment_text.strip()
        self.dirty = True

    def insert_comment_at_ply(self, ply: int, comment_text: str) -> None:
        """Insert or replace comment at a specific mainline ply."""
        if not self.loaded:
            return
        node = self.loaded.game
        for _ in range(ply):
            if not node.variations:
                return
            node = node.variations[0]
        node.comment = comment_text.strip()
        self.dirty   = True

    # moves / variations

    def _activate_or_create(self, mv: chess.Move) -> chess.pgn.GameNode:
        assert self.current_node is not None
        for i, v in enumerate(self.current_node.variations):
            if v.move == mv:
                if i != 0:
                    self.current_node.variations.insert(
                        0, self.current_node.variations.pop(i)
                    )
                return self.current_node.variations[0]
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

    # export / save

    def export_pgn(self) -> str:
        if not self.loaded:
            return str(chess.pgn.Game())
        return str(self.loaded.game)

    def export_pgn_to_file(self, path: Path) -> None:
        """Write current game PGN to a standalone file. Marks game clean."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.export_pgn(), encoding="utf-8")
        self.dirty = False

    def replace_in_library_file(self, store=None, source_id: int | None = None) -> bool:
        """
        Replace the original game in the source library PGN file with the
        current annotated version. Updates downstream byte offsets in the
        SQLite index directly — no re-index required.
        """
        if self.source_pgn_path is None or self.source_offset is None:
            raise ValueError(
                "No source file tracked for this game. "
                "Use Save As to save to a new file instead."
            )

        src_path = Path(self.source_pgn_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Source library not found: {src_path}")

        raw = src_path.read_bytes()

        # Decode slice from byte offset so python-chess can find the game end
        slice_text    = raw[self.source_offset:].decode("utf-8", errors="replace")
        buf           = io.StringIO(slice_text)
        original_game = chess.pgn.read_game(buf)
        if original_game is None:
            raise RuntimeError(
                f"Could not re-parse original game at byte offset {self.source_offset}"
            )

        chars_consumed    = buf.tell()
        bytes_consumed    = len(slice_text[:chars_consumed].encode("utf-8", errors="replace"))
        original_end_byte = self.source_offset + bytes_consumed

        replacement_bytes = (self.export_pgn() + "\n\n").encode("utf-8")
        delta             = len(replacement_bytes) - bytes_consumed

        new_raw = raw[: self.source_offset] + replacement_bytes + raw[original_end_byte:]

        # Atomic write
        tmp_fd, tmp_path = tempfile.mkstemp(dir=src_path.parent, suffix=".pgn.tmp")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(new_raw)
            os.replace(tmp_path, src_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        # Update downstream offsets in the index — no re-index needed
        if delta != 0 and store is not None and source_id is not None:
            conn = store._connect()
            try:
                conn.execute(
                    """UPDATE games
                          SET offset_bytes = offset_bytes + ?
                        WHERE source_id = ? AND offset_bytes > ?""",
                    (delta, source_id, self.source_offset),
                )
                conn.commit()
            finally:
                conn.close()

        self.dirty = False
        return True

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
