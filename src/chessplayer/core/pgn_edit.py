from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from chessplayer.core.game_session import GameSession, MoveResult


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

    def uci_to_san(self, base_ply: int, uci_list: list[str]) -> list[str]:
        """
        Convert a list of UCI strings to SAN notation, replaying from base_ply
        on the mainline.  Pure read — does NOT touch current_node or session.
        Returns SAN strings; falls back to raw UCI if a move is illegal.
        """
        if not self.loaded:
            return list(uci_list)
        # Build a throw-away board at base_ply
        board = chess.Board()
        node  = self.loaded.game
        for _ in range(base_ply):
            if not node.variations:
                break
            node = node.variations[0]
            board.push(node.move)
        result: list[str] = []
        for uci in uci_list:
            try:
                mv = chess.Move.from_uci(uci)
                if mv not in board.legal_moves:
                    result.append(uci)
                    break
                result.append(board.san(mv))
                board.push(mv)
            except Exception:
                result.append(uci)
                break
        return result

    def san_to_uci(self, base_ply: int, san_tokens: list[str]) -> tuple[list[str], str]:
        """
        Convert a list of SAN tokens to UCI strings, replaying from base_ply
        on the mainline.  Pure read — does NOT touch current_node or session.
        Returns (uci_list, error_message).  error_message is empty on full success.
        Stops at the first illegal/unrecognised token and reports it.
        """
        if not self.loaded:
            return [], "No game loaded."
        board = chess.Board()
        node  = self.loaded.game
        for _ in range(base_ply):
            if not node.variations:
                break
            node = node.variations[0]
            board.push(node.move)
        result: list[str] = []
        for token in san_tokens:
            token = token.strip()
            if not token:
                continue
            try:
                mv = board.parse_san(token)
            except Exception:
                return result, f'Unrecognised move "{token}" at position after {base_ply} moves.'
            result.append(mv.uci())
            board.push(mv)
        return result, ""


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

    def navigate_to_node(self, node: "chess.pgn.GameNode") -> bool:
        """
        Navigate to any node in the tree (mainline or variation).
        Walks up through parents to build the full move path, resets the board,
        then replays each move WITHOUT promoting any variation to mainline.
        """
        if not self.loaded:
            return False

        # Build path from node back to root
        path: list[chess.Move] = []
        n = node
        while n.move is not None:
            path.append(n.move)
            n = n.parent  # type: ignore[assignment]
        path.reverse()

        self.session.reset()
        self.current_node = self.loaded.game

        for mv in path:
            matched = None
            for v in self.current_node.variations:
                if v.move == mv:
                    matched = v
                    break
            if matched is None:
                return False  # tree mismatch — bail without corrupting state
            self.session.board.push(mv)
            self.session._undo.append(mv)
            self.session._redo.clear()
            self.current_node = matched  # never promotes; just walks to the node

        return True

    def promote_variation(self, node: "chess.pgn.GameNode") -> bool:
        """
        Move this variation one slot earlier in its parent's variation list.
        When it reaches index 0 it becomes the mainline.
        """
        if not self.loaded or node.parent is None or node.move is None:
            return False
        parent = node.parent
        try:
            idx = parent.variations.index(node)
        except ValueError:
            return False
        if idx == 0:
            return False  # already mainline
        parent.variations[idx - 1], parent.variations[idx] = (
            parent.variations[idx],
            parent.variations[idx - 1],
        )
        self.dirty = True
        return True

    def demote_variation(self, node: "chess.pgn.GameNode") -> bool:
        """
        Move this variation one slot later in its parent's variation list.
        When the mainline (index 0) is demoted it becomes a variation.
        """
        if not self.loaded or node.parent is None or node.move is None:
            return False
        parent = node.parent
        try:
            idx = parent.variations.index(node)
        except ValueError:
            return False
        if idx >= len(parent.variations) - 1:
            return False  # already last
        parent.variations[idx], parent.variations[idx + 1] = (
            parent.variations[idx + 1],
            parent.variations[idx],
        )
        self.dirty = True
        return True

    def delete_variation(self, node: "chess.pgn.GameNode") -> bool:
        """
        Remove an entire variation subtree rooted at `node` from its parent.
        Works on any node in the tree — mainline or variation.
        If node is at index 0 (the mainline continuation) it is still removed;
        the next sibling (if any) becomes the new mainline.
        After deletion, current_node is reset to the nearest safe ancestor.
        """
        if not self.loaded or node.parent is None or node.move is None:
            return False
        parent = node.parent
        try:
            parent.variations.remove(node)
        except ValueError:
            return False
        # If we were sitting on or below the deleted node, retreat to parent
        n = self.current_node
        while n is not None and n.move is not None:
            if n is node:
                # navigate to parent position
                self.navigate_to_node(parent) if parent.move is not None else self._reset_to_root()
                break
            n = n.parent  # type: ignore[assignment]
        self.dirty = True
        return True

    def delete_from_node(self, node: "chess.pgn.GameNode") -> bool:
        """
        Delete `node` and every move after it in its line, but keep the moves
        before it intact.  Equivalent to truncating the game at this point.
        Works on mainline and variation nodes at any nesting depth.
        """
        if not self.loaded or node.move is None:
            return False
        if node.parent is None:
            # Deleting from the very first move — wipe all variations from root
            self.loaded.game.variations.clear()
            self._reset_to_root()
            self.dirty = True
            return True
        parent = node.parent
        try:
            idx = parent.variations.index(node)
        except ValueError:
            return False
        # Drop this node and everything after it in the parent's variations list
        # (keep siblings before idx — they are other branches, not continuations)
        # Actually we only want to remove THIS node (and implicitly its subtree);
        # other siblings at the same level are unrelated branches and must stay.
        parent.variations.remove(node)
        # After deletion, navigate to parent if we were on or below node
        n = self.current_node
        while n is not None and n.move is not None:
            if n is node:
                if parent.move is not None:
                    self.navigate_to_node(parent)
                else:
                    self._reset_to_root()
                break
            n = n.parent  # type: ignore[assignment]
        self.dirty = True
        return True

    def _reset_to_root(self) -> None:
        """Navigate to the root (before move 1)."""
        self.session.reset()
        self.current_node = self.loaded.game  # type: ignore[union-attr]

    def _activate_or_create(self, mv: chess.Move) -> chess.pgn.GameNode:
        """Find or create a child node for mv and PROMOTE it to index 0 (user move)."""
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

    def _find_or_create_no_promote(self, mv: chess.Move) -> chess.pgn.GameNode:
        """Find or create a child node for mv WITHOUT reordering existing variations."""
        assert self.current_node is not None
        for v in self.current_node.variations:
            if v.move == mv:
                return v
        new = self.current_node.add_variation(mv)
        self.dirty = True
        return new

    def _is_on_mainline(self) -> bool:
        """Return True if current_node sits on the mainline (always variations[0])."""
        node = self.current_node
        while node is not None and node.parent is not None:
            if node.parent.variations[0] is not node:
                return False
            node = node.parent
        return True

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
        # Only promote to mainline when already on the mainline;
        # inside a variation just append so we don't displace the existing continuation.
        if self._is_on_mainline():
            self.current_node = self._activate_or_create(mv)
        else:
            self.current_node = self._find_or_create_no_promote(mv)
        return res

    def resolve_promotion(self, promo: str) -> MoveResult:
        if not self._pending_promo_prefix:
            return MoveResult(ok=False, reason="no_pending_promotion")
        res = self.session.try_promotion(self._pending_promo_prefix, promo)
        if not res.ok:
            return res
        mv = chess.Move.from_uci(res.uci)  # type: ignore[arg-type]
        assert self.current_node is not None
        if self._is_on_mainline():
            self.current_node = self._activate_or_create(mv)
        else:
            self.current_node = self._find_or_create_no_promote(mv)
        self._pending_promo_prefix = None
        return res

    def apply_uci_move(self, uci: str, promote: bool = True) -> MoveResult:
        """
        Apply a UCI move to the tree.
        promote=True  → used for user moves; found variations are promoted to mainline.
        promote=False → used when adding engine PV lines; existing order is preserved.
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
        if promote:
            self.current_node = self._activate_or_create(mv)
        else:
            self.current_node = self._find_or_create_no_promote(mv)
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

    def replace_in_library_file(self) -> bool:
        """
        Replace the original game in the source library PGN file with the
        current annotated version. Rewrites the file atomically via temp file.
        Requires source_pgn_path and source_offset to be set.

        NOTE: source_offset is a BYTE offset from the SQLite index.
        We work in bytes throughout to avoid byte/char mismatch on files
        that contain non-ASCII characters (accented player names etc.).
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

        # Decode only the slice from the byte offset so python-chess can parse
        # the original game and tell us how many CHARACTERS it consumed.
        slice_text = raw[self.source_offset:].decode("utf-8", errors="replace")
        buf        = io.StringIO(slice_text)
        original_game = chess.pgn.read_game(buf)
        if original_game is None:
            raise RuntimeError(
                f"Could not re-parse original game at byte offset {self.source_offset}"
            )

        # Convert the character-count consumed back to a byte count.
        # re-encode the consumed slice to get its byte length exactly.
        chars_consumed    = buf.tell()
        bytes_consumed    = len(slice_text[:chars_consumed].encode("utf-8", errors="replace"))
        original_end_byte = self.source_offset + bytes_consumed

        replacement_bytes = (self.export_pgn() + "\n\n").encode("utf-8")
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

        self.dirty = False
        return True

    # helpers

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
