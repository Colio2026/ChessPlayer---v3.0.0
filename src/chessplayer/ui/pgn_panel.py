from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.pgn_edit import PgnEditor


class _Token:
    """A single move token: char positions in PGN string + ply index."""
    __slots__ = ("start", "end", "ply")

    def __init__(self, start: int, end: int, ply: int) -> None:
        self.start = start
        self.end   = end
        self.ply   = ply


class PgnPanel(QWidget):
    """
    PGN display and annotation panel.

    Left-click a move  → navigate board to that position
    Right-click a move → context menu:
        • Navigate to this move
        • Insert / edit comment   (opens multiline dialog)
        • [Request Coach]         (stub, Phase F)
    Comments shown inline as amber italic { like this }
    Current move highlighted blue/bold
    """

    navigate_requested = Signal(int)   # emits ply (1-based)
    header_changed     = Signal()

    def __init__(self, editor: PgnEditor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._editor    = editor
        self._token_map: list[_Token] = []
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        toolbar   = QWidget()
        toolbar_l = QHBoxLayout(toolbar)
        toolbar_l.setContentsMargins(4, 2, 4, 2)

        self._edit_hdr_btn = QPushButton("Edit field")
        self._edit_hdr_btn.setToolTip("Edit an existing PGN header field")
        self._edit_hdr_btn.clicked.connect(self._edit_header)
        toolbar_l.addWidget(self._edit_hdr_btn)

        self._add_hdr_btn = QPushButton("Add field")
        self._add_hdr_btn.setToolTip("Add a new PGN header field")
        self._add_hdr_btn.clicked.connect(self._add_header)
        toolbar_l.addWidget(self._add_hdr_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setToolTip("Save game (Ctrl+S)")
        self._save_btn.setEnabled(False)   # enabled by window.py
        toolbar_l.addWidget(self._save_btn)

        self._save_as_btn = QPushButton("Save As")
        self._save_as_btn.setToolTip("Save game to a new file")
        self._save_as_btn.setEnabled(False)   # enabled by window.py
        toolbar_l.addWidget(self._save_as_btn)

        self._save_lib_btn = QPushButton("Save to Library")
        self._save_lib_btn.setToolTip(
            "Replace the original game in the source library and re-index"
        )
        self._save_lib_btn.setEnabled(False)   # enabled by window.py when source is known
        toolbar_l.addWidget(self._save_lib_btn)

        toolbar_l.addStretch(1)

        self._dirty_label = QLabel("")
        self._dirty_label.setStyleSheet("color: #FFA500; font-style: italic;")
        toolbar_l.addWidget(self._dirty_label)

        layout.addWidget(toolbar)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QTextEdit.WidgetWidth)
        self._text.setFont(QFont("Consolas", 10))
        self._text.setContextMenuPolicy(Qt.CustomContextMenu)
        self._text.customContextMenuRequested.connect(self._on_right_click)
        self._text.mousePressEvent = self._on_mouse_press
        layout.addWidget(self._text, 1)

    # ── public API ────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild PGN display. Call on every position change."""
        pgn = self._editor.export_pgn()
        self._text.setPlainText(pgn)
        self._token_map = self._build_token_map(pgn)
        self._apply_comment_formatting(pgn)
        self._highlight_current_move()
        self._update_dirty_indicator()
        self._update_save_buttons()

    def enable_save(self, save_slot, save_as_slot) -> None:
        """
        Called by window.py to wire up and enable the Save / Save As buttons.
        """
        self._save_btn.setEnabled(True)
        self._save_btn.clicked.connect(save_slot)
        self._save_as_btn.setEnabled(True)
        self._save_as_btn.clicked.connect(save_as_slot)

    def enable_save_to_library(self, slot) -> None:
        """Called by window.py when a library source is active."""
        self._save_lib_btn.setEnabled(True)
        try:
            self._save_lib_btn.clicked.disconnect()
        except RuntimeError:
            pass
        self._save_lib_btn.clicked.connect(slot)

    def disable_save_to_library(self) -> None:
        self._save_lib_btn.setEnabled(False)

    # ── token map ─────────────────────────────────────────────────────────────

    def _build_token_map(self, pgn: str) -> list[_Token]:
        tokens: list[_Token] = []
        move_text_start = self._find_move_text_start(pgn)
        ply = 1
        i   = move_text_start
        n   = len(pgn)

        while i < n:
            if pgn[i].isspace():
                i += 1
                continue

            token_start = i

            if pgn[i] == "{":
                end = pgn.find("}", i)
                i   = end + 1 if end != -1 else n
                continue

            while i < n and not pgn[i].isspace():
                i += 1
            token = pgn[token_start:i]

            if self._is_non_move_token(token):
                continue

            tokens.append(_Token(start=token_start, end=i, ply=ply))
            ply += 1

        return tokens

    @staticmethod
    def _find_move_text_start(pgn: str) -> int:
        char_pos     = 0
        past_headers = False
        for line in pgn.splitlines(keepends=True):
            stripped = line.strip()
            if stripped.startswith("["):
                past_headers = True
                char_pos    += len(line)
            elif past_headers and stripped == "":
                char_pos += len(line)
            elif past_headers:
                return char_pos
            else:
                char_pos += len(line)
        return char_pos

    @staticmethod
    def _is_non_move_token(token: str) -> bool:
        if not token:
            return True
        if token.rstrip(".").isdigit():
            return True
        if token.endswith("...") and token[:-3].isdigit():
            return True
        if token in ("1-0", "0-1", "1/2-1/2", "*"):
            return True
        if token.startswith("$") and token[1:].isdigit():
            return True
        return False

    def _token_at_char(self, char_pos: int) -> _Token | None:
        for tok in self._token_map:
            if tok.start <= char_pos < tok.end:
                return tok
        return None

    # ── mouse handling ────────────────────────────────────────────────────────

    def _on_mouse_press(self, event) -> None:
        QTextEdit.mousePressEvent(self._text, event)
        if event.button() != Qt.LeftButton:
            return
        cursor   = self._text.cursorForPosition(event.pos())
        char_pos = cursor.position()
        tok      = self._token_at_char(char_pos)
        if tok is not None:
            self.navigate_requested.emit(tok.ply)

    def _on_right_click(self, pos) -> None:
        cursor   = self._text.cursorForPosition(pos)
        char_pos = cursor.position()
        tok      = self._token_at_char(char_pos)

        menu = QMenu(self)

        if tok is not None:
            pgn        = self._editor.export_pgn()
            token_text = pgn[tok.start:tok.end]
            move_label = (
                f"move {(tok.ply + 1) // 2} "
                f"({'White' if tok.ply % 2 == 1 else 'Black'})  {token_text}"
            )

            nav_action = menu.addAction(f"Navigate → {token_text}")
            nav_action.triggered.connect(
                lambda checked=False, p=tok.ply: self.navigate_requested.emit(p)
            )

            menu.addSeparator()

            existing = self._editor.get_comment_at_ply(tok.ply)
            comment_label = "Edit comment" if existing else "Insert comment"
            comment_action = menu.addAction(f"{comment_label}  [{move_label}]")
            comment_action.triggered.connect(
                lambda checked=False, p=tok.ply: self._insert_comment_at(p)
            )

            menu.addSeparator()

        coach_action = menu.addAction("Request Coach Analysis  (coming Phase F)")
        coach_action.setEnabled(False)

        menu.exec(self._text.mapToGlobal(pos))

    # ── comment insertion ─────────────────────────────────────────────────────

    def _insert_comment_at(self, ply: int) -> None:
        existing = self._editor.get_comment_at_ply(ply)
        move_num = (ply + 1) // 2
        side     = "White" if ply % 2 == 1 else "Black"

        text, ok = QInputDialog.getMultiLineText(
            self,
            "Comment",
            f"Comment for move {move_num} ({side}):",
            existing,
        )
        if not ok:
            return
        self._editor.insert_comment_at_ply(ply, text.strip())
        self.refresh()

    # ── formatting ────────────────────────────────────────────────────────────

    def _highlight_current_move(self) -> None:
        current_ply = len(self._editor.session.board.move_stack)
        if current_ply == 0 or not self._token_map:
            return

        tok = next((t for t in self._token_map if t.ply == current_ply), None)
        if tok is None:
            return

        # Clear all formatting first, then re-apply comments, then highlight
        cur = self._text.textCursor()
        cur.select(QTextCursor.Document)
        cur.setCharFormat(QTextCharFormat())
        cur.clearSelection()

        self._apply_comment_formatting(self._text.toPlainText())

        cur.setPosition(tok.start)
        cur.setPosition(tok.end, QTextCursor.KeepAnchor)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#0A84FF"))
        fmt.setFontWeight(QFont.Bold)
        fmt.setBackground(QColor("#1A2A3A"))
        cur.mergeCharFormat(fmt)
        self._text.setTextCursor(cur)
        self._text.ensureCursorVisible()

    def _apply_comment_formatting(self, pgn: str) -> None:
        """Render { comment } blocks in amber italic."""
        i = 0
        n = len(pgn)
        while i < n:
            if pgn[i] == "{":
                start = i
                end   = pgn.find("}", i)
                if end == -1:
                    break
                end += 1
                cur = self._text.textCursor()
                cur.setPosition(start)
                cur.setPosition(end, QTextCursor.KeepAnchor)
                fmt = QTextCharFormat()
                fmt.setForeground(QColor("#D4A017"))
                fmt.setFontItalic(True)
                cur.mergeCharFormat(fmt)
                i = end
            else:
                i += 1

    def _update_dirty_indicator(self) -> None:
        self._dirty_label.setText("● unsaved" if self._editor.dirty else "")

    def _update_save_buttons(self) -> None:
        # Save to Library only makes sense if a source file is tracked
        has_source = (
            self._editor.source_pgn_path is not None
            and self._editor.source_offset is not None
        )
        self._save_lib_btn.setEnabled(has_source)

    # ── header editing ────────────────────────────────────────────────────────

    def _edit_header(self) -> None:
        hdrs = self._editor.headers()
        if not hdrs:
            self._editor.new_freeplay()
            hdrs = self._editor.headers()
        keys    = sorted(hdrs.keys())
        key, ok = QInputDialog.getItem(
            self, "Edit PGN field", "Field:", keys, 0, False
        )
        if not ok or not key:
            return
        val, ok2 = QInputDialog.getText(
            self, "Edit PGN field", f"Value for {key}:",
            text=str(hdrs.get(key, ""))
        )
        if not ok2:
            return
        self._editor.set_header(key, val)
        self.header_changed.emit()
        self.refresh()

    def _add_header(self) -> None:
        key, ok = QInputDialog.getText(self, "Add PGN field", "Field name:")
        if not ok or not key:
            return
        val, ok2 = QInputDialog.getText(
            self, "Add PGN field", f"Value for {key}:"
        )
        if not ok2:
            return
        self._editor.add_header(key, val)
        self.header_changed.emit()
        self.refresh()
