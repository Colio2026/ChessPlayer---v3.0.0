from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import chess.pgn

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.pgn_edit import PgnEditor

# ── Colours ───────────────────────────────────────────────────────────────────
_COL_HIGHLIGHT  = QColor("#1A3A5C")   # current move — dark blue
_COL_COMMENT_FG = QColor("#D4A017")   # amber for comments
_COL_HEADER_BG  = QColor("#1E1E1E")   # header table bg
_FONT_MONO      = QFont("Consolas", 10)
_FONT_COMMENT   = QFont("Consolas", 9)
_FONT_COMMENT.setItalic(True)

# UserRole data keys stored on QTableWidgetItem
_PLY_ROLE  = Qt.UserRole        # int ply (1-based), None for comment rows
_KIND_ROLE = Qt.UserRole + 1    # "move" | "comment"


# ── Internal row descriptors ──────────────────────────────────────────────────

@dataclass
class _MoveRow:
    move_num:   int
    white_san:  Optional[str]
    white_ply:  Optional[int]
    black_san:  Optional[str]
    black_ply:  Optional[int]


@dataclass
class _CommentRow:
    ply:     int       # which move this comment belongs to
    text:    str
    side:    str       # "White" | "Black"  — cosmetic only


# ── Panel ─────────────────────────────────────────────────────────────────────

class MoveNotationPanel(QWidget):
    """
    Score-card style move notation panel.

    Layout (top to bottom):
      ┌──────────────────────────────────────┐
      │  TOOLBAR  (save buttons, dirty dot)  │
      ├──────────────────────────────────────┤
      │  HEADER TABLE  (Field | Value)       │
      │  right-click → edit / add field      │
      ├──────────────────────────────────────┤
      │  MOVE TABLE   (#  | White | Black)   │
      │   comment sub-rows in amber italic   │
      └──────────────────────────────────────┘

    Signals:
        navigate_requested(int)  — ply (1-based) user wants to jump to
        header_changed()         — after any header edit
    """

    navigate_requested = Signal(int)
    header_changed     = Signal()

    def __init__(self, editor: PgnEditor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._editor = editor
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar   = QWidget()
        toolbar_l = QHBoxLayout(toolbar)
        toolbar_l.setContentsMargins(4, 3, 4, 3)
        toolbar_l.setSpacing(6)

        self._save_btn = QPushButton("Save")
        self._save_btn.setEnabled(False)
        toolbar_l.addWidget(self._save_btn)

        self._save_as_btn = QPushButton("Save As")
        self._save_as_btn.setEnabled(False)
        toolbar_l.addWidget(self._save_as_btn)

        self._save_lib_btn = QPushButton("Save to Library")
        self._save_lib_btn.setEnabled(False)
        toolbar_l.addWidget(self._save_lib_btn)

        toolbar_l.addStretch(1)

        self._dirty_label = QLabel("")
        self._dirty_label.setStyleSheet("color: #FFA500; font-style: italic;")
        toolbar_l.addWidget(self._dirty_label)

        layout.addWidget(toolbar)

        # Splitter: header table on top, move table below
        splitter = QSplitter(Qt.Vertical)

        # Header table
        self._header_table = QTableWidget()
        self._header_table.setColumnCount(2)
        self._header_table.setHorizontalHeaderLabels(["Field", "Value"])
        self._header_table.horizontalHeader().setStretchLastSection(True)
        self._header_table.verticalHeader().setVisible(False)
        self._header_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._header_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._header_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._header_table.setFont(_FONT_MONO)
        self._header_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._header_table.customContextMenuRequested.connect(
            self._on_header_right_click
        )
        self._header_table.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Minimum
        )
        splitter.addWidget(self._header_table)

        # Move table
        self._move_table = QTableWidget()
        self._move_table.setColumnCount(3)
        self._move_table.setHorizontalHeaderLabels(["#", "White", "Black"])
        self._move_table.horizontalHeader().setStretchLastSection(True)
        self._move_table.horizontalHeader().setDefaultSectionSize(120)
        self._move_table.setColumnWidth(0, 40)
        self._move_table.verticalHeader().setVisible(False)
        self._move_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._move_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._move_table.setFont(_FONT_MONO)
        self._move_table.setShowGrid(False)
        self._move_table.setAlternatingRowColors(True)
        self._move_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._move_table.customContextMenuRequested.connect(
            self._on_move_right_click
        )
        self._move_table.cellClicked.connect(self._on_cell_clicked)
        splitter.addWidget(self._move_table)

        splitter.setSizes([160, 600])
        layout.addWidget(splitter, 1)

    # ── public API ────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild both tables. Call on every position change."""
        self._rebuild_header_table()
        self._rebuild_move_table()
        self._dirty_label.setText("● unsaved" if self._editor.dirty else "")
        self._update_save_lib_btn()

    def enable_save(self, save_slot, save_as_slot) -> None:
        self._save_btn.setEnabled(True)
        self._save_btn.clicked.connect(save_slot)
        self._save_as_btn.setEnabled(True)
        self._save_as_btn.clicked.connect(save_as_slot)

    def enable_save_to_library(self, slot) -> None:
        self._save_lib_btn.setEnabled(True)
        try:
            self._save_lib_btn.clicked.disconnect()
        except RuntimeError:
            pass
        self._save_lib_btn.clicked.connect(slot)

    def disable_save_to_library(self) -> None:
        self._save_lib_btn.setEnabled(False)

    # ── header table ──────────────────────────────────────────────────────────

    def _rebuild_header_table(self) -> None:
        headers = self._editor.headers()
        self._header_table.setRowCount(0)

        # Priority fields shown first, rest alphabetically
        priority = ["Event", "Site", "Date", "Round",
                    "White", "Black", "Result", "WhiteElo", "BlackElo", "ECO"]
        keys = [k for k in priority if k in headers]
        keys += sorted(k for k in headers if k not in priority)

        self._header_table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            key_item = QTableWidgetItem(key)
            key_item.setFont(_FONT_MONO)
            key_item.setForeground(QColor("#888888"))
            val_item = QTableWidgetItem(str(headers.get(key, "")))
            val_item.setFont(_FONT_MONO)
            self._header_table.setItem(row, 0, key_item)
            self._header_table.setItem(row, 1, val_item)

        self._header_table.resizeRowsToContents()
        # Resize the widget to fit all rows exactly
        total_h = self._header_table.horizontalHeader().height()
        for r in range(self._header_table.rowCount()):
            total_h += self._header_table.rowHeight(r)
        self._header_table.setMaximumHeight(total_h + 4)

    def _on_header_right_click(self, pos) -> None:
        item = self._header_table.itemAt(pos)
        menu = QMenu(self)

        if item is not None:
            row      = self._header_table.row(item)
            key_item = self._header_table.item(row, 0)
            key      = key_item.text() if key_item else ""

            edit_action = menu.addAction(f"Edit  \"{key}\"")
            edit_action.triggered.connect(
                lambda checked=False, k=key: self._edit_header_field(k)
            )
            menu.addSeparator()

        add_action = menu.addAction("Add field")
        add_action.triggered.connect(self._add_header_field)
        menu.exec(self._header_table.mapToGlobal(pos))

    def _edit_header_field(self, key: str) -> None:
        current = self._editor.headers().get(key, "")
        val, ok = QInputDialog.getText(
            self, "Edit field", f"{key}:", text=current
        )
        if not ok:
            return
        self._editor.set_header(key, val)
        self.header_changed.emit()
        self.refresh()

    def _add_header_field(self) -> None:
        key, ok = QInputDialog.getText(self, "Add field", "Field name:")
        if not ok or not key.strip():
            return
        val, ok2 = QInputDialog.getText(self, "Add field", f"Value for {key}:")
        if not ok2:
            return
        self._editor.add_header(key.strip(), val)
        self.header_changed.emit()
        self.refresh()

    # ── move table ────────────────────────────────────────────────────────────

    def _build_display_rows(self) -> list[_MoveRow | _CommentRow]:
        """Walk the mainline and produce a flat list of display rows."""
        if not self._editor.loaded:
            return []

        # Collect mainline nodes
        mainline: list[chess.pgn.GameNode] = []
        node = self._editor.loaded.game
        while node.variations:
            node = node.variations[0]
            mainline.append(node)

        rows: list[_MoveRow | _CommentRow] = []

        for i in range(0, len(mainline), 2):
            white_node = mainline[i]
            black_node = mainline[i + 1] if i + 1 < len(mainline) else None
            move_num   = i // 2 + 1
            white_ply  = i + 1
            black_ply  = i + 2 if black_node else None

            try:
                white_san = white_node.san()
            except Exception:
                white_san = white_node.move.uci() if white_node.move else "?"

            black_san = None
            if black_node:
                try:
                    black_san = black_node.san()
                except Exception:
                    black_san = black_node.move.uci() if black_node.move else "?"

            rows.append(_MoveRow(
                move_num=move_num,
                white_san=white_san, white_ply=white_ply,
                black_san=black_san, black_ply=black_ply,
            ))

            # Comment sub-rows immediately after the move row
            if white_node.comment and white_node.comment.strip():
                rows.append(_CommentRow(
                    ply=white_ply,
                    text=white_node.comment.strip(),
                    side="White",
                ))
            if black_node and black_node.comment and black_node.comment.strip():
                rows.append(_CommentRow(
                    ply=black_ply,
                    text=black_node.comment.strip(),
                    side="Black",
                ))

        return rows

    def _rebuild_move_table(self) -> None:
        current_ply = len(self._editor.session.board.move_stack)
        display_rows = self._build_display_rows()

        self._move_table.setRowCount(0)
        self._move_table.setRowCount(len(display_rows))

        scroll_to_row: int | None = None

        for table_row, dr in enumerate(display_rows):
            if isinstance(dr, _MoveRow):
                # Column 0: move number
                num_item = QTableWidgetItem(str(dr.move_num))
                num_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
                num_item.setForeground(QColor("#666666"))
                num_item.setData(_KIND_ROLE, "move")
                num_item.setData(_PLY_ROLE, None)
                self._move_table.setItem(table_row, 0, num_item)

                # Column 1: White move
                w_item = QTableWidgetItem(dr.white_san or "")
                w_item.setData(_PLY_ROLE, dr.white_ply)
                w_item.setData(_KIND_ROLE, "move")
                if dr.white_ply and dr.white_ply == current_ply:
                    w_item.setBackground(_COL_HIGHLIGHT)
                    scroll_to_row = table_row
                self._move_table.setItem(table_row, 1, w_item)

                # Column 2: Black move
                b_item = QTableWidgetItem(dr.black_san or "")
                b_item.setData(_PLY_ROLE, dr.black_ply)
                b_item.setData(_KIND_ROLE, "move")
                if dr.black_ply and dr.black_ply == current_ply:
                    b_item.setBackground(_COL_HIGHLIGHT)
                    scroll_to_row = table_row
                self._move_table.setItem(table_row, 2, b_item)

            elif isinstance(dr, _CommentRow):
                # Span all 3 columns for the comment
                self._move_table.setSpan(table_row, 0, 1, 3)
                side_prefix = "  ♟" if dr.side == "Black" else "  ♙"
                c_item = QTableWidgetItem(f"{side_prefix}  {dr.text}")
                c_item.setFont(_FONT_COMMENT)
                c_item.setForeground(_COL_COMMENT_FG)
                c_item.setData(_PLY_ROLE, dr.ply)
                c_item.setData(_KIND_ROLE, "comment")
                self._move_table.setItem(table_row, 0, c_item)

        self._move_table.resizeRowsToContents()

        # Scroll to keep current move visible
        if scroll_to_row is not None:
            self._move_table.scrollToItem(
                self._move_table.item(scroll_to_row, 1),
                QAbstractItemView.PositionAtCenter,
            )

    # ── move table interactions ───────────────────────────────────────────────

    def _on_cell_clicked(self, row: int, col: int) -> None:
        item = self._move_table.item(row, col)
        if item is None:
            return
        kind = item.data(_KIND_ROLE)
        ply  = item.data(_PLY_ROLE)
        if kind == "move" and ply is not None:
            self.navigate_requested.emit(ply)

    def _on_move_right_click(self, pos) -> None:
        item = self._move_table.itemAt(pos)
        if item is None:
            return

        kind = item.data(_KIND_ROLE)
        ply  = item.data(_PLY_ROLE)
        if ply is None:
            return

        move_num = (ply + 1) // 2
        side     = "White" if ply % 2 == 1 else "Black"

        # Get the SAN for the menu label
        san = ""
        if kind == "move":
            san = item.text()
        label = f"move {move_num} ({side})  {san}".strip()

        menu = QMenu(self)

        nav_action = menu.addAction(f"Navigate → {san or label}")
        nav_action.triggered.connect(
            lambda checked=False, p=ply: self.navigate_requested.emit(p)
        )

        menu.addSeparator()

        existing = self._editor.get_comment_at_ply(ply)
        comment_label = "Edit comment" if existing else "Insert comment"
        comment_action = menu.addAction(f"{comment_label}  [{label}]")
        comment_action.triggered.connect(
            lambda checked=False, p=ply, s=side: self._insert_comment_at(p, s)
        )

        menu.addSeparator()

        coach_stub = menu.addAction("Request Coach Analysis  (coming Phase F)")
        coach_stub.setEnabled(False)

        menu.exec(self._move_table.mapToGlobal(pos))

    def _insert_comment_at(self, ply: int, side: str) -> None:
        move_num = (ply + 1) // 2
        existing = self._editor.get_comment_at_ply(ply)
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

    # ── save button state ─────────────────────────────────────────────────────

    def _update_save_lib_btn(self) -> None:
        has_source = (
            self._editor.source_pgn_path is not None
            and self._editor.source_offset is not None
        )
        self._save_lib_btn.setEnabled(has_source)
