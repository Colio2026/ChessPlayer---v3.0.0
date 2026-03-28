from __future__ import annotations

import html as _html
import re
from typing import Optional

import chess
import chess.pgn

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont
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
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from chessplayer.core.pgn_edit import PgnEditor
from chessplayer.ui.comment_dialog import CommentDialog


class PgnPanel(QWidget):
    """
    PGN viewer with collapsible variations.

    Rendered as HTML inside a QTextBrowser.
    - Mainline moves are clickable links  (href="nav:PLY")
    - Each variation has a ▶/▼ toggle    (href="toggle:VAR_ID")
    - Collapsed → ▶ ( ... )
    - Expanded  → ▼ ( full moves in grey italic )
    - Comments rendered amber italic inline
    - Current move highlighted blue/bold
    - Right-click mainline move → navigate / insert comment / coach stub
    """

    navigate_requested          = Signal(int)     # ply (1-based) — mainline
    navigate_node_requested     = Signal(object)  # chess.pgn.GameNode — variation
    promote_variation_requested = Signal(object)  # GameNode to promote
    demote_variation_requested  = Signal(object)  # GameNode to demote
    delete_variation_requested  = Signal(object)  # GameNode: remove whole branch
    delete_from_node_requested  = Signal(object)  # GameNode: truncate from here
    comment_line_clicked        = Signal(int, list)  # (base_ply, uci_list)
    coach_help_requested        = Signal()           # right-click → request coach note
    header_changed              = Signal()

    # ── CSS ───────────────────────────────────────────────────────────────────
    _CSS = """
        body  { background:#161616; color:#C8C8C8;
                font-family:'Segoe UI','SF Pro Text',Arial,sans-serif;
                font-size:10.5pt; margin:10px 12px; line-height:2.0; }
        a     { text-decoration:none; }

        /* ── moves ───────────────────────────────────── */
        /* No background on move tokens — clean, flat, Apple-esque */
        .mp         { display:inline-block;
                      padding:0px 3px;
                      margin:0 1px;
                      white-space:nowrap; }
        .mp:hover   { color:#FFFFFF; }

        /* Move number — barely-there, 2pt smaller */
        .mn  { color:#363636; font-size:8.5pt; padding-right:1px; }

        /* Normal move */
        .mv  { color:#C8C8C8; }

        /* Current move — no box, just bright blue + bold */
        .mp.cur-pill { background:none; }
        .cur { color:#4DB8FF; font-weight:700; }

        /* ── comments ───────────────────────────────── */
        .cmt  { color:#C89520; font-style:italic; }

        /* ── variations — 1.5pt smaller, recessed ───── */
        .var  { color:#5A5A5A; font-style:italic; font-size:9pt; }
        .tog  { color:#3A9E50; font-size:9pt; }
        .tog:hover { color:#5CC870; }

        /* ── header table ────────────────────────────── */
        .htbl { width:100%; border-collapse:collapse;
                border:1px solid #282828;
                border-radius:6px;
                margin-bottom:12px;
                font-size:9pt; }
        .htbl td          { padding:5px 12px;
                            border-bottom:1px solid #222222; }
        .htbl tr:last-child td { border-bottom:none; }
        .htbl tr:nth-child(odd)  td { background:#1A1A1A; }
        .htbl tr:nth-child(even) td { background:#1E1E1E; }
        .hfkey { color:#484848; width:28%; white-space:nowrap; }
        .hfval { color:#C0C0C0; }
        .htitle-row td    { background:#1E1E1E !important;
                            padding:10px 12px 8px 12px;
                            border-bottom:1px solid #2A2A2A; }
        .htitle           { color:#EEEEEE; font-size:11pt; font-weight:600; }
        .hresult-1-0      { color:#6EC07A; font-weight:bold; }
        .hresult-0-1      { color:#E06C6C; font-weight:bold; }
        .hresult-draw     { color:#C8A96E; font-weight:bold; }
        .hresult-other    { color:#888888; font-weight:bold; }
    """

    def __init__(self, editor: PgnEditor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._editor    = editor
        self._collapsed: set[str] = set()   # var_ids that are collapsed
        # rebuilt each render: var_id -> chess.pgn.GameNode (first node of variation)
        self._var_map:   dict[str, chess.pgn.GameNode] = {}
        self._var_counter = 0
        # nav_id -> ply for right-click (mainline only)
        self._ply_map:      dict[str, int]    = {}
        self._main_node_map: dict[str, object] = {}   # nav_id -> GameNode for mainline moves
        self._node_map:      dict[str, object] = {}   # vn_id  -> GameNode for variation moves
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        toolbar   = QWidget()
        toolbar_l = QHBoxLayout(toolbar)
        toolbar_l.setContentsMargins(4, 2, 4, 2)
        toolbar_l.setSpacing(6)

        self._edit_hdr_btn = QPushButton("Edit field")
        self._edit_hdr_btn.clicked.connect(self._edit_header)
        toolbar_l.addWidget(self._edit_hdr_btn)

        self._add_hdr_btn = QPushButton("Add field")
        self._add_hdr_btn.clicked.connect(self._add_header)
        toolbar_l.addWidget(self._add_hdr_btn)

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
        self._dirty_label.setStyleSheet("color:#FFA500; font-style:italic;")
        toolbar_l.addWidget(self._dirty_label)

        layout.addWidget(toolbar)

        # ── Native header table (Consolas, matches old Moves tab) ──────
        _FONT = QFont("Consolas", 10)
        self._header_table = QTableWidget()
        self._header_table.setColumnCount(2)
        self._header_table.setHorizontalHeaderLabels(["Field", "Value"])
        self._header_table.horizontalHeader().setStretchLastSection(True)
        self._header_table.verticalHeader().setVisible(False)
        self._header_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._header_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._header_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._header_table.setFont(_FONT)
        self._header_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._header_table.customContextMenuRequested.connect(
            self._on_header_right_click
        )
        self._header_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._header_table.setAlternatingRowColors(True)
        self._hdr_font = _FONT

        # ── Splitter: header table (top) + move browser (bottom) ─────────
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._header_table)

        self._browser = QTextBrowser()
        self._browser.setOpenLinks(False)
        self._browser.setReadOnly(True)
        self._browser.anchorClicked.connect(self._on_anchor_clicked)
        self._browser.setContextMenuPolicy(Qt.CustomContextMenu)
        self._browser.customContextMenuRequested.connect(self._on_right_click)
        splitter.addWidget(self._browser)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        layout.addWidget(splitter, 1)

    # ── public API ────────────────────────────────────────────────────────────

    def _rebuild_header_table(self) -> None:
        """Populate the native QTableWidget with current PGN headers."""
        if not self._editor.loaded:
            self._header_table.setRowCount(0)
            return
        headers  = self._editor.headers()
        priority = ["Event", "Site", "Date", "Round",
                    "White", "Black", "Result", "WhiteElo", "BlackElo", "ECO"]
        keys  = [k for k in priority if k in headers]
        keys += sorted(k for k in headers if k not in priority)
        self._header_table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            ki = QTableWidgetItem(key)
            ki.setFont(self._hdr_font)
            ki.setForeground(QColor("#888888"))
            vi = QTableWidgetItem(str(headers.get(key, "")))
            vi.setFont(self._hdr_font)
            self._header_table.setItem(row, 0, ki)
            self._header_table.setItem(row, 1, vi)
        self._header_table.resizeRowsToContents()
        total_h = self._header_table.horizontalHeader().height()
        for r in range(self._header_table.rowCount()):
            total_h += self._header_table.rowHeight(r)
        self._header_table.setMaximumHeight(total_h + 4)

    def _on_header_right_click(self, pos) -> None:
        item = self._header_table.itemAt(pos)
        if item is None:
            return
        row = self._header_table.row(item)
        key = (self._header_table.item(row, 0) or QTableWidgetItem("")).text()
        menu = QMenu(self)
        ea = menu.addAction(f'Edit "{key}"')
        ea.triggered.connect(lambda checked=False, k=key: self._edit_header_field(k))
        menu.addAction("Add field").triggered.connect(self._add_header)
        menu.exec(self._header_table.mapToGlobal(pos))

    def refresh(self) -> None:
        self._rebuild_header_table()
        vscroll = self._browser.verticalScrollBar().value()

        self._var_counter   = 0
        self._var_map       = {}
        self._ply_map       = {}
        self._main_node_map = {}
        self._node_map      = {}

        html = self._build_html()
        self._browser.setHtml(html)
        self._browser.verticalScrollBar().setValue(vscroll)
        self._update_dirty_indicator()
        self._update_save_lib_btn()

        # Scroll to current move — variation nodes use the "vncur" anchor
        current_node = self._editor.current_node
        if current_node is not None and current_node.move is not None:
            # Try named anchor set during render; fall back to ply-based
            self._browser.scrollToAnchor("vncur")
            current_ply = len(self._editor.session.board.move_stack)
            self._browser.scrollToAnchor(f"ply{current_ply}")

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

    # ── HTML builder ──────────────────────────────────────────────────────────

    def _build_html(self) -> str:
        parts = [f"<html><head><style>{self._CSS}</style></head><body>"]

        # Headers
        if self._editor.loaded:
            current_ply  = len(self._editor.session.board.move_stack)
            current_node = self._editor.current_node
            self._render_mainline(
                self._editor.loaded.game,
                parts,
                current_ply,
                current_node,
                ply=1,
                force_movenumber=True,
            )

        parts.append("</body></html>")
        return "".join(parts)

    def _render_mainline(
        self,
        node: chess.pgn.GameNode,
        out: list[str],
        current_ply: int,
        current_node: "chess.pgn.GameNode | None",
        ply: int,
        force_movenumber: bool,
    ) -> int:
        """
        Walk the mainline from node, appending HTML to out.
        Returns the ply after the last mainline move rendered.
        Uses node identity (not ply index) to detect the current move so that
        navigating into a variation doesn't falsely highlight a mainline move.
        """
        need_movenumber = force_movenumber

        while node.variations:
            main = node.variations[0]

            # Move number + SAN wrapped in a pill
            board = node.board()
            if board.turn == chess.WHITE:
                num_str = f'<span class="mn">{board.fullmove_number}.</span> '
                need_movenumber = False
            elif need_movenumber:
                num_str = f'<span class="mn">{board.fullmove_number}...</span> '
                need_movenumber = False
            else:
                num_str = ""

            # The move itself
            try:
                san = main.san()
            except Exception:
                san = main.move.uci() if main.move else "?"

            is_current = (main is current_node)
            cls        = "cur" if is_current else "mv"
            anchor     = f'<a name="ply{ply}"></a>' if is_current else ""
            nav_id     = f"nav{ply}"
            self._ply_map[nav_id]       = ply
            self._main_node_map[nav_id] = main

            # Pill: number + move link together
            pill_cls = "mp cur-pill" if is_current else "mp"
            out.append(
                f'{anchor}<span class="{pill_cls}">'
                f'{num_str}<a href="{nav_id}" class="{cls}">{_e(san)}</a>'
                f'</span> '
            )

            # Comment on this move — parens become clickable coach lines
            if main.comment and main.comment.strip():
                try:
                    _cmt = _comment_to_html_pgn(main.comment.strip(), ply, self._editor)
                except Exception:
                    import traceback; traceback.print_exc()
                    _cmt = _e(main.comment.strip())
                out.append(f'<span class="cmt">{{ {_cmt} }}</span> ')

            # Variation blocks (indices 1+)
            for var_node in node.variations[1:]:
                self._render_variation_block(var_node, out, current_ply, current_node, ply)
                need_movenumber = True

            node = main
            ply += 1

        return ply

    def _render_variation_block(
        self,
        var_node: chess.pgn.GameNode,
        out: list[str],
        current_ply: int,
        current_node: "chess.pgn.GameNode | None",
        parent_ply: int,
    ) -> None:
        """Render one variation block with a ▶/▼ toggle."""
        var_id = f"v{self._var_counter}"
        self._var_counter += 1
        self._var_map[var_id] = var_node

        collapsed = var_id in self._collapsed
        arrow     = "▶" if collapsed else "▼"

        out.append(f'<a href="toggle:{var_id}" class="tog">{arrow}</a> ')

        if collapsed:
            out.append('<span class="var">( ... )</span> ')
        else:
            out.append('<span class="var">( ')
            self._render_variation_moves(var_node, out, current_ply, current_node, force_movenumber=True)
            out.append(') </span>')

    def _render_variation_moves(
        self,
        node: chess.pgn.GameNode,
        out: list[str],
        current_ply: int,
        current_node: "chess.pgn.GameNode | None",
        force_movenumber: bool,
    ) -> None:
        """
        Render variation moves as clickable links.
        Uses node identity to highlight the current move even inside variations.
        Recursively renders nested sub-variations.
        """
        need_movenumber = force_movenumber

        while node is not None and node.move is not None:
            board = node.parent.board() if node.parent else chess.Board()

            if board.turn == chess.WHITE:
                out.append(f'<span class="mn">{board.fullmove_number}.</span> ')
                need_movenumber = False
            elif need_movenumber:
                out.append(f'<span class="mn">{board.fullmove_number}...</span> ')
                need_movenumber = False

            try:
                san = node.san()
            except Exception:
                san = node.move.uci() if node.move else "?"

            node_id = f"vn{len(self._node_map)}"
            self._node_map[node_id] = node

            is_current = (node is current_node)
            cls        = "cur" if is_current else "var"
            anchor     = f'<a name="vncur"></a>' if is_current else ""

            out.append(f'{anchor}<a href="{node_id}" class="{cls}">{_e(san)}</a> ')

            if node.comment and node.comment.strip():
                _vply = len(node.board().move_stack)
                _cmt  = _comment_to_html_pgn(node.comment.strip(), _vply, self._editor)
                out.append(f'<span class="cmt">{{ {_cmt} }}</span> ')

            # Nested sub-variations branching from this node (indices 1+)
            for sub_var in (node.variations[1:] if node.variations else []):
                self._render_variation_block(sub_var, out, current_ply, current_node, parent_ply=0)
                need_movenumber = True

            # Descend to main continuation of this variation
            node = node.variations[0] if node.variations else None



    # ── anchor clicks ─────────────────────────────────────────────────────────

    def _on_anchor_clicked(self, url: QUrl) -> None:
        href = url.toString()

        # coach:PLY_uci1_uci2_... — coach line link
        # Format avoids URL authority parsing issues (no // no host)
        if href.startswith("coach:"):
            parts    = href[len("coach:"):].split("_")
            if len(parts) >= 2:
                try:
                    base_ply = int(parts[0])
                except ValueError:
                    return
                uci_list = [u for u in parts[1:] if u]
                if uci_list:
                    self.comment_line_clicked.emit(base_ply, uci_list)
            return


        if href.startswith("toggle:"):
            var_id = href[7:]
            if var_id in self._collapsed:
                self._collapsed.discard(var_id)
            else:
                self._collapsed.add(var_id)
            self.refresh()
            return

        # nav:PLY — navigate mainline
        if href.startswith("nav") and href[3:].isdigit():
            ply = int(href[3:])
            self.navigate_requested.emit(ply)
            return

        # vn:NODE_ID — navigate variation move
        if href.startswith("vn") and href in self._node_map:
            node = self._node_map[href]
            self.navigate_node_requested.emit(node)
            return

    # ── right-click context menu ───────────────────────────────────────────────

    def _on_right_click(self, pos) -> None:
        anchor = self._browser.anchorAt(pos)
        menu   = QMenu(self)

        # ── mainline move ──────────────────────────────────────────────────
        ply = None
        if anchor and anchor.startswith("nav") and anchor[3:].isdigit():
            ply = int(anchor[3:])

        if ply is not None:
            move_num = (ply + 1) // 2
            side     = "White" if ply % 2 == 1 else "Black"
            nav_id   = f"nav{ply}"

            nav_action = menu.addAction(f"Navigate → move {move_num} ({side})")
            nav_action.triggered.connect(
                lambda checked=False, p=ply: self.navigate_requested.emit(p)
            )

            menu.addSeparator()

            existing      = self._editor.get_comment_at_ply(ply)
            comment_label = "Edit comment" if existing else "Insert comment"
            comment_action = menu.addAction(
                f"{comment_label}  [move {move_num} ({side})]"
            )
            comment_action.triggered.connect(
                lambda checked=False, p=ply: self._insert_comment_at(p)
            )

            # Promote/demote available when there are sibling variations
            main_node = self._main_node_map.get(nav_id)
            if main_node is not None and main_node.parent is not None:
                parent    = main_node.parent
                siblings  = parent.variations
                var_idx   = siblings.index(main_node) if main_node in siblings else -1
                var_count = len(siblings)
                if var_count > 1:
                    menu.addSeparator()
                    promote_ml = menu.addAction("⬆  Promote / move up")
                    promote_ml.setEnabled(var_idx > 0)
                    promote_ml.triggered.connect(
                        lambda checked=False, n=main_node: self._do_promote_variation(n)
                    )
                    demote_ml = menu.addAction("⬇  Demote to variation")
                    demote_ml.setEnabled(0 <= var_idx < var_count - 1)
                    demote_ml.triggered.connect(
                        lambda checked=False, n=main_node: self._do_demote_variation(n)
                    )

            menu.addSeparator()

            # Delete options — available for every mainline move
            if main_node is not None:
                del_from_action = menu.addAction(f"✂  Delete from move {move_num} onwards")
                del_from_action.triggered.connect(
                    lambda checked=False, n=main_node: self._do_delete_from_node(n)
                )
                # "Delete variation" only makes sense when this node IS a variation (idx > 0)
                if main_node.parent is not None:
                    siblings2 = main_node.parent.variations
                    idx2      = siblings2.index(main_node) if main_node in siblings2 else 0
                    if idx2 > 0:
                        del_var_action = menu.addAction("🗑  Delete this variation branch")
                        del_var_action.triggered.connect(
                            lambda checked=False, n=main_node: self._do_delete_variation(n)
                        )

            menu.addSeparator()

        # ── variation move ─────────────────────────────────────────────────
        var_node = None
        if anchor and anchor.startswith("vn") and anchor in self._node_map:
            var_node = self._node_map[anchor]

        if var_node is not None:
            parent    = var_node.parent
            var_idx   = parent.variations.index(var_node) if parent else -1
            var_count = len(parent.variations) if parent else 0

            nav_var_action = menu.addAction("Navigate here")
            nav_var_action.triggered.connect(
                lambda checked=False, n=var_node: self.navigate_node_requested.emit(n)
            )

            menu.addSeparator()

            promote_action = menu.addAction("⬆  Promote variation")
            promote_action.setEnabled(var_idx > 0)
            promote_action.triggered.connect(
                lambda checked=False, n=var_node: self._do_promote_variation(n)
            )

            demote_action = menu.addAction("⬇  Demote variation")
            demote_action.setEnabled(0 <= var_idx < var_count - 1)
            demote_action.triggered.connect(
                lambda checked=False, n=var_node: self._do_demote_variation(n)
            )

            menu.addSeparator()

            del_from_var = menu.addAction("✂  Delete from this move onwards")
            del_from_var.triggered.connect(
                lambda checked=False, n=var_node: self._do_delete_from_node(n)
            )

            del_var_action = menu.addAction("🗑  Delete entire variation branch")
            del_var_action.triggered.connect(
                lambda checked=False, n=var_node: self._do_delete_variation(n)
            )

            menu.addSeparator()

        coach_action = menu.addAction("♟  Request Coach Note")
        coach_action.triggered.connect(self.coach_help_requested.emit)
        menu.exec(self._browser.mapToGlobal(pos))


    def _do_promote_variation(self, node) -> None:
        self.promote_variation_requested.emit(node)

    def _do_demote_variation(self, node) -> None:
        self.demote_variation_requested.emit(node)

    def _do_delete_variation(self, node) -> None:
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Delete variation",
            "Delete this entire variation branch?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.delete_variation_requested.emit(node)

    def _do_delete_from_node(self, node) -> None:
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Delete moves",
            "Delete this move and everything after it in this line?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.delete_from_node_requested.emit(node)

    # ── comment insertion ─────────────────────────────────────────────────────

    def _insert_comment_at(self, ply: int) -> None:
        existing = self._editor.get_comment_at_ply(ply)
        move_num = (ply + 1) // 2
        dlg = CommentDialog(
            parent   = self,
            title    = f"Comment for move {move_num}:",
            initial  = existing,
            editor   = self._editor,
            base_ply = ply,
        )
        from PySide6.QtWidgets import QDialog
        if dlg.exec() != QDialog.Accepted:
            return
        self._editor.insert_comment_at_ply(ply, dlg.text())
        self.refresh()

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
        if not ok or not key.strip():
            return
        val, ok2 = QInputDialog.getText(
            self, "Add PGN field", f"Value for {key}:"
        )
        if not ok2:
            return
        self._editor.add_header(key.strip(), val)
        self.header_changed.emit()
        self.refresh()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _update_dirty_indicator(self) -> None:
        self._dirty_label.setText("● unsaved" if self._editor.dirty else "")

    def _update_save_lib_btn(self) -> None:
        has_source = (
            self._editor.source_pgn_path is not None
            and self._editor.source_offset is not None
        )
        self._save_lib_btn.setEnabled(has_source)


def _render_header_card(hdrs: dict) -> str:
    """Render PGN headers as a styled table above the moves."""
    white   = hdrs.get("White",   "?")
    black   = hdrs.get("Black",   "?")
    result  = hdrs.get("Result",  "*")
    event   = hdrs.get("Event",   "")
    site    = hdrs.get("Site",    "")
    date    = hdrs.get("Date",    "")
    round_  = hdrs.get("Round",   "")
    eco     = hdrs.get("ECO",     "")
    opening = hdrs.get("Opening", "")
    welo    = hdrs.get("WhiteElo", "")
    belo    = hdrs.get("BlackElo", "")

    if result == "1-0":
        rc = "hresult-1-0"
    elif result == "0-1":
        rc = "hresult-0-1"
    elif result in ("1/2-1/2", "\u00bd-\u00bd"):
        rc = "hresult-draw"
    else:
        rc = "hresult-other"

    welo_str = (' <span style="color:#555;">({})</span>'.format(_e(welo))
                if welo and welo != "?" else "")
    belo_str = (' <span style="color:#555;">({})</span>'.format(_e(belo))
                if belo and belo != "?" else "")
    title_html = (
        '<span class="htitle">{}{}</span>'
        ' <span style="color:#444;"> vs </span> '
        '<span class="htitle">{}{}</span>'
        '&nbsp;&nbsp;<span class="{}">{}</span>'
    ).format(_e(white), welo_str, _e(black), belo_str, rc, _e(result))

    def row(key, val):
        return (
            '<tr><td class="hfkey">{}</td>'
            '<td class="hfval">{}</td></tr>'
        ).format(_e(key), val)

    rows = []

    if event and event not in ("?", ""):
        event_val = _e(event)
        if site and site not in ("?", ""):
            event_val += ' <span style="color:#555;">\u2014 {}</span>'.format(_e(site))
        rows.append(row("Event", event_val))
    elif site and site not in ("?", ""):
        rows.append(row("Site", _e(site)))

    if date and date not in ("?", ""):
        date_val = _e(date)
        if round_ and round_ not in ("?", ""):
            date_val += ' <span style="color:#555;">\u00b7 R{}</span>'.format(_e(round_))
        rows.append(row("Date", date_val))

    if eco and eco not in ("?", ""):
        eco_val = '<span style="color:#7A9EC0;">{}</span>'.format(_e(eco))
        if opening and opening not in ("?", ""):
            eco_val += ' <span style="color:#666;">{}</span>'.format(_e(opening))
        rows.append(row("ECO", eco_val))

    shown = {"White", "Black", "Result", "Event", "Site", "Date", "Round",
             "ECO", "Opening", "WhiteElo", "BlackElo"}
    for k, v in hdrs.items():
        if k not in shown and v not in ("?", ""):
            rows.append(row(k, '<span style="color:#888;">{}</span>'.format(_e(v))))

    return (
        '<table class="htbl">'
        '<tr class="htitle-row"><td colspan="2">{}</td></tr>'
        '{}'
        '</table>'
    ).format(title_html, "".join(rows))


def _e(text: str) -> str:
    """HTML-escape a string."""
    return _html.escape(str(text))


# Matches (Nf3 Nc6 Bb5) or (1.Nf3 d5) — parenthesised SAN sequences in comments
_PAREN_RE    = re.compile(r'\(([^)]+)\)')
_MOVE_NUM_RE = re.compile(r'\d+\.+\s*')


def _strip_move_numbers(text: str) -> list[str]:
    return [t for t in _MOVE_NUM_RE.sub(' ', text).split() if t]


def _comment_to_html_pgn(raw: str, base_ply: int, editor) -> str:
    """
    Convert a raw PGN comment to HTML for QTextBrowser.
    Text in parentheses is tried as SAN — valid moves become per-move cyan links.
    href format: coach://{base_ply}/{uci1 … uciN}
    """
    if not editor or not editor.loaded:
        return _e(raw)

    parts = _PAREN_RE.split(raw)
    out: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            out.append(_e(part))
            continue

        san_tokens = _strip_move_numbers(part.strip())
        if not san_tokens:
            out.append(_e(f'({part})'))
            continue

        uci_list, _err = editor.san_to_uci(base_ply, san_tokens)
        # Fallback: comment on a Black move often starts with that same move,
        # e.g. commenting e5 with '(e5 Nf3...)'. Try ply-1 (before the move).
        effective_ply = base_ply
        if not uci_list and base_ply > 0:
            uci_list, _err = editor.san_to_uci(base_ply - 1, san_tokens)
            effective_ply = base_ply - 1
        if not uci_list:
            out.append(_e(f'({part})'))
            continue

        san_list         = editor.uci_to_san(effective_ply, uci_list)
        remaining_tokens = san_tokens[len(uci_list):]

        links: list[str] = []
        for idx, (uci, san) in enumerate(zip(uci_list, san_list)):
            prefix_uci = "_".join(uci_list[: idx + 1])
            href       = "coach:{}_{}".format(effective_ply, prefix_uci)
            is_white   = (effective_ply + idx) % 2 == 0
            move_num   = (effective_ply + idx) // 2 + 1
            if is_white:
                num_html = '<span style="color:#888888;">{}. </span>'.format(move_num)
            elif idx == 0:
                num_html = '<span style="color:#888888;">{}... </span>'.format(move_num)
            else:
                num_html = ""
            links.append(
                '{}'
                '<a href="{}" style="color:#4FC3F7; text-decoration:underline;">{}</a>'.format(
                    num_html, href, _e(san)
                )
            )

        inner = " ".join(links)
        if remaining_tokens:
            inner += " " + _e(" ".join(remaining_tokens))
        out.append('(' + inner + ')')

    return "".join(out)
