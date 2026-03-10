from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor, QTextCharFormat
from PySide6.QtWidgets import QMessageBox, QInputDialog

from pgn.continuations import common_continuations_from_store


class BrowserMixin:
    def _refresh_games(self, query) -> None:
        if not self._store:
            self._table_model.set_rows([])
            self._current_table_game_ids = []
            self._refresh_common_continuations()
            return

        page_size = int(self._config.get("browsing", {}).get("page_size", 200))
        rows = self._store.list_games(
            query=query,
            multisort=self._sort,
            page=0,
            page_size=page_size,
            source_id=self._active_source_id,
        )
        self._table_model.set_rows(rows)

        ids: list[int] = []
        for r in rows:
            gid = r.get("game_id") if isinstance(r, dict) else None
            if isinstance(gid, int):
                ids.append(gid)
        self._current_table_game_ids = ids

        self._refresh_common_continuations()

    def _open_selected_game(self) -> None:
        if not self._store:
            QMessageBox.information(self, "No index", "No index.sqlite found. Build index first.")
            return
        sel = self._table.selectionModel()
        if not sel or not sel.hasSelection():
            return
        row = sel.selectedRows()[0].row()
        game_id = self._table_model.game_id_at(row)
        if game_id is None:
            return
        try:
            pgn = self._store.open_game_pgn_text(game_id)
            self._editor.load_pgn_text(pgn)
            self._board_model.rebuild()
            self._on_position_changed()
        except Exception as e:
            QMessageBox.critical(self, "Open game failed", str(e))

    def _set_pgn_text_and_highlight(self) -> None:
        pgn = self._editor.export_pgn()
        self._pgn_text.setPlainText(pgn)

        san = self._editor.current_san()
        if not san:
            return

        pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(san) + r"(?![A-Za-z0-9_])")
        matches = list(pattern.finditer(pgn))
        if not matches:
            return
        m = matches[-1]
        start, end = m.start(), m.end()

        cur = self._pgn_text.textCursor()
        cur.setPosition(start)
        cur.setPosition(end, QTextCursor.KeepAnchor)

        fmt = QTextCharFormat()
        fmt.setBackground(Qt.blue)
        cur.setCharFormat(fmt)

        self._pgn_text.setTextCursor(cur)

    def _edit_header(self) -> None:
        hdrs = self._editor.headers()
        if not hdrs:
            self._editor.new_freeplay()
            hdrs = self._editor.headers()

        keys = sorted(hdrs.keys())
        key, ok = QInputDialog.getItem(self, "Edit PGN field", "Field:", keys, 0, False)
        if not ok or not key:
            return

        val, ok2 = QInputDialog.getText(self, "Edit PGN field", f"Value for {key}:", text=str(hdrs.get(key, "")))
        if not ok2:
            return

        self._editor.set_header(key, val)
        self._on_position_changed()

    def _add_header(self) -> None:
        key, ok = QInputDialog.getText(self, "Add PGN field", "Field name:")
        if not ok or not key:
            return
        val, ok2 = QInputDialog.getText(self, "Add PGN field", f"Value for {key}:")
        if not ok2:
            return
        self._editor.add_header(key, val)
        self._on_position_changed()

    def _refresh_common_continuations(self) -> None:
        if not self._store or not self._current_table_game_ids:
            self._cont_model.setStringList([])
            return

        prefix = self._editor.played_prefix_uci()
        conts = common_continuations_from_store(
            self._store,
            self._current_table_game_ids,
            prefix_uci=prefix,
            max_games=int(self._config.get("browsing", {}).get("continuation_sample_games", 200)),
            max_out=30,
        )
        items = [f"{c.san}    ({c.count})" for c in conts]
        self._cont_model.setStringList(items)
