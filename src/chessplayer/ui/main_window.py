from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, QStringListModel
from PySide6.QtGui import QTextCursor, QTextCharFormat
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QCheckBox,
    QTextEdit,
    QTableView,
    QMessageBox,
    QTabWidget,
    QListView,
    QInputDialog,
    QComboBox,
)

from core.pgn_edit import PgnEditor
from ui.board_model import BoardListModel, BoardBridge
from utils.paths import resolve_path

from ui.query_builder import QueryBuilder
from ui.game_table_model import GameTableModel
from pgn.query import default_multisort
from pgn.store import IndexHandle, PgnStore
from pgn.continuations import common_continuations_from_store

from engine.uci_engine import UciEngine


class MainWindow(QMainWindow):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self._config = config
        self.setWindowTitle("CHESSPLAYER 3.0.0")

        # Core editor / board
        self._editor = PgnEditor()
        self._editor.new_freeplay()

        pieces_dir = resolve_path(self._config["paths"]["pieces_dir"])
        self._board_model = BoardListModel(self._editor, pieces_dir=pieces_dir)
        self._bridge = BoardBridge(self._editor, self._board_model)

        # Store
        self._store: PgnStore | None = None
        self._sort = default_multisort()
        self._current_table_game_ids: list[int] = []

        data_dir = resolve_path(self._config["paths"]["data_dir"])
        db_path = data_dir / "index.sqlite"
        if db_path.exists():
            self._store = PgnStore(IndexHandle(db_path=db_path))

        # Engine
        self._engine: UciEngine | None = None
        self._engine_enabled = False
        self._engine_plays = False  # if enabled, engine makes opponent moves
        self._engine_side = "Black"  # engine plays as Black by default
        self._engine_movetime_ms = 250

        root = QWidget()
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)

        main_split = QSplitter(Qt.Horizontal)

        # LEFT: games browser
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(8, 8, 8, 8)

        left_l.addWidget(QLabel("Games"))

        self._query = QueryBuilder()
        self._query.query_changed.connect(lambda q: self._refresh_games(q))
        left_l.addWidget(self._query)

        self._table_model = GameTableModel()
        self._table = QTableView()
        self._table.setModel(self._table_model)
        self._table.setSelectionBehavior(QTableView.SelectRows)
        self._table.setSelectionMode(QTableView.SingleSelection)
        self._table.doubleClicked.connect(self._open_selected_game)
        left_l.addWidget(self._table, 1)

        btn_row = QWidget()
        btn_l = QHBoxLayout(btn_row)
        btn_l.setContentsMargins(0, 0, 0, 0)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(lambda: self._refresh_games(self._query.current_query()))
        btn_l.addWidget(self._refresh_btn)

        self._open_btn = QPushButton("Open")
        self._open_btn.clicked.connect(self._open_selected_game)
        btn_l.addWidget(self._open_btn)

        btn_l.addStretch(1)
        left_l.addWidget(btn_row)

        main_split.addWidget(left)

        # RIGHT: board + tabs
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(8, 8, 8, 8)

        # Status + board controls
        top = QWidget()
        top_l = QHBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)

        self._status = QLabel("Ready")
        top_l.addWidget(self._status)
        top_l.addStretch(1)

        self._flip = QCheckBox("Flip")
        self._flip.stateChanged.connect(lambda s: self._bridge.setFlipped(bool(s)))
        top_l.addWidget(self._flip)

        back_btn = QPushButton("Back")
        back_btn.clicked.connect(self._on_back)
        top_l.addWidget(back_btn)

        fwd_btn = QPushButton("Forward")
        fwd_btn.clicked.connect(self._on_forward)
        top_l.addWidget(fwd_btn)

        right_l.addWidget(top)

        # Engine controls row
        eng = QWidget()
        eng_l = QHBoxLayout(eng)
        eng_l.setContentsMargins(0, 0, 0, 0)

        self._engine_toggle = QCheckBox("Stockfish")
        self._engine_toggle.stateChanged.connect(lambda s: self._set_engine_enabled(bool(s)))
        eng_l.addWidget(self._engine_toggle)

        self._engine_play_toggle = QCheckBox("Play vs Engine")
        self._engine_play_toggle.stateChanged.connect(lambda s: self._set_engine_plays(bool(s)))
        eng_l.addWidget(self._engine_play_toggle)

        eng_l.addWidget(QLabel("Engine side:"))
        self._engine_side_combo = QComboBox()
        self._engine_side_combo.addItems(["Black", "White"])
        self._engine_side_combo.currentTextChanged.connect(lambda t: setattr(self, "_engine_side", t))
        eng_l.addWidget(self._engine_side_combo)

        eng_l.addWidget(QLabel("MoveTime ms:"))
        self._engine_time_combo = QComboBox()
        self._engine_time_combo.addItems(["100", "250", "500", "1000"])
        self._engine_time_combo.setCurrentText("250")
        self._engine_time_combo.currentTextChanged.connect(lambda t: setattr(self, "_engine_movetime_ms", int(t)))
        eng_l.addWidget(self._engine_time_combo)

        eng_l.addStretch(1)
        right_l.addWidget(eng)

        # QML board
        self._board_view = QQuickWidget()
        self._board_view.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self._board_view.rootContext().setContextProperty("piecesModel", self._board_model)
        self._board_view.rootContext().setContextProperty("bridge", self._bridge)
        qml_path = resolve_path("src/chessplayer/qml/BoardView.qml")
        self._board_view.setSource(QUrl.fromLocalFile(str(qml_path)))
        errs = self._board_view.errors()
        if errs:
            QMessageBox.critical(self, "QML errors", "\n".join([e.toString() for e in errs]))
        right_l.addWidget(self._board_view, 3)

        tabs = QTabWidget()

        # Common continuations tab (replaces tree model for now)
        cont_tab = QWidget()
        cont_l = QVBoxLayout(cont_tab)
        cont_l.setContentsMargins(0, 0, 0, 0)
        cont_l.addWidget(QLabel("Common continuations from loaded base (sampled)"))
        self._cont_model = QStringListModel()
        self._cont_view = QListView()
        self._cont_view.setModel(self._cont_model)
        cont_l.addWidget(self._cont_view, 1)
        tabs.addTab(cont_tab, "Variations")

        # PGN tab (highlight current move token)
        pgn_tab = QWidget()
        pgn_l = QVBoxLayout(pgn_tab)
        pgn_l.setContentsMargins(0, 0, 0, 0)

        hdr_row = QWidget()
        hdr_l = QHBoxLayout(hdr_row)
        hdr_l.setContentsMargins(0, 0, 0, 0)

        self._edit_hdr_btn = QPushButton("Edit PGN field")
        self._edit_hdr_btn.clicked.connect(self._edit_header)
        hdr_l.addWidget(self._edit_hdr_btn)

        self._add_hdr_btn = QPushButton("Add field")
        self._add_hdr_btn.clicked.connect(self._add_header)
        hdr_l.addWidget(self._add_hdr_btn)

        hdr_l.addStretch(1)
        pgn_l.addWidget(hdr_row)

        self._pgn_text = QTextEdit()
        self._pgn_text.setReadOnly(True)
        pgn_l.addWidget(self._pgn_text, 1)
        tabs.addTab(pgn_tab, "PGN")

        right_l.addWidget(tabs, 2)

        main_split.addWidget(right)
        main_split.setSizes([520, 900])

        outer.addWidget(main_split)
        self.setCentralWidget(root)

        # Bridge signals
        self._bridge.statusChanged.connect(self._status.setText)
        self._bridge.fenChanged.connect(lambda _fen: self._on_position_changed())
        self._bridge.moveMade.connect(lambda _san: self._after_user_move())

        # initial UI
        self._on_position_changed()
        self._refresh_games(self._query.current_query())

    # ---------------- engine ----------------

    def _engine_exe_path(self) -> Path:
        # default to your known layout if config lacks it
        default_rel = "assets/engines/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
        rel = self._config.get("paths", {}).get("engine_exe", default_rel)
        return resolve_path(rel)

    def _set_engine_enabled(self, enabled: bool) -> None:
        self._engine_enabled = enabled
        if not enabled:
            if self._engine:
                self._engine.stop()
            self._engine = None
            return

        exe = self._engine_exe_path()
        if not exe.exists():
            QMessageBox.critical(self, "Stockfish missing", f"Engine not found:\n{exe}")
            self._engine_toggle.setChecked(False)
            return

        try:
            self._engine = UciEngine(exe)
            self._engine.start()
        except Exception as e:
            QMessageBox.critical(self, "Stockfish start failed", str(e))
            self._engine_toggle.setChecked(False)
            self._engine = None

    def _set_engine_plays(self, enabled: bool) -> None:
        self._engine_plays = enabled

    def _engine_should_move_now(self) -> bool:
        if not (self._engine_enabled and self._engine_plays and self._engine):
            return False
        turn_is_white = self._editor.session.board.turn  # True=White to move
        engine_is_white = (self._engine_side == "White")
        return turn_is_white == engine_is_white

    def _do_engine_move_if_needed(self) -> None:
        if not self._engine_should_move_now():
            return
        assert self._engine is not None
        prefix = self._editor.played_prefix_uci()
        try:
            bm = self._engine.bestmove_movetime(prefix, movetime_ms=self._engine_movetime_ms)
            if bm.uci == "0000":
                return
            res = self._editor.apply_uci_move(bm.uci)
            if res.ok:
                self._board_model.rebuild()
                self._on_position_changed()
        except Exception as e:
            QMessageBox.critical(self, "Stockfish error", str(e))

    # ---------------- navigation ----------------

    def _on_back(self) -> None:
        if self._editor.step_back():
            self._board_model.rebuild()
            self._on_position_changed()

    def _on_forward(self) -> None:
        if self._editor.step_forward_mainline():
            self._board_model.rebuild()
            self._on_position_changed()

    # ---------------- games browser ----------------

    def _refresh_games(self, query) -> None:
        if not self._store:
            return
        page_size = int(self._config.get("browsing", {}).get("page_size", 200))
        rows = self._store.list_games(query=query, multisort=self._sort, page=0, page_size=page_size)
        self._table_model.set_rows(rows)

        # capture ids for continuation sampling
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

    # ---------------- PGN display + highlight ----------------

    def _set_pgn_text_and_highlight(self) -> None:
        pgn = self._editor.export_pgn()
        self._pgn_text.setPlainText(pgn)

        san = self._editor.current_san()
        if not san:
            return

        # find last occurrence of SAN token in text (best-effort token match)
        # token boundaries: whitespace or punctuation
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
        fmt.setBackground(Qt.yellow)
        cur.setCharFormat(fmt)

        # keep view stable: do not jump scroll aggressively
        # but ensure highlight is visible
        self._pgn_text.setTextCursor(cur)

    # ---------------- headers UI ----------------

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

    # ---------------- continuations (Variations tab) ----------------

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

    # ---------------- lifecycle hooks ----------------

    def _on_position_changed(self) -> None:
        self._set_pgn_text_and_highlight()
        self._refresh_common_continuations()

    def _after_user_move(self) -> None:
        self._on_position_changed()
        self._do_engine_move_if_needed()