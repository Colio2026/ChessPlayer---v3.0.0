from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, QObject, Signal, QThread
from PySide6.QtGui import QAction, QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableView,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.pgn_edit import PgnEditor
from engine.uci_engine import UciEngine
from pgn.continuations import root_continuation_stats_from_store
from pgn.indexer import build_or_rebuild_index_for_source
from pgn.query import default_multisort
from pgn.store import IndexHandle, PgnStore, SourceRecord
from ui.board_model import BoardBridge, BoardListModel
from ui.continuation_stats_model import ContinuationStatsModel
from ui.game_table_model import GameTableModel
from ui.query_builder import QueryBuilder
from utils.paths import resolve_path


class _IndexWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(str, str, str)
    failed = Signal(str)

    def __init__(self, config: dict, source_type: str, source_path: str) -> None:
        super().__init__()
        self._config = config
        self._source_type = source_type
        self._source_path = source_path

    def run(self) -> None:
        try:
            db_path = build_or_rebuild_index_for_source(
                cfg=self._config,
                source_type=self._source_type,
                source_path=self._source_path,
                progress_cb=lambda count, message: self.progress.emit(count, message),
                cancel_cb=None,
            )
            self.finished.emit(str(db_path), self._source_type, self._source_path)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self._config = config
        self.setWindowTitle("CHESSPLAYER 3.0.0")

        self._editor = PgnEditor()
        self._editor.new_freeplay()

        pieces_dir = resolve_path(self._config["paths"]["pieces_dir"])
        self._board_model = BoardListModel(self._editor, pieces_dir=pieces_dir)
        self._bridge = BoardBridge(self._editor, self._board_model)

        self._store: PgnStore | None = None
        self._sort = default_multisort()
        self._active_source_id: int | None = None
        self._active_source_type: str | None = None
        self._active_source_path: str | None = None
        self._updating_source_combo = False

        self._index_thread: QThread | None = None
        self._index_worker: _IndexWorker | None = None
        self._progress_dialog: QProgressDialog | None = None
        self._last_indexed_game_count = 0

        self._engine: UciEngine | None = None
        self._engine_enabled = False
        self._engine_plays = False
        self._engine_side = "Black"
        self._engine_movetime_ms = 250

        data_dir = resolve_path(self._config["paths"]["data_dir"])
        db_path = data_dir / "index.sqlite"
        if db_path.exists():
            self._store = PgnStore(IndexHandle(db_path=db_path))

        self._build_menu_bar()
        self._build_toolbar()
        self._build_ui()

        self._bridge.statusChanged.connect(self._status.setText)
        self._bridge.fenChanged.connect(lambda _fen: self._on_position_changed())
        self._bridge.moveMade.connect(lambda _san: self._after_user_move())

        self._choose_initial_source()
        self._on_position_changed()
        self._refresh_games(self._query.current_query())

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        self._load_pgn_file_action = QAction("Load PGN Library...", self)
        self._load_pgn_file_action.triggered.connect(self._load_pgn_library_file)
        file_menu.addAction(self._load_pgn_file_action)

        self._load_pgn_dir_action = QAction("Load PGN Folder...", self)
        self._load_pgn_dir_action.triggered.connect(self._load_pgn_library_directory)
        file_menu.addAction(self._load_pgn_dir_action)

        self._reindex_action = QAction("Reindex Current Library", self)
        self._reindex_action.triggered.connect(self._reindex_current_source)
        file_menu.addAction(self._reindex_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("File")
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        toolbar.addAction(self._load_pgn_file_action)
        toolbar.addAction(self._load_pgn_dir_action)
        toolbar.addAction(self._reindex_action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Active Library:"))

        self._source_combo = QComboBox()
        self._source_combo.setMinimumWidth(420)
        self._source_combo.currentIndexChanged.connect(self._on_source_combo_changed)
        toolbar.addWidget(self._source_combo)

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)

        main_split = QSplitter(Qt.Horizontal)

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
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(False)
        self._table.horizontalHeader().setStretchLastSection(True)
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

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(8, 8, 8, 8)

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

        cont_tab = QWidget()
        cont_l = QVBoxLayout(cont_tab)
        cont_l.setContentsMargins(0, 0, 0, 0)
        cont_l.addWidget(QLabel("Most-played first moves from the default base"))
        self._cont_model = ContinuationStatsModel()
        self._cont_view = QTableView()
        self._cont_view.setModel(self._cont_model)
        self._cont_view.setSelectionMode(QTableView.NoSelection)
        self._cont_view.setEditTriggers(QTableView.NoEditTriggers)
        self._cont_view.setAlternatingRowColors(True)
        self._cont_view.setShowGrid(False)
        self._cont_view.verticalHeader().setVisible(False)
        self._cont_view.horizontalHeader().setStretchLastSection(False)
        self._cont_view.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 6):
            self._cont_view.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        cont_l.addWidget(self._cont_view, 1)
        tabs.addTab(cont_tab, "Variations")

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

    def _source_label(self, src: SourceRecord) -> str:
        return f"[{src.source_type}] {src.path}"

    def _set_active_source(self, src: SourceRecord | None) -> None:
        if src is None:
            self._active_source_id = None
            self._active_source_type = None
            self._active_source_path = None
        else:
            self._active_source_id = src.source_id
            self._active_source_type = src.source_type
            self._active_source_path = src.path
        self._update_source_combo()

    def _choose_initial_source(self) -> None:
        self._update_source_combo()
        if not self._store:
            return
        sources = self._store.list_sources()
        if sources and self._active_source_id is None:
            self._set_active_source(sources[0])

    def _update_source_combo(self) -> None:
        self._updating_source_combo = True
        try:
            self._source_combo.clear()
            if not self._store:
                self._source_combo.setEnabled(False)
                return
            sources = self._store.list_sources()
            if not sources:
                self._source_combo.setEnabled(False)
                return
            self._source_combo.setEnabled(True)
            current_index = -1
            for idx, src in enumerate(sources):
                self._source_combo.addItem(self._source_label(src), src.source_id)
                if self._active_source_id == src.source_id:
                    current_index = idx
            self._source_combo.setCurrentIndex(current_index if current_index >= 0 else 0)
        finally:
            self._updating_source_combo = False

    def _on_source_combo_changed(self, index: int) -> None:
        if self._updating_source_combo or not self._store or index < 0:
            return
        source_id = self._source_combo.itemData(index)
        if not isinstance(source_id, int):
            return
        for src in self._store.list_sources():
            if src.source_id == source_id:
                self._set_active_source(src)
                self._refresh_games(self._query.current_query())
                return

    def _ensure_progress_dialog(self) -> None:
        if self._progress_dialog is None:
            dlg = QProgressDialog("Indexing PGN library...", None, 0, 0, self)
            dlg.setWindowTitle("Indexing library")
            dlg.setWindowModality(Qt.WindowModal)
            dlg.setMinimumDuration(0)
            dlg.setAutoClose(False)
            dlg.setAutoReset(False)
            dlg.setCancelButton(None)
            self._progress_dialog = dlg

    def _show_progress_dialog(self, message: str) -> None:
        self._ensure_progress_dialog()
        assert self._progress_dialog is not None
        self._progress_dialog.setLabelText(message)
        self._progress_dialog.setRange(0, 0)
        self._progress_dialog.show()
        QApplication.processEvents()

    def _hide_progress_dialog(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.hide()

    def _on_index_progress(self, count: int, message: str) -> None:
        self._last_indexed_game_count = count
        self._status.setText(message)
        self._ensure_progress_dialog()
        assert self._progress_dialog is not None
        self._progress_dialog.setLabelText(f"{message}\n\nGames indexed: {count}")
        QApplication.processEvents()

    def _set_indexing_ui_busy(self, busy: bool, message: str | None = None) -> None:
        self._load_pgn_file_action.setEnabled(not busy)
        self._load_pgn_dir_action.setEnabled(not busy)
        self._reindex_action.setEnabled(not busy)
        self._refresh_btn.setEnabled(not busy)
        if message:
            self._status.setText(message)
        elif not busy:
            self._status.setText("Ready")

    def _start_index_job(self, source_type: str, source_path: str) -> None:
        if self._index_thread is not None:
            QMessageBox.information(self, "Indexing busy", "A PGN indexing job is already running.")
            return
        self._last_indexed_game_count = 0
        self._set_indexing_ui_busy(True, f"Indexing {source_path} ...")
        self._show_progress_dialog(f"Starting index for:\n{source_path}")

        self._index_thread = QThread(self)
        self._index_worker = _IndexWorker(self._config, source_type, source_path)
        self._index_worker.moveToThread(self._index_thread)
        self._index_thread.started.connect(self._index_worker.run)
        self._index_worker.progress.connect(self._on_index_progress)
        self._index_worker.finished.connect(self._on_index_job_finished)
        self._index_worker.failed.connect(self._on_index_job_failed)
        self._index_worker.finished.connect(self._index_thread.quit)
        self._index_worker.failed.connect(self._index_thread.quit)
        self._index_thread.finished.connect(self._cleanup_index_job)
        self._index_thread.start()

    def _cleanup_index_job(self) -> None:
        if self._index_worker is not None:
            self._index_worker.deleteLater()
            self._index_worker = None
        if self._index_thread is not None:
            self._index_thread.deleteLater()
            self._index_thread = None

    def _on_index_job_finished(self, db_path: str, source_type: str, source_path: str) -> None:
        self._hide_progress_dialog()
        self._store = PgnStore(IndexHandle(db_path=Path(db_path)))

        chosen: SourceRecord | None = None
        source_id = self._store.get_source_id_by_path(source_path, source_type)
        if source_id is not None:
            for src in self._store.list_sources():
                if src.source_id == source_id:
                    chosen = src
                    break
        if chosen is None:
            sources = self._store.list_sources()
            chosen = sources[0] if sources else None

        self._set_active_source(chosen)
        self._refresh_games(self._query.current_query())
        self._table.viewport().update()
        QApplication.processEvents()
        self._set_indexing_ui_busy(False, f"Indexed {self._last_indexed_game_count} games")

    def _on_index_job_failed(self, message: str) -> None:
        self._hide_progress_dialog()
        self._set_indexing_ui_busy(False, "Indexing failed")
        QMessageBox.critical(self, "Indexing failed", message)

    def _load_pgn_library_file(self) -> None:
        start_dir = str(resolve_path(self._config["paths"]["data_dir"]))
        path, _ = QFileDialog.getOpenFileName(self, "Load PGN Library", start_dir, "PGN Files (*.pgn);;All Files (*)")
        if path:
            self._start_index_job("archive_file", path)

    def _load_pgn_library_directory(self) -> None:
        start_dir = str(resolve_path(self._config["paths"]["data_dir"]))
        path = QFileDialog.getExistingDirectory(self, "Load PGN Folder", start_dir)
        if path:
            self._start_index_job("directory", path)

    def _reindex_current_source(self) -> None:
        if not self._store or not self._active_source_type or not self._active_source_path:
            QMessageBox.information(self, "No source", "No active PGN source is selected.")
            return
        self._start_index_job(self._active_source_type, self._active_source_path)

    def _engine_exe_path(self) -> Path:
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
        except Exception as exc:
            QMessageBox.critical(self, "Stockfish start failed", str(exc))
            self._engine_toggle.setChecked(False)
            self._engine = None

    def _set_engine_plays(self, enabled: bool) -> None:
        self._engine_plays = enabled

    def _engine_should_move_now(self) -> bool:
        if not (self._engine_enabled and self._engine_plays and self._engine):
            return False
        turn_is_white = self._editor.session.board.turn
        engine_is_white = self._engine_side == "White"
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
        except Exception as exc:
            QMessageBox.critical(self, "Stockfish error", str(exc))

    def _on_back(self) -> None:
        if self._editor.step_back():
            self._board_model.rebuild()
            self._on_position_changed()

    def _on_forward(self) -> None:
        if self._editor.step_forward_mainline():
            self._board_model.rebuild()
            self._on_position_changed()

    def _refresh_games(self, query) -> None:
        if not self._store:
            self._table_model.set_rows([])
            self._cont_model.set_rows([])
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
        self._refresh_variations_tab()

    def _default_variation_source_id(self) -> int | None:
        if not self._store:
            return None
        active_cfg = self._config.get("pgn_sources", {}).get("active_source", {})
        cfg_path = active_cfg.get("path")
        cfg_type = active_cfg.get("type")
        if isinstance(cfg_path, str):
            source_id = self._store.get_source_id_by_path(cfg_path, cfg_type if isinstance(cfg_type, str) else None)
            if source_id is not None:
                return source_id
        return self._active_source_id

    def _refresh_variations_tab(self) -> None:
        if not self._store:
            self._cont_model.set_rows([])
            return
        source_id = self._default_variation_source_id()
        if source_id is None:
            self._cont_model.set_rows([])
            return
        game_ids = self._store.list_game_ids_for_source(source_id)
        stats = root_continuation_stats_from_store(self._store, game_ids, max_out=50)
        self._cont_model.set_rows(stats)
        self._cont_view.resizeColumnsToContents()

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
        except Exception as exc:
            QMessageBox.critical(self, "Open game failed", str(exc))

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

        start, end = matches[-1].start(), matches[-1].end()
        cur = self._pgn_text.textCursor()
        cur.setPosition(start)
        cur.setPosition(end, QTextCursor.KeepAnchor)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#0A84FF"))
        fmt.setFontWeight(QFont.Bold)
        cur.mergeCharFormat(fmt)
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

    def _on_position_changed(self) -> None:
        self._set_pgn_text_and_highlight()
        self._refresh_variations_tab()

    def _after_user_move(self) -> None:
        self._on_position_changed()
        self._do_engine_move_if_needed()
