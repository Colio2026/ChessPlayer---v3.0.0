from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl, Signal, QObject
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableView,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from chessplayer.core.pgn_edit import PgnEditor
from chessplayer.pgn.indexer import build_or_rebuild_index_for_source
from chessplayer.pgn.query import default_multisort
from chessplayer.pgn.store import IndexHandle, PgnStore, SourceRecord
from chessplayer.ui.board_model import BoardBridge, BoardListModel
from chessplayer.ui.coach_board import CoachBoardWidget
from chessplayer.ui.coach_panel import CoachPanel
from chessplayer.ui.engine_panel import EnginePanel
from chessplayer.ui.eval_bar import EvalBar
from chessplayer.ui.game_table_model import GameTableModel
from chessplayer.ui.pgn_panel import PgnPanel
from chessplayer.ui.query_builder import QueryBuilder
from chessplayer.ui.variations_panel import VariationsPanel
from chessplayer.utils.paths import resolve_path


# ─── Index worker ─────────────────────────────────────────────────────────────

class _IndexWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(str, str, str)
    failed   = Signal(str)

    def __init__(self, config: dict, source_type: str, source_path: str) -> None:
        super().__init__()
        self._config      = config
        self._source_type = source_type
        self._source_path = source_path

    def run(self) -> None:
        try:
            db_path = build_or_rebuild_index_for_source(
                cfg=self._config,
                source_type=self._source_type,
                source_path=self._source_path,
                progress_cb=lambda count, msg: self.progress.emit(count, msg),
                cancel_cb=None,
            )
            self.finished.emit(str(db_path), self._source_type, self._source_path)
        except Exception as exc:
            self.failed.emit(str(exc))


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    # F2 — single signal the coach widget connects to
    # emitted with (fen, move_history_uci) on every move and navigation
    coachRequested = Signal(str, list)

    def __init__(self, config: dict) -> None:
        super().__init__()
        self._config = config
        self.setWindowTitle("CHESSPLAYER 3.0.0")

        self._editor      = PgnEditor()
        self._editor.new_freeplay()
        pieces_dir        = resolve_path(self._config["paths"]["pieces_dir"])
        self._board_model = BoardListModel(self._editor, pieces_dir=pieces_dir)
        self._bridge      = BoardBridge(self._editor, self._board_model)

        self._store:              PgnStore | None = None
        self._sort                                = default_multisort()
        self._active_source_id:   int | None      = None
        self._active_source_type: str | None      = None
        self._active_source_path: str | None      = None
        self._active_board: str                   = "main"
        self._updating_source_combo               = False

        self._index_thread:    QThread | None          = None
        self._index_worker:    _IndexWorker | None     = None
        self._progress_dialog: QProgressDialog | None  = None
        self._last_indexed_game_count                  = 0

        data_dir = resolve_path(self._config["paths"]["data_dir"])
        db_path  = data_dir / "index.sqlite"
        if db_path.exists():
            self._store = PgnStore(IndexHandle(db_path=db_path))

        self._build_menu_bar()
        self._build_toolbar()
        self._build_ui()
        self._build_shortcuts()

        # Bridge signals
        self._bridge.statusChanged.connect(self._status.setText)
        self._bridge.fenChanged.connect(lambda _fen: self._on_position_changed())
        self._bridge.moveMade.connect(lambda _san: self._after_user_move())

        # Engine
        self._engine_panel.move_ready.connect(self._on_engine_move)
        self._engine_panel.eval_updated.connect(self._on_eval_updated)
        self._engine_panel.eval_updated.connect(self._coach_panel.on_pv_updated)
        self._engine_panel.all_pvs_updated.connect(self._coach_panel.on_all_pvs_updated)
        self._engine_panel.pv_line_clicked.connect(self._on_pv_line_clicked)

        # Variations
        self._variations_panel.status_message.connect(self._status.setText)
        self._variations_panel.move_selected.connect(self._on_variation_move_selected)

        # Coach panel signals
        self._coach_panel.coach_help_requested.connect(self._on_coach_help_requested)
        self._coach_panel.gm_load_requested.connect(self._on_gm_load_requested)
        self._coach_panel.pv_line_load_requested.connect(
            lambda fen, uci, san, title: self._on_pv_line_load_requested(fen, uci, san, title)
        )
        self._coach_panel.weakness_squares_ready.connect(
            self._coach_board.set_weakness_squares
        )
        # PGN panel right-click "Request Coach Note" → force analysis + insert
        self._pgn_panel.coach_help_requested.connect(self._coach_panel.request_help)

        # PGN panel — navigate on click, wire save buttons
        self._pgn_panel.navigate_requested.connect(self._navigate_to_ply)
        self._pgn_panel.navigate_node_requested.connect(self._navigate_to_node)
        self._pgn_panel.promote_variation_requested.connect(self._promote_variation)
        self._pgn_panel.demote_variation_requested.connect(self._demote_variation)
        self._pgn_panel.delete_variation_requested.connect(self._delete_variation)
        self._pgn_panel.delete_from_node_requested.connect(self._delete_from_node)
        self._pgn_panel.comment_line_clicked.connect(self._on_comment_line_clicked)
        self._pgn_panel.enable_save(self._save_game, self._save_game_as)
        self._pgn_panel.enable_save_to_library(self._save_to_library)

        self._choose_initial_source()
        self._on_position_changed()
        self._refresh_games(self._query.current_query())

    # ── Menu bar ──────────────────────────────────────────────────────────────

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

        self._save_action = QAction("Save Game", self)
        self._save_action.setShortcut(QKeySequence("Ctrl+S"))
        self._save_action.triggered.connect(self._save_game)
        file_menu.addAction(self._save_action)

        self._save_as_action = QAction("Save Game As...", self)
        self._save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._save_as_action.triggered.connect(self._save_game_as)
        file_menu.addAction(self._save_as_action)

        self._save_lib_action = QAction("Save to Library", self)
        self._save_lib_action.triggered.connect(self._save_to_library)
        file_menu.addAction(self._save_lib_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    # ── Toolbar ───────────────────────────────────────────────────────────────

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

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root       = QWidget()
        outer      = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        main_split = QSplitter(Qt.Horizontal)

        # Left: Game browser (top) + Coach Board (bottom, hidden until note-line click)
        left   = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(8, 8, 8, 8)
        left_l.setSpacing(4)

        # ── game archive ─────────────────────────────────────────────────
        archive_widget = QWidget()
        archive_l      = QVBoxLayout(archive_widget)
        archive_l.setContentsMargins(0, 0, 0, 0)
        archive_l.addWidget(QLabel("Games"))

        self._query = QueryBuilder()
        self._query.query_changed.connect(lambda q: self._refresh_games(q))
        archive_l.addWidget(self._query)

        self._table_model = GameTableModel()
        self._table       = QTableView()
        self._table.setModel(self._table_model)
        self._table.setSelectionBehavior(QTableView.SelectRows)
        self._table.setSelectionMode(QTableView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.doubleClicked.connect(self._open_selected_game)
        archive_l.addWidget(self._table, 1)

        btn_row = QWidget()
        btn_l   = QHBoxLayout(btn_row)
        btn_l.setContentsMargins(0, 0, 0, 0)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(
            lambda: self._refresh_games(self._query.current_query())
        )
        btn_l.addWidget(self._refresh_btn)
        self._open_btn = QPushButton("Open")
        self._open_btn.clicked.connect(self._open_selected_game)
        btn_l.addWidget(self._open_btn)
        btn_l.addStretch(1)
        archive_l.addWidget(btn_row)

        # ── coach board ───────────────────────────────────────────────────
        pieces_dir_cb = resolve_path(self._config["paths"]["pieces_dir"])
        qml_path_cb   = resolve_path("src/chessplayer/qml/BoardView.qml")
        self._coach_board = CoachBoardWidget(
            pieces_dir = pieces_dir_cb,
            qml_path   = qml_path_cb,
            parent     = self,
        )

        # Vertical splitter: archive on top, coach board below
        self._left_split = QSplitter(Qt.Vertical)
        self._left_split.addWidget(archive_widget)
        self._left_split.addWidget(self._coach_board)
        self._left_split.setCollapsible(0, False)
        self._left_split.setCollapsible(1, True)
        self._coach_board.hide()   # hidden until a line is explicitly loaded
        self._coach_board.closed.connect(self._on_coach_board_closed)
        left_l.addWidget(self._left_split, 1)

        main_split.addWidget(left)

        # Right: Board + panels
        right   = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(8, 8, 8, 8)

        top   = QWidget()
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

        self._engine_panel = EnginePanel(self._config, self)
        right_l.addWidget(self._engine_panel)

        # Eval bar + board + panels side by side
        board_split = QSplitter(Qt.Horizontal)

        self._eval_bar = EvalBar(self)
        board_split.addWidget(self._eval_bar)

        self._board_view = QQuickWidget()
        self._board_view.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self._board_view.rootContext().setContextProperty("piecesModel",       self._board_model)
        self._board_view.rootContext().setContextProperty("bridge",            self._bridge)
        self._board_view.rootContext().setContextProperty("squareIndicators",  [])
        qml_path = resolve_path("src/chessplayer/qml/BoardView.qml")
        self._board_view.setSource(QUrl.fromLocalFile(str(qml_path)))
        errs = self._board_view.errors()
        if errs:
            QMessageBox.critical(
                self, "QML errors", "\n".join([e.toString() for e in errs])
            )
        board_split.addWidget(self._board_view)

        tabs = QTabWidget()
        self._variations_panel = VariationsPanel(self._config, self)
        tabs.addTab(self._variations_panel, "Variations")
        self._pgn_panel = PgnPanel(self._editor, self)
        tabs.addTab(self._pgn_panel, "PGN")
        self._coach_panel = CoachPanel(self._config, self)
        tabs.addTab(self._coach_panel, "Coach")
        board_split.addWidget(tabs)

        board_split.setSizes([28, 520, 420])
        board_split.setCollapsible(0, False)
        right_l.addWidget(board_split, 1)
        main_split.addWidget(right)
        main_split.setSizes([520, 900])

        outer.addWidget(main_split)
        self.setCentralWidget(root)

    # ── Source management ─────────────────────────────────────────────────────

    def _choose_initial_source(self) -> None:
        self._update_source_combo()
        if not self._store:
            return
        sources = self._store.list_sources()
        if sources and self._active_source_id is None:
            self._set_active_source(sources[0])

    def _set_active_source(self, src: SourceRecord | None) -> None:
        if src is None:
            self._active_source_id   = None
            self._active_source_type = None
            self._active_source_path = None
        else:
            self._active_source_id   = src.source_id
            self._active_source_type = src.source_type
            self._active_source_path = src.path
        self._update_source_combo()
        self._variations_panel.set_source(
            self._store, self._active_source_id, self._active_source_path
        )

    def _source_label(self, src: SourceRecord) -> str:
        return f"[{src.source_type}] {src.path}"

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
            self._source_combo.setCurrentIndex(
                current_index if current_index >= 0 else 0
            )
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

    # ── Index job ─────────────────────────────────────────────────────────────

    def _ensure_progress_dialog(self) -> None:
        if self._progress_dialog is None:
            dlg = QProgressDialog("Working...", None, 0, 0, self)
            dlg.setWindowTitle("Please wait")
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
            QMessageBox.information(
                self, "Indexing busy", "A PGN indexing job is already running."
            )
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

    def _on_index_job_finished(
        self, db_path: str, source_type: str, source_path: str
    ) -> None:
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
            chosen  = sources[0] if sources else None

        self._set_active_source(chosen)
        self._refresh_games(self._query.current_query())
        self._set_indexing_ui_busy(
            False, f"Indexed {self._last_indexed_game_count} games"
        )
        # Auto-build / update the move tree so the Variations tab is ready.
        # Incremental: only processes games added since the last tree build.
        self._variations_panel.start_tree_build(incremental=True)

    def _on_index_job_failed(self, message: str) -> None:
        self._hide_progress_dialog()
        self._set_indexing_ui_busy(False, "Indexing failed")
        QMessageBox.critical(self, "Indexing failed", message)

    def _load_pgn_library_file(self) -> None:
        start_dir = str(resolve_path(self._config["paths"]["data_dir"]))
        path, _   = QFileDialog.getOpenFileName(
            self, "Load PGN Library", start_dir,
            "PGN Files (*.pgn);;All Files (*)"
        )
        if path:
            self._start_index_job("archive_file", path)

    def _load_pgn_library_directory(self) -> None:
        start_dir = str(resolve_path(self._config["paths"]["data_dir"]))
        path      = QFileDialog.getExistingDirectory(
            self, "Load PGN Folder", start_dir
        )
        if path:
            self._start_index_job("directory", path)

    def _reindex_current_source(self) -> None:
        if (
            not self._store
            or not self._active_source_type
            or not self._active_source_path
        ):
            QMessageBox.information(
                self, "No source", "No active PGN source is selected."
            )
            return
        self._start_index_job(self._active_source_type, self._active_source_path)

    # ── Save game ─────────────────────────────────────────────────────────────

    def _save_game(self) -> bool:
        """
        Ctrl+S: if we have a tracked standalone save path use it,
        otherwise fall through to Save As.
        """
        # If the game was loaded from a library we don't overwrite it here —
        # that's Save to Library. For standalone files we track _save_path.
        save_path = getattr(self, "_save_path", None)
        if save_path:
            try:
                self._editor.export_pgn_to_file(Path(save_path))
                self._pgn_panel.refresh()
                self._status.setText(f"Saved → {save_path}")
                return True
            except Exception as exc:
                QMessageBox.critical(self, "Save failed", str(exc))
                return False
        return self._save_game_as()

    def _save_game_as(self) -> bool:
        """Save to a user-chosen file."""
        start_dir = str(resolve_path(self._config["paths"]["data_dir"]))
        path, _   = QFileDialog.getSaveFileName(
            self, "Save Game As", start_dir, "PGN Files (*.pgn);;All Files (*)"
        )
        if not path:
            return False
        if not path.lower().endswith(".pgn"):
            path += ".pgn"
        try:
            self._editor.export_pgn_to_file(Path(path))
            self._save_path = path        # remember for future Ctrl+S
            self._status.setText(f"Saved → {path}")
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False

    def _save_to_library(self) -> None:
        """
        Replace the original game in the source library file and re-index.
        """
        if (
            self._editor.source_pgn_path is None
            or self._editor.source_offset is None
        ):
            QMessageBox.information(
                self,
                "Not a library game",
                "This game was not opened from a library.\n"
                "Use Save As to save it to a file, then load that file as a library.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Save to Library",
            f"Replace the original game in:\n{self._editor.source_pgn_path}\n\n"
            "This will rewrite the library file and trigger a re-index. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            self._editor.replace_in_library_file()
            self._status.setText("Saved to library — re-indexing...")
            # Re-index so the library reflects the updated game
            if self._active_source_type and self._active_source_path:
                self._start_index_job(
                    self._active_source_type, self._active_source_path
                )
        except Exception as exc:
            QMessageBox.critical(self, "Save to Library failed", str(exc))

    # ── Navigation ────────────────────────────────────────────────────────────

    def _navigate_to_ply(self, ply: int) -> None:
        """Called when user clicks a mainline move in the PGN panel."""
        if self._editor.navigate_to_ply(ply):
            self._board_model.rebuild()
            self._on_position_changed()

    def _navigate_to_node(self, node) -> None:
        """Called when user clicks a variation move in the PGN panel."""
        if self._editor.navigate_to_node(node):
            self._board_model.rebuild()
            self._on_position_changed()

    def _promote_variation(self, node) -> None:
        """Promote a variation one slot toward mainline and refresh."""
        if self._editor.promote_variation(node):
            self._board_model.rebuild()
            self._on_position_changed()

    def _demote_variation(self, node) -> None:
        """Demote a variation one slot away from mainline and refresh."""
        if self._editor.demote_variation(node):
            self._board_model.rebuild()
            self._on_position_changed()

    def _delete_variation(self, node) -> None:
        """Remove an entire variation branch from the tree."""
        if self._editor.delete_variation(node):
            self._board_model.rebuild()
            self._on_position_changed()

    def _delete_from_node(self, node) -> None:
        """Truncate the game tree at node, removing it and everything after."""
        if self._editor.delete_from_node(node):
            self._board_model.rebuild()
            self._on_position_changed()

    def _on_back(self) -> None:
        if self._editor.step_back():
            self._board_model.rebuild()
            self._on_position_changed()

    def _on_forward(self) -> None:
        if self._editor.step_forward_mainline():
            self._board_model.rebuild()
            self._on_position_changed()

    # ── Game browser ──────────────────────────────────────────────────────────

    def _refresh_games(self, query) -> None:
        self._table_model.reset_query(
            store=self._store,
            query=query,
            sort=self._sort,
            source_id=self._active_source_id,
        )

    def _open_selected_game(self) -> None:
        if not self._store:
            QMessageBox.information(
                self, "No index", "No index.sqlite found. Build index first."
            )
            return
        sel = self._table.selectionModel()
        if not sel or not sel.hasSelection():
            return
        row     = sel.selectedRows()[0].row()
        game_id = self._table_model.game_id_at(row)
        if game_id is None:
            return

        # Check dirty before replacing current game
        if self._editor.dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved changes",
                "The current game has unsaved changes. Discard and open new game?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        try:
            # Get the pgn_path and offset so Save to Library knows where it came from
            conn = self._store._connect()
            row_data = conn.execute(
                "SELECT pgn_path, offset_bytes FROM games WHERE game_id=?",
                (game_id,)
            ).fetchone()
            conn.close()

            pgn = self._store.open_game_pgn_text(game_id)
            self._editor.load_pgn_text(pgn)

            # Track source for Save to Library
            if row_data:
                from pathlib import Path as _Path
                self._editor.source_pgn_path = _Path(row_data[0])
                self._editor.source_offset   = int(row_data[1])

            self._save_path = None   # clear any previous standalone save path
            self._board_model.rebuild()
            self._on_position_changed()
            # Show starting position on coach board so it's never empty
            import chess as _chess
            self._coach_board.load_line(
                _chess.Board().fen(), [], []
            )
        except Exception as exc:
            QMessageBox.critical(self, "Open game failed", str(exc))

    # ── Position change ───────────────────────────────────────────────────────

    def _on_position_changed(self) -> None:
        print(f"[MAIN] _on_position_changed: editor.loaded={hasattr(self._editor, 'loaded') and self._editor.loaded is not None}")
        # Coach board is tied to a specific position — clear it whenever the main board moves
        if self._coach_board.isVisible():
            self._coach_board.clear()
            self._coach_board.hide()
        prefix        = self._editor.played_prefix_uci()
        white_to_move = self._editor.session.board.turn
        self._pgn_panel.refresh()
        self._variations_panel.refresh(prefix)
        # F2 — notify coach of every position change
        self.coachRequested.emit(
            self._editor.session.board.fen(), list(prefix)
        )
        self._coach_panel.queue_analysis(self._editor.session.board, history=[], side='white' if white_to_move else 'black')
        # Always trigger analysis so eval updates on navigation, game load, etc.
        self._engine_panel.trigger_analysis(prefix, white_to_move)

    def _after_user_move(self) -> None:
        self._on_position_changed()   # already calls trigger_analysis inside

    def _on_engine_move(self, uci: str) -> None:
        res = self._editor.apply_uci_move(uci)
        if res.ok:
            self._board_model.rebuild()
            self._on_position_changed()

    def _on_variation_move_selected(self, san: str) -> None:
        """Play a continuation clicked in the Variations tab."""
        try:
            move = self._editor.session.board.parse_san(san)
        except Exception:
            return
        res = self._editor.apply_uci_move(move.uci())
        if res.ok:
            self._board_model.rebuild()
            self._on_position_changed()

    # ── Dirty close prompt ────────────────────────────────────────────────────

    def _on_pv_line_clicked(self, base_moves: list, pv_uci: list) -> None:
        """
        Navigate the board through a clicked Stockfish PV line.
        Resets to the base position, then applies each PV move as a variation.
        """
        if not pv_uci:
            return

        # Step 1: navigate back to the base position (end of real game moves)
        self._editor.navigate_to_ply(len(base_moves))

        # Step 2: push each PV move as a variation without reordering existing moves
        for uci in pv_uci:
            res = self._editor.apply_uci_move(uci, promote=False)
            if not res.ok:
                break

        self._board_model.rebuild()
        self._on_position_changed()

    def _on_comment_line_clicked(self, base_ply: int, uci_list: list) -> None:
        """
        Show the clicked note line on the Coach Board.
        uci_list is the PREFIX up to the clicked move, so the coach board
        always lands on exactly the move that was clicked.
        The main board is never touched.
        """
        if not uci_list or not self._editor.loaded:
            return
        # Build a throw-away board at base_ply (mainline)
        import chess
        board = chess.Board()
        node  = self._editor.loaded.game
        for _ in range(base_ply):
            if not node.variations:
                break
            node = node.variations[0]
            board.push(node.move)
        base_fen = board.fen()
        # SAN labels for the full clicked prefix
        san_list = self._editor.uci_to_san(base_ply, uci_list)
        # clicked_idx = index of the move that was clicked (last in prefix)
        clicked_idx = len(uci_list) - 1
        self._coach_board.show()
        self._coach_board.load_line(
            base_fen, uci_list, san_list, start_idx=clicked_idx
        )
        self._set_active_board("coach")

    def _on_coach_board_closed(self) -> None:
        self._coach_board.hide()
        self._set_active_board("main")

    def _on_eval_updated(self, result) -> None:
        white_to_move = self._editor.session.board.turn
        self._eval_bar.update_eval(result, white_to_move)

    def _on_coach_help_requested(self, output) -> None:
        """
        Insert a CoachOutput as a PGN comment at the current move.
        Plan sentences + top recommended moves as parenthesised SAN tokens.
        The notation panels convert those tokens to clickable coach:// links.
        """
        import chess as _chess
        board = self._editor.session.board

        # Build SAN strings for up to 3 top move_flags
        priority = {"engine_best": 0, "engine_good": 1,
                    "attack_target": 2, "kingside_break": 3}
        ranked = sorted(
            output.move_flags,
            key=lambda f: priority.get(f.get("flag", ""), 99)
        )
        san_tokens: list[str] = []
        seen: set[str] = set()
        for f in ranked:
            uci = f.get("move", "")
            if uci and uci not in seen:
                try:
                    san_tokens.append(board.san(_chess.Move.from_uci(uci)))
                    seen.add(uci)
                except Exception:
                    pass
            if len(san_tokens) == 3:
                break

        plan_text = "  ".join(output.plan_sentences[:2])
        rec_part  = (
            "  Recommended: (" + " ".join(san_tokens) + ")"
            if san_tokens else ""
        )
        comment = f"\u265f {output.headline}  {plan_text}{rec_part}"

        self._editor.insert_comment(comment)
        self._pgn_panel.refresh()
        self._status.setText("Coach note inserted.")

    def _on_pv_line_load_requested(
        self, base_fen: str, pv_uci: list, pv_san: list, title: str = ''
    ) -> None:
        """
        Load a Stockfish PV line into the Coach Board so the user can
        step through it move by move using the vertical move list.
        """
        if not pv_uci or not base_fen:
            return
        san_labels = pv_san if pv_san else pv_uci
        self._coach_board.set_title(title or "Coach Line")
        self._coach_board.load_line(base_fen, pv_uci, san_labels)
        self._coach_board.show()
        # Ensure the coach board splitter pane is open
        sizes = self._left_split.sizes()
        if sizes[1] < 80:
            total = sum(sizes)
            self._left_split.setSizes([max(total - 300, 100), 300])
        self._set_active_board("coach")

    def _on_gm_load_requested(self, prec) -> None:
        """
        Load a GM precedent:
          1. Parse game, navigate to matched ply.
          2. Load continuation to the coach board.
          3. Insert continuation as a variation in the current game.
        """
        if not self._store:
            return
        try:
            pgn_text = self._store.open_game_pgn_text(int(prec.game_id))
        except Exception:
            return

        import chess.pgn as _pgn, io as _io, chess as _chess

        game = _pgn.read_game(_io.StringIO(pgn_text))
        if game is None:
            return

        base_fen = game.headers.get("FEN", _chess.Board().fen())
        board    = _chess.Board(base_fen)
        node     = game
        for _i in range(prec.ply):
            if not node.variations:
                break
            node = node.variations[0]
            board.push(node.move)

        # Extract up to 8 continuation moves
        cont_board = board.copy()
        uci_list:  list[str] = []
        san_list:  list[str] = []
        cont_node = node
        for _j in range(8):
            if not cont_node.variations:
                break
            cont_node = cont_node.variations[0]
            move      = cont_node.move
            san_list.append(cont_board.san(move))
            cont_board.push(move)
            uci_list.append(move.uci())

        # Load to coach board
        self._coach_board.show()
        self._coach_board.load_line(board.fen(), uci_list, san_list)

        # Insert as variation from current position
        current_ply = len(self._editor.session.board.move_stack)
        self._editor.navigate_to_ply(current_ply)
        for uci in uci_list:
            res = self._editor.apply_uci_move(uci, promote=False)
            if not res.ok:
                break

        self._board_model.rebuild()
        self._on_position_changed()

    def closeEvent(self, event) -> None:
        if self._editor.dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved changes",
                "The current game has unsaved changes.\n\nSave before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                saved = self._save_game()
                if not saved:
                    event.ignore()
                    return
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        event.accept()

    def _build_shortcuts(self) -> None:
        """Wire all keyboard shortcuts.  Called once after _build_ui."""
        def sc(key, slot):
            QShortcut(QKeySequence(key), self).activated.connect(slot)

        sc("Left",    self._kb_back)
        sc("Right",   self._kb_forward)
        sc("Ctrl+S",  self._save_game)
        sc("Ctrl+F",  lambda: self._bridge.setFlipped(not self._board_model.flipped()))
        sc("Ctrl+E",  self._toggle_engine_panel)

        # Track focus: main board clicked
        self._board_view.mousePressEvent = self._make_focus_handler(
            "main", self._board_view.__class__.mousePressEvent
        )
        # Track focus: coach board back/forward buttons
        self._coach_board._back_btn.clicked.connect(
            lambda: self._set_active_board("coach")
        )
        self._coach_board._fwd_btn.clicked.connect(
            lambda: self._set_active_board("coach")
        )
        # Coach board becomes active when a line loads
        # (load_line is called from _on_comment_line_clicked)

    def _make_focus_handler(self, board_name: str, original):
        widget = self._board_view
        def handler(event):
            self._set_active_board(board_name)
            original(widget, event)
        return handler

    def _set_active_board(self, name: str) -> None:
        self._active_board = name

    def _toggle_engine_panel(self) -> None:
        visible = self._engine_panel.isVisible()
        self._engine_panel.setVisible(not visible)

    def _kb_back(self) -> None:
        """Left arrow — step back on whichever board was last used."""
        if self._active_board == "coach":
            self._coach_board._on_back()
        else:
            self._on_back()

    def _kb_forward(self) -> None:
        """Right arrow — step forward on whichever board was last used."""
        if self._active_board == "coach":
            self._coach_board._on_forward()
        else:
            self._on_forward()

    def closeEvent(self, event) -> None:
        if self._editor.dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved changes",
                "The current game has unsaved changes.\n\nSave before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                saved = self._save_game()
                if not saved:
                    event.ignore()
                    return
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        event.accept()
