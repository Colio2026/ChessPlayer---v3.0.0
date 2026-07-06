from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from chessplayer.pgn.continuations import query_continuations
from chessplayer.pgn.move_tree import MoveTree, build_tree
from chessplayer.pgn.store import PgnStore
from chessplayer.ui.continuation_stats_model import ContinuationStatsModel


class _TreeWorker(QObject):
    progress    = Signal(int, str)
    total_known = Signal(int)   # emitted once with game count before scanning starts
    finished    = Signal()
    failed      = Signal(str)

    def __init__(
        self,
        config: dict,
        store: PgnStore,
        source_id: int,
        source_path: str,
        incremental: bool = False,
    ) -> None:
        super().__init__()
        self._config      = config
        self._store       = store
        self._source_id   = source_id
        self._source_path = source_path
        self._incremental = incremental

    def run(self) -> None:
        try:
            self.total_known.emit(0)   # start indeterminate; build_tree reports actual total
            build_tree(
                cfg=self._config,
                source_path=self._source_path,
                pgn_store=self._store,
                source_id=self._source_id,
                progress_cb=lambda count, msg: self.progress.emit(count, msg),
                cancel_cb=None,
                total_cb=lambda t: self.total_known.emit(t),
                incremental=self._incremental,
            )
            self.finished.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class VariationsPanel(QWidget):
    """
    Variations tab.

    Workflow
    --------
    1. When a PGN library is loaded, a tree build/update is triggered
       automatically in the background.
    2. On next launch the tree loads from disk in < 1 second.
    3. Every position change calls refresh(prefix_uci).
       The tree query is a single dict lookup — < 1 ms regardless of
       library size.
    4. "Rebuild Tree" — full rescan (minutes for large databases).
       A progress bar shows exact percentage during the scan.
    5. "Update Tree"  — incremental: only new games added since the last
       build (near-instant for small additions).

    A 100 ms debounce on refresh() means rapid Back/Forward clicks never
    trigger more than one lookup per navigation stop.
    """

    status_message = Signal(str)
    move_selected  = Signal(str)   # emits SAN of the clicked continuation
    tree_built     = Signal()      # emitted when a build/update completes successfully

    def __init__(self, config: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config       = config
        self._move_tree:   MoveTree | None    = None
        self._store:       PgnStore | None    = None
        self._source_id:   int | None         = None
        self._source_path: str | None         = None
        self._thread:      QThread | None     = None
        self._worker:      _TreeWorker | None = None
        self._pending_prefix: list[str]       = []

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(100)
        self._debounce.timeout.connect(self._run_query)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Top row: buttons + status label
        top_row   = QWidget()
        top_row_l = QHBoxLayout(top_row)
        top_row_l.setContentsMargins(0, 0, 0, 0)
        top_row_l.setSpacing(6)

        self._rebuild_btn = QPushButton("Rebuild Tree")
        self._rebuild_btn.setToolTip(
            "Discard and rebuild the move tree from scratch.\n"
            "Use this if the library was heavily edited."
        )
        self._rebuild_btn.clicked.connect(self._on_rebuild_clicked)
        top_row_l.addWidget(self._rebuild_btn)

        self._update_btn = QPushButton("Update Tree")
        self._update_btn.setToolTip(
            "Add games imported since the last build to the existing tree.\n"
            "Much faster than a full rebuild."
        )
        self._update_btn.clicked.connect(self._on_update_clicked)
        top_row_l.addWidget(self._update_btn)

        self._tree_label = QLabel("No tree built")
        self._tree_label.setStyleSheet("color:gray; font-style:italic;")
        top_row_l.addWidget(self._tree_label)
        top_row_l.addStretch(1)
        layout.addWidget(top_row)

        # Progress bar — hidden when idle, shown during build
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(0)   # indeterminate until total_known
        self._progress_bar.setFixedHeight(14)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet(
            "QProgressBar {"
            "  background:#1A1A1A; border:1px solid #37474F;"
            "  border-radius:3px; color:#CFD8DC; font-size:9px;"
            "}"
            "QProgressBar::chunk {"
            "  background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "    stop:0 #1565C0, stop:1 #42A5F5);"
            "  border-radius:2px;"
            "}"
        )
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        # Status line (games matched)
        self._match_label = QLabel("")
        self._match_label.setStyleSheet("color:#666666; font-size:8.5pt; font-style:italic;")
        layout.addWidget(self._match_label)

        layout.addWidget(QLabel("Continuations from current position:"))

        # Table
        self._model = ContinuationStatsModel()
        self._view  = QTableView()
        self._view.setModel(self._model)
        self._view.setSelectionMode(QTableView.NoSelection)
        self._view.setEditTriggers(QTableView.NoEditTriggers)
        self._view.setAlternatingRowColors(True)
        self._view.setShowGrid(False)
        self._view.verticalHeader().setVisible(False)
        self._view.horizontalHeader().setStretchLastSection(False)
        self._view.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 6):
            self._view.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
        self._view.activated.connect(self._on_row_activated)
        layout.addWidget(self._view, 1)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_source(
        self,
        store: PgnStore | None,
        source_id: int | None,
        source_path: str | None,
    ) -> None:
        """Called by MainWindow when the active library changes."""
        self._store       = store
        self._source_id   = source_id
        self._source_path = source_path
        self._move_tree   = None
        self._model.set_rows([])
        self._match_label.setText("")
        self._try_load_existing_tree()

    def refresh(self, prefix_uci: list[str]) -> None:
        """
        Called by MainWindow on every position change.
        Debounced — the query runs 100 ms after the last call.
        """
        self._pending_prefix = list(prefix_uci)
        self._debounce.start()

    def start_tree_build(self, incremental: bool = True) -> None:
        """
        Trigger a tree build/update programmatically (called by MainWindow
        after indexing finishes).

        incremental=True  → only process new games (fast, default)
        incremental=False → full rebuild from scratch
        """
        if not self._store or self._source_id is None or not self._source_path:
            return
        if self._thread is not None:
            return   # already running; the in-progress build will cover it
        self._start_worker(incremental=incremental)

    # ── Tree management ───────────────────────────────────────────────────────

    def _try_load_existing_tree(self) -> None:
        if not self._source_path:
            self._set_tree_label("No source selected", "gray")
            return
        tree = MoveTree.load(self._config, self._source_path)
        if tree.is_built():
            self._move_tree = tree
            self._set_tree_label(
                f"Tree ready ✓  ({tree.total_games():,} games)", "#4CAF50"
            )
        else:
            self._move_tree = None
            self._set_tree_label("No tree — will build automatically on load", "gray")

    def _on_rebuild_clicked(self) -> None:
        if not self._store or self._source_id is None or not self._source_path:
            QMessageBox.information(
                self, "No source",
                "Load a PGN library before building the move tree."
            )
            return
        if self._thread is not None:
            QMessageBox.information(self, "Busy", "A tree build is already running.")
            return

        reply = QMessageBox.question(
            self, "Rebuild move tree",
            "This will discard the existing tree and rescan every game.\n"
            "It may take several minutes for large databases.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._start_worker(incremental=False)

    def _on_update_clicked(self) -> None:
        if not self._store or self._source_id is None or not self._source_path:
            QMessageBox.information(
                self, "No source",
                "Load a PGN library before updating the move tree."
            )
            return
        if self._thread is not None:
            QMessageBox.information(self, "Busy", "A tree build is already running.")
            return
        self._start_worker(incremental=True)

    def _start_worker(self, incremental: bool) -> None:
        verb = "Updating" if incremental else "Building"
        self._rebuild_btn.setEnabled(False)
        self._update_btn.setEnabled(False)
        self._set_tree_label(f"{verb}…", "orange")
        self.status_message.emit(f"{verb} move tree…")

        # Progress bar: indeterminate until total_known fires
        self._progress_bar.setMaximum(0)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat(f"{verb} tree…")
        self._progress_bar.show()

        self._thread = QThread(self)
        self._worker = _TreeWorker(
            config=self._config,
            store=self._store,
            source_id=self._source_id,
            source_path=self._source_path,
            incremental=incremental,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.total_known.connect(self._on_total_known)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()

    def _on_total_known(self, total: int) -> None:
        """Switch progress bar from indeterminate to percentage mode."""
        if total > 0:
            self._progress_bar.setMaximum(total)
            self._progress_bar.setFormat("%v / %m games  (%p%)")
        # total == 0 → keep indeterminate (incremental fast path)

    def _on_progress(self, count: int, message: str) -> None:
        self.status_message.emit(message)
        self._set_tree_label(message, "orange")
        if self._progress_bar.maximum() > 0:
            self._progress_bar.setValue(count)
        QApplication.processEvents()

    def _on_finished(self) -> None:
        self._progress_bar.hide()
        self._rebuild_btn.setEnabled(True)
        self._update_btn.setEnabled(True)
        self._try_load_existing_tree()
        self.status_message.emit("Move tree ready")
        self.tree_built.emit()

    def _on_failed(self, message: str) -> None:
        self._progress_bar.hide()
        self._rebuild_btn.setEnabled(True)
        self._update_btn.setEnabled(True)
        self._set_tree_label("Build failed", "red")
        QMessageBox.critical(self, "Tree build failed", message)

    def _cleanup(self) -> None:
        if self._worker:
            self._worker.deleteLater()
            self._worker = None
        if self._thread:
            self._thread.deleteLater()
            self._thread = None

    # ── Query (debounced) ─────────────────────────────────────────────────────

    def _run_query(self) -> None:
        stats = query_continuations(
            self._move_tree, self._pending_prefix, max_out=30
        )
        self._model.set_rows(stats)
        self._view.resizeColumnsToContents()

        if self._move_tree and self._move_tree.is_built():
            if stats:
                total = sum(s.count for s in stats)
                self._match_label.setText(
                    f"{total:,} games reached this position"
                )
            else:
                self._match_label.setText("No games reached this position.")
        else:
            self._match_label.setText("")

    def _on_row_activated(self, index) -> None:
        """Emit the SAN of the row the user clicked or pressed Enter on."""
        row = self._model._rows
        if 0 <= index.row() < len(row):
            self.move_selected.emit(row[index.row()].san)

    def _set_tree_label(self, text: str, color: str) -> None:
        self._tree_label.setText(text)
        self._tree_label.setStyleSheet(f"color:{color}; font-style:italic;")
