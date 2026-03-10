from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from pgn.continuations import query_continuations
from pgn.move_tree import MoveTree, build_tree
from pgn.store import PgnStore
from ui.continuation_stats_model import ContinuationStatsModel

from PySide6.QtCore import QObject, QThread
from PySide6.QtWidgets import QApplication


class _TreeWorker(QObject):
    progress = Signal(int, str)
    finished = Signal()
    failed   = Signal(str)

    def __init__(
        self,
        config: dict,
        store: PgnStore,
        source_id: int,
        source_path: str,
    ) -> None:
        super().__init__()
        self._config      = config
        self._store       = store
        self._source_id   = source_id
        self._source_path = source_path

    def run(self) -> None:
        try:
            build_tree(
                cfg=self._config,
                source_path=self._source_path,
                pgn_store=self._store,
                source_id=self._source_id,
                progress_cb=lambda count, msg: self.progress.emit(count, msg),
                cancel_cb=None,
            )
            self.finished.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class VariationsPanel(QWidget):
    """
    Variations tab panel.

    Owns the move tree lifecycle:
      - "Initialize" button triggers the one-time background build
      - Auto-loads an existing tree on source change
      - refresh(prefix_uci) is called by MainWindow on every position change

    Completely isolated from the game browser and PGN panel.
    """

    status_message = Signal(str)    # forwards progress messages to MainWindow status bar

    def __init__(self, config: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config      = config
        self._move_tree:  MoveTree | None   = None
        self._store:      PgnStore | None   = None
        self._source_id:  int | None        = None
        self._source_path: str | None       = None
        self._thread:     QThread | None    = None
        self._worker:     _TreeWorker | None = None

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Init row
        init_row   = QWidget()
        init_row_l = QHBoxLayout(init_row)
        init_row_l.setContentsMargins(0, 0, 0, 0)

        self._init_btn = QPushButton("Initialize Game Base Move Tree")
        self._init_btn.setToolTip(
            "Analyses every game in the active library and builds a position tree.\n"
            "Slow once — instant every time after that."
        )
        self._init_btn.clicked.connect(self._on_init_clicked)
        init_row_l.addWidget(self._init_btn)

        self._tree_label = QLabel("No tree built")
        self._tree_label.setStyleSheet("color: gray; font-style: italic;")
        init_row_l.addWidget(self._tree_label)
        init_row_l.addStretch(1)
        layout.addWidget(init_row)

        layout.addWidget(QLabel("Continuations from current position:"))

        # Continuation table
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
        layout.addWidget(self._view, 1)

    # ── public API ────────────────────────────────────────────────────────────

    def set_source(
        self,
        store: PgnStore | None,
        source_id: int | None,
        source_path: str | None,
    ) -> None:
        """
        Called by MainWindow when the active library changes.
        Resets the tree and tries to load an existing one from disk.
        """
        self._store       = store
        self._source_id   = source_id
        self._source_path = source_path
        self._move_tree   = None
        self._model.set_rows([])
        self._try_load_existing_tree()

    def refresh(self, prefix_uci: list[str]) -> None:
        """
        Called by MainWindow on every board position change.
        Queries the in-memory tree — instant if built, no-op if not.
        """
        stats = query_continuations(self._move_tree, prefix_uci, max_out=50)
        self._model.set_rows(stats)
        self._view.resizeColumnsToContents()

    # ── tree management ───────────────────────────────────────────────────────

    def _try_load_existing_tree(self) -> None:
        if not self._source_path:
            self._set_tree_label("No source selected", "gray")
            return

        tree = MoveTree.load(self._config, self._source_path)
        if tree.is_built():
            self._move_tree = tree
            self._set_tree_label("Tree ready ✓", "green")
        else:
            self._move_tree = None
            self._set_tree_label("No tree — click Initialize to build", "gray")

    def _on_init_clicked(self) -> None:
        if not self._store or self._source_id is None or not self._source_path:
            QMessageBox.information(
                self, "No source",
                "Load a PGN library before building the move tree."
            )
            return

        if self._thread is not None:
            QMessageBox.information(
                self, "Busy", "A tree build is already running."
            )
            return

        reply = QMessageBox.question(
            self,
            "Build move tree",
            "This will analyse every game in the active library.\n"
            "It may take several minutes for large databases.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._init_btn.setEnabled(False)
        self._set_tree_label("Building tree...", "orange")
        self.status_message.emit("Building move tree...")

        self._thread = QThread(self)
        self._worker = _TreeWorker(
            config=self._config,
            store=self._store,
            source_id=self._source_id,
            source_path=self._source_path,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()

    def _on_progress(self, _count: int, message: str) -> None:
        self.status_message.emit(message)
        QApplication.processEvents()

    def _on_finished(self) -> None:
        self._init_btn.setEnabled(True)
        self._try_load_existing_tree()
        self.status_message.emit("Move tree built successfully")

    def _on_failed(self, message: str) -> None:
        self._init_btn.setEnabled(True)
        self._set_tree_label("Build failed", "red")
        QMessageBox.critical(self, "Tree build failed", message)

    def _cleanup(self) -> None:
        if self._worker:
            self._worker.deleteLater()
            self._worker = None
        if self._thread:
            self._thread.deleteLater()
            self._thread = None

    def _set_tree_label(self, text: str, color: str) -> None:
        self._tree_label.setText(text)
        self._tree_label.setStyleSheet(
            f"color: {color}; font-style: italic;"
        )
