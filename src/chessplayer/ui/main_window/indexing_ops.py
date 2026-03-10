from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressDialog

from pgn.store import IndexHandle, PgnStore, SourceRecord
from utils.paths import resolve_path

from .worker import IndexWorker


class IndexingMixin:
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
        self._index_worker = IndexWorker(self._config, source_type, source_path)
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

        source_id = self._store.get_source_id_by_path(source_path, source_type)
        chosen: SourceRecord | None = None
        if source_id is not None:
            for src in self._store.list_sources():
                if src.source_id == source_id:
                    chosen = src
                    break

        if chosen is None:
            sources = self._store.list_sources()
            chosen = sources[0] if sources else None

        self._set_active_source(chosen, save_config=True)
        self._update_source_combo()
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
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load PGN Library",
            start_dir,
            "PGN Files (*.pgn);;All Files (*)",
        )
        if not path:
            return
        self._start_index_job("archive_file", path)

    def _load_pgn_library_directory(self) -> None:
        start_dir = str(resolve_path(self._config["paths"]["data_dir"]))
        path = QFileDialog.getExistingDirectory(
            self,
            "Load PGN Folder",
            start_dir,
        )
        if not path:
            return
        self._start_index_job("directory", path)

    def _reindex_current_source(self) -> None:
        if not self._store or not self._active_source_type or not self._active_source_path:
            QMessageBox.information(self, "No source", "No active PGN source is selected.")
            return
        self._start_index_job(self._active_source_type, self._active_source_path)
