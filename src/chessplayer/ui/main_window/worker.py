from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from pgn.indexer import build_or_rebuild_index_for_source


class IndexWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(str, str, str)  # db_path, source_type, source_path
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
        except Exception as e:
            self.failed.emit(str(e))
