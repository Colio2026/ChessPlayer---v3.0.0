from __future__ import annotations

from config.loader import save_user_config_patch
from pgn.store import SourceRecord


class SourceStateMixin:
    def _save_active_source_to_user_config(self) -> None:
        patch = {
            "ui": {
                "last_source_id": self._active_source_id,
                "last_source_type": self._active_source_type,
                "last_source_path": self._active_source_path,
            }
        }
        save_user_config_patch(patch)

    def _restore_active_source_from_config(self) -> None:
        if not self._store:
            self._update_source_combo()
            return

        ui_cfg = self._config.get("ui", {})
        wanted_id = ui_cfg.get("last_source_id")
        wanted_type = ui_cfg.get("last_source_type")
        wanted_path = ui_cfg.get("last_source_path")

        sources = self._store.list_sources()
        if not sources:
            self._active_source_id = None
            self._active_source_type = None
            self._active_source_path = None
            self._update_source_combo()
            return

        chosen: SourceRecord | None = None

        if isinstance(wanted_id, int):
            for src in sources:
                if src.source_id == wanted_id:
                    chosen = src
                    break

        if chosen is None and wanted_path:
            source_id = self._store.get_source_id_by_path(
                wanted_path,
                wanted_type if isinstance(wanted_type, str) else None,
            )
            if source_id is not None:
                for src in sources:
                    if src.source_id == source_id:
                        chosen = src
                        break

        if chosen is None:
            chosen = sources[0]

        self._set_active_source(chosen, save_config=False)
        self._update_source_combo()

    def _source_label(self, src: SourceRecord) -> str:
        return f"[{src.source_type}] {src.path}"

    def _set_active_source(self, src: SourceRecord | None, save_config: bool = True) -> None:
        if src is None:
            self._active_source_id = None
            self._active_source_type = None
            self._active_source_path = None
        else:
            self._active_source_id = src.source_id
            self._active_source_type = src.source_type
            self._active_source_path = src.path

        self._update_source_combo()

        if save_config:
            self._save_active_source_to_user_config()

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
                if self._active_source_id is not None and src.source_id == self._active_source_id:
                    current_index = idx

            if current_index >= 0:
                self._source_combo.setCurrentIndex(current_index)
            else:
                self._source_combo.setCurrentIndex(0)
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
                self._set_active_source(src, save_config=True)
                self._refresh_games(self._query.current_query())
                return
