from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QWidget,
)

from engine.uci_engine import UciEngine
from utils.paths import resolve_path


class EnginePanel(QWidget):
    """
    Self-contained engine control bar.

    Exposes a single signal: move_ready(uci: str)
    MainWindow connects to this and applies the move to the editor.

    All engine state lives here. MainWindow never touches UciEngine directly.
    """

    move_ready = Signal(str)   # emits UCI string when engine has chosen a move

    def __init__(self, config: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config

        self._engine:          UciEngine | None = None
        self._enabled:         bool             = False
        self._plays:           bool             = False
        self._side:            str              = "Black"
        self._movetime_ms:     int              = 250

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._toggle = QCheckBox("Stockfish")
        self._toggle.stateChanged.connect(lambda s: self._on_toggle(bool(s)))
        layout.addWidget(self._toggle)

        self._play_toggle = QCheckBox("Play vs Engine")
        self._play_toggle.stateChanged.connect(lambda s: self._set_plays(bool(s)))
        layout.addWidget(self._play_toggle)

        layout.addWidget(QLabel("Side:"))
        self._side_combo = QComboBox()
        self._side_combo.addItems(["Black", "White"])
        self._side_combo.currentTextChanged.connect(lambda t: setattr(self, "_side", t))
        layout.addWidget(self._side_combo)

        layout.addWidget(QLabel("MoveTime ms:"))
        self._time_combo = QComboBox()
        self._time_combo.addItems(["100", "250", "500", "1000"])
        self._time_combo.setCurrentText("250")
        self._time_combo.currentTextChanged.connect(
            lambda t: setattr(self, "_movetime_ms", int(t))
        )
        layout.addWidget(self._time_combo)
        layout.addStretch(1)

    # ── public API ────────────────────────────────────────────────────────────

    def trigger_move_if_needed(self, prefix_uci: list[str], white_to_move: bool) -> None:
        """
        Called by MainWindow after every position change.
        If the engine is active and it is the engine's turn, analyses and
        emits move_ready(uci).
        """
        if not self._should_move(white_to_move):
            return
        assert self._engine is not None
        try:
            bm = self._engine.analyze_movetime(prefix_uci, movetime_ms=self._movetime_ms)
            if bm.uci and bm.uci != "0000":
                self.move_ready.emit(bm.uci)
        except Exception as exc:
            QMessageBox.critical(self, "Stockfish error", str(exc))

    def stop_engine(self) -> None:
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
            self._engine = None
        self._enabled = False
        self._toggle.setChecked(False)

    # ── internal ─────────────────────────────────────────────────────────────

    def _exe_path(self) -> Path:
        default = (
            "assets/engines/stockfish-windows-x86-64-avx2"
            "/stockfish/stockfish-windows-x86-64-avx2.exe"
        )
        rel = self._config.get("paths", {}).get("engine_exe", default)
        return resolve_path(rel)

    def _on_toggle(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self.stop_engine()
            return
        exe = self._exe_path()
        if not exe.exists():
            QMessageBox.critical(self, "Stockfish missing", f"Engine not found:\n{exe}")
            self._toggle.setChecked(False)
            self._enabled = False
            return
        try:
            self._engine = UciEngine(exe)
            self._engine.start()
        except Exception as exc:
            QMessageBox.critical(self, "Stockfish start failed", str(exc))
            self._toggle.setChecked(False)
            self._enabled = False
            self._engine = None

    def _set_plays(self, plays: bool) -> None:
        self._plays = plays

    def _should_move(self, white_to_move: bool) -> bool:
        if not (self._enabled and self._plays and self._engine):
            return False
        engine_is_white = self._side == "White"
        return white_to_move == engine_is_white
