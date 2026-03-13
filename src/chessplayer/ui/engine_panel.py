from __future__ import annotations

from pathlib import Path
from typing import Optional

import chess

from PySide6.QtCore import QMutex, QObject, QThread, Signal, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from chessplayer.engine.uci_engine import BestMove, UciEngine, _parse_multipv_line
from chessplayer.utils.paths import resolve_path

_PV_UCI_ROLE = Qt.UserRole


# ── Reader thread ─────────────────────────────────────────────────────────────

class _ReaderThread(QObject):
    """
    Permanently reads Stockfish stdout on a background thread.
    Emits line_received for every line that comes out.
    Runs until stop_flag is set.
    """
    line_received = Signal(str)

    def __init__(self, engine: UciEngine) -> None:
        super().__init__()
        self._engine    = engine
        self._stop_flag = False

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        while not self._stop_flag:
            line = self._engine.readline()
            if line:
                self.line_received.emit(line)


# ── Panel ─────────────────────────────────────────────────────────────────────

class EnginePanel(QWidget):
    """
    Lichess-style streaming engine panel.

    Architecture:
      - One persistent Stockfish process + reader thread.
      - "go infinite" (or "go depth N") starts analysis.
      - Every info line updates the PV table in real time as depth increases.
      - Position changes send "stop", wait for bestmove, restart immediately.
      - No polling timer. No blocking. No per-move subprocess.

    State machine:
      IDLE       → engine off
      ANALYZING  → go infinite running, info lines streaming
      STOPPING   → "stop" sent, waiting for bestmove line
    """

    move_ready      = Signal(str)
    eval_updated    = Signal(object)   # BestMove | None
    pv_line_clicked = Signal(object, object)

    _MOVETIME_OPTIONS: list[tuple[str, int]] = [
        ("0.5s",  500),
        ("1s",    1000),
        ("2s",    2000),
        ("5s",    5000),
        ("10s",   10000),
        ("20s",   20000),
        ("30s",   30000),
        ("60s",   60000),
        ("120s",  120000),
        ("180s",  180000),
    ]

    _STATE_IDLE      = "idle"
    _STATE_ANALYZING = "analyzing"
    _STATE_STOPPING  = "stopping"

    def __init__(self, config: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config

        self._engine:        UciEngine | None     = None
        self._reader:        _ReaderThread | None  = None
        self._reader_thread: QThread | None        = None

        self._state          = self._STATE_IDLE
        self._paused         = False
        self._plays          = False
        self._side           = "Black"
        self._movetime_ms    = 1000
        self._num_lines      = 1
        self._max_moves      = 12
        self._depth: int | None = None
        self._threads        = 1
        self._hash_mb        = 64

        self._last_moves:         list[str] = []
        self._last_white_to_move: bool      = True
        self._pending_moves:      list[str] | None = None  # queued while stopping
        self._pending_wtm:        bool      = True

        # Per-depth accumulator: dict[multipv_rank] -> BestMove-ish dict
        self._pv_data: dict[int, dict] = {}

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        # Row 1
        row1_w = QWidget()
        row1   = QHBoxLayout(row1_w)
        row1.setContentsMargins(4, 2, 4, 2)
        row1.setSpacing(8)

        self._analysis_toggle = QCheckBox("Analysis Mode")
        self._analysis_toggle.setToolTip("Start Stockfish streaming analysis")
        self._analysis_toggle.stateChanged.connect(
            lambda s: self._on_analysis_toggled(bool(s))
        )
        row1.addWidget(self._analysis_toggle)

        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.setEnabled(False)
        self._pause_btn.setFixedWidth(90)
        self._pause_btn.setToolTip(
            "Pause analysis. Click a PV line to navigate to that position."
        )
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        row1.addWidget(self._pause_btn)

        self._play_toggle = QCheckBox("Play vs Engine")
        self._play_toggle.setEnabled(False)
        self._play_toggle.stateChanged.connect(self._set_plays)
        row1.addWidget(self._play_toggle)

        row1.addWidget(QLabel("Side:"))
        self._side_combo = QComboBox()
        self._side_combo.addItems(["Black", "White"])
        self._side_combo.currentTextChanged.connect(
            lambda t: setattr(self, "_side", t)
        )
        row1.addWidget(self._side_combo)

        row1.addStretch(1)
        outer.addWidget(row1_w)

        # Row 2
        row2_w = QWidget()
        row2   = QHBoxLayout(row2_w)
        row2.setContentsMargins(4, 0, 4, 2)
        row2.setSpacing(8)

        row2.addWidget(QLabel("MoveTime:"))
        self._time_combo = QComboBox()
        for label, _ in self._MOVETIME_OPTIONS:
            self._time_combo.addItem(label)
        self._time_combo.setCurrentText("1s")
        self._time_combo.currentTextChanged.connect(
            lambda t: (self._on_movetime_changed(t), self._restart_if_analyzing())
        )
        row2.addWidget(self._time_combo)

        row2.addWidget(QLabel("Lines:"))
        self._lines_combo = QComboBox()
        self._lines_combo.addItems(["1", "2", "3", "4", "5"])
        self._lines_combo.setCurrentText("1")
        self._lines_combo.currentTextChanged.connect(self._on_lines_changed)
        self._lines_combo.currentTextChanged.connect(lambda _: self._restart_analysis())
        row2.addWidget(self._lines_combo)

        row2.addWidget(QLabel("Line len:"))
        self._linelen_combo = QComboBox()
        self._linelen_combo.addItems(["8", "12", "16", "20", "25", "30"])
        self._linelen_combo.setCurrentText("12")
        self._linelen_combo.setToolTip("Moves shown per PV line")
        self._linelen_combo.currentTextChanged.connect(
            lambda t: (setattr(self, "_max_moves", int(t)), self._restart_if_analyzing())
        )
        row2.addWidget(self._linelen_combo)

        row2.addWidget(QLabel("Depth:"))
        self._depth_combo = QComboBox()
        self._depth_combo.addItems(["∞", "5", "10", "15", "20", "25", "30"])
        self._depth_combo.setCurrentText("∞")
        self._depth_combo.setToolTip("∞ = go infinite (like Lichess). Fixed depth stops there.")
        self._depth_combo.currentTextChanged.connect(
            lambda t: (self._on_depth_changed(t), self._restart_if_analyzing())
        )
        row2.addWidget(self._depth_combo)

        row2.addWidget(QLabel("Threads:"))
        self._threads_combo = QComboBox()
        self._threads_combo.addItems(["1", "2", "4", "8"])
        self._threads_combo.setCurrentText("1")
        self._threads_combo.currentTextChanged.connect(
            lambda t: (setattr(self, "_threads", int(t)), self._restart_if_analyzing())
        )
        row2.addWidget(self._threads_combo)

        row2.addWidget(QLabel("Hash MB:"))
        self._hash_combo = QComboBox()
        self._hash_combo.addItems(["16", "64", "128", "256", "512"])
        self._hash_combo.setCurrentText("64")
        self._hash_combo.currentTextChanged.connect(
            lambda t: (setattr(self, "_hash_mb", int(t)), self._restart_if_analyzing())
        )
        row2.addWidget(self._hash_combo)

        row2.addStretch(1)
        outer.addWidget(row2_w)

        # PV table
        self._pv_table = QTableWidget()
        self._pv_table.setColumnCount(3)
        self._pv_table.setHorizontalHeaderLabels(["#", "Score", "Line"])
        self._pv_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self._pv_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self._pv_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._pv_table.setColumnWidth(0, 24)
        self._pv_table.setColumnWidth(1, 60)
        self._pv_table.verticalHeader().setVisible(False)
        self._pv_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._pv_table.setSelectionMode(QTableWidget.SingleSelection)
        self._pv_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._pv_table.setShowGrid(False)
        self._pv_table.setFont(QFont("Consolas", 9))
        self._pv_table.setAlternatingRowColors(True)
        self._pv_table.setMaximumHeight(0)
        self._pv_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._pv_table.customContextMenuRequested.connect(self._on_pv_right_click)
        outer.addWidget(self._pv_table)

    # ── public API ────────────────────────────────────────────────────────────

    def trigger_analysis(self, prefix_uci: list[str], white_to_move: bool) -> None:
        """Called by MainWindow on every position change."""
        self._last_moves         = list(prefix_uci)
        self._last_white_to_move = white_to_move

        if not self._engine or self._paused:
            return

        if self._state == self._STATE_ANALYZING:
            # Queue new position and send stop — restart happens on bestmove
            self._pending_moves = list(prefix_uci)
            self._pending_wtm   = white_to_move
            self._state         = self._STATE_STOPPING
            self._engine.send_stop()
        elif self._state == self._STATE_IDLE:
            self._start_go(prefix_uci, white_to_move)
        elif self._state == self._STATE_STOPPING:
            # Already stopping — just update what we'll restart with
            self._pending_moves = list(prefix_uci)
            self._pending_wtm   = white_to_move

    def trigger_move_if_needed(self, prefix_uci: list[str], white_to_move: bool) -> None:
        self.trigger_analysis(prefix_uci, white_to_move)

    def stop_engine(self) -> None:
        if self._engine:
            if self._state != self._STATE_IDLE:
                try:
                    self._engine.send_stop()
                except Exception:
                    pass
        if self._reader:
            self._reader.stop()
        if self._reader_thread:
            self._reader_thread.quit()
            self._reader_thread.wait(2000)
            self._reader_thread.deleteLater()
            self._reader_thread = None
        if self._reader:
            self._reader.deleteLater()
            self._reader = None
        if self._engine:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

        self._state       = self._STATE_IDLE
        self._paused      = False
        self._pending_moves = None
        self._pv_data     = {}

        self._analysis_toggle.setChecked(False)
        self._pause_btn.setEnabled(False)
        self._pause_btn.setText("⏸ Pause")
        self._play_toggle.setEnabled(False)
        self._play_toggle.setChecked(False)
        self._pv_table.setRowCount(0)
        self._pv_table.setMaximumHeight(0)
        self.eval_updated.emit(None)

    # ── pause ─────────────────────────────────────────────────────────────────

    def _on_pause_clicked(self) -> None:
        if not self._paused:
            self._paused = True
            self._pause_btn.setText("▶ Resume")
            self._pv_table.setStyleSheet("QTableWidget { border: 1px solid #D4A017; }")
            if self._state == self._STATE_ANALYZING and self._engine:
                self._engine.send_stop()
                self._state = self._STATE_STOPPING
                self._pending_moves = None   # don't restart after stop
        else:
            self._paused = False
            self._pause_btn.setText("⏸ Pause")
            self._pv_table.setStyleSheet("")
            if self._state == self._STATE_IDLE and self._engine:
                self._start_go(self._last_moves, self._last_white_to_move)

    # ── PV click → navigate ───────────────────────────────────────────────────

    def _on_pv_right_click(self, pos) -> None:
        item = self._pv_table.itemAt(pos)
        if item is None:
            return
        row       = self._pv_table.row(item)
        rank_item = self._pv_table.item(row, 0)
        if rank_item is None:
            return
        pv_uci = rank_item.data(_PV_UCI_ROLE)
        if not pv_uci:
            return

        # Get score label for menu title
        score_item = self._pv_table.item(row, 1)
        score_txt  = score_item.text() if score_item else ""
        rank       = rank_item.text()

        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)

        add_action = menu.addAction(f"Add line {rank} ({score_txt}) to PGN")
        add_action.triggered.connect(
            lambda checked=False, p=list(pv_uci[:25]):
                self.pv_line_clicked.emit(list(self._last_moves), p)
        )

        menu.exec(self._pv_table.mapToGlobal(pos))

    # ── engine start ──────────────────────────────────────────────────────────

    def _on_analysis_toggled(self, on: bool) -> None:
        if on:
            self._start_engine()
        else:
            self.stop_engine()

    def _start_engine(self) -> None:
        exe = self._exe_path()
        if not exe.exists():
            QMessageBox.critical(self, "Stockfish missing", f"Engine not found:\n{exe}")
            self._analysis_toggle.setChecked(False)
            return
        try:
            self._engine = UciEngine(exe)
            self._engine.start()
        except Exception as exc:
            QMessageBox.critical(self, "Stockfish start failed", str(exc))
            self._analysis_toggle.setChecked(False)
            self._engine = None
            return

        # Start permanent reader thread
        self._reader        = _ReaderThread(self._engine)
        self._reader_thread = QThread(self)
        self._reader.moveToThread(self._reader_thread)
        self._reader_thread.started.connect(self._reader.run)
        self._reader.line_received.connect(self._on_line)
        self._reader_thread.start()

        self._play_toggle.setEnabled(True)
        self._pause_btn.setEnabled(True)
        self._pv_table.setMaximumHeight(120)
        self._state = self._STATE_IDLE

        # Begin analysis of current position immediately
        self._start_go(self._last_moves, self._last_white_to_move)

    def _start_go(self, moves: list[str], white_to_move: bool) -> None:
        if not self._engine:
            return
        self._pv_data            = {}
        self._last_moves         = list(moves)
        self._last_white_to_move = white_to_move
        self._state              = self._STATE_ANALYZING
        try:
            self._engine.start_analysis(
                moves,
                multipv  = self._num_lines,
                threads  = self._threads,
                hash_mb  = self._hash_mb,
                depth    = self._depth,
            )
        except Exception as exc:
            self._state = self._STATE_IDLE
            QMessageBox.critical(self, "Stockfish error", str(exc))

    # ── line handler ──────────────────────────────────────────────────────────

    def _on_line(self, line: str) -> None:
        """Called on main thread via Qt signal for every line from Stockfish."""

        if line.startswith("info ") and "depth" in line and "pv" in line:
            sc_cp, sc_mate, pv, rank = _parse_multipv_line(line)
            if rank is None:
                rank = 1
            if pv:
                self._pv_data[rank] = {
                    "score_cp": sc_cp,
                    "mate_in":  sc_mate,
                    "pv":       pv,
                }
                self._refresh_pv_table()

                # Update eval bar from line 1
                if rank == 1:
                    bm = BestMove(
                        uci      = pv[0] if pv else "0000",
                        score_cp = sc_cp,
                        mate_in  = sc_mate,
                        pv_uci   = pv,
                    )
                    self.eval_updated.emit(bm)

        elif line.startswith("bestmove"):
            parts        = line.split()
            bestmove_uci = parts[1] if len(parts) >= 2 else "0000"

            # Emit engine move if it's the engine's turn
            if (
                self._should_move(self._last_white_to_move)
                and bestmove_uci
                and bestmove_uci != "0000"
                and not self._paused
            ):
                self.move_ready.emit(bestmove_uci)

            # Decide what to do next
            if self._pending_moves is not None and not self._paused:
                # Restart with queued position
                moves = self._pending_moves
                wtm   = self._pending_wtm
                self._pending_moves = None
                self._start_go(moves, wtm)
            else:
                self._state = self._STATE_IDLE

    # ── PV table ─────────────────────────────────────────────────────────────

    def _refresh_pv_table(self) -> None:
        rows = sorted(self._pv_data.keys())
        self._pv_table.setRowCount(len(rows))

        for row_idx, rank in enumerate(rows):
            data = self._pv_data[rank]
            pv   = data["pv"]

            rank_item = QTableWidgetItem(str(rank))
            rank_item.setTextAlignment(0x0004 | 0x0080)
            rank_item.setData(_PV_UCI_ROLE, pv)
            self._pv_table.setItem(row_idx, 0, rank_item)

            score_item = QTableWidgetItem(self._format_score(data))
            score_item.setTextAlignment(0x0004 | 0x0080)
            if rank == 1:
                score_item.setForeground(QColor("#4CAF50"))
            self._pv_table.setItem(row_idx, 1, score_item)

            line_item = QTableWidgetItem(self._pv_to_san(pv))
            self._pv_table.setItem(row_idx, 2, line_item)

        self._pv_table.resizeRowsToContents()

    @staticmethod
    def _format_score(data: dict) -> str:
        if data.get("mate_in") is not None:
            m = data["mate_in"]
            return f"M{abs(m)}" if m > 0 else f"M-{abs(m)}"
        cp = data.get("score_cp")
        if cp is not None:
            pawns = cp / 100.0
            return f"{'+' if pawns >= 0 else ''}{pawns:.2f}"
        return "—"

    def _pv_to_san(self, pv_uci: list[str]) -> str:
        if not pv_uci:
            return ""
        try:
            board = chess.Board()
            for uci in self._last_moves:
                board.push_uci(uci)
            parts: list[str] = []
            for uci in pv_uci:
                try:
                    mv = chess.Move.from_uci(uci)
                except Exception:
                    break
                if mv not in board.legal_moves:
                    break
                if board.turn == chess.WHITE:
                    parts.append(f"{board.fullmove_number}.")
                parts.append(board.san(mv))
                board.push(mv)
            return " ".join(parts)
        except Exception:
            return " ".join(pv_uci)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _on_threads_changed(self, text: str) -> None:
        try:
            self._threads = int(text)
        except ValueError:
            self._threads = 1
        self._restart_if_analyzing()

    def _on_hash_changed(self, text: str) -> None:
        try:
            self._hash_mb = int(text)
        except ValueError:
            self._hash_mb = 64
        self._restart_if_analyzing()

    def _restart_if_analyzing(self) -> None:
        """Call after any setting change that requires engine reconfiguration."""
        if self._engine and self._state != self._STATE_IDLE and not self._paused:
            self._pending_moves = list(self._last_moves)
            self._pending_wtm   = self._last_white_to_move
            self._state         = self._STATE_STOPPING
            self._engine.send_stop()

    def _restart_analysis(self) -> None:
        """Restart analysis with current settings. Called when any param changes."""
        if not self._engine or self._paused:
            return
        if self._state == self._STATE_ANALYZING:
            self._pending_moves = list(self._last_moves)
            self._pending_wtm   = self._last_white_to_move
            self._state         = self._STATE_STOPPING
            self._engine.send_stop()
        elif self._state == self._STATE_IDLE:
            self._start_go(self._last_moves, self._last_white_to_move)

    def _exe_path(self) -> Path:
        default = (
            "assets/engines/stockfish-windows-x86-64-avx2"
            "/stockfish/stockfish-windows-x86-64-avx2.exe"
        )
        return resolve_path(self._config.get("paths", {}).get("engine_exe", default))

    def _set_plays(self, plays: bool) -> None:
        self._plays = bool(plays)
        if self._plays and self._state == self._STATE_IDLE and self._engine and not self._paused:
            self._start_go(self._last_moves, self._last_white_to_move)

    def _should_move(self, white_to_move: bool) -> bool:
        if not (self._plays and self._engine):
            return False
        return white_to_move == (self._side == "White")

    def _on_movetime_changed(self, label: str) -> None:
        for lbl, ms in self._MOVETIME_OPTIONS:
            if lbl == label:
                self._movetime_ms = ms
                return

    def _on_lines_changed(self, text: str) -> None:
        try:
            self._num_lines = int(text)
        except ValueError:
            self._num_lines = 1

    def _on_depth_changed(self, text: str) -> None:
        self._depth = None if text in ("∞", "Off") else int(text)

