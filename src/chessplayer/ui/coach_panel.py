"""
ui/coach_panel.py
==================
Coach tab — front-facing GUI for the chess_coach backend.

Sits as the 4th tab alongside Variations / Moves / PGN.

Behaviour
---------
- Toggle button (Coach OFF / Coach ON) activates analysis.
  When OFF, no analysis runs and the panel shows its last result dimmed.
- queue_analysis() is called by MainWindow on every position change.
  The panel only acts when the toggle is ON.
- request_help() forces one analysis run, auto-toggles ON, and flags
  the result to be inserted into the PGN via coach_help_requested.
- Clicking a GM precedent row expands an inline detail section with a
  single "Open in Coach Board + Insert Variation" button.

Signals → MainWindow
---------------------
    coach_help_requested(CoachOutput)
        Insert output as a PGN comment at the current move.
    gm_load_requested(GMPrecedent)
        Load the game to the coach board and insert its continuation
        as a variation in the current game.
    weakness_squares_ready(list[str])
        Forward to CoachBoardWidget.set_weakness_squares() after every
        successful analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

# chess_coach internal imports are bare: "from core.X import ..."
# So sys.path needs src/chess_coach/ on it.
# This file: <project_root>/src/chessplayer/ui/coach_panel.py
# parents[2] = <project_root>/src/  →  / "chess_coach"
_chess_coach_dir = Path(__file__).resolve().parents[2] / "chess_coach"
if str(_chess_coach_dir) not in sys.path:
    sys.path.insert(0, str(_chess_coach_dir))

import chess

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)


# ── Workers ───────────────────────────────────────────────────────────────────

class _InitWorker(QObject):
    ready  = Signal(object)
    failed = Signal(str)

    def __init__(self, config: dict) -> None:
        super().__init__()
        self._config = config

    def run(self) -> None:
        try:
            from chess_coach.core.strategy_engine import StrategyEngine
            self.ready.emit(StrategyEngine.from_config(self._config))
        except Exception as exc:
            self.failed.emit(str(exc))


class _AnalysisWorker(QObject):
    finished = Signal(object, int)
    failed   = Signal(str, int)

    def __init__(self, engine, board, history, side, token):
        super().__init__()
        self._engine  = engine
        self._board   = board.copy()
        self._history = [b.copy() for b in history]
        self._side    = side
        self._token   = token

    def run(self) -> None:
        try:
            out = self._engine.analyse(
                self._board,
                player_side=self._side,
                history_boards=self._history,
            )
            self.finished.emit(out, self._token)
        except Exception as exc:
            self.failed.emit(str(exc), self._token)


# ── Panel ─────────────────────────────────────────────────────────────────────

class CoachPanel(QWidget):

    coach_help_requested   = Signal(object)   # CoachOutput
    gm_load_requested      = Signal(object)   # GMPrecedent
    weakness_squares_ready = Signal(list)

    _DEBOUNCE_MS = 400
    _COLOURS = {
        "blitz":    "#EF5350",
        "flank":    "#42A5F5",
        "fortress": "#66BB6A",
        "feint":    "#AB47BC",
        "general":  "#78909C",
    }

    def __init__(self, config: dict, parent=None) -> None:
        super().__init__(parent)
        self._config          = config
        self._engine          = None
        self._ready           = False
        self._active          = False
        self._token           = 0
        self._last_output     = None
        self._insert_pending  = False
        self._selected_prec   = None

        self._pending_board:   chess.Board | None = None
        self._pending_history: list               = []
        self._pending_side:    str                = "white"

        self._init_thread = self._init_worker = None
        self._ana_thread  = self._ana_worker  = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(self._DEBOUNCE_MS)
        self._debounce.timeout.connect(self._fire_analysis)

        self._build_ui()
        self._start_init()

    # ── Public API ────────────────────────────────────────────────────────────

    def queue_analysis(self, board, history=None, side="white") -> None:
        """Queue analysis. Only fires when toggle is ON."""
        print(f"[COACH] queue_analysis called: active={self._active}, ready={self._ready}, board={board.fen()[:10]}...")
        self._pending_board   = board.copy()
        self._pending_history = [b.copy() for b in (history or [])]
        self._pending_side    = side
        if self._active and self._ready:
            print("[COACH] Starting debounce...")
            self._debounce.start()

    def request_help(self) -> None:
        """Force one analysis + flag result for PGN insertion. Auto-activates."""
        if not self._active:
            self._set_active(True)
        if self._last_output is not None:
            self.coach_help_requested.emit(self._last_output)
        else:
            self._insert_pending = True
            if self._ready and self._pending_board is not None:
                self._debounce.start()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _start_init(self) -> None:
        self._set_status("Coach initialising\u2026", busy=True)
        print("Initializing")
        self._init_thread = QThread(self)
        self._init_worker = _InitWorker(self._config)
        self._init_worker.moveToThread(self._init_thread)
        self._init_thread.started.connect(self._init_worker.run)
        self._init_worker.ready.connect(self._on_init_ready)
        self._init_worker.failed.connect(self._on_init_failed)
        self._init_worker.ready.connect(self._init_thread.quit)
        self._init_worker.failed.connect(self._init_thread.quit)
        self._init_thread.finished.connect(self._init_thread.deleteLater)
        self._init_thread.start()
        
    @Slot(object)
    def _on_init_ready(self, engine) -> None:
        print("[COACH] Init READY")
        self._engine = engine
        self._ready  = True
        self._toggle_btn.setEnabled(True)
        self._set_status("Coach ready", busy=False)
        if self._active and self._pending_board is not None:
            self._debounce.start()

    @Slot(str)
    def _on_init_failed(self, msg: str) -> None:
        self._set_status(f"Coach unavailable: {msg}", busy=False)

    # ── Toggle ────────────────────────────────────────────────────────────────

    def _set_active(self, active: bool) -> None:
        self._active = active
        if active:
            self._toggle_btn.setText("\u23f9  Coach ON")
            self._toggle_btn.setStyleSheet(
                "background:#1B3A1B; color:#66BB6A; font-weight:bold;"
                " border:1px solid #2E5A2E; border-radius:4px; padding:4px 10px;"
            )
            if self._ready and self._pending_board is not None:
                self._debounce.start()
        else:
            self._debounce.stop()
            self._toggle_btn.setText("\u25b6  Coach OFF")
            self._toggle_btn.setStyleSheet(
                "background:#2A2A2A; color:#546E7A; font-weight:bold;"
                " border:1px solid #37474F; border-radius:4px; padding:4px 10px;"
            )

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _fire_analysis(self) -> None:
        print(f"[COACH] _fire_analysis: ready={self._ready}, pending={self._pending_board is not None}, ana_thread={self._ana_thread is not None}")
        if not self._ready or self._pending_board is None:
            self._set_status("No board to analyze", busy=False)
            return
        if self._ana_thread is not None:
            print("[COACH] Ana thread busy, restarting debounce")
            self._debounce.start()
            return
        self._token += 1
        token = self._token
        self._set_status("Analysing\u2026", busy=True)
        print(f"[COACH] Starting analysis token={token}")
        self._ana_thread = QThread(self)
        self._ana_worker = _AnalysisWorker(
            self._engine, self._pending_board,
            self._pending_history, self._pending_side, token,
        )
        self._ana_worker.moveToThread(self._ana_thread)
        self._ana_thread.started.connect(self._ana_worker.run)
        self._ana_worker.finished.connect(self._on_done)
        self._ana_worker.failed.connect(self._on_failed)
        self._ana_worker.finished.connect(self._ana_thread.quit)
        self._ana_worker.failed.connect(self._ana_thread.quit)
        self._ana_thread.finished.connect(self._cleanup_ana)
        self._ana_thread.start()

    @Slot(object, int)
    def _on_done(self, output, token: int) -> None:
        print(f"[COACH] _on_done: token={token}, current={self._token}, strategy={getattr(output, 'strategy_primary', 'None')}")
        if token != self._token:
            print("[COACH] Token mismatch, ignoring")
            return
        self._last_output = output
        self._insert_btn.setEnabled(True)
        self.weakness_squares_ready.emit(output.weakness_squares)
        if self._active:
            print("[COACH] Rendering output")
            self._render(output)
        self._set_status("Ready", busy=False)
        if self._insert_pending:
            self._insert_pending = False
            self.coach_help_requested.emit(output)

    @Slot(str, int)
    def _on_failed(self, msg: str, token: int) -> None:
        if token == self._token:
            self._set_status(f"Error: {msg}", busy=False)

    def _cleanup_ana(self) -> None:
        print("[COACH] Cleaning up ana thread")
        if self._ana_worker:
            self._ana_worker.deleteLater()
            self._ana_worker = None
        self._ana_thread = None
        if self._ana_thread:
            self._ana_thread.deleteLater()

    # ── Render ────────────────────────────────────────────────────────────────

    def _render(self, output) -> None:
        strat  = output.strategy_primary
        colour = self._COLOURS.get(strat, "#78909C")

        self._badge.setText(strat.upper())
        self._badge.setStyleSheet(
            f"background:{colour}; color:white; font-weight:bold;"
            f" padding:3px 10px; border-radius:4px; font-size:11px;"
        )
        self._conf_lbl.setText(f"{output.confidence:.0%}  \u00b7  {output.phase}")

        if output.strategy_secondary:
            sc = self._COLOURS.get(output.strategy_secondary, "#78909C")
            self._sec_badge.setText(output.strategy_secondary.upper())
            self._sec_badge.setStyleSheet(
                f"background:{sc}; color:white; font-weight:bold;"
                f" padding:2px 8px; border-radius:4px; font-size:10px;"
            )
            self._sec_label.show(); self._sec_badge.show()
        else:
            self._sec_label.hide(); self._sec_badge.hide()

        self._headline_lbl.setText(output.headline)
        self._headline_lbl.setStyleSheet(
            f"color:#E0E0E0; font-size:13px; font-weight:bold; padding:8px;"
            f" background:#1E2A2E; border-radius:4px; border-left:3px solid {colour};"
        )
        self._plan_lbl.setText("".join(
            f'<p style="margin:0 0 8px 0;">{s}</p>' for s in output.plan_sentences
        ))

        if output.tactic_hints:
            self._tactics_lbl.setText("".join(
                f'<p style="margin:0 0 6px 0;">\u26a1 {h}</p>'
                for h in output.tactic_hints
            ))
            self._tactics_frame.show()
        else:
            self._tactics_frame.hide()

        if output.weakness_squares:
            self._weak_lbl.setText(
                "Weak squares:  " + "  \u00b7  ".join(output.weakness_squares)
            )
            self._weak_frame.show()
        else:
            self._weak_frame.hide()

        self._precedents_list.clear()
        self._detail_frame.hide()
        self._selected_prec = None
        if output.gm_precedents:
            for prec in output.gm_precedents:
                mn    = prec.ply // 2 + 1
                label = f"\u265f  {prec.player}  \u2014  move {mn}"
                item  = QListWidgetItem(label)
                item.setData(Qt.UserRole, prec)
                item.setToolTip(f"Key move: {prec.key_move}\n{prec.annotation}")
                self._precedents_list.addItem(item)
            self._precedents_frame.show()
        else:
            self._precedents_frame.hide()

    # ── Status ────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, busy: bool = False) -> None:
        self._status_lbl.setVisible(bool(msg)); self._status_lbl.setText(msg)
        self._spinner.setVisible(busy)

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Toolbar
        tb = QHBoxLayout()
        self._toggle_btn = QPushButton("\u25b6  Coach OFF")
        self._toggle_btn.setFixedHeight(28)
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.setStyleSheet(
            "background:#2A2A2A; color:#546E7A; font-weight:bold;"
            " border:1px solid #37474F; border-radius:4px; padding:4px 10px;"
        )
        self._toggle_btn.clicked.connect(lambda: self._set_active(not self._active))
        tb.addWidget(self._toggle_btn)
        tb.addStretch(1)
        self._insert_btn = QPushButton("\U0001f4cb  Insert Note")
        self._insert_btn.setFixedHeight(28)
        self._insert_btn.setEnabled(False)
        self._insert_btn.setStyleSheet(
            "background:#1B2A3A; color:#4FC3F7; font-weight:bold;"
            " border:1px solid #1E4A6A; border-radius:4px; padding:4px 10px;"
        )
        self._insert_btn.setToolTip(
            "Insert coach recommendation as a PGN comment at the current move.\n"
            "The recommended moves appear as clickable links in the notation panel."
        )
        self._insert_btn.clicked.connect(self._on_insert_clicked)
        tb.addWidget(self._insert_btn)
        root.addLayout(tb)

        # Status
        sr = QHBoxLayout()
        self._spinner = QLabel("\u23f3"); self._spinner.setFixedWidth(22); self._spinner.hide()
        sr.addWidget(self._spinner)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#78909C; font-size:11px;")
        self._status_lbl.hide()
        sr.addWidget(self._status_lbl, 1)
        root.addLayout(sr)

        # Badge row
        br = QHBoxLayout(); br.setSpacing(8)
        self._badge = QLabel("\u2014")
        self._badge.setStyleSheet(
            "background:#37474F; color:#90A4AE; font-weight:bold;"
            " padding:3px 10px; border-radius:4px; font-size:11px;"
        )
        br.addWidget(self._badge)
        self._sec_label = QLabel("also:"); self._sec_label.setStyleSheet("color:#546E7A; font-size:10px;"); self._sec_label.hide()
        br.addWidget(self._sec_label)
        self._sec_badge = QLabel(""); self._sec_badge.hide()
        br.addWidget(self._sec_badge)
        br.addStretch(1)
        self._conf_lbl = QLabel(""); self._conf_lbl.setStyleSheet("color:#78909C; font-size:11px;")
        br.addWidget(self._conf_lbl)
        root.addLayout(br)

        # Scroll area
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame); scroll.setStyleSheet("background:transparent;")
        content = QWidget(); cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(10)

        self._headline_lbl = QLabel("")
        self._headline_lbl.setWordWrap(True)
        self._headline_lbl.setStyleSheet(
            "color:#E0E0E0; font-size:13px; font-weight:bold; padding:8px;"
            " background:#1E2A2E; border-radius:4px; border-left:3px solid #42A5F5;"
        )
        cl.addWidget(self._headline_lbl)

        self._plan_lbl = QLabel("")
        self._plan_lbl.setWordWrap(True); self._plan_lbl.setTextFormat(Qt.RichText)
        self._plan_lbl.setStyleSheet(
            "color:#CFD8DC; font-size:12px; padding:8px; background:#1A1A1A; border-radius:4px;"
        )
        cl.addWidget(self._plan_lbl)

        # Tactics
        self._tactics_frame = QFrame()
        self._tactics_frame.setStyleSheet(
            "QFrame{background:#1A1F1A;border:1px solid #2E4A2E;border-radius:4px;}"
        )
        tl = QVBoxLayout(self._tactics_frame); tl.setContentsMargins(8, 6, 8, 6); tl.setSpacing(4)
        QLabel("\u26a1 Tactics", self._tactics_frame).setStyleSheet("color:#A5D6A7;font-weight:bold;font-size:11px;")
        tl.addWidget(QLabel("\u26a1 Tactics"))
        self._tactics_lbl = QLabel(""); self._tactics_lbl.setWordWrap(True); self._tactics_lbl.setTextFormat(Qt.RichText)
        self._tactics_lbl.setStyleSheet("color:#C8E6C9;font-size:11px;")
        tl.addWidget(self._tactics_lbl)
        self._tactics_frame.hide(); cl.addWidget(self._tactics_frame)

        # Weakness
        self._weak_frame = QFrame()
        self._weak_frame.setStyleSheet(
            "QFrame{background:#1F1A1A;border:1px solid #4A2E2E;border-radius:4px;}"
        )
        wl = QVBoxLayout(self._weak_frame); wl.setContentsMargins(8, 4, 8, 4)
        self._weak_lbl = QLabel(""); self._weak_lbl.setWordWrap(True)
        self._weak_lbl.setStyleSheet("color:#FFCDD2;font-size:11px;")
        wl.addWidget(self._weak_lbl)
        self._weak_frame.hide(); cl.addWidget(self._weak_frame)

        # GM Precedents
        self._precedents_frame = QFrame()
        self._precedents_frame.setStyleSheet(
            "QFrame{background:#1A1A2A;border:1px solid #2E2E4A;border-radius:4px;}"
        )
        pl = QVBoxLayout(self._precedents_frame); pl.setContentsMargins(8, 6, 8, 6); pl.setSpacing(4)
        prec_hdr = QLabel("\u265f  GM Precedents")
        prec_hdr.setStyleSheet("color:#B39DDB;font-weight:bold;font-size:11px;")
        pl.addWidget(prec_hdr)
        prec_sub = QLabel("Click a game to see details")
        prec_sub.setStyleSheet("color:#546E7A;font-size:10px;")
        pl.addWidget(prec_sub)
        self._precedents_list = QListWidget()
        self._precedents_list.setFixedHeight(78)
        self._precedents_list.setStyleSheet(
            "QListWidget{background:#12121E;border:none;color:#CE93D8;font-size:11px;}"
            "QListWidget::item:hover{background:#1E1E3A;}"
            "QListWidget::item:selected{background:#2A2A5A;}"
        )
        self._precedents_list.itemClicked.connect(self._on_prec_clicked)
        pl.addWidget(self._precedents_list)

        # Inline detail card
        self._detail_frame = QFrame()
        self._detail_frame.setStyleSheet(
            "QFrame{background:#111122;border:1px solid #3A3A6A;border-radius:4px;}"
        )
        dl = QVBoxLayout(self._detail_frame); dl.setContentsMargins(10, 8, 10, 8); dl.setSpacing(6)
        self._detail_player = QLabel("")
        self._detail_player.setStyleSheet("color:#E1BEE7;font-weight:bold;font-size:12px;")
        dl.addWidget(self._detail_player)
        self._detail_move = QLabel(""); self._detail_move.setStyleSheet("color:#B39DDB;font-size:11px;")
        dl.addWidget(self._detail_move)
        self._detail_ann = QLabel(""); self._detail_ann.setWordWrap(True)
        self._detail_ann.setStyleSheet("color:#9E9E9E;font-size:11px;font-style:italic;")
        dl.addWidget(self._detail_ann)
        self._load_btn = QPushButton("\u25b6  Open in Coach Board  +  Insert Variation")
        self._load_btn.setStyleSheet(
            "background:#1B2A3A;color:#4FC3F7;font-weight:bold;"
            "border:1px solid #1E4A6A;border-radius:4px;padding:5px 8px;"
        )
        self._load_btn.clicked.connect(self._on_load_prec)
        dl.addWidget(self._load_btn)
        self._detail_frame.hide()
        pl.addWidget(self._detail_frame)

        self._precedents_frame.hide()
        cl.addWidget(self._precedents_frame)
        cl.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_insert_clicked(self) -> None:
        if self._last_output is not None:
            if not self._active:
                self._set_active(True)
            self.coach_help_requested.emit(self._last_output)
        else:
            self.request_help()

    def _on_prec_clicked(self, item: QListWidgetItem) -> None:
        prec = item.data(Qt.UserRole)
        if prec is None:
            return
        self._selected_prec = prec
        mn   = prec.ply // 2 + 1
        side = "White" if prec.ply % 2 == 0 else "Black"
        self._detail_player.setText(f"\u265f {prec.player}")
        self._detail_move.setText(f"Move {mn} ({side})  \u2014  key move: {prec.key_move}")
        self._detail_ann.setText(prec.annotation or "")
        self._detail_ann.setVisible(bool(prec.annotation))
        self._detail_frame.show()

    def _on_load_prec(self) -> None:
        if self._selected_prec is not None:
            self.gm_load_requested.emit(self._selected_prec)

    def closeEvent(self, event) -> None:
        self._debounce.stop()
        if self._engine:
            try: self._engine.close()
            except Exception: pass
        super().closeEvent(event)

