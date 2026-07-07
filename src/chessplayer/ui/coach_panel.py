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
- on_all_pvs_updated() receives all MultiPV engine lines and triggers
  a multi-analysis run: up to 3 PV lines are analysed in one background
  worker and each result fills its own analysis card.
- request_help() forces one analysis run, auto-toggles ON, and flags
  the result to be inserted into the PGN via coach_help_requested.
- Clicking a GM precedent row expands an inline detail section.

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
    pv_line_load_requested(str, list, list, str)
        (base_fen, pv_uci, pv_san, title) — load a PV line into the
        Coach Board widget.
"""
from __future__ import annotations

import sys
from pathlib import Path

# chess_coach internal imports are bare: "from core.X import ..."
# So sys.path needs src/chess_coach/ on it.
_chess_coach_dir = Path(__file__).resolve().parents[2] / "chess_coach"
if str(_chess_coach_dir) not in sys.path:
    sys.path.insert(0, str(_chess_coach_dir))

import chess

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
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


class _MultiAnalysisWorker(QObject):
    """
    Run analyse_from_pv for up to 3 PV lines sequentially in one thread.
    Emits line_ready after each line completes so the UI can update
    incrementally without waiting for all 3 to finish.
    Falls back to plain analyse() when no PV lines are available.
    """
    line_ready = Signal(object, int)   # (CoachOutput, line_idx 0-based)
    all_done   = Signal(int)           # token
    failed     = Signal(str, int)      # (msg, token)

    def __init__(self, engine, board, side, pvs_with_scores, token):
        super().__init__()
        self._engine   = engine
        self._board    = board.copy()
        self._side     = side
        self._pvs      = pvs_with_scores   # list[(pv_uci, score_cp)]
        self._token    = token

    def run(self) -> None:
        if not self._pvs:
            try:
                out = self._engine.analyse(self._board, player_side=self._side)
                self.line_ready.emit(out, 0)
            except Exception as exc:
                self.failed.emit(str(exc), self._token)
        else:
            for i, (pv_uci, score_cp) in enumerate(self._pvs[:3]):
                try:
                    out = self._engine.analyse_from_pv(
                        self._board, pv_uci, self._side, score_cp=score_cp
                    )
                    self.line_ready.emit(out, i)
                except Exception as exc:
                    self.failed.emit(f"Line {i + 1}: {exc}", self._token)
        self.all_done.emit(self._token)


# ── Panel ─────────────────────────────────────────────────────────────────────

class CoachPanel(QWidget):

    coach_help_requested   = Signal(object)              # CoachOutput
    gm_load_requested      = Signal(object)              # GMPrecedent
    weakness_squares_ready = Signal(list)
    pv_line_load_requested = Signal(str, list, list, str)  # (base_fen, pv_uci, pv_san, title)

    _DEBOUNCE_MS = 1200
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

        self._pending_board:        chess.Board | None = None
        self._pending_history:      list               = []
        self._pending_side:         str                = "white"
        self._multi_bms:            list               = []   # BestMove objects from engine
        self._pending_analysis_fen: str                = ''
        self._multi_results:        list               = [None, None, None]
        self._last_analyzed_pv_key: str                = ''   # prevents re-run of same PVs
        self._had_pvs_at_fire:      bool               = False

        self._init_thread  = self._init_worker  = None
        self._multi_thread = self._multi_worker = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(self._DEBOUNCE_MS)
        self._debounce.timeout.connect(self._fire_analysis)

        self._build_ui()
        self._start_init()

    # ── Public API ────────────────────────────────────────────────────────────

    def queue_analysis(self, board, history=None, side="white") -> None:
        """Queue analysis for the new board position. Only fires when toggle is ON."""
        print(f"[COACH] queue_analysis: active={self._active}, board={board.fen()[:10]}...")
        self._pending_board    = board.copy()
        self._pending_history  = [b.copy() for b in (history or [])]
        self._pending_side     = side
        self._multi_bms        = []     # new position — any cached PVs are now stale
        self._last_analyzed_pv_key = ''
        if side == "white":
            self._side_lbl.setText("♙ White")
            self._side_lbl.setStyleSheet(
                "background:#1B2A2A; color:#80CBC4; font-weight:bold; font-size:11px;"
                " border:1px solid #2E4A4A; border-radius:4px; padding:2px 10px;"
            )
        else:
            self._side_lbl.setText("♟ Black")
            self._side_lbl.setStyleSheet(
                "background:#2A1B2A; color:#CE93D8; font-weight:bold; font-size:11px;"
                " border:1px solid #4A2E4A; border-radius:4px; padding:2px 10px;"
            )
        if self._active and self._ready:
            self._debounce.start()

    @Slot(object)
    def on_pv_updated(self, best_move) -> None:
        """Rank-1 PV update — on_all_pvs_updated is the primary trigger now."""
        pass   # kept for the signal connection in window.py

    @Slot(list)
    def on_all_pvs_updated(self, pvs: list) -> None:
        """
        Receives every MultiPV update from the engine panel.
        Stores all lines and triggers (re-)analysis if the best line changed.
        """
        new_bms = [bm for bm in pvs if getattr(bm, 'pv_uci', None)]
        if not new_bms:
            return
        self._multi_bms = new_bms
        # Skip if the same PVs were already analysed
        key = '|'.join(
            ' '.join((bm.pv_uci or [])[:3]) for bm in new_bms[:3]
        )
        if key == self._last_analyzed_pv_key:
            return
        if self._active and self._ready and self._multi_thread is None:
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
        self._set_status("Coach initialising…", busy=True)
        print("[COACH] Initializing")
        self._init_thread = QThread(self)
        self._init_worker = _InitWorker(self._config)
        self._init_worker.moveToThread(self._init_thread)
        self._init_thread.started.connect(self._init_worker.run)
        self._init_worker.ready.connect(self._on_init_ready)
        self._init_worker.failed.connect(self._on_init_failed)
        self._init_worker.ready.connect(self._init_thread.quit)
        self._init_worker.failed.connect(self._init_thread.quit)
        self._init_thread.finished.connect(self._init_thread.deleteLater)
        self._init_thread.finished.connect(self._cleanup_init)
        self._init_thread.start()

    def _cleanup_init(self) -> None:
        self._init_worker = None
        self._init_thread = None

    @Slot(object)
    def _on_init_ready(self, engine) -> None:
        print("[COACH] Init READY")
        self._engine = engine
        self._ready  = True
        self._toggle_btn.setEnabled(True)
        self._set_status("Coach ready — turn ON to analyse", busy=False)
        if self._active and self._pending_board is not None:
            self._debounce.start()

    @Slot(str)
    def _on_init_failed(self, msg: str) -> None:
        self._set_status(f"Coach unavailable: {msg}", busy=False)

    # ── Toggle ────────────────────────────────────────────────────────────────

    def _set_active(self, active: bool) -> None:
        self._active = active
        if active:
            self._toggle_btn.setText("⏹  Coach ON")
            self._toggle_btn.setStyleSheet(
                "background:#1B3A1B; color:#66BB6A; font-weight:bold;"
                " border:1px solid #2E5A2E; border-radius:4px; padding:4px 10px;"
            )
            if self._ready and self._pending_board is not None:
                self._debounce.start()
        else:
            self._debounce.stop()
            self._toggle_btn.setText("▶  Coach OFF")
            self._toggle_btn.setStyleSheet(
                "background:#2A2A2A; color:#546E7A; font-weight:bold;"
                " border:1px solid #37474F; border-radius:4px; padding:4px 10px;"
            )

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _fire_analysis(self) -> None:
        print(f"[COACH] _fire_analysis: ready={self._ready}, pvs={len(self._multi_bms)}, thread_busy={self._multi_thread is not None}")
        if not self._ready or self._pending_board is None:
            self._set_status("No board to analyze", busy=False)
            return
        if self._multi_thread is not None:
            self._debounce.start()   # try again after next debounce interval
            return

        self._token += 1
        token = self._token
        self._pending_analysis_fen = self._pending_board.fen()
        self._had_pvs_at_fire = bool(self._multi_bms)
        self._set_status("Analysing…", busy=True)

        # Hide all cards — they re-appear as each line_ready signal fires
        for card in self._line_cards:
            card.hide()

        pvs_with_scores = [
            (list(bm.pv_uci), getattr(bm, 'score_cp', None))
            for bm in self._multi_bms[:3]
        ] if self._multi_bms else []

        self._multi_results = [None, None, None]

        print(f"[COACH] Starting multi-analysis token={token}, {len(pvs_with_scores)} PV line(s)")
        self._multi_thread = QThread(self)
        self._multi_worker = _MultiAnalysisWorker(
            self._engine, self._pending_board, self._pending_side,
            pvs_with_scores, token,
        )
        self._multi_worker.moveToThread(self._multi_thread)
        self._multi_thread.started.connect(self._multi_worker.run)
        self._multi_worker.line_ready.connect(self._on_line_ready)
        self._multi_worker.all_done.connect(self._on_multi_done)
        self._multi_worker.failed.connect(self._on_multi_failed)
        self._multi_worker.all_done.connect(self._multi_thread.quit)
        self._multi_worker.failed.connect(self._multi_thread.quit)
        self._multi_thread.finished.connect(self._cleanup_multi)
        self._multi_thread.start()

    @Slot(object, int)
    def _on_line_ready(self, output, line_idx: int) -> None:
        print(f"[COACH] _on_line_ready: idx={line_idx}, strategy={getattr(output, 'strategy_primary', '?')}")
        self._multi_results[line_idx] = output
        if self._active:
            self._render_line_card(line_idx, output)
        if line_idx == 0:
            self._last_output = output
            self._insert_btn.setEnabled(True)
            self.weakness_squares_ready.emit(output.weakness_squares)
            if self._active:
                self._render_shared_sections(output)
            if self._insert_pending:
                self._insert_pending = False
                self.coach_help_requested.emit(output)

    @Slot(int)
    def _on_multi_done(self, token: int) -> None:
        print(f"[COACH] _on_multi_done: token={token}, current={self._token}")
        if token != self._token:
            return
        self._last_analyzed_pv_key = '|'.join(
            ' '.join((bm.pv_uci or [])[:3]) for bm in self._multi_bms[:3]
        ) if self._multi_bms else ''
        self._set_status("Ready", busy=False)

    @Slot(str, int)
    def _on_multi_failed(self, msg: str, token: int) -> None:
        if token == self._token:
            self._set_status(f"Error: {msg}", busy=False)

    def _cleanup_multi(self) -> None:
        print("[COACH] Cleaning up multi thread")
        if self._multi_worker:
            self._multi_worker.deleteLater()
            self._multi_worker = None
        if self._multi_thread:
            self._multi_thread.deleteLater()
            self._multi_thread = None
        # If analysis started without PVs but engine lines have since arrived, re-fire once
        if self._active and self._ready and not self._had_pvs_at_fire and self._multi_bms:
            self._had_pvs_at_fire = True
            self._debounce.start()

    # ── Render ────────────────────────────────────────────────────────────────

    def _render_shared_sections(self, output) -> None:
        """Update strategy badge, tactics, weakness squares, and GM precedents from line 0."""
        strat  = output.strategy_primary
        colour = self._COLOURS.get(strat, "#78909C")

        self._badge.setText(strat.upper())
        self._badge.setStyleSheet(
            f"background:{colour}; color:white; font-weight:bold;"
            f" padding:3px 10px; border-radius:4px; font-size:11px;"
        )
        self._conf_lbl.setText(f"{output.confidence:.0%}  ·  {output.phase}")

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

        if output.tactic_hints:
            self._tactics_lbl.setText("".join(
                f'<p style="margin:0 0 6px 0;">⚡ {h}</p>'
                for h in output.tactic_hints
            ))
            self._tactics_frame.show()
        else:
            self._tactics_frame.hide()

        if output.weakness_squares:
            self._weak_lbl.setText(
                "Weak squares:  " + "  ·  ".join(output.weakness_squares)
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
                label = f"♟  {prec.player}  —  move {mn}"
                item  = QListWidgetItem(label)
                item.setData(Qt.UserRole, prec)
                item.setToolTip(f"Key move: {prec.key_move}\n{prec.annotation}")
                self._precedents_list.addItem(item)
            self._precedents_frame.show()
        else:
            self._precedents_frame.hide()

    def _render_line_card(self, idx: int, output) -> None:
        """Populate and show the analysis card for the given line index."""
        if idx >= len(self._line_cards):
            return
        card = self._line_cards[idx]
        refs = self._line_card_refs[idx]
        bm   = self._multi_bms[idx] if idx < len(self._multi_bms) else None

        # Score display
        if bm and getattr(bm, 'score_cp', None) is not None:
            score_str   = f"{bm.score_cp / 100.0:+.2f}"
            score_color = "#A5D6A7" if bm.score_cp >= 0 else "#EF9A9A"
        elif bm and getattr(bm, 'mate_in', None) is not None:
            score_str   = f"M{bm.mate_in}"
            score_color = "#FFD54F"
        else:
            score_str   = "?"
            score_color = "#78909C"

        refs['rank_lbl'].setText("#Best Line" if idx == 0 else f"#Line {idx + 1}")
        refs['score_lbl'].setText(score_str)
        refs['score_lbl'].setStyleSheet(
            f"color:{score_color}; font-family:monospace; font-size:11px;"
            " font-weight:bold;"
        )

        # Headline with per-strategy left border colour
        colour = self._COLOURS.get(output.strategy_primary, "#37474F")
        refs['headline_lbl'].setText(output.headline)
        refs['headline_lbl'].setStyleSheet(
            f"color:#E0E0E0; font-size:12px; font-weight:bold; padding:5px;"
            f" background:#1A2830; border-radius:3px; border-left:3px solid {colour};"
        )

        # Moves text
        pv_text = getattr(output, 'pv_line_text', '')
        refs['moves_lbl'].setText(pv_text)

        # Plan sentences — all for line 0, first 2 for lines 1+
        sentences = output.plan_sentences if idx == 0 else output.plan_sentences[:2]
        refs['plan_lbl'].setText("".join(
            f'<p style="margin:0 0 5px 0;">{s}</p>' for s in sentences
        ))

        # Load button — wire per-card with captured closure values
        pv_uci = list(getattr(output, 'pv_uci', []))
        pv_san = list(getattr(output, 'pv_san', []))
        fen    = self._pending_analysis_fen
        title  = f"Best Line  {score_str}" if idx == 0 else f"Line #{idx + 1}  {score_str}"
        try:
            refs['load_btn'].clicked.disconnect()
        except RuntimeError:
            pass
        refs['load_btn'].clicked.connect(
            lambda _c=False, f=fen, u=pv_uci, s=pv_san, t=title:
                self.pv_line_load_requested.emit(f, u, s, t)
        )

        # Metrics table (shown for all lines that have data)
        if getattr(output, 'metrics_table', None):
            rows = output.metrics_table
            mt   = refs['metrics_table']
            mt.setRowCount(len(rows))
            for r, (label, before, after, delta) in enumerate(rows):
                mt.setItem(r, 0, QTableWidgetItem(label))
                mt.setItem(r, 1, QTableWidgetItem(f"{before:+.2f}"))
                mt.setItem(r, 2, QTableWidgetItem(f"{after:+.2f}"))
                d_item = QTableWidgetItem(f"{delta:+.2f}")
                if delta > 0.05:
                    d_item.setForeground(QColor("#A5D6A7"))
                elif delta < -0.05:
                    d_item.setForeground(QColor("#EF9A9A"))
                else:
                    d_item.setForeground(QColor("#78909C"))
                mt.setItem(r, 3, d_item)
            mt.resizeRowsToContents()
            hdr_h = mt.horizontalHeader().height()
            row_h = sum(mt.rowHeight(r) for r in range(len(rows)))
            mt.setFixedHeight(hdr_h + row_h + 2)
            refs['metrics_sub'].show()
            mt.show()
        else:
            refs['metrics_sub'].hide()
            refs['metrics_table'].hide()

        card.show()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _make_line_card(self, idx: int) -> tuple:
        """Build one analysis card for a single PV line; returns (frame, refs-dict)."""
        card = QFrame()
        card.setStyleSheet(
            "QFrame{background:#0F1920; border:1px solid #1E2E3A; border-radius:4px;}"
        )
        cl = QVBoxLayout(card)
        cl.setContentsMargins(8, 6, 8, 8)
        cl.setSpacing(5)

        # Header: rank label + score + load button
        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        rank_lbl = QLabel(f"#Line {idx + 1}")
        rank_lbl.setStyleSheet("color:#546E7A; font-weight:bold; font-size:11px;")
        hdr.addWidget(rank_lbl)
        score_lbl = QLabel("")
        score_lbl.setStyleSheet(
            "color:#A5D6A7; font-family:monospace; font-size:11px; font-weight:bold;"
        )
        hdr.addWidget(score_lbl)
        hdr.addStretch(1)
        load_btn = QPushButton("♞  Load in Coach Board")
        load_btn.setFixedHeight(22)
        load_btn.setStyleSheet(
            "background:#1B2A3A; color:#4FC3F7; font-weight:bold; font-size:10px;"
            " border:1px solid #1E4A6A; border-radius:3px; padding:1px 8px;"
        )
        hdr.addWidget(load_btn)
        cl.addLayout(hdr)

        # Headline
        headline_lbl = QLabel("")
        headline_lbl.setWordWrap(True)
        headline_lbl.setStyleSheet(
            "color:#E0E0E0; font-size:12px; font-weight:bold; padding:5px;"
            " background:#1A2830; border-radius:3px; border-left:3px solid #37474F;"
        )
        cl.addWidget(headline_lbl)

        # Moves (monospace, wrapping)
        moves_lbl = QLabel("")
        moves_lbl.setWordWrap(True)
        moves_lbl.setStyleSheet(
            "color:#B0BEC5; font-family:monospace; font-size:11px; padding:2px 0;"
        )
        cl.addWidget(moves_lbl)

        # Plan
        plan_lbl = QLabel("")
        plan_lbl.setWordWrap(True)
        plan_lbl.setTextFormat(Qt.RichText)
        plan_lbl.setStyleSheet("color:#90A4AE; font-size:11px; padding:2px 0;")
        cl.addWidget(plan_lbl)

        # Metrics subtitle + table
        metrics_sub = QLabel(
            "Before = position NOW  ·  After = position after the full line plays out  ·  Δ = your gain"
        )
        metrics_sub.setStyleSheet("color:#455A64; font-size:10px; padding:2px 0;")
        metrics_sub.hide()
        cl.addWidget(metrics_sub)

        metrics_table = QTableWidget(0, 4)
        metrics_table.setHorizontalHeaderLabels(["Term", "Before", "After", "Δ"])
        metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3):
            metrics_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
        metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        metrics_table.setSelectionMode(QTableWidget.NoSelection)
        metrics_table.verticalHeader().setVisible(False)
        metrics_table.setShowGrid(True)
        metrics_table.setAlternatingRowColors(True)
        metrics_table.setStyleSheet("""
            QTableWidget {
                background:#0E0E1C; alternate-background-color:#131320;
                color:#CFD8DC; font-size:10px; border:none; gridline-color:#1E1E3A;
            }
            QHeaderView::section {
                background:#1A1A2A; color:#90CAF9; font-weight:bold; font-size:10px;
                border:none; border-bottom:1px solid #2A2A4A; padding:2px 6px;
            }
            QTableWidget::item { padding:2px 4px; }
        """)
        metrics_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        metrics_table.hide()
        cl.addWidget(metrics_table)

        card.hide()

        refs = {
            'rank_lbl':      rank_lbl,
            'score_lbl':     score_lbl,
            'load_btn':      load_btn,
            'headline_lbl':  headline_lbl,
            'moves_lbl':     moves_lbl,
            'plan_lbl':      plan_lbl,
            'metrics_sub':   metrics_sub,
            'metrics_table': metrics_table,
        }
        return card, refs

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Toolbar
        tb = QHBoxLayout()
        self._toggle_btn = QPushButton("▶  Coach OFF")
        self._toggle_btn.setFixedHeight(28)
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.setStyleSheet(
            "background:#2A2A2A; color:#546E7A; font-weight:bold;"
            " border:1px solid #37474F; border-radius:4px; padding:4px 10px;"
        )
        self._toggle_btn.clicked.connect(lambda: self._set_active(not self._active))
        tb.addWidget(self._toggle_btn)
        self._side_lbl = QLabel("♙ White")
        self._side_lbl.setFixedHeight(28)
        self._side_lbl.setStyleSheet(
            "background:#1B2A2A; color:#80CBC4; font-weight:bold; font-size:11px;"
            " border:1px solid #2E4A4A; border-radius:4px; padding:2px 10px;"
        )
        tb.addWidget(self._side_lbl)
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

        # Status row
        sr = QHBoxLayout()
        self._spinner = QLabel("⏳")
        self._spinner.setFixedWidth(22)
        self._spinner.hide()
        sr.addWidget(self._spinner)
        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color:#78909C; font-size:11px;")
        self._status_lbl.hide()
        sr.addWidget(self._status_lbl, 1)
        root.addLayout(sr)

        # Badge row (position-level strategy)
        br = QHBoxLayout()
        br.setSpacing(8)
        self._badge = QLabel("—")
        self._badge.setStyleSheet(
            "background:#37474F; color:#90A4AE; font-weight:bold;"
            " padding:3px 10px; border-radius:4px; font-size:11px;"
        )
        br.addWidget(self._badge)
        self._sec_label = QLabel("also:")
        self._sec_label.setStyleSheet("color:#546E7A; font-size:10px;")
        self._sec_label.hide()
        br.addWidget(self._sec_label)
        self._sec_badge = QLabel("")
        self._sec_badge.hide()
        br.addWidget(self._sec_badge)
        br.addStretch(1)
        self._conf_lbl = QLabel("")
        self._conf_lbl.setStyleSheet("color:#78909C; font-size:11px;")
        br.addWidget(self._conf_lbl)
        root.addLayout(br)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background:transparent;")
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)

        # Tactics — shown first so it's immediately visible without scrolling
        self._tactics_frame = QFrame()
        self._tactics_frame.setStyleSheet(
            "QFrame{background:#1A1F1A;border:1px solid #2E4A2E;border-radius:4px;}"
        )
        tl = QVBoxLayout(self._tactics_frame)
        tl.setContentsMargins(8, 6, 8, 6)
        tl.setSpacing(4)
        _tactics_hdr = QLabel("⚡ Tactics")
        _tactics_hdr.setStyleSheet("color:#A5D6A7;font-weight:bold;font-size:11px;")
        tl.addWidget(_tactics_hdr)
        self._tactics_lbl = QLabel("")
        self._tactics_lbl.setWordWrap(True)
        self._tactics_lbl.setTextFormat(Qt.RichText)
        self._tactics_lbl.setStyleSheet("color:#C8E6C9;font-size:11px;")
        tl.addWidget(self._tactics_lbl)
        self._tactics_frame.hide()
        cl.addWidget(self._tactics_frame)

        # Three analysis cards (one per PV line) — below tactics
        self._line_cards: list[QFrame] = []
        self._line_card_refs: list[dict] = []
        for i in range(3):
            card, refs = self._make_line_card(i)
            self._line_cards.append(card)
            self._line_card_refs.append(refs)
            cl.addWidget(card)

        # Weakness squares
        self._weak_frame = QFrame()
        self._weak_frame.setStyleSheet(
            "QFrame{background:#1F1A1A;border:1px solid #4A2E2E;border-radius:4px;}"
        )
        wl = QVBoxLayout(self._weak_frame)
        wl.setContentsMargins(8, 4, 8, 4)
        self._weak_lbl = QLabel("")
        self._weak_lbl.setWordWrap(True)
        self._weak_lbl.setStyleSheet("color:#FFCDD2;font-size:11px;")
        wl.addWidget(self._weak_lbl)
        self._weak_frame.hide()
        cl.addWidget(self._weak_frame)

        # GM Precedents
        self._precedents_frame = QFrame()
        self._precedents_frame.setStyleSheet(
            "QFrame{background:#1A1A2A;border:1px solid #2E2E4A;border-radius:4px;}"
        )
        pl = QVBoxLayout(self._precedents_frame)
        pl.setContentsMargins(8, 6, 8, 6)
        pl.setSpacing(4)
        prec_hdr = QLabel("♟  GM Precedents")
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

        # Inline GM detail card
        self._detail_frame = QFrame()
        self._detail_frame.setStyleSheet(
            "QFrame{background:#111122;border:1px solid #3A3A6A;border-radius:4px;}"
        )
        dl = QVBoxLayout(self._detail_frame)
        dl.setContentsMargins(10, 8, 10, 8)
        dl.setSpacing(6)
        self._detail_player = QLabel("")
        self._detail_player.setStyleSheet("color:#E1BEE7;font-weight:bold;font-size:12px;")
        dl.addWidget(self._detail_player)
        self._detail_move = QLabel("")
        self._detail_move.setStyleSheet("color:#B39DDB;font-size:11px;")
        dl.addWidget(self._detail_move)
        self._detail_ann = QLabel("")
        self._detail_ann.setWordWrap(True)
        self._detail_ann.setStyleSheet("color:#9E9E9E;font-size:11px;font-style:italic;")
        dl.addWidget(self._detail_ann)
        self._load_btn = QPushButton("▶  Open in Coach Board  +  Insert Variation")
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

    # ── Status ────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, busy: bool = False) -> None:
        self._status_lbl.setVisible(bool(msg))
        self._status_lbl.setText(msg)
        self._spinner.setVisible(busy)

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
        self._detail_player.setText(f"♟ {prec.player}")
        self._detail_move.setText(f"Move {mn} ({side})  —  key move: {prec.key_move}")
        self._detail_ann.setText(prec.annotation or "")
        self._detail_ann.setVisible(bool(prec.annotation))
        self._detail_frame.show()

    def _on_load_prec(self) -> None:
        if self._selected_prec is not None:
            self.gm_load_requested.emit(self._selected_prec)

    def closeEvent(self, event) -> None:
        self._debounce.stop()
        if self._engine:
            try:
                self._engine.close()
            except Exception:
                pass
        super().closeEvent(event)
