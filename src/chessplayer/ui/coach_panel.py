"""
ui/coach_panel.py
==================
Coach tab — front-facing GUI for the chess_coach backend.

Layout (per docs/layoutdesign.md)
----------------------------------
Toolbar:  [Coach ON/OFF]  [♙ White]  [↺ Refresh]  [Insert Note]
Panel:
  1. Opening / Endgame Bubble  — appears when ECO matched or tablebase probed
  2. Weak Squares Bubble       — appears when weak_square concept fires
  3. SF Line Bubble #1         — top Stockfish line; always shown after first analysis
  4. SF Line Bubble #2         — second line; shown when second PV available

Analysis is manual: the user presses Refresh. The SF engine runs continuously
in the background; only the ML + RAG layer is gated behind the button.

Signals → MainWindow
---------------------
    coach_help_requested(CoachOutput)      Insert output as PGN comment.
    gm_load_requested(GMPrecedent)         Load GM game to coach board.
    weakness_squares_ready(list[str])      Forward to CoachBoardWidget overlay.
    pv_line_load_requested(str,list,list,str)  Load PV line to coach board.
"""
from __future__ import annotations

import sys
from pathlib import Path

_chess_coach_dir = Path(__file__).resolve().parents[2] / "chess_coach"
if str(_chess_coach_dir) not in sys.path:
    sys.path.insert(0, str(_chess_coach_dir))

import chess

from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
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
            from chess_coach.coach.nimzo_net_engine import NimzoNetEngine
            self.ready.emit(NimzoNetEngine.from_config(self._config))
            return
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[COACH] NimzoNetEngine failed to load: {exc} — falling back to StrategyEngine")
        try:
            from chess_coach.core.strategy_engine import StrategyEngine
            self.ready.emit(StrategyEngine.from_config(self._config))
        except Exception as exc:
            self.failed.emit(str(exc))


class _MultiAnalysisWorker(QObject):
    """Run context + per-line analysis in one background thread.

    Signals
    -------
    context_ready(dict)     Opening/endgame bubble + weak squares data.
    line_ready(dict, int)   Per-line result dict + line index (0 or 1).
    all_done(int)           Token when all work is finished.
    failed(str, int)        Error message + token.
    """
    context_ready = Signal(object)          # dict
    line_ready    = Signal(object, int)     # (dict, line_idx)
    all_done      = Signal(int)
    failed        = Signal(str, int)

    def __init__(self, engine, board, side, pvs_with_scores, token):
        super().__init__()
        self._engine = engine
        self._board  = board.copy()
        self._side   = side
        self._pvs    = pvs_with_scores      # list[(pv_uci, score_cp)]
        self._token  = token

    def run(self) -> None:
        eco_code = None
        try:
            # Step 1: context (opening/endgame/weak squares)
            ctx = {}
            if hasattr(self._engine, 'get_context'):
                ctx = self._engine.get_context(self._board, player_side=self._side)
                eco_code = ctx.get('eco_code')
                self.context_ready.emit(ctx)
            else:
                # Legacy strategy engine fallback — emit empty context
                self.context_ready.emit({})

            # Step 2: up to 2 PV lines
            if hasattr(self._engine, 'analyse_line'):
                pvs = self._pvs[:2]
                if not pvs:
                    # No PVs yet — run context-position analysis as a placeholder line
                    line = self._engine.analyse_line(self._board, [], None, eco_code=eco_code)
                    self.line_ready.emit(line, 0)
                else:
                    for i, (pv_uci, score_cp) in enumerate(pvs):
                        try:
                            line = self._engine.analyse_line(
                                self._board, pv_uci, score_cp,
                                eco_code=eco_code, player_side=self._side,
                            )
                            self.line_ready.emit(line, i)
                        except Exception as exc:
                            self.failed.emit(f"Line {i + 1}: {exc}", self._token)
            else:
                # Legacy engine fallback
                pvs = self._pvs[:2]
                if not pvs:
                    out = self._engine.analyse(self._board, player_side=self._side)
                    self.line_ready.emit({'_legacy': out}, 0)
                else:
                    for i, (pv_uci, score_cp) in enumerate(pvs):
                        try:
                            out = self._engine.analyse_from_pv(
                                self._board, pv_uci, self._side, score_cp=score_cp
                            )
                            self.line_ready.emit({'_legacy': out}, i)
                        except Exception as exc:
                            self.failed.emit(f"Line {i + 1}: {exc}", self._token)

        except Exception as exc:
            self.failed.emit(str(exc), self._token)

        self.all_done.emit(self._token)


# ── Panel ─────────────────────────────────────────────────────────────────────

class CoachPanel(QWidget):

    coach_help_requested      = Signal(object)
    gm_load_requested         = Signal(object)
    weakness_squares_ready    = Signal(list)           # flat list (legacy)
    weakness_per_side_ready   = Signal(list, list)     # (player_sq, opponent_sq)
    pv_line_load_requested    = Signal(str, list, list, str)

    _COLOURS = {
        "blitz":            "#EF5350",
        "flank":            "#42A5F5",
        "fortress":         "#66BB6A",
        "feint":            "#AB47BC",
        "mating_attack":    "#EF5350",
        "passed_pawn":      "#FF7043",
        "outpost":          "#26A69A",
        "space_advantage":  "#42A5F5",
        "pawn_storm":       "#FF5722",
        "pawn_majority":    "#9CCC65",
        "blockade":         "#FFA726",
        "prophylaxis":      "#66BB6A",
        "initiative":       "#FFEE58",
        "development_lead": "#26C6DA",
        "piece_activity":   "#78909C",
        "king_activity":    "#AB47BC",
        "positional":       "#78909C",
        "general":          "#78909C",
    }

    def __init__(self, config: dict, parent=None) -> None:
        super().__init__(parent)
        self._config         = config
        self._engine         = None
        self._ready          = False
        self._active         = False
        self._token          = 0
        self._last_output    = None          # CoachOutput for Insert Note
        self._last_context   = {}            # most recent context dict
        self._insert_pending = False

        self._pending_board:       chess.Board | None = None
        self._pending_side:        str                = 'white'
        self._pending_history_uci: list[str]          = []
        self._multi_bms:           list               = []
        self._pending_analysis_fen: str               = ''

        self._init_thread  = self._init_worker  = None
        self._multi_thread = self._multi_worker = None

        self._build_ui()
        self._start_init()

    # ── Public API ────────────────────────────────────────────────────────────

    def queue_analysis(self, board, history=None, side='white') -> None:
        """Store the latest board position and side. Does NOT auto-trigger analysis."""
        self._pending_history_uci = list(history) if history else []
        # Rebuild the board from the explicit UCI history so move_stack is always complete.
        # The board received from the editor may have been set via FEN and lacks move_stack.
        if self._pending_history_uci:
            _b = chess.Board()
            for _uci in self._pending_history_uci:
                try:
                    _mv = chess.Move.from_uci(_uci)
                    if _mv in _b.legal_moves:
                        _b.push(_mv)
                    else:
                        break
                except Exception:
                    break
            self._pending_board = _b
        else:
            self._pending_board = board.copy()
        self._pending_side  = side
        self._multi_bms     = []
        side_text   = '♙ White' if side == 'white' else '♟ Black'
        side_style  = (
            "background:#1B2A2A; color:#80CBC4; font-weight:bold; font-size:11px;"
            " border:1px solid #2E4A4A; border-radius:4px; padding:2px 10px;"
            if side == 'white' else
            "background:#2A1B2A; color:#CE93D8; font-weight:bold; font-size:11px;"
            " border:1px solid #4A2E4A; border-radius:4px; padding:2px 10px;"
        )
        self._side_lbl.setText(side_text)
        self._side_lbl.setStyleSheet(side_style)

    @Slot(object)
    def on_pv_updated(self, best_move) -> None:
        pass   # kept for signal connection in window.py

    @Slot(list)
    def on_all_pvs_updated(self, pvs: list) -> None:
        """Store updated PV lines and auto-fire if coach is active and position has changed."""
        new_bms = [bm for bm in pvs if getattr(bm, 'pv_uci', None)]
        if new_bms:
            self._multi_bms = new_bms
        # Auto-refresh: if coach is active and SF has computed PVs for a position that
        # hasn't been analysed yet (piece moved to a different square), fire automatically.
        if (self._active and self._ready and self._multi_thread is None
                and self._pending_board is not None):
            current_fen = self._pending_board.fen()
            if current_fen != self._pending_analysis_fen:
                self._fire_analysis()

    def request_help(self) -> None:
        """Force one analysis + flag for PGN insertion. Auto-activates coach."""
        if not self._active:
            self._set_active(True)
        if self._last_output is not None:
            self.coach_help_requested.emit(self._last_output)
        else:
            self._insert_pending = True
            if self._ready and self._pending_board is not None:
                self._fire_analysis()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _start_init(self) -> None:
        self._set_status('Coach initialising…', busy=True)
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
        self._engine = engine
        self._ready  = True
        self._toggle_btn.setEnabled(True)
        self._refresh_btn.setEnabled(self._active)
        self._set_status('Coach ready — press Refresh to analyse', busy=False)

    @Slot(str)
    def _on_init_failed(self, msg: str) -> None:
        self._set_status(f'Coach unavailable: {msg}', busy=False)

    # ── Toggle ────────────────────────────────────────────────────────────────

    def _set_active(self, active: bool) -> None:
        self._active = active
        self._refresh_btn.setEnabled(active and self._ready)
        if active:
            self._toggle_btn.setText('⏹  Coach ON')
            self._toggle_btn.setStyleSheet(
                "background:#1B3A1B; color:#66BB6A; font-weight:bold;"
                " border:1px solid #2E5A2E; border-radius:4px; padding:4px 10px;"
            )
        else:
            self._toggle_btn.setText('▶  Coach OFF')
            self._toggle_btn.setStyleSheet(
                "background:#2A2A2A; color:#546E7A; font-weight:bold;"
                " border:1px solid #37474F; border-radius:4px; padding:4px 10px;"
            )

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _fire_analysis(self) -> None:
        if not self._ready or self._pending_board is None:
            self._set_status('No board to analyse', busy=False)
            return
        if self._multi_thread is not None:
            self._set_status('Analysis already running…', busy=True)
            return

        self._token += 1
        token = self._token
        self._pending_analysis_fen = self._pending_board.fen()
        self._set_status('Analysing…', busy=True)

        for card in self._line_cards:
            card.hide()
        self._context_bubble.hide()
        self._weak_bubble.hide()

        pvs_with_scores = [
            (list(bm.pv_uci), getattr(bm, 'score_cp', None))
            for bm in self._multi_bms[:2]
        ] if self._multi_bms else []

        self._multi_thread = QThread(self)
        self._multi_worker = _MultiAnalysisWorker(
            self._engine, self._pending_board, self._pending_side,
            pvs_with_scores, token,
        )
        self._multi_worker.moveToThread(self._multi_thread)
        self._multi_thread.started.connect(self._multi_worker.run)
        self._multi_worker.context_ready.connect(self._on_context_ready)
        self._multi_worker.line_ready.connect(self._on_line_ready)
        self._multi_worker.all_done.connect(self._on_multi_done)
        self._multi_worker.failed.connect(self._on_multi_failed)
        self._multi_worker.all_done.connect(self._multi_thread.quit)
        self._multi_worker.failed.connect(self._multi_thread.quit)
        self._multi_thread.finished.connect(self._cleanup_multi)
        self._multi_thread.start()

    @Slot(object)
    def _on_context_ready(self, ctx: dict) -> None:
        if not self._active or not ctx:
            return
        self._last_context = ctx
        self._render_context(ctx)

    @Slot(object, int)
    def _on_line_ready(self, line_data, line_idx: int) -> None:
        if not self._active:
            return
        if '_legacy' in line_data:
            # Legacy engine path — use old renderer
            out = line_data['_legacy']
            self._render_line_card_legacy(line_idx, out)
            if line_idx == 0:
                self._last_output = out
                self._insert_btn.setEnabled(True)
                self.weakness_squares_ready.emit(out.weakness_squares)
        else:
            self._render_line_card(line_idx, line_data)
            if line_idx == 0:
                self._insert_btn.setEnabled(True)
                # Build a minimal CoachOutput for insert_note compatibility
                self._last_output = self._line_to_coach_output(line_data)
                if self._insert_pending:
                    self._insert_pending = False
                    self.coach_help_requested.emit(self._last_output)

    @Slot(int)
    def _on_multi_done(self, token: int) -> None:
        if token != self._token:
            return
        self._set_status('Ready', busy=False)

    @Slot(str, int)
    def _on_multi_failed(self, msg: str, token: int) -> None:
        if token == self._token:
            self._set_status(f'Error: {msg}', busy=False)

    def _cleanup_multi(self) -> None:
        if self._multi_worker:
            self._multi_worker.deleteLater()
            self._multi_worker = None
        if self._multi_thread:
            self._multi_thread.deleteLater()
            self._multi_thread = None

    # ── Compat helper ─────────────────────────────────────────────────────────

    def _line_to_coach_output(self, line: dict):
        """Construct a minimal CoachOutput from a line dict for Insert Note."""
        from chess_coach.core.data_types import CoachOutput
        badge   = line.get('theme_badge', 'general')
        from chess_coach.core.data_types import STRATEGIES
        strategy = badge if badge in set(STRATEGIES) else 'general'
        hints    = line.get('plan_sentences') or (line.get('tier1_hints') or []) + (line.get('tier2_hints') or [])
        ctx      = self._last_context
        return CoachOutput(
            strategy_primary   = strategy,
            strategy_secondary = None,
            confidence         = line.get('confidence', 0.5),
            phase              = line.get('phase', 'middlegame'),
            headline           = f"{badge.upper().replace('_', ' ')} — after move 1",
            plan_sentences     = hints if hints else ['The position has been analysed.'],
            weakness_squares   = ctx.get('weak_squares', []) if ctx else [],
            rag_annotation     = line.get('rag_annotation', ''),
            rag_annotation_src = line.get('rag_src', ''),
        )

    # ── Render — context bubble ────────────────────────────────────────────────

    def _render_context(self, ctx: dict) -> None:
        """Update Opening/Endgame bubble and Weak Squares bubble from context dict."""
        # Opening/endgame bubble
        wdl          = ctx.get('wdl')
        opening_name = ctx.get('opening_name', '')
        opening_rag  = ctx.get('opening_rag', '')
        opening_src  = ctx.get('opening_rag_src', '')
        opening_depth= ctx.get('opening_depth', 0)
        endgame_type = ctx.get('endgame_type', '')
        endgame_rag  = ctx.get('endgame_rag', '')
        endgame_src  = ctx.get('endgame_rag_src', '')
        dtz          = ctx.get('dtz')

        show_bubble = False

        if wdl is not None:
            # Tablebase mode
            show_bubble = True
            eg_label = endgame_type.replace('_', ' ').title() if endgame_type else 'Endgame'
            if wdl == 2:
                wdl_chip = '<span style="color:#A5D6A7;font-weight:bold;">WIN</span>'
                if dtz is not None:
                    wdl_chip += f' <span style="color:#78909C;">(DTZ {abs(dtz)})</span>'
            elif wdl == 0:
                wdl_chip = '<span style="color:#FFD54F;font-weight:bold;">DRAW</span>'
            else:
                wdl_chip = '<span style="color:#EF9A9A;font-weight:bold;">LOSS</span>'
                if dtz is not None:
                    wdl_chip += f' <span style="color:#78909C;">(DTZ {abs(dtz)})</span>'

            html = (
                f'<p style="margin:0 0 4px 0;color:#90A4AE;font-size:10px;">'
                f'TABLEBASE POSITION</p>'
                f'<p style="margin:0 0 4px 0;color:#E0E0E0;font-weight:bold;">'
                f'{eg_label} — {wdl_chip}</p>'
            )
            if endgame_rag:
                html += (
                    f'<p style="margin:4px 0 2px 0;color:#B0BEC5;font-size:11px;'
                    f'font-style:italic;">"{endgame_rag}"</p>'
                )
                if endgame_src:
                    html += (
                        f'<p style="margin:0;color:#546E7A;font-size:10px;">'
                        f'— {endgame_src}</p>'
                    )
            self._ctx_lbl.setText(html)
            self._ctx_lbl.setTextFormat(Qt.RichText)

        elif opening_name:
            # Opening mode
            show_bubble      = True
            opening_moves    = ctx.get('opening_moves_san', [])
            opening_conts    = ctx.get('opening_continuations', [])
            depth_str        = (f'  <span style="color:#546E7A;">move {opening_depth}</span>'
                                if opening_depth else '')
            html = (
                f'<p style="margin:0 0 4px 0;color:#90A4AE;font-size:10px;">OPENING</p>'
                f'<p style="margin:0 0 4px 0;color:#E0E0E0;font-weight:bold;">'
                f'{opening_name}{depth_str}</p>'
            )
            # Move sequence — mirror the actual game moves that are in theory
            if opening_moves:
                parts = []
                for i, san in enumerate(opening_moves):
                    if i % 2 == 0:
                        parts.append(f'{i // 2 + 1}. {san}')
                    else:
                        parts.append(san)
                line_text = '&nbsp;&nbsp;'.join(parts)
                html += (
                    f'<p style="margin:2px 0 4px 0;color:#80CBC4;'
                    f'font-family:monospace;font-size:11px;">{line_text}</p>'
                )
            # Theory continuations from ECO database
            if opening_conts:
                cont_text = ' / '.join(opening_conts)
                html += (
                    f'<p style="margin:0 0 4px 0;color:#546E7A;font-size:10px;">'
                    f'Continues: {cont_text}</p>'
                )
            if opening_rag:
                html += (
                    f'<p style="margin:4px 0 2px 0;color:#B0BEC5;font-size:11px;'
                    f'font-style:italic;">"{opening_rag}"</p>'
                )
                if opening_src:
                    html += (
                        f'<p style="margin:0;color:#546E7A;font-size:10px;">'
                        f'— {opening_src}</p>'
                    )
            self._ctx_lbl.setText(html)
            self._ctx_lbl.setTextFormat(Qt.RichText)

        if show_bubble:
            self._context_bubble.show()
        else:
            self._context_bubble.hide()

        # Weak squares bubble
        weak_squares = ctx.get('weak_squares', [])
        weak_action  = ctx.get('weak_action', '')
        weak_rag     = ctx.get('weak_rag', '')
        weak_src     = ctx.get('weak_rag_src', '')

        weak_squares_opp = ctx.get('weak_squares_opp', [])
        self.weakness_squares_ready.emit(weak_squares)
        self.weakness_per_side_ready.emit(weak_squares, weak_squares_opp)

        if weak_squares:
            sq_text = '  ·  '.join(weak_squares)
            html = (
                f'<p style="margin:0 0 4px 0;color:#90A4AE;font-size:10px;">WEAK SQUARES</p>'
                f'<p style="margin:0 0 4px 0;color:#FFCDD2;font-weight:bold;">{sq_text}</p>'
            )
            if weak_action:
                html += f'<p style="margin:2px 0;color:#EF9A9A;font-size:11px;">{weak_action}</p>'
            if weak_rag:
                html += (
                    f'<p style="margin:4px 0 2px 0;color:#B0BEC5;font-size:11px;'
                    f'font-style:italic;">"{weak_rag}"</p>'
                )
                if weak_src:
                    html += f'<p style="margin:0;color:#546E7A;font-size:10px;">— {weak_src}</p>'
            self._weak_lbl.setText(html)
            self._weak_lbl.setTextFormat(Qt.RichText)
            self._weak_bubble.show()
        else:
            self._weak_bubble.hide()

    # ── Render — line card (new) ───────────────────────────────────────────────

    def _render_line_card(self, idx: int, line: dict) -> None:
        if idx >= len(self._line_cards):
            return
        card = self._line_cards[idx]
        refs = self._line_card_refs[idx]
        bm   = self._multi_bms[idx] if idx < len(self._multi_bms) else None

        badge           = line.get('theme_badge', 'general')
        score_cp        = line.get('score_cp')
        plan_sentences  = line.get('plan_sentences') or []
        rag_text        = line.get('rag_annotation', '')
        rag_src         = line.get('rag_src', '')
        metrics         = line.get('sf_eval_rows') or []   # (term, white, black, total)

        # Score
        if score_cp is not None:
            score_str   = f'{score_cp / 100.0:+.2f}'
            score_color = '#A5D6A7' if score_cp >= 0 else '#EF9A9A'
        elif bm and getattr(bm, 'mate_in', None) is not None:
            score_str   = f'M{bm.mate_in}'
            score_color = '#FFD54F'
        else:
            score_str   = '?'
            score_color = '#78909C'

        # Theme badge + confidence
        colour          = self._COLOURS.get(badge, '#78909C')
        confidence_pct  = int(line.get('confidence', 0) * 100)
        badge_text      = badge.replace('_', ' ').title()
        badge_display   = f"{badge_text} ({confidence_pct}%)" if confidence_pct else badge_text
        refs['badge_lbl'].setText(badge_display)
        refs['badge_lbl'].setStyleSheet(
            f"background:{colour}; color:white; font-weight:bold;"
            f" padding:2px 8px; border-radius:4px; font-size:11px;"
        )
        refs['score_lbl'].setText(score_str)
        refs['score_lbl'].setStyleSheet(
            f"color:{score_color}; font-family:monospace; font-size:11px; font-weight:bold;"
        )

        # Generate SAN for the FULL PV line (coach board gets all moves)
        pv_uci = list(bm.pv_uci) if bm and getattr(bm, 'pv_uci', None) else []
        pv_san_full: list[str] = []
        if pv_uci and self._pending_analysis_fen:
            try:
                _b = chess.Board(self._pending_analysis_fen)
                for uci in pv_uci:
                    mv = chess.Move.from_uci(uci)
                    if mv in _b.legal_moves:
                        pv_san_full.append(_b.san(mv))
                        _b.push(mv)
                    else:
                        break
            except Exception:
                pass
        pv_san = pv_san_full   # full list goes to coach board via load button
        refs['moves_lbl'].setText('  '.join(pv_san_full[:5]))
        fen    = self._pending_analysis_fen
        title  = f'Best Line {score_str}' if idx == 0 else f'Line #{idx + 1} {score_str}'
        try:
            refs['load_btn'].clicked.disconnect()
        except RuntimeError:
            pass
        refs['load_btn'].clicked.connect(
            lambda _c=False, f=fen, u=pv_uci, s=pv_san, t=title:
                self.pv_line_load_requested.emit(f, u, s, t)
        )

        # SF eval table: Term / Before / After / Δ
        mt = refs['metrics_table']
        if metrics:
            mt.setRowCount(len(metrics))
            for r, row in enumerate(metrics):
                term   = row[0] if len(row) > 0 else ''
                before = row[1] if len(row) > 1 else ''
                after  = row[2] if len(row) > 2 else ''
                delta  = row[3] if len(row) > 3 else ''
                mt.setItem(r, 0, QTableWidgetItem(term))
                mt.setItem(r, 1, QTableWidgetItem(before))
                mt.setItem(r, 2, QTableWidgetItem(after))
                try:
                    d_val  = float(delta.replace(' ', ''))
                    d_item = QTableWidgetItem(delta)
                    d_item.setForeground(
                        QColor('#A5D6A7') if d_val > 0.05 else
                        QColor('#EF9A9A') if d_val < -0.05 else
                        QColor('#78909C')
                    )
                    mt.setItem(r, 3, d_item)
                except (ValueError, AttributeError):
                    mt.setItem(r, 3, QTableWidgetItem(delta))
            mt.resizeRowsToContents()
            hdr_h = mt.horizontalHeader().height()
            row_h = sum(mt.rowHeight(r) for r in range(len(metrics)))
            mt.setFixedHeight(hdr_h + row_h + 2)
            refs['metrics_sub'].show()
            refs['metrics_ba_lbl'].show()
            mt.show()
        else:
            refs['metrics_sub'].hide()
            refs['metrics_ba_lbl'].hide()
            mt.hide()

        # Part 1 — Assembled narrative sentences from NimzoNet → phrase DB
        if plan_sentences:
            p1_html = ''.join(
                f'<p style="margin:0 0 5px 0;color:#B0BEC5;">{s}</p>'
                for s in plan_sentences
            )
            refs['part1_lbl'].setText(p1_html)
            refs['part1_lbl'].setTextFormat(Qt.RichText)
            refs['part1_lbl'].show()
        else:
            refs['part1_lbl'].hide()

        # Part 2 — Precedent
        if rag_text:
            p2_html = (
                f'<p style="margin:0 0 3px 0;color:#B0BEC5;font-style:italic;">'
                f'"{rag_text}"</p>'
            )
            if rag_src:
                p2_html += f'<p style="margin:0;color:#546E7A;font-size:10px;">— {rag_src}</p>'
            refs['part2_frame'].show()
            refs['part2_lbl'].setText(p2_html)
            refs['part2_lbl'].setTextFormat(Qt.RichText)
        else:
            refs['part2_frame'].hide()

        card.show()

    # ── Render — line card (legacy engine) ────────────────────────────────────

    def _render_line_card_legacy(self, idx: int, output) -> None:
        """Render using a CoachOutput from the legacy StrategyEngine."""
        if idx >= len(self._line_cards):
            return
        card = self._line_cards[idx]
        refs = self._line_card_refs[idx]
        bm   = self._multi_bms[idx] if idx < len(self._multi_bms) else None

        score_str   = '?'
        score_color = '#78909C'
        if bm and getattr(bm, 'score_cp', None) is not None:
            score_str   = f'{bm.score_cp / 100.0:+.2f}'
            score_color = '#A5D6A7' if bm.score_cp >= 0 else '#EF9A9A'

        badge  = output.strategy_primary
        colour = self._COLOURS.get(badge, '#37474F')
        refs['badge_lbl'].setText(badge.upper())
        refs['badge_lbl'].setStyleSheet(
            f"background:{colour}; color:white; font-weight:bold;"
            f" padding:2px 8px; border-radius:4px; font-size:11px;"
        )
        refs['score_lbl'].setText(score_str)
        refs['score_lbl'].setStyleSheet(
            f"color:{score_color}; font-family:monospace; font-size:11px; font-weight:bold;"
        )

        pv_text = getattr(output, 'pv_line_text', '')
        refs['moves_lbl'].setText(pv_text)

        sentences = output.plan_sentences
        refs['part1_lbl'].setText(''.join(
            f'<p style="margin:0 0 5px 0;color:#90A4AE;">{s}</p>' for s in sentences
        ))
        refs['part1_lbl'].setTextFormat(Qt.RichText)
        refs['part1_lbl'].show()
        refs['part2_frame'].hide()
        refs['metrics_sub'].hide()
        refs['metrics_table'].hide()

        card.show()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _make_line_card(self, idx: int) -> tuple:
        """Build one SF line bubble. Returns (frame, refs-dict)."""
        card = QFrame()
        card.setStyleSheet(
            "QFrame{background:#0F1920; border:1px solid #1E2E3A; border-radius:4px;}"
        )
        cl = QVBoxLayout(card)
        cl.setContentsMargins(8, 6, 8, 8)
        cl.setSpacing(5)

        # Header row: theme badge + score + "after 1 move" + load button
        hdr = QHBoxLayout()
        hdr.setSpacing(6)

        badge_lbl = QLabel('—')
        badge_lbl.setStyleSheet(
            "background:#37474F; color:#90A4AE; font-weight:bold;"
            " padding:2px 8px; border-radius:4px; font-size:11px;"
        )
        hdr.addWidget(badge_lbl)

        score_lbl = QLabel('')
        score_lbl.setStyleSheet(
            "color:#A5D6A7; font-family:monospace; font-size:11px; font-weight:bold;"
        )
        hdr.addWidget(score_lbl)

        after_lbl = QLabel('Recommended line')
        after_lbl.setStyleSheet("color:#455A64; font-size:10px;")
        hdr.addWidget(after_lbl)

        hdr.addStretch(1)

        load_btn = QPushButton('♞  Load in Coach Board')
        load_btn.setFixedHeight(22)
        load_btn.setStyleSheet(
            "background:#1B2A3A; color:#4FC3F7; font-weight:bold; font-size:10px;"
            " border:1px solid #1E4A6A; border-radius:3px; padding:1px 8px;"
        )
        hdr.addWidget(load_btn)
        cl.addLayout(hdr)

        # PV moves
        moves_lbl = QLabel('')
        moves_lbl.setWordWrap(True)
        moves_lbl.setStyleSheet(
            "color:#78909C; font-family:monospace; font-size:11px; padding:2px 0;"
        )
        cl.addWidget(moves_lbl)

        # Metrics table header
        metrics_sub = QLabel('Stockfish Eval Terms — for the recommended line')
        metrics_sub.setStyleSheet("color:#546E7A; font-size:10px; padding:2px 0; font-style:italic;")
        metrics_sub.hide()
        cl.addWidget(metrics_sub)

        # Before/After column sub-header
        ba_lbl = QLabel('Before = position NOW  ·  After = position after the full line plays out  ·  Δ = your gain')
        ba_lbl.setStyleSheet("color:#37474F; font-size:9px; padding:0 0 1px 0;")
        ba_lbl.hide()
        cl.addWidget(ba_lbl)

        # Metrics table
        metrics_table = QTableWidget(0, 4)
        metrics_table.setHorizontalHeaderLabels(['Term', 'Before', 'After', 'Δ'])
        metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3):
            metrics_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
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

        # Part 1 — Pattern
        part1_lbl = QLabel('')
        part1_lbl.setWordWrap(True)
        part1_lbl.setTextFormat(Qt.RichText)
        part1_lbl.setStyleSheet("color:#90A4AE; font-size:11px; padding:2px 0;")
        part1_lbl.hide()
        cl.addWidget(part1_lbl)

        # Part 2 — Precedent blockquote
        part2_frame = QFrame()
        part2_frame.setStyleSheet(
            "QFrame{background:#131825; border-left:3px solid #2A3A5A; border-radius:2px;}"
        )
        p2l = QVBoxLayout(part2_frame)
        p2l.setContentsMargins(8, 5, 8, 5)
        hdr2 = QLabel('PRECEDENT')
        hdr2.setStyleSheet("color:#546E7A; font-size:9px; font-weight:bold; letter-spacing:1px;")
        p2l.addWidget(hdr2)
        part2_lbl = QLabel('')
        part2_lbl.setWordWrap(True)
        part2_lbl.setTextFormat(Qt.RichText)
        part2_lbl.setStyleSheet("color:#78909C; font-size:10px;")
        p2l.addWidget(part2_lbl)
        part2_frame.hide()
        cl.addWidget(part2_frame)

        card.hide()

        refs = {
            'badge_lbl':     badge_lbl,
            'score_lbl':     score_lbl,
            'load_btn':      load_btn,
            'moves_lbl':     moves_lbl,
            'metrics_sub':   metrics_sub,
            'metrics_ba_lbl': ba_lbl,
            'metrics_table': metrics_table,
            'part1_lbl':     part1_lbl,
            'part2_frame':   part2_frame,
            'part2_lbl':     part2_lbl,
        }
        return card, refs

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── Toolbar ───────────────────────────────────────────────────────────
        tb = QHBoxLayout()
        tb.setSpacing(6)

        self._toggle_btn = QPushButton('▶  Coach OFF')
        self._toggle_btn.setFixedHeight(28)
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.setStyleSheet(
            "background:#2A2A2A; color:#546E7A; font-weight:bold;"
            " border:1px solid #37474F; border-radius:4px; padding:4px 10px;"
        )
        self._toggle_btn.clicked.connect(lambda: self._set_active(not self._active))
        tb.addWidget(self._toggle_btn)

        self._side_lbl = QLabel('♙ White')
        self._side_lbl.setFixedHeight(28)
        self._side_lbl.setStyleSheet(
            "background:#1B2A2A; color:#80CBC4; font-weight:bold; font-size:11px;"
            " border:1px solid #2E4A4A; border-radius:4px; padding:2px 10px;"
        )
        tb.addWidget(self._side_lbl)

        self._refresh_btn = QPushButton('↺  Refresh')
        self._refresh_btn.setFixedHeight(28)
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.setStyleSheet(
            "background:#1A2A1A; color:#81C784; font-weight:bold;"
            " border:1px solid #2E5A2E; border-radius:4px; padding:4px 10px;"
        )
        self._refresh_btn.clicked.connect(self._fire_analysis)
        tb.addWidget(self._refresh_btn)

        tb.addStretch(1)

        self._insert_btn = QPushButton('\U0001f4cb  Insert Note')
        self._insert_btn.setFixedHeight(28)
        self._insert_btn.setEnabled(False)
        self._insert_btn.setStyleSheet(
            "background:#1B2A3A; color:#4FC3F7; font-weight:bold;"
            " border:1px solid #1E4A6A; border-radius:4px; padding:4px 10px;"
        )
        self._insert_btn.setToolTip(
            "Insert coach recommendation as a PGN comment at the current move."
        )
        self._insert_btn.clicked.connect(self._on_insert_clicked)
        tb.addWidget(self._insert_btn)
        root.addLayout(tb)

        # ── Status row ────────────────────────────────────────────────────────
        sr = QHBoxLayout()
        self._spinner = QLabel('⏳')
        self._spinner.setFixedWidth(22)
        self._spinner.hide()
        sr.addWidget(self._spinner)
        self._status_lbl = QLabel('')
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color:#78909C; font-size:11px;")
        self._status_lbl.hide()
        sr.addWidget(self._status_lbl, 1)
        root.addLayout(sr)

        # ── Scroll area content ───────────────────────────────────────────────
        scroll  = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background:transparent;")
        content = QWidget()
        cl      = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)

        # Opening / Endgame Bubble
        self._context_bubble = QFrame()
        self._context_bubble.setStyleSheet(
            "QFrame{background:#1A1A2A; border:1px solid #2E2E5A; border-radius:4px;}"
        )
        ctxl = QVBoxLayout(self._context_bubble)
        ctxl.setContentsMargins(10, 8, 10, 8)
        self._ctx_lbl = QLabel('')
        self._ctx_lbl.setWordWrap(True)
        self._ctx_lbl.setTextFormat(Qt.RichText)
        ctxl.addWidget(self._ctx_lbl)
        self._context_bubble.hide()
        cl.addWidget(self._context_bubble)

        # Weak Squares Bubble
        self._weak_bubble = QFrame()
        self._weak_bubble.setStyleSheet(
            "QFrame{background:#1F1A1A; border:1px solid #4A2E2E; border-radius:4px;}"
        )
        wbl = QVBoxLayout(self._weak_bubble)
        wbl.setContentsMargins(10, 8, 10, 8)
        self._weak_lbl = QLabel('')
        self._weak_lbl.setWordWrap(True)
        self._weak_lbl.setTextFormat(Qt.RichText)
        wbl.addWidget(self._weak_lbl)
        self._weak_bubble.hide()
        cl.addWidget(self._weak_bubble)

        # SF Line Bubbles (2 maximum)
        self._line_cards: list[QFrame] = []
        self._line_card_refs: list[dict] = []
        for i in range(2):
            card, refs = self._make_line_card(i)
            self._line_cards.append(card)
            self._line_card_refs.append(refs)
            cl.addWidget(card)

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

    def closeEvent(self, event) -> None:
        if self._engine:
            try:
                self._engine.close()
            except Exception:
                pass
        super().closeEvent(event)
