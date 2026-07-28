"""
coach/nimzo_net_engine.py
==========================
Bridge from the MoE concept classifier to the live application.

Public API
----------
    engine = NimzoNetEngine.from_config(config)
    context = engine.get_context(board)          # → dict; opening, endgame, weak squares
    line    = engine.analyse_line(board, pv_uci, score_cp)  # → dict; per-PV explanation
    output  = engine.analyse(board, player_side)            # → CoachOutput (insert-note compat)
    engine.close()
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import torch

_PROJ_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from chess_coach.core.data_types        import CoachOutput, STRATEGIES
from chess_coach.core.board_utils       import get_phase
from chess_coach.core.conflict_resolver import ResolverResult
from chess_coach.coach.narrator         import assemble
from chess_coach.database.phrase_db     import PhraseDB
from chess_coach.ml.concept_signal_adapter import (
    adapt, infer_strategy,
    _ACTION_HINTS, TIER1_CONCEPTS, TIER2_CONCEPTS, TIER4_CONCEPTS,
)
from chess_coach.ml.paths               import CLASSIFIER_BEST as _CHECKPOINT
from chess_coach.ml.evaluate            import load_thresholds

_VALID_STRATEGIES = set(STRATEGIES)


class NimzoNetEngine:

    def __init__(self, model, phrase_db: PhraseDB, retriever=None, sf_exe: str = '') -> None:
        self._model     = model
        self._phrase_db = phrase_db
        self._retriever = retriever
        self._sf_exe    = sf_exe            # path to stockfish binary for eval terms

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "NimzoNetEngine":
        from chess_coach.ml.classifier import ChessConceptClassifier, MoEConceptClassifier

        ckpt_path = Path(
            config.get('coach', {}).get('nimzo_checkpoint', str(_CHECKPOINT))
        )
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"NimzoNet checkpoint not found: {ckpt_path}\n"
                "Run the training pipeline first: .\\retrain_and_reparse.ps1"
            )

        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        sd   = ckpt['state_dict']

        is_phase6 = any(k.startswith("gate_network.") for k in sd)
        is_phase5 = any(k.startswith("nnue_proj")     for k in sd) and not is_phase6
        is_phase4 = any(k.startswith("spatial_proj")  for k in sd) and not is_phase5 and not is_phase6

        if is_phase6:
            model = MoEConceptClassifier()
        else:
            model = ChessConceptClassifier(phase4=is_phase4, phase5=is_phase5)
        model.load_state_dict(sd)
        model.eval()

        db_path = config.get('coach', {}).get('phrase_db', '')
        phrase_db = PhraseDB(db_path)

        retriever = None
        try:
            from chess_coach.rag.retriever import RAGRetriever
            retriever = RAGRetriever()
        except Exception:
            pass

        # Resolve SF exe relative to project root (same path the engine panel uses)
        _sf_raw = config.get('paths', {}).get('engine_exe', '')
        sf_exe = ''
        if _sf_raw:
            _sf_path = Path(_sf_raw)
            sf_exe = str(_sf_path if _sf_path.is_absolute() else _PROJ_ROOT / _sf_path)

        return cls(model, phrase_db, retriever, sf_exe=sf_exe)

    # ── History helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_history_from_board(board: chess.Board):
        """Extract move history from a board's move_stack.

        Returns (history_uci, history_rich, history_fens).
        history_rich is the per-move dict list consumed by the GRU.
        """
        from chess_coach.rag.coach import _build_history_rich
        moves = list(board.move_stack)
        history_uci = [m.uci() for m in moves]
        if not history_uci:
            return [], [], [chess.STARTING_FEN]
        history_fens, history_rich = _build_history_rich(chess.STARTING_FEN, history_uci)
        return history_uci, history_rich, history_fens

    # ── Tablebase probe ───────────────────────────────────────────────────────

    @staticmethod
    def _probe_tablebase(board: chess.Board):
        """Probe Syzygy tablebase. Returns (wdl, dtz) or (None, None)."""
        try:
            import chess.syzygy
            from chess_coach.ml.paths import SYZYGY_PATH
            syzygy_path = _PROJ_ROOT / 'data' / 'syzygy'
            if not syzygy_path.exists():
                return None, None
            with chess.syzygy.open_tablebase(str(syzygy_path)) as tb:
                wdl = tb.probe_wdl(board)
                dtz = tb.probe_dtz(board)
                return wdl, dtz
        except Exception:
            return None, None

    # ── RAG helpers ───────────────────────────────────────────────────────────

    def _retrieve_one(self, fen: str, history_fens: list, concepts: list, eco_code: str | None) -> tuple[str, str]:
        """Retrieve top-1 RAG annotation. Returns (text, source) or ('', '').

        Allows ECO-only retrieval when concepts is empty and eco_code is set,
        so the opening bubble gets commentary even before any ML concept fires.
        Annotations in non-English languages are automatically translated.
        """
        if not self._retriever:
            return '', ''
        if not concepts and not eco_code:
            return '', ''
        _MIN_CHARS = 200
        try:
            anns = self._retriever.retrieve(
                fen,
                history_fens=history_fens or None,
                concepts=concepts,
                eco_override=eco_code,
                n=4,
            )
            if anns:
                # Prefer the longest annotation that meets the minimum length;
                # fall back to the longest available if none reach it.
                best_text, best_src = '', ''
                long_text, long_src = '', ''
                for ann in anns:
                    t = ann.get('annotation', '')
                    s = ann.get('game', '') or ann.get('eco', eco_code or '')
                    if len(t) > len(long_text):
                        long_text, long_src = t, s
                    if len(t) >= _MIN_CHARS and len(t) > len(best_text):
                        best_text, best_src = t, s
                chosen_text = best_text if best_text else long_text
                chosen_src  = best_src  if best_text else long_src
                return self._ensure_english(chosen_text), chosen_src
        except Exception:
            pass
        return '', ''

    @staticmethod
    def _ensure_english(text: str) -> str:
        """Translate non-English text to English. Silently returns original on failure."""
        if not text or len(text) < 20:
            return text
        try:
            from langdetect import detect, LangDetectException
            try:
                lang = detect(text)
            except LangDetectException:
                return text
            if lang == 'en':
                return text
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source=lang, target='en').translate(text)
            return translated if translated else text
        except Exception:
            return text

    # ── SF eval term breakdown ────────────────────────────────────────────────

    def _get_sf_eval_terms(self, fen: str) -> list[tuple[str, str, str, str]]:
        """Open a one-shot SF process, run 'eval', return classical term breakdown.

        Returns list of (term, white, black, total) string tuples, or [] on failure.
        Only the classical evaluation table is returned even when NNUE is active.
        """
        if not self._sf_exe:
            return []
        try:
            import subprocess
            proc = subprocess.Popen(
                [self._sf_exe],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True,
            )
            stdout, _ = proc.communicate(
                input=f'position fen {fen}\neval\nquit\n',
                timeout=4,
            )
            return self._parse_sf_eval(stdout)
        except Exception:
            return []

    @staticmethod
    def _parse_sf_eval(output: str) -> list[tuple[str, str, str, str]]:
        """Parse the ASCII table from SF's 'eval' command output."""
        rows: list[tuple[str, str, str, str]] = []
        in_classical = False
        for line in output.splitlines():
            # Locate the classical eval section (works with NNUE and classic builds)
            if 'classical' in line.lower():
                in_classical = True
                continue
            # Stop at NNUE section or second blank separator
            if in_classical and ('nnue' in line.lower() and 'contributing' in line.lower()):
                break
            if not in_classical:
                continue
            if not line.strip() or line.startswith('+--') or line.startswith('|--'):
                continue
            if '|' not in line:
                continue
            parts = [p.strip() for p in line.strip().strip('|').split('|')]
            if len(parts) < 2:
                continue
            term = parts[0].strip()
            if not term or term == 'Term':
                continue
            white = parts[1].strip() if len(parts) > 1 else ''
            black = parts[2].strip() if len(parts) > 2 else ''
            total = parts[3].strip() if len(parts) > 3 else ''
            rows.append((term, white, black, total))
        return rows

    # ── Deep RAG walk-back ────────────────────────────────────────────────────

    def _retrieve_line_rag(
        self,
        board: chess.Board,
        pv_uci: list[str],
        history_fens: list[str],
        concept_names: list[str],
        max_depth: int = 10,
    ) -> tuple[str, str]:
        """Walk back from max_depth moves deep in the PV to find a relevant annotation.

        Algorithm:
          1. Push min(max_depth, len(pv_uci)) moves.
          2. Retrieve RAG for that position.
          3. If found, run relevance check (annotation mentions first-move destination).
          4. If blank or irrelevant, decrement depth and try again.
          5. At depth=1 (post-first-move), accept any non-empty annotation.
        """
        if not pv_uci:
            return '', ''

        # SAN and destination square of the first recommended move (for relevance check)
        first_dest = pv_uci[0][2:4] if len(pv_uci[0]) >= 4 else ''
        first_san  = ''
        try:
            mv = chess.Move.from_uci(pv_uci[0])
            if mv in board.legal_moves:
                first_san = board.san(mv).lower().replace('+', '').replace('#', '')
        except Exception:
            pass

        _MIN_RAG = 200   # minimum annotation length worth accepting
        depth = min(max_depth, len(pv_uci))
        best_text, best_src = '', ''   # keep best even if relevance check fails

        while depth >= 1:
            b = board.copy()
            # Build history_fens that INCLUDES the PV moves up to this depth.
            # This is what differentiates the two SF lines — each PV produces a
            # distinct trajectory so the retriever identifies a different opening.
            pv_history = list(history_fens)
            pushed = 0
            for uci in pv_uci[:depth]:
                try:
                    mv = chess.Move.from_uci(uci)
                    if mv in b.legal_moves:
                        pv_history.append(b.fen())   # position BEFORE push
                        b.push(mv)
                        pushed += 1
                    else:
                        break
                except Exception:
                    break

            if pushed == 0:
                depth -= 1
                continue

            text, src = self._retrieve_one(b.fen(), pv_history, concept_names, None)
            if text:
                # Keep running best — prefer longer annotations
                if len(text) > len(best_text):
                    best_text, best_src = text, src
                # Accept immediately only if annotation is long enough AND relevant
                tl = text.lower()
                long_enough = len(text) >= _MIN_RAG
                relevant    = (depth == 1 or first_dest in tl or
                               (first_san and first_san in tl))
                if long_enough and relevant:
                    return text, src

            depth -= 1

        return best_text, best_src

    # ── Public API — full context (opening/endgame/weak squares) ─────────────

    def get_context(self, board: chess.Board, player_side: str = 'white') -> dict:
        """Compute all context data for the opening/endgame bubble and weak squares.

        Returns a dict consumed by the panel's context_ready signal handler.
        Keys: eco_code, opening_name, opening_depth, opening_rag, opening_rag_src,
              wdl, dtz, endgame_type, endgame_rag, endgame_rag_src,
              weak_squares, weak_action, weak_rag, weak_rag_src,
              phase, current_concepts
        """
        _, history_rich, history_fens = self._build_history_from_board(board)
        fen   = board.fen()
        phase = get_phase(board)

        # ── Opening identification ────────────────────────────────────────────
        eco_code     = None
        opening_name = ''
        opening_depth = 0
        opening_rag  = ''
        opening_rag_src = ''

        opening_moves_san:    list[str] = []
        opening_continuations: list[str] = []

        if self._retriever and len(history_fens) > 1:
            try:
                opening = self._retriever.identify_opening(history_fens)
                if opening:
                    eco_code      = opening.get('eco', '')
                    base_name     = opening.get('opening', '')
                    variation     = opening.get('variation', '')
                    opening_name  = f"{base_name}, {variation}" if variation else base_name
                    if eco_code:
                        opening_name = f"{eco_code} {opening_name}".strip()
                    matched_ply   = opening.get('depth', 0)
                    opening_depth = (matched_ply // 2) + 1
                    opening_rag, opening_rag_src = self._retrieve_one(
                        fen, history_fens, [], eco_code
                    )

                    # Build SAN list for moves played that are part of this opening
                    _b = chess.Board()
                    for i, mv in enumerate(board.move_stack):
                        if i >= matched_ply:
                            break
                        try:
                            opening_moves_san.append(_b.san(mv))
                            _b.push(mv)
                        except Exception:
                            break

                    # Theory continuations: legal moves from deepest match that extend ECO
                    if matched_ply < len(history_fens):
                        deepest_fen = history_fens[matched_ply]
                        try:
                            from chess_coach.rag.retriever import _norm_fen as _rnf
                            eco_prefix = eco_code[:2] if eco_code else ''
                            _b2 = chess.Board(deepest_fen)
                            for _mv in list(_b2.legal_moves):
                                _san = _b2.san(_mv)
                                _b2.push(_mv)
                                _entry = self._retriever._eco_db.get(_rnf(_b2.fen()))
                                if (_entry and
                                        _entry.get('eco', '').startswith(eco_prefix) and
                                        _entry.get('depth', 0) > matched_ply):
                                    opening_continuations.append(_san)
                                _b2.pop()
                                if len(opening_continuations) >= 4:
                                    break
                        except Exception:
                            pass
            except Exception:
                pass

        # ── Tablebase probe ───────────────────────────────────────────────────
        wdl, dtz, endgame_type, endgame_rag, endgame_rag_src = None, None, '', '', ''
        if len(board.piece_map()) <= 7:
            wdl, dtz = self._probe_tablebase(board)

        # ── Current position concept classification ───────────────────────────
        concepts = self._model.predict_concepts(
            fen, history_rich=history_rich or None, eco_code=eco_code
        )

        # ── Endgame type from Tier 4 concepts (drives RAG retrieval key) ─────
        if wdl is not None:
            tier4 = [(n, p) for n, p in concepts if n in TIER4_CONCEPTS]
            if tier4:
                endgame_type = tier4[0][0]
                endgame_rag, endgame_rag_src = self._retrieve_one(
                    fen, history_fens, [endgame_type], eco_code
                )

        # ── Weak squares ─────────────────────────────────────────────────────
        concept_map      = dict(concepts)
        weak_squares     = []
        weak_squares_opp = []
        weak_action      = ''
        weak_rag         = ''
        weak_rag_src     = ''

        # Weak square overlay — computed directly from the board, not gated on
        # the concept classifier. Uses the same _outpost_squares_bb logic used
        # to generate training labels, so the definition is perfectly consistent.
        # Only EMPTY squares are shown (no point flagging squares we already occupy).
        try:
            from tools.label_positions import _outpost_squares_bb
            _player_color = chess.WHITE if player_side == 'white' else chess.BLACK
            _opp_color    = not _player_color
            _my_weak_bb   = _outpost_squares_bb(board, _opp_color)   & ~board.occupied
            _opp_weak_bb  = _outpost_squares_bb(board, _player_color) & ~board.occupied
            weak_squares     = [chess.square_name(sq)
                                for sq in chess.scan_forward(_my_weak_bb)][:8]
            weak_squares_opp = [chess.square_name(sq)
                                for sq in chess.scan_forward(_opp_weak_bb)][:8]
        except Exception:
            pass

        # Bubble text — still gated on classifier so it only appears when the
        # classifier has detected a significant structural weakness
        if 'weak_square' in concept_map:
            weak_action = _ACTION_HINTS.get('weak_square', '')
            weak_rag, weak_rag_src = self._retrieve_one(
                fen, history_fens, ['weak_square'], eco_code
            )

        return {
            'eco_code':              eco_code,
            'opening_name':          opening_name,
            'opening_depth':         opening_depth,
            'opening_moves_san':     opening_moves_san,
            'opening_continuations': opening_continuations,
            'opening_rag':           opening_rag,
            'opening_rag_src':       opening_rag_src,
            'wdl':                   wdl,
            'dtz':                   dtz,
            'endgame_type':          endgame_type,
            'endgame_rag':           endgame_rag,
            'endgame_rag_src':       endgame_rag_src,
            'weak_squares':          weak_squares,
            'weak_squares_opp':      weak_squares_opp,
            'weak_action':           weak_action,
            'weak_rag':              weak_rag,
            'weak_rag_src':          weak_rag_src,
            'phase':                 phase,
            'current_concepts':      concepts,
        }

    # ── Public API — per-line analysis ────────────────────────────────────────

    def analyse_line(
        self,
        board:        chess.Board,
        pv_uci:       list[str],
        score_cp:     int | None,
        eco_code:     str | None = None,
        player_side:  str = 'white',
    ) -> dict:
        """Analyse one Stockfish PV line.

        Pushes pv_uci[0] only (1 move deep) and classifies the resulting position.
        Returns a dict consumed by the panel's line_ready signal handler.
        Keys: theme_badge, confidence, score_cp, metrics_rows,
              plan_sentences, rag_annotation, rag_src, phase, post_move_fen
        """
        _, history_rich, history_fens = self._build_history_from_board(board)
        current_fen = board.fen()
        phase       = get_phase(board)

        # Push first PV move to get post-move position
        b_post        = board.copy()
        post_move_fen = current_fen
        if pv_uci:
            try:
                move = chess.Move.from_uci(pv_uci[0])
                if move in b_post.legal_moves:
                    b_post.push(move)
                    post_move_fen = b_post.fen()
            except Exception:
                pass

        # Classify both positions
        current_concepts = self._model.predict_concepts(
            current_fen, history_rich=history_rich or None, eco_code=eco_code
        )
        post_concepts = self._model.predict_concepts(
            post_move_fen, history_rich=history_rich or None, eco_code=eco_code
        )

        # Theme badge from post-move Tier 1
        primary, _, confidence, _ = infer_strategy(post_concepts)
        theme_badge = primary if primary != 'general' else 'positional'

        # Working concept list — fall back to threshold=0.25 in early positions
        # where calibrated thresholds rarely fire
        _working_concepts = post_concepts
        if not any(n in TIER1_CONCEPTS or n in TIER2_CONCEPTS for n, _ in post_concepts):
            _working_concepts = self._model.predict_concepts(
                post_move_fen, history_rich=history_rich or None,
                eco_code=eco_code, threshold=0.25,
            )

        # Board-specific sentences from NimzoNet concepts + actual piece/square positions
        from chess_coach.coach.board_sentences import build_board_sentences
        plan_sentences = build_board_sentences(_working_concepts, b_post, player_side, phase)

        # Fallback to action hints when no board-specific sentence could be produced
        if not plan_sentences:
            plan_sentences = [_ACTION_HINTS[n] for n, _ in _working_concepts
                              if n in _ACTION_HINTS][:3]
        if not plan_sentences:
            plan_sentences = [{
                'opening':    'Develop pieces toward the centre and prepare to castle.',
                'middlegame': 'Improve your worst-placed piece and create a concrete threat.',
                'endgame':    'Activate the king — in the endgame an active king is decisive.',
            }.get(phase, 'Seek an improvement in your position.')]

        # SF eval Before/After/Δ table
        # "Before" = current position; "After" = full PV line pushed
        b_full = board.copy()
        for uci in pv_uci:
            try:
                mv = chess.Move.from_uci(uci)
                if mv in b_full.legal_moves:
                    b_full.push(mv)
                else:
                    break
            except Exception:
                break
        full_line_fen = b_full.fen()

        before_raw = self._get_sf_eval_terms(current_fen)
        after_raw  = self._get_sf_eval_terms(full_line_fen)
        before_dict = {row[0]: row[3] for row in before_raw}   # term → total string
        sf_eval_rows: list[tuple[str, str, str, str]] = []
        for row in after_raw:
            term       = row[0]
            before_str = before_dict.get(term, '0.00')
            after_str  = row[3]
            try:
                delta_val = float(after_str.replace(' ', '')) - float(before_str.replace(' ', ''))
                delta_str = f'{delta_val:+.2f}'
            except (ValueError, AttributeError):
                delta_str = ''
            sf_eval_rows.append((term, before_str, after_str, delta_str))

        # Deep RAG walk-back: start at 10 PV moves, walk back until relevant annotation found
        concept_names = [n for n, _ in (post_concepts or _working_concepts)[:6]]
        rag_annotation, rag_src = self._retrieve_line_rag(
            board, pv_uci, history_fens, concept_names, max_depth=10
        )

        return {
            'theme_badge':    theme_badge,
            'confidence':     confidence,
            'score_cp':       score_cp,
            'sf_eval_rows':   sf_eval_rows,   # list[(term, white, black, total)]
            'plan_sentences': plan_sentences,
            'rag_annotation': rag_annotation,
            'rag_src':        rag_src,
            'phase':          phase,
            'post_move_fen':  post_move_fen,
        }

    # ── Public API — legacy CoachOutput (for Insert Note compatibility) ───────

    def analyse(
        self,
        board:       chess.Board,
        player_side: str = 'white',
    ) -> CoachOutput:
        """Return a CoachOutput for the current position (used by Insert Note)."""
        _, history_rich, _ = self._build_history_from_board(board)
        fen   = board.fen()
        phase = get_phase(board)

        concepts = self._model.predict_concepts(
            fen, history_rich=history_rich or None
        )
        signals  = adapt(concepts, board, phase, player_side)
        primary, secondary, confidence, tie_band = infer_strategy(concepts)

        result = ResolverResult(
            primary    = primary,
            secondary  = secondary,
            confidence = confidence,
            tie_band   = tie_band,
        )

        weakness_squares = list({sq for sig in signals for sq in sig.key_squares})

        return assemble(
            result           = result,
            phase            = phase,
            signals          = signals,
            player_side      = player_side,
            phrase_db        = self._phrase_db,
            gm_precedents    = [],
            move_flags       = [],
            weakness_squares = weakness_squares,
        )

    def analyse_from_pv(
        self,
        board:       chess.Board,
        pv_uci:      list[str],
        player_side: str,
        score_cp:    int | None = None,
    ) -> CoachOutput:
        """Legacy shim — still used by Insert Note fallback path."""
        if pv_uci:
            try:
                b2   = board.copy()
                move = chess.Move.from_uci(pv_uci[0])
                if move in b2.legal_moves:
                    b2.push(move)
                    return self.analyse(b2, player_side)
            except Exception:
                pass
        return self.analyse(board, player_side)

    def close(self) -> None:
        pass
