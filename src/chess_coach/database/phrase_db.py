"""
database/phrase_db.py
======================
Queries the phrases and tactics tables in chess_coach.db.
Fills MetricSignal.fragment fields before the narrator runs.

The phrase database IS the language model — deterministic, auditable,
editable. No AI calls at runtime.

DB file: chess_coach.db (separate from index.sqlite — phrase content
is not tied to the game index).

Tables
------
    phrases  — positional coaching phrases keyed by strategy/metric/severity
    tactics  — tactical pattern descriptions keyed by tactic_type

Fragment assembly order (spec Section 7):
    Slot 1: diagnosis  — what is wrong
    Slot 2: evidence   — why we know
    Slot 3: plan       — what to do
    Slot 4: urgency    — why now
    Slot 5: tactic_hint (separate table, 0-3 per output)

Placeholders in phrase_text are filled from MetricSignal data:
    {file}   → key_squares[0][0] if available
    {square} → key_squares[0]
    {piece}  → key_pieces[0]
    {side}   → 'White' | 'Black'
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from core.data_types import MetricSignal

_FRAGMENT_ORDER = ('diagnosis', 'evidence', 'plan', 'urgency')

_CREATE_PHRASES = """
CREATE TABLE IF NOT EXISTS phrases (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy      TEXT NOT NULL,
    phase         TEXT NOT NULL DEFAULT 'any',
    metric        TEXT NOT NULL,
    severity      TEXT NOT NULL DEFAULT 'any',
    fragment_type TEXT NOT NULL,
    cause_tag     TEXT NOT NULL DEFAULT '',
    phrase_text   TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'original',
    voice         TEXT NOT NULL DEFAULT 'nimzowitsch',
    priority      INTEGER NOT NULL DEFAULT 5
);
CREATE INDEX IF NOT EXISTS idx_phrases_lookup
    ON phrases (strategy, metric, fragment_type, severity);
"""

_CREATE_TACTICS = """
CREATE TABLE IF NOT EXISTS tactics (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    tactic_type    TEXT NOT NULL,
    phase          TEXT NOT NULL DEFAULT 'any',
    severity       TEXT NOT NULL DEFAULT 'any',
    phrase_text    TEXT NOT NULL,
    strategic_link TEXT NOT NULL DEFAULT 'any',
    source         TEXT NOT NULL DEFAULT 'original'
);
CREATE INDEX IF NOT EXISTS idx_tactics_type ON tactics (tactic_type);
"""

# ── Seed data — ~50 phrases drawn from Nimzowitsch's My System ───────────────
# Format: (strategy, phase, metric, severity, fragment_type, cause_tag,
#          phrase_text, source, voice, priority)
_SEED_PHRASES = [
    # ── BLITZ / KING SAFETY ──────────────────────────────────────────────────
    ('blitz', 'middlegame', 'king_exposure', 'critical', 'diagnosis', 'missing_pawn_shield',
     'The pawn cover before the {side} king has been shattered — the {square} square gapes like an open wound.',
     'My System Ch.4', 'nimzowitsch', 9),
    ('blitz', 'middlegame', 'king_exposure', 'critical', 'evidence', 'open_file',
     'The {file}-file stands open as a highway for the rook — the king can find no shelter.',
     'My System Ch.4', 'nimzowitsch', 9),
    ('blitz', 'middlegame', 'king_exposure', 'critical', 'plan', 'attacker_concentration',
     'Concentrate every available piece upon the breach. The attack must be relentless — hesitation is capitulation.',
     'My System Ch.5', 'nimzowitsch', 8),
    ('blitz', 'middlegame', 'king_exposure', 'critical', 'urgency', 'missing_pawn_shield',
     'The window is open now. One consolidating move from {side} seals it forever — strike before it closes.',
     'My System Ch.5', 'nimzowitsch', 9),
    ('blitz', 'any', 'king_exposure', 'high', 'diagnosis', 'missing_pawn_shield',
     'The king stands insufficiently defended — the shelter at {square} has been compromised.',
     'My System Ch.4', 'nimzowitsch', 7),
    ('blitz', 'any', 'king_exposure', 'high', 'plan', 'attacker_concentration',
     'Pile up the attackers on the weakened zone. Two pieces on a target are good; three are decisive.',
     'My System Ch.4', 'nimzowitsch', 7),
    ('blitz', 'any', 'sacrifice_delta', 'any', 'diagnosis', 'material_sacrifice',
     'Material has been invested as a down payment on the attack. The dividend must be collected now.',
     'My System Ch.6', 'nimzowitsch', 8),
    ('blitz', 'any', 'sacrifice_delta', 'any', 'urgency', 'material_sacrifice',
     'A sacrifice demands repayment. Do not allow the opponent time to consolidate what they have received.',
     'My System Ch.6', 'nimzowitsch', 8),

    # ── FLANK / SQUEEZE ──────────────────────────────────────────────────────
    ('flank', 'middlegame', 'space_delta_queenside', 'high', 'diagnosis', 'space_advantage',
     'The queenside belongs to us. The opponent\'s pieces are cramped and without prospect of relief.',
     'My System Ch.7', 'nimzowitsch', 8),
    ('flank', 'middlegame', 'space_delta_kingside', 'high', 'diagnosis', 'space_advantage',
     'The kingside has been seized. Every square forward of the fifth rank is ours to use and theirs to fear.',
     'My System Ch.7', 'nimzowitsch', 8),
    ('flank', 'any', 'piece_mobility_ratio', 'high', 'evidence', 'mobility_advantage',
     'Count the moves available to each side. Ours are many; theirs are few. This is the squeeze working.',
     'My System Ch.8', 'nimzowitsch', 8),
    ('flank', 'any', 'piece_mobility_ratio', 'high', 'plan', 'mobility_advantage',
     'Reduce the opponent\'s options to zero. When every piece is bound, the position collapses of its own weight.',
     'My System Ch.8', 'nimzowitsch', 9),
    ('flank', 'any', 'outpost_occupation', 'any', 'diagnosis', 'outpost',
     'The outpost at {square} is inviolable — no enemy pawn can disturb the piece planted there.',
     'My System Ch.3', 'nimzowitsch', 9),
    ('flank', 'any', 'outpost_occupation', 'any', 'plan', 'outpost',
     'Occupy the outpost and make it the pivot of all operations. A knight on {square} is worth a rook on an open file.',
     'My System Ch.3', 'nimzowitsch', 9),
    ('flank', 'any', 'bad_piece', 'any', 'evidence', 'bad_bishop',
     'The opponent\'s bishop is a tall pawn — locked behind its own pawns with no prospect of release.',
     'My System Ch.2', 'nimzowitsch', 8),
    ('flank', 'any', 'bad_piece', 'any', 'plan', 'bad_piece',
     'Maintain the pawn structure that entombs the bad piece. Do not trade it away — let it suffer.',
     'My System Ch.2', 'nimzowitsch', 8),
    ('flank', 'middlegame', 'space_delta_queenside', 'moderate', 'urgency', 'space_trend',
     'The queenside grip is tightening with every move. Act on it before the opponent finds the freeing break.',
     'My System Ch.7', 'nimzowitsch', 7),
    ('flank', 'endgame', 'space_delta_queenside', 'any', 'plan', 'space_advantage',
     'In the endgame the space advantage becomes a passed pawn. Convert the grip into a concrete advantage.',
     'My System Ch.7', 'nimzowitsch', 8),

    # ── FORTRESS / BLOCKADE ──────────────────────────────────────────────────
    ('fortress', 'any', 'pawn_fixedness', 'high', 'diagnosis', 'fixed_structure',
     'The position is locked. This is not a weakness — it is a wall. Behind it the king sits secure.',
     'My System Ch.9', 'nimzowitsch', 9),
    ('fortress', 'any', 'pawn_fixedness', 'high', 'plan', 'fixed_structure',
     'Seal every open file. Place a blockading piece on every advanced enemy pawn. Let them starve.',
     'My System Ch.9', 'nimzowitsch', 9),
    ('fortress', 'any', 'eval_deficit', 'high', 'diagnosis', 'positional_deficit',
     'The position is objectively worse. Accept it. The task now is to construct a wall they cannot breach.',
     'My System Ch.10', 'nimzowitsch', 8),
    ('fortress', 'any', 'eval_deficit', 'high', 'evidence', 'positional_deficit',
     'Objectively worse positions have survived for centuries behind an impenetrable blockade.',
     'My System Ch.10', 'nimzowitsch', 7),
    ('fortress', 'any', 'eval_deficit', 'high', 'plan', 'positional_deficit',
     'Find the key square — the one that, if held, renders the opponent\'s entire plan futile. Occupy it forever.',
     'My System Ch.10', 'nimzowitsch', 9),
    ('fortress', 'any', 'eval_deficit', 'critical', 'urgency', 'positional_deficit',
     'One crack in the wall and the fortress falls. Every defensive resource must be committed now.',
     'My System Ch.10', 'nimzowitsch', 9),
    ('fortress', 'any', 'overextension', 'any', 'evidence', 'opponent_pawn_overextended',
     'The opponent\'s pawn at {square} has advanced without support — an overextension ripe for attack.',
     'My System Ch.9', 'nimzowitsch', 8),
    ('fortress', 'any', 'overextension', 'any', 'plan', 'opponent_pawn_overextended',
     'Attack the overextended pawn. A pawn that cannot be supported must fall.',
     'My System Ch.9', 'nimzowitsch', 8),

    # ── FEINT / MISDIRECTION ─────────────────────────────────────────────────
    ('feint', 'any', 'outpost_occupation', 'any', 'diagnosis', 'latent_threat',
     'The quiet move prepares what the opponent does not yet see. Let them commit to the wrong wing.',
     'My System Ch.11', 'nimzowitsch', 8),
    ('feint', 'any', 'pawn_fixedness', 'any', 'plan', 'positional_tension',
     'Hold the tension. Release it only when the moment is perfect. Premature action wastes the preparation.',
     'My System Ch.11', 'nimzowitsch', 9),
    ('feint', 'any', 'piece_mobility_ratio', 'moderate', 'evidence', 'quiet_preparation',
     'The position appears equal to the eye — but one side has prepared a secret weapon in silence.',
     'My System Ch.11', 'nimzowitsch', 7),

    # ── GENERAL / STRUCTURAL ─────────────────────────────────────────────────
    ('general', 'any', 'pawn_fixedness', 'high', 'diagnosis', 'fixed_structure',
     'The pawn chain is the skeleton of the position. Its character — fixed or fluid — determines the plan.',
     'My System Ch.1', 'nimzowitsch', 7),
    ('general', 'any', 'passed_pawn', 'critical', 'diagnosis', 'advanced_passed_pawn',
     'The passed pawn is a criminal who should be put under lock and key. Here it runs free at {square}.',
     'My System Ch.2', 'nimzowitsch', 9),
    ('general', 'any', 'passed_pawn', 'critical', 'plan', 'advanced_passed_pawn',
     'Support the passed pawn from behind. Every piece must contribute to its advance.',
     'My System Ch.2', 'nimzowitsch', 9),
    ('general', 'any', 'passed_pawn', 'high', 'urgency', 'advanced_passed_pawn',
     'The passed pawn becomes more powerful with every move it advances. Do not delay.',
     'My System Ch.2', 'nimzowitsch', 8),
    ('general', 'any', 'weak_pawns', 'high', 'diagnosis', 'isolated_pawn',
     'The isolated pawn at {square} is a chronic weakness — it can be defended but never made strong.',
     'My System Ch.1', 'nimzowitsch', 8),
    ('general', 'any', 'weak_pawns', 'high', 'plan', 'isolated_pawn',
     'Blockade the isolated pawn. Place a piece in front of it and let it wither.',
     'My System Ch.1', 'nimzowitsch', 8),
    ('general', 'any', 'weak_pawns', 'moderate', 'diagnosis', 'doubled_pawn',
     'The doubled pawns on the {file}-file create a permanent structural weakness.',
     'My System Ch.1', 'nimzowitsch', 7),
    ('general', 'opening', 'piece_mobility_ratio', 'high', 'plan', 'mobility_advantage',
     'Development demands activity. Every piece must have a purpose; every tempo must be spent wisely.',
     'My System Ch.1', 'nimzowitsch', 6),

    # ── TACTIC HINTS (strategy=tactic, uses tactic metric names) ────────────
    ('tactic', 'any', 'tactic_pin', 'any', 'tactic_hint', 'pin',
     'The {piece} on {square} is pinned to the {target} — it cannot move without grave consequence.',
     'My System Ch.5', 'nimzowitsch', 8),
    ('tactic', 'any', 'tactic_fork', 'any', 'tactic_hint', 'fork',
     'A fork on {square} attacks two pieces simultaneously — one of them must be lost.',
     'original', 'neutral', 8),
    ('tactic', 'any', 'tactic_skewer', 'any', 'tactic_hint', 'skewer',
     'The skewer forces the {piece} to step aside, exposing the piece behind it.',
     'original', 'neutral', 7),
    ('tactic', 'any', 'tactic_discovery', 'any', 'tactic_hint', 'discovery',
     'Moving the {piece} uncovers a devastating attack from the piece behind it.',
     'original', 'neutral', 7),
]

_SEED_TACTICS = [
    # (tactic_type, phase, severity, phrase_text, strategic_link, source)
    ('pin',       'any', 'any',    'The {piece} on {square} is pinned — exploiting it is the tactical backbone of the attack.',
     'blitz', 'My System Ch.5'),
    ('fork',      'any', 'any',    'The fork on {square} wins material by force — one of the attacked pieces must be surrendered.',
     'any', 'original'),
    ('skewer',    'any', 'any',    'A skewer through {square} forces the higher-value piece to flee, losing what stands behind.',
     'blitz', 'original'),
    ('discovery', 'any', 'any',    'The discovered attack reveals a hidden threat — the opponent must defend two things at once.',
     'blitz', 'original'),
    ('blockade',  'any', 'any',    'The blockading piece on {square} renders the enemy pawn permanently inert.',
     'fortress', 'My System Ch.9'),
]


# ── Public API ────────────────────────────────────────────────────────────────

class PhraseDB:
    """
    Queries phrases and tactics tables to fill coaching fragments.

    Parameters
    ----------
    db_path : str
        Path to chess_coach.db. Created and seeded on first open if absent.
    """

    def __init__(self, db_path: str = '') -> None:
        self.db_path   = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._available = False

        if db_path:
            self._open_or_create(db_path)

    @property
    def is_available(self) -> bool:
        return self._available

    def get_fragments(
        self,
        strategy: str,
        phase: str,
        signals: list[MetricSignal],
        max_per_slot: int = 1,
    ) -> dict[str, list[str]]:
        """
        Return filled phrases for each fragment slot.

        Returns
        -------
        dict with keys: 'diagnosis', 'evidence', 'plan', 'urgency'
        Each value is a list of assembled phrase strings (length 0–max_per_slot).
        """
        if not self._available or not signals:
            return {slot: [] for slot in _FRAGMENT_ORDER}

        result: dict[str, list[str]] = {slot: [] for slot in _FRAGMENT_ORDER}
        used_ids: set[int] = set()

        # Sort signals by score descending — lead with the strongest signal
        ranked = sorted(signals, key=lambda s: s.score, reverse=True)

        for slot in _FRAGMENT_ORDER:
            for sig in ranked:
                if len(result[slot]) >= max_per_slot:
                    break
                phrase, phrase_id = self._query_phrase(
                    strategy, phase, sig, slot, exclude=used_ids
                )
                if phrase:
                    filled = _fill(phrase, sig)
                    result[slot].append(filled)
                    used_ids.add(phrase_id)

        return result

    def get_tactic_hints(
        self,
        tactic_signals: list[MetricSignal],
        phase: str,
        max_hints: int = 3,
    ) -> list[str]:
        """Return filled tactic hint phrases for the given tactic signals."""
        if not self._available or not tactic_signals:
            return []

        hints: list[str] = []
        for sig in sorted(tactic_signals, key=lambda s: s.score, reverse=True)[:max_hints]:
            tactic_type = sig.metric_name.replace('tactic_', '')
            row = self._query_tactic(tactic_type, phase)
            if row:
                hints.append(_fill(row, sig))

        return hints

    def close(self) -> None:
        if self._conn:
            try: self._conn.close()
            except Exception: pass
            self._conn = None
            self._available = False

    # ── Internal ───────────────────────────────────────────────────────────

    def _open_or_create(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_CREATE_PHRASES)
            self._conn.executescript(_CREATE_TACTICS)
            self._conn.commit()
            self._seed_if_empty()
            self._available = True
        except sqlite3.Error:
            self._conn = None

    def _seed_if_empty(self) -> None:
        assert self._conn
        count = self._conn.execute("SELECT COUNT(*) FROM phrases").fetchone()[0]
        if count > 0:
            return

        self._conn.executemany(
            """INSERT INTO phrases
               (strategy, phase, metric, severity, fragment_type,
                cause_tag, phrase_text, source, voice, priority)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            _SEED_PHRASES,
        )
        self._conn.executemany(
            """INSERT INTO tactics
               (tactic_type, phase, severity, phrase_text, strategic_link, source)
               VALUES (?,?,?,?,?,?)""",
            _SEED_TACTICS,
        )
        self._conn.commit()

    def _query_phrase(
        self,
        strategy: str,
        phase: str,
        sig: MetricSignal,
        fragment_type: str,
        exclude: set[int],
    ) -> tuple[Optional[str], int]:
        """Return (phrase_text, id) or (None, -1) if nothing matches."""
        assert self._conn
        exclude_sql = f"AND id NOT IN ({','.join('?' * len(exclude))})" if exclude else ''
        params = [strategy, sig.metric_name, fragment_type, sig.severity,
                  phase, sig.cause, *exclude, sig.metric_name, fragment_type,
                  sig.severity, 'any', sig.cause, *exclude]

        sql = f"""
            SELECT id, phrase_text FROM phrases
            WHERE strategy = ?
              AND metric   = ?
              AND fragment_type = ?
              AND (severity = ? OR severity = 'any')
              AND (phase    = ? OR phase    = 'any')
              AND (cause_tag = '' OR cause_tag = ?)
              {exclude_sql}
            ORDER BY priority DESC
            LIMIT 1
        """
        row = self._conn.execute(sql, [
            strategy, sig.metric_name, fragment_type, sig.severity,
            phase, sig.cause, *exclude
        ]).fetchone()

        if row:
            return row['phrase_text'], row['id']

        # Fallback 1: relax strategy to 'general'
        row = self._conn.execute(sql, [
            'general', sig.metric_name, fragment_type, sig.severity,
            phase, sig.cause, *exclude
        ]).fetchone()

        if row:
            return row['phrase_text'], row['id']

        # Fallback 2: relax severity to 'any' — handles mismatched signal severity
        sql_any_sev = f"""
            SELECT id, phrase_text FROM phrases
            WHERE strategy = ?
              AND metric   = ?
              AND fragment_type = ?
              AND (phase = ? OR phase = 'any')
              AND (cause_tag = '' OR cause_tag = ?)
              {exclude_sql}
            ORDER BY priority DESC
            LIMIT 1
        """
        for strat in (strategy, 'general'):
            row = self._conn.execute(sql_any_sev, [
                strat, sig.metric_name, fragment_type,
                phase, sig.cause, *exclude
            ]).fetchone()
            if row:
                return row['phrase_text'], row['id']

        return None, -1

    def _query_tactic(self, tactic_type: str, phase: str) -> Optional[str]:
        assert self._conn
        row = self._conn.execute(
            """SELECT phrase_text FROM tactics
               WHERE tactic_type = ?
                 AND (phase = ? OR phase = 'any')
               ORDER BY id LIMIT 1""",
            (tactic_type, phase),
        ).fetchone()
        return row['phrase_text'] if row else None


# ── Placeholder filling ───────────────────────────────────────────────────────

def _fill(phrase: str, sig: MetricSignal) -> str:
    """Replace {placeholders} in a phrase with MetricSignal data."""
    sq  = sig.key_squares[0] if sig.key_squares else ''
    pc  = sig.key_pieces[0]  if sig.key_pieces  else ''
    fil = sq[0] if sq else ''
    side = 'White' if sig.side == 'white' else 'Black'
    target = sig.key_squares[1] if len(sig.key_squares) > 1 else ''

    return (phrase
            .replace('{square}', sq)
            .replace('{file}',   fil)
            .replace('{piece}',  pc)
            .replace('{side}',   side)
            .replace('{target}', target))
