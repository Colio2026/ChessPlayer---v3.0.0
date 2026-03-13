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

from chess_coach.core.data_types import MetricSignal

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

    # ── BLITZ — expanded ─────────────────────────────────────────────────────
    ('blitz', 'any', 'king_exposure', 'moderate', 'diagnosis', '',
     'The shelter around the {side} king is eroding — the danger is not yet acute, but the signs are unmistakable.',
     'My System Ch.4', 'nimzowitsch', 7),
    ('blitz', 'any', 'king_exposure', 'moderate', 'evidence', '',
     'The king has been left with fewer defenders than the position demands. The vulnerability is real.',
     'My System Ch.4', 'nimzowitsch', 7),
    ('blitz', 'any', 'king_exposure', 'moderate', 'plan', '',
     'Direct every available resource toward the exposed king. The developing attack must not lose its momentum.',
     'My System Ch.5', 'nimzowitsch', 7),
    ('blitz', 'any', 'king_exposure', 'moderate', 'urgency', '',
     'The opponent has one consolidating move available. Do not grant it — press the attack without delay.',
     'My System Ch.5', 'nimzowitsch', 7),
    ('blitz', 'any', 'king_exposure', 'high', 'evidence', '',
     'The pieces are converging on the exposed king from multiple directions. The geometry of attack is unmistakable.',
     'My System Ch.4', 'nimzowitsch', 8),
    ('blitz', 'any', 'king_exposure', 'high', 'urgency', '',
     'High exposure demands high urgency. Every move spent on anything other than the king is a move wasted.',
     'My System Ch.5', 'nimzowitsch', 8),
    ('blitz', 'any', 'king_exposure', 'moderate', 'diagnosis', 'open_file',
     'The {file}-file stands open like a loaded cannon aimed at the {side} king — the barrel must be used.',
     'My System Ch.4', 'nimzowitsch', 8),
    ('blitz', 'any', 'king_exposure', 'moderate', 'plan', 'open_file',
     'Double the rooks on the {file}-file and let the pressure on the exposed king become irresistible.',
     'My System Ch.4', 'nimzowitsch', 8),
    ('blitz', 'any', 'piece_mobility_ratio', 'high', 'evidence', '',
     'Our pieces command far more squares than theirs. This disparity in activity is the true measure of the attack.',
     'My System Ch.8', 'nimzowitsch', 7),
    ('blitz', 'any', 'piece_mobility_ratio', 'high', 'plan', '',
     'Exploit the activity advantage immediately — inactive pieces on the wrong side of the board are wasted ammunition.',
     'My System Ch.8', 'nimzowitsch', 7),
    ('blitz', 'any', 'space_delta_kingside', 'high', 'plan', 'space_advantage',
     'The kingside space must be converted into a direct assault. Advance the pawns and let the pieces follow.',
     'My System Ch.7', 'nimzowitsch', 8),
    ('blitz', 'any', 'space_delta_kingside', 'high', 'urgency', 'space_advantage',
     'Kingside space without attack is rent unpaid. The time to collect the dividend is now.',
     'My System Ch.7', 'nimzowitsch', 8),
    ('blitz', 'any', 'sacrifice_delta', 'any', 'evidence', 'material_sacrifice',
     'The sacrifice is sound. Positional compensation — open lines, exposed king, active pieces — outweighs the material.',
     'My System Ch.6', 'nimzowitsch', 8),
    ('blitz', 'any', 'sacrifice_delta', 'any', 'plan', 'material_sacrifice',
     'The compensation for the sacrificed material lies in the attack. Prosecute it without hesitation.',
     'My System Ch.6', 'nimzowitsch', 8),
    ('blitz', 'any', 'tactic_pin', 'any', 'diagnosis', 'pin',
     'The pin on {square} immobilises a key defender — the entire attack is built upon this single tactical fact.',
     'My System Ch.5', 'nimzowitsch', 8),
    ('blitz', 'opening', 'king_exposure', 'high', 'diagnosis', '',
     'The opponent has neglected development and left the king in the centre. Such neglect carries its own punishment.',
     'My System Ch.1', 'nimzowitsch', 8),
    ('blitz', 'opening', 'king_exposure', 'high', 'plan', '',
     'Open the centre immediately. A king in the centre when the position is open is a king in mortal danger.',
     'My System Ch.1', 'nimzowitsch', 9),
    ('blitz', 'endgame', 'king_exposure', 'moderate', 'diagnosis', '',
     'In the endgame even a modest king exposure can prove fatal — the king cannot hide behind material.',
     'My System Ch.4', 'nimzowitsch', 7),

    # ── FLANK — expanded ─────────────────────────────────────────────────────
    ('flank', 'any', 'space_delta_queenside', 'moderate', 'diagnosis', '',
     'A modest queenside space advantage is establishing itself. The seeds of the squeeze have been planted.',
     'My System Ch.7', 'nimzowitsch', 7),
    ('flank', 'any', 'space_delta_queenside', 'moderate', 'evidence', '',
     'The opponent\'s queenside pieces lack scope. The space advantage, though modest, is a real constraint.',
     'My System Ch.7', 'nimzowitsch', 7),
    ('flank', 'any', 'space_delta_queenside', 'moderate', 'plan', '',
     'Expand the queenside grip methodically. Every pawn advance that gains space without weakening is a gain.',
     'My System Ch.7', 'nimzowitsch', 7),
    ('flank', 'any', 'space_delta_queenside', 'moderate', 'urgency', 'space_trend',
     'The grip is growing but not yet decisive. Act before the opponent finds the freeing pawn break.',
     'My System Ch.7', 'nimzowitsch', 7),
    ('flank', 'any', 'space_delta_kingside', 'high', 'evidence', 'space_advantage',
     'The kingside is ours to command. Every square beyond the fourth rank belongs to our pieces alone.',
     'My System Ch.7', 'nimzowitsch', 8),
    ('flank', 'any', 'space_delta_kingside', 'high', 'plan', 'space_advantage',
     'Use the kingside space to manoeuvre pieces to their optimal squares. The squeeze will follow naturally.',
     'My System Ch.7', 'nimzowitsch', 8),
    ('flank', 'any', 'space_delta_kingside', 'moderate', 'evidence', '',
     'A modest kingside space edge exists — the opponent\'s pieces are slightly cramped on that wing.',
     'My System Ch.7', 'nimzowitsch', 7),
    ('flank', 'any', 'space_delta_kingside', 'moderate', 'plan', '',
     'Press the modest kingside advantage with piece activity, not pawn advances. Preserve the structure.',
     'My System Ch.7', 'nimzowitsch', 7),
    ('flank', 'any', 'pawn_fixedness', 'high', 'diagnosis', 'fixed_structure',
     'The fixed pawn structure has handed us a permanent positional map. The plan is written in the pawns.',
     'My System Ch.9', 'nimzowitsch', 8),
    ('flank', 'any', 'pawn_fixedness', 'high', 'plan', 'fixed_structure',
     'In a fixed structure the winning method is the gradual squeeze — restrict, regroup, tighten, and then strike.',
     'My System Ch.9', 'nimzowitsch', 8),
    ('flank', 'any', 'piece_mobility_ratio', 'moderate', 'diagnosis', '',
     'A modest mobility edge has emerged. Our pieces move more freely — this is the beginning of the squeeze.',
     'My System Ch.8', 'nimzowitsch', 7),
    ('flank', 'any', 'piece_mobility_ratio', 'moderate', 'plan', '',
     'Convert the modest mobility advantage into concrete pressure. Improve the worst-placed piece first.',
     'My System Ch.8', 'nimzowitsch', 7),
    ('flank', 'any', 'bad_piece', 'any', 'urgency', 'bad_bishop',
     'The bad bishop has suffered long enough. The moment to exploit its immobility has arrived — act now.',
     'My System Ch.2', 'nimzowitsch', 8),
    ('flank', 'any', 'bad_piece', 'any', 'urgency', 'bad_piece',
     'A restricted piece is a liability that grows with time. Press against it now before the opponent reorganises.',
     'My System Ch.2', 'nimzowitsch', 8),
    ('flank', 'any', 'outpost_occupation', 'any', 'evidence', 'outpost',
     'The piece on {square} controls the position like a lighthouse — every enemy plan must navigate around it.',
     'My System Ch.3', 'nimzowitsch', 8),
    ('flank', 'any', 'outpost_occupation', 'any', 'urgency', 'outpost',
     'The outpost at {square} is available now. Occupy it before the opponent closes the square with a pawn.',
     'My System Ch.3', 'nimzowitsch', 8),
    ('flank', 'any', 'passed_pawn', 'any', 'plan', 'advanced_passed_pawn',
     'The flank strategy finds its ultimate expression in the passed pawn. Create it and let it lead the endgame.',
     'My System Ch.2', 'nimzowitsch', 7),
    ('flank', 'endgame', 'space_delta_queenside', 'high', 'diagnosis', 'space_advantage',
     'In the endgame the queenside space advantage becomes a passed pawn. The conversion must begin now.',
     'My System Ch.7', 'nimzowitsch', 8),
    ('flank', 'endgame', 'piece_mobility_ratio', 'high', 'plan', 'mobility_advantage',
     'Active pieces in the endgame are more valuable than in any other phase. Every tempo is decisive.',
     'My System Ch.8', 'nimzowitsch', 8),

    # ── FORTRESS — expanded ───────────────────────────────────────────────────
    ('fortress', 'any', 'pawn_fixedness', 'moderate', 'diagnosis', 'fixed_structure',
     'The position is becoming locked. The fortress takes shape — passive resistance is the order of the day.',
     'My System Ch.9', 'nimzowitsch', 7),
    ('fortress', 'any', 'pawn_fixedness', 'moderate', 'evidence', 'fixed_structure',
     'The fixed pawns confirm the defensive wall is viable. The structure supports a prolonged resistance.',
     'My System Ch.9', 'nimzowitsch', 7),
    ('fortress', 'any', 'pawn_fixedness', 'moderate', 'plan', 'fixed_structure',
     'Use the moderate lockdown to stabilise the position. Place pieces on their optimal defensive squares.',
     'My System Ch.9', 'nimzowitsch', 7),
    ('fortress', 'any', 'pawn_fixedness', 'moderate', 'urgency', 'fixed_structure',
     'Fix the remaining mobile pawns now. Every pawn that advances becomes a target; every fixed pawn is a wall.',
     'My System Ch.9', 'nimzowitsch', 7),
    ('fortress', 'any', 'eval_deficit', 'moderate', 'diagnosis', 'positional_deficit',
     'The position is modestly worse. This is not catastrophe — it is a challenge that clear-eyed defence can meet.',
     'My System Ch.10', 'nimzowitsch', 7),
    ('fortress', 'any', 'eval_deficit', 'moderate', 'evidence', 'positional_deficit',
     'The evaluation confirms a modest but real inferiority. The task is containment, not counterattack.',
     'My System Ch.10', 'nimzowitsch', 7),
    ('fortress', 'any', 'eval_deficit', 'moderate', 'plan', 'positional_deficit',
     'Identify the weakest point in the defensive line and reinforce it. A fortress held at one point is held everywhere.',
     'My System Ch.10', 'nimzowitsch', 7),
    ('fortress', 'any', 'eval_deficit', 'moderate', 'urgency', 'positional_deficit',
     'Contain the modest deficit before it compounds. One passive move becomes two; two become a lost position.',
     'My System Ch.10', 'nimzowitsch', 7),
    ('fortress', 'any', 'king_exposure', 'moderate', 'evidence', '',
     'The king is not adequately sheltered. Every undefended pawn near it is a crack in the fortress wall.',
     'My System Ch.9', 'nimzowitsch', 7),
    ('fortress', 'any', 'king_exposure', 'moderate', 'plan', '',
     'Shore up the king\'s position before all else. A fortress with an exposed king is a fortress in name only.',
     'My System Ch.9', 'nimzowitsch', 8),
    ('fortress', 'any', 'weak_pawns', 'high', 'diagnosis', 'isolated_pawn',
     'The opponent\'s isolated pawn at {square} is the fortress\'s chosen target — a weakness that cannot be healed.',
     'My System Ch.1', 'nimzowitsch', 8),
    ('fortress', 'any', 'weak_pawns', 'high', 'plan', 'isolated_pawn',
     'Plant a blockading piece in front of the isolated pawn and let it wither. Do not trade it — make it suffer.',
     'My System Ch.1', 'nimzowitsch', 9),
    ('fortress', 'any', 'piece_mobility_ratio', 'moderate', 'diagnosis', '',
     'Our pieces are restricted — this is not a failure. In the fortress strategy, restriction is the design.',
     'My System Ch.9', 'nimzowitsch', 7),
    ('fortress', 'any', 'passed_pawn', 'high', 'urgency', 'advanced_passed_pawn',
     'The enemy passed pawn at {square} is the single greatest threat to the fortress. It must be blockaded immediately.',
     'My System Ch.2', 'nimzowitsch', 9),
    ('fortress', 'endgame', 'eval_deficit', 'high', 'plan', 'positional_deficit',
     'In the endgame a fortress is held by technique, not hope. Place king and rook on the ideal defensive squares.',
     'My System Ch.10', 'nimzowitsch', 8),
    ('fortress', 'endgame', 'pawn_fixedness', 'high', 'plan', 'fixed_structure',
     'A fixed pawn structure in the endgame is a draw certificate — if the blockading piece cannot be dislodged.',
     'My System Ch.9', 'nimzowitsch', 8),

    # ── FEINT — expanded ──────────────────────────────────────────────────────
    ('feint', 'any', 'outpost_occupation', 'any', 'evidence', 'latent_threat',
     'The piece on the outpost conceals a threat the opponent has not yet perceived. This ignorance is our advantage.',
     'My System Ch.11', 'nimzowitsch', 8),
    ('feint', 'any', 'outpost_occupation', 'any', 'plan', 'latent_threat',
     'Use the outpost as the pivot of the misdirection. Let the opponent commit to the wrong wing first.',
     'My System Ch.11', 'nimzowitsch', 8),
    ('feint', 'any', 'outpost_occupation', 'any', 'urgency', 'latent_threat',
     'Occupy {square} now, before the opponent perceives the plan. Once the outpost is ours the feint can begin.',
     'My System Ch.11', 'nimzowitsch', 8),
    ('feint', 'any', 'space_delta_queenside', 'any', 'diagnosis', '',
     'The queenside activity is the decoy. The opponent\'s attention is drawn there — the real blow falls elsewhere.',
     'My System Ch.11', 'nimzowitsch', 8),
    ('feint', 'any', 'space_delta_queenside', 'any', 'plan', '',
     'Maintain the queenside pressure as a threat-in-being. When the opponent overcommits there, switch wings.',
     'My System Ch.11', 'nimzowitsch', 8),
    ('feint', 'any', 'pawn_fixedness', 'any', 'diagnosis', 'positional_tension',
     'The fixed structure is not equilibrium — it is tension held in reserve, awaiting the correct moment of release.',
     'My System Ch.11', 'nimzowitsch', 8),
    ('feint', 'any', 'pawn_fixedness', 'any', 'urgency', 'positional_tension',
     'Release the accumulated tension only at the moment of maximum effect. Premature release wastes everything.',
     'My System Ch.11', 'nimzowitsch', 9),
    ('feint', 'any', 'piece_mobility_ratio', 'any', 'plan', '',
     'The quiet position is the feint\'s natural habitat. Use the mobility edge to complete the preparation unseen.',
     'My System Ch.11', 'nimzowitsch', 7),
    ('feint', 'any', 'piece_mobility_ratio', 'any', 'urgency', '',
     'Every piece is poised. The preparation is complete. The moment to act has arrived — do not hesitate further.',
     'My System Ch.11', 'nimzowitsch', 8),
    ('feint', 'any', 'king_exposure', 'moderate', 'diagnosis', '',
     'The king appears safely sheltered — and that appearance is the feint\'s greatest weapon.',
     'My System Ch.11', 'nimzowitsch', 8),
    ('feint', 'middlegame', 'pawn_fixedness', 'any', 'evidence', 'positional_tension',
     'The tension in the pawn structure has been maintained deliberately. Nothing here is accidental.',
     'My System Ch.11', 'nimzowitsch', 7),

    # ── GENERAL — expanded ────────────────────────────────────────────────────
    ('general', 'any', 'passed_pawn', 'critical', 'evidence', 'advanced_passed_pawn',
     'The passed pawn at {square} is the decisive factor. Every other consideration is secondary to its advance.',
     'My System Ch.2', 'nimzowitsch', 9),
    ('general', 'any', 'passed_pawn', 'high', 'diagnosis', 'advanced_passed_pawn',
     'A powerful passed pawn stands at {square}. It is a potential queen — treat it with corresponding respect.',
     'My System Ch.2', 'nimzowitsch', 8),
    ('general', 'any', 'passed_pawn', 'high', 'plan', 'advanced_passed_pawn',
     'Advance the passed pawn with the support of king and rook. The rook belongs behind it, pushing from the rear.',
     'My System Ch.2', 'nimzowitsch', 8),
    ('general', 'endgame', 'passed_pawn', 'critical', 'plan', 'advanced_passed_pawn',
     'The king must march to escort the passed pawn. In the endgame the king is not a liability but a weapon.',
     'My System Ch.2', 'nimzowitsch', 9),
    ('general', 'endgame', 'passed_pawn', 'high', 'urgency', 'advanced_passed_pawn',
     'Every tempo in a passed pawn ending is worth a pawn. Advance now — the opponent has no time to build a blockade.',
     'My System Ch.2', 'nimzowitsch', 8),
    ('general', 'any', 'weak_pawns', 'high', 'evidence', 'isolated_pawn',
     'The isolated pawn at {square} can be pressured from in front, behind, and from both flanks. It is indefensible.',
     'My System Ch.1', 'nimzowitsch', 8),
    ('general', 'any', 'weak_pawns', 'high', 'urgency', 'isolated_pawn',
     'The isolated pawn will not defend itself. Attack it now while every piece can participate in the assault.',
     'My System Ch.1', 'nimzowitsch', 8),
    ('general', 'any', 'weak_pawns', 'moderate', 'plan', 'doubled_pawn',
     'The doubled pawns on the {file}-file are a structural fault. Exploit them methodically — they cannot run.',
     'My System Ch.1', 'nimzowitsch', 7),
    ('general', 'any', 'weak_pawns', 'moderate', 'urgency', 'doubled_pawn',
     'Press against the doubled pawns before the opponent untangles the formation. Structural faults are time-sensitive.',
     'My System Ch.1', 'nimzowitsch', 7),
    ('general', 'any', 'weak_pawns', 'moderate', 'evidence', 'doubled_pawn',
     'The doubled pawns on the {file}-file rob the opponent of a connected majority — a permanent positional liability.',
     'My System Ch.1', 'nimzowitsch', 7),
    ('general', 'any', 'piece_mobility_ratio', 'moderate', 'diagnosis', '',
     'A modest mobility advantage has emerged. The better-placed pieces are not a decoration — they are a plan.',
     'My System Ch.8', 'nimzowitsch', 6),
    ('general', 'any', 'piece_mobility_ratio', 'moderate', 'evidence', '',
     'The move count tells the story — our pieces command more squares. This is the first measure of positional advantage.',
     'My System Ch.8', 'nimzowitsch', 6),
    ('general', 'any', 'pawn_fixedness', 'moderate', 'diagnosis', 'fixed_structure',
     'The pawn structure is becoming defined. Fixed pawns dictate fixed plans — read the structure and follow it.',
     'My System Ch.1', 'nimzowitsch', 6),
    ('general', 'any', 'pawn_fixedness', 'moderate', 'plan', 'fixed_structure',
     'In a moderately fixed position the correct plan emerges from the pawn structure itself. Do not impose — deduce.',
     'My System Ch.1', 'nimzowitsch', 6),
    ('general', 'any', 'king_exposure', 'high', 'diagnosis', '',
     'The king stands dangerously exposed. Whatever the strategy, this must be addressed before all other considerations.',
     'My System Ch.4', 'nimzowitsch', 8),
    ('general', 'any', 'king_exposure', 'high', 'evidence', '',
     'Multiple structural indicators confirm that the king is insufficiently sheltered. The position demands action.',
     'My System Ch.4', 'nimzowitsch', 8),
    ('general', 'any', 'outpost_occupation', 'any', 'diagnosis', 'outpost',
     'The square at {square} is an outpost in the classical sense — no enemy pawn can ever challenge a piece placed there.',
     'My System Ch.3', 'nimzowitsch', 8),
    ('general', 'any', 'outpost_occupation', 'any', 'plan', 'outpost',
     'A piece on {square} cannot be driven away. Place it there and build the entire plan around its permanence.',
     'My System Ch.3', 'nimzowitsch', 8),
    ('general', 'any', 'overextension', 'any', 'diagnosis', 'opponent_pawn_overextended',
     'The pawn at {square} has advanced without adequate support. Overextension is an invitation — accept it.',
     'My System Ch.9', 'nimzowitsch', 8),
    ('general', 'any', 'overextension', 'any', 'plan', 'opponent_pawn_overextended',
     'Attack the base of the overextended chain. A pawn that cannot be supported by another pawn must fall.',
     'My System Ch.9', 'nimzowitsch', 8),
    ('general', 'any', 'overextension', 'any', 'urgency', 'opponent_pawn_overextended',
     'The overextended pawn at {square} is vulnerable now. If the opponent consolidates it ceases to be a weakness.',
     'My System Ch.9', 'nimzowitsch', 8),
    ('general', 'opening', 'king_exposure', 'critical', 'diagnosis', '',
     'The king has been left in the centre in an open position. This is not a positional deficiency — it is a crisis.',
     'My System Ch.1', 'nimzowitsch', 9),
    ('general', 'opening', 'king_exposure', 'critical', 'plan', '',
     'Open the centre at once. A king in the centre of an open game is a target that cannot long survive.',
     'My System Ch.1', 'nimzowitsch', 9),
    ('general', 'endgame', 'weak_pawns', 'high', 'urgency', 'isolated_pawn',
     'In the endgame the isolated pawn is not merely weak — it is lost. Attack it with king and rook together.',
     'My System Ch.1', 'nimzowitsch', 9),
    ('general', 'endgame', 'passed_pawn', 'moderate', 'plan', 'advanced_passed_pawn',
     'Even a modest passed pawn in the endgame demands respect. Centralise the king and escort it forward.',
     'My System Ch.2', 'nimzowitsch', 7),

    # ── HEADLINE fragments ────────────────────────────────────────────────────
    ('blitz', 'any', 'king_exposure', 'critical', 'headline', '',
     'A direct assault on the {side} king — the position demands immediate and decisive action.',
     'My System Ch.5', 'nimzowitsch', 9),
    ('blitz', 'any', 'king_exposure', 'high', 'headline', '',
     'The {side} king stands exposed — the attacking pieces must be mobilised at once.',
     'My System Ch.4', 'nimzowitsch', 8),
    ('blitz', 'any', 'sacrifice_delta', 'any', 'headline', '',
     'Material has been invested in the attack — the position demands that the attack be pressed to a conclusion.',
     'My System Ch.6', 'nimzowitsch', 8),
    ('flank', 'any', 'space_delta_queenside', 'high', 'headline', '',
     'A queenside squeeze is in operation — the opponent\'s pieces are cramped and without prospect.',
     'My System Ch.7', 'nimzowitsch', 8),
    ('flank', 'any', 'outpost_occupation', 'any', 'headline', '',
     'The outpost at {square} is the positional cornerstone — all play revolves around its occupation.',
     'My System Ch.3', 'nimzowitsch', 8),
    ('flank', 'any', 'piece_mobility_ratio', 'high', 'headline', '',
     'A positional squeeze is taking hold — the opponent\'s pieces are running out of room.',
     'My System Ch.8', 'nimzowitsch', 7),
    ('fortress', 'any', 'eval_deficit', 'high', 'headline', '',
     'The position is worse — the task is to construct an impenetrable defensive wall.',
     'My System Ch.10', 'nimzowitsch', 8),
    ('fortress', 'any', 'pawn_fixedness', 'high', 'headline', '',
     'The fixed structure is the fortress foundation — passive resistance is the correct strategy.',
     'My System Ch.9', 'nimzowitsch', 8),
    ('feint', 'any', 'pawn_fixedness', 'any', 'headline', '',
     'Tension is being held deliberately — the opponent does not yet understand which wing will be struck.',
     'My System Ch.11', 'nimzowitsch', 8),
    ('feint', 'any', 'outpost_occupation', 'any', 'headline', '',
     'A quiet preparation conceals a decisive plan — patience is the most lethal weapon.',
     'My System Ch.11', 'nimzowitsch', 8),
    ('general', 'any', 'passed_pawn', 'critical', 'headline', '',
     'The passed pawn at {square} is the dominant factor — all other considerations are secondary.',
     'My System Ch.2', 'nimzowitsch', 9),
    ('general', 'any', 'king_exposure', 'high', 'headline', '',
     'King safety is the paramount concern — the position must be handled with the utmost care.',
     'My System Ch.4', 'nimzowitsch', 8),
    ('general', 'any', 'outpost_occupation', 'any', 'headline', '',
     'The outpost at {square} defines the character of the position — occupy it and the plan becomes clear.',
     'My System Ch.3', 'nimzowitsch', 7),
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
    ('zwischenzug', 'any', 'any',
     'Before recapturing, an in-between move changes the calculation entirely — the opponent must answer a new threat first.',
     'blitz', 'original'),
    ('deflection', 'any', 'any',
     'The defender of {square} can be driven from its post — once deflected, the target becomes undefended.',
     'any', 'original'),
    ('pin', 'middlegame', 'strong',
     'The {piece} on {square} is pinned to the {target} — a paralysed defender that cannot fulfil its duty.',
     'blitz', 'My System Ch.5'),
    ('fork', 'middlegame', 'strong',
     'The fork on {square} attacks two pieces simultaneously — one of them must be abandoned.',
     'any', 'original'),
    ('skewer', 'any', 'strong',
     'The skewer through {square} forces the {piece} to step aside, losing what stands behind.',
     'blitz', 'original'),
    ('discovery', 'middlegame', 'strong',
     'The discovered attack from behind the {piece} cannot be met — the opponent faces two threats at once.',
     'blitz', 'original'),
    ('deflection', 'endgame', 'any',
     'In the endgame a single deflection can decide everything. Drive the defender away from {square}.',
     'fortress', 'original'),
    ('blockade', 'endgame', 'any',
     'The blockading piece on {square} in the endgame stops a future queen — it is worth more than its material value.',
     'fortress', 'My System Ch.9'),
    ('zwischenzug', 'middlegame', 'any',
     'The in-between move disrupts the expected sequence — the opponent is unprepared for the threat they had discounted.',
     'feint', 'original'),
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
