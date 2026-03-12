# Phrase Database — Expansion Guide

The phrase database (`chess_coach.db`, table `phrases`) currently holds ~50 seed
phrases. The narrator needs **300+ phrases** to cover all strategy/metric/severity
combinations without falling back to `action_hint` text.

This document defines exactly what is missing, what each phrase must do, and how
to write or import one correctly.

---

## How a Phrase Gets Used

When `StrategyEngine.analyse()` runs, the narrator assembles a 4-sentence coaching
paragraph by filling four slots in order:

| Slot | Fragment Type | Role | Example |
|------|--------------|------|---------|
| 1 | `diagnosis` | What is wrong | *"The pawn cover before the Black king has been shattered."* |
| 2 | `evidence` | Why we know | *"Ng5 and Qd3 are already bearing down on the weakened zone."* |
| 3 | `plan` | What to do | *"The g4 pawn break tears open the h-file for the rook."* |
| 4 | `urgency` | Why right now | *"This window closes the moment Black plays Bf8 — do not delay."* |

Each slot is filled by the highest-priority phrase that matches:
- `strategy` = the firing strategy (blitz / flank / fortress / feint / general)
- `metric` = the MetricSignal's `metric_name`
- `severity` = the MetricSignal's `severity` (critical / high / moderate / mild)
- `phase` = the game phase (opening / middlegame / endgame / any)
- `cause_tag` = the MetricSignal's `cause` field (optional, narrows further)

If no exact match, the query relaxes: strategy → `general`, then severity → `any`.
So a phrase tagged `strategy='general', severity='any'` acts as a universal fallback.

---

## Current Coverage — What Exists

| Strategy | Metric | Severities Covered | Slots Covered |
|----------|--------|--------------------|---------------|
| blitz | king_exposure | critical, high | diagnosis, evidence, plan, urgency |
| blitz | sacrifice_delta | any | diagnosis, urgency |
| flank | space_delta_queenside | high, moderate | diagnosis, plan, urgency |
| flank | space_delta_kingside | high | diagnosis |
| flank | piece_mobility_ratio | high | evidence, plan |
| flank | outpost_occupation | any | diagnosis, plan |
| flank | bad_piece | any | evidence, plan |
| fortress | pawn_fixedness | high | diagnosis, plan |
| fortress | eval_deficit | high, critical | diagnosis, evidence, plan, urgency |
| fortress | overextension | any | evidence, plan |
| feint | outpost_occupation | any | diagnosis |
| feint | pawn_fixedness | any | plan |
| feint | piece_mobility_ratio | moderate | evidence |
| general | pawn_fixedness | high | diagnosis |
| general | passed_pawn | critical, high | diagnosis, plan, urgency |
| general | weak_pawns | high, moderate | diagnosis, plan |
| general | piece_mobility_ratio | high | plan |
| tactic | tactic_pin | any | tactic_hint |
| tactic | tactic_fork | any | tactic_hint |
| tactic | tactic_skewer | any | tactic_hint |
| tactic | tactic_discovery | any | tactic_hint |

---

## What Is Missing — Priority Order

### PRIORITY 1 — These cause fallback to `action_hint` text on every run

#### Blitz (attack positions fire constantly — needs full coverage)
- `blitz` / `king_exposure` / `moderate` — all 4 slots  
- `blitz` / `piece_mobility_ratio` / `high` — evidence, plan  
- `blitz` / `pawn_break_availability` / any — diagnosis, plan, urgency  
- `blitz` / `tactic_pin` / any — diagnosis, plan (pin as attack instrument)  
- `blitz` / `tactic_fork` / any — diagnosis, urgency  
- `blitz` / `space_delta_kingside` / high — plan, urgency (kingside space = attack corridor)  

#### Flank (trend positions — most Carlsen games will fire these)
- `flank` / `space_delta_queenside` / moderate — all 4 slots  
- `flank` / `space_delta_kingside` / moderate, high — evidence, plan  
- `flank` / `pawn_fixedness` / high — diagnosis, plan (locked = squeeze viable)  
- `flank` / `piece_mobility_ratio` / moderate — diagnosis, plan  
- `flank` / `bad_piece` / any — urgency (when to exploit the bad piece)  
- `flank` / `outpost_occupation` / any — evidence, urgency  
- `flank` / `passed_pawn` / any — plan (passed pawn as squeeze endpoint)  

#### Fortress (needs more variety — currently only 7 phrases)
- `fortress` / `pawn_fixedness` / moderate — all 4 slots  
- `fortress` / `king_exposure` / moderate — evidence, plan (defensive king safety)  
- `fortress` / `weak_pawns` / high — diagnosis, plan (defend, don't trade)  
- `fortress` / `piece_mobility_ratio` / moderate — diagnosis (restricted pieces = fortress working)  
- `fortress` / `eval_deficit` / moderate — all 4 slots  
- `fortress` / `passed_pawn` / high — urgency (enemy passed pawn = fortress threat)  

#### Feint (only 3 phrases — nearly always falls back)
- `feint` / `outpost_occupation` / any — evidence, plan, urgency  
- `feint` / `space_delta_queenside` / any — diagnosis (feint queenside, strike kingside)  
- `feint` / `pawn_fixedness` / any — diagnosis, urgency  
- `feint` / `piece_mobility_ratio` / any — plan, urgency  
- `feint` / `king_exposure` / moderate — diagnosis (feint toward king that looks safe)  

---

### PRIORITY 2 — Phase-specific phrases (currently most are tagged `any`)

Every major metric needs **opening**, **middlegame**, and **endgame** variants
because the advice changes dramatically by phase:

- `general` / `passed_pawn` / critical / **endgame** — king must escort it  
- `general` / `passed_pawn` / high / **middlegame** — create a second weakness  
- `general` / `weak_pawns` / high / **endgame** — isolated pawn becomes a lost pawn  
- `flank` / `space_delta_queenside` / any / **endgame** — space converts to passed pawn  
- `blitz` / `king_exposure` / critical / **opening** — development punishment  
- `fortress` / `eval_deficit` / high / **endgame** — technique to hold the draw  

---

### PRIORITY 3 — Cause-tag variants (most specific, highest priority when matched)

The `cause_tag` column lets phrases match exactly the *reason* a signal fired.
These make the output feel precise rather than generic.

Needed cause-tag phrases for:
- `cause_tag = 'open_file'` — for rook-on-open-file evidence (blitz + flank)
- `cause_tag = 'bad_bishop'` — Nimzowitsch has extensive writing on this
- `cause_tag = 'isolated_pawn'` — blockade plan is different from doubled pawn plan
- `cause_tag = 'doubled_pawn'` — trade vs advance decision
- `cause_tag = 'advanced_passed_pawn'` — escort plan by rank (rank 6 vs rank 7)
- `cause_tag = 'opponent_pawn_overextended'` — attack the head of the chain
- `cause_tag = 'missing_pawn_shield'` — which specific file (g, h, f)
- `cause_tag = 'attacker_concentration'` — pile-up language

---

### PRIORITY 4 — Tactic table expansion

Currently only `pin`, `fork`, `skewer`, `discovery`. Needed:
- `zwischenzug` — in-between move that disrupts the sequence
- `deflection` — force a defender away from its duty
- `blockade` (tactic variant) — block a passed pawn with a piece
- All existing types need **phase variants** (opening / middlegame / endgame)
- All existing types need **strategic_link variants** (blitz vs fortress use same pin differently)

---

## How to Add Phrases

### Option A — Direct SQL insert (permanent, version-controlled)

Add rows to the `_SEED_PHRASES` tuple in `database/phrase_db.py`. Format:

```python
(strategy, phase, metric, severity, fragment_type, cause_tag,
 phrase_text, source, voice, priority)
```

Example:
```python
('blitz', 'middlegame', 'pawn_break_availability', 'high', 'plan', '',
 'The g4-g5 thrust rips open the h-file. Play it now before {side} shores up the kingside.',
 'My System Ch.5', 'nimzowitsch', 8),
```

The DB is seeded from `_SEED_PHRASES` on first creation. To force a re-seed on
an existing DB, delete `chess_coach.db` and let `StrategyEngine` recreate it.

### Option B — Anthropic API artifact (bulk generation)

An artifact has already been prepared that uses the Claude API to generate
Nimzowitsch-voice phrases in bulk given a strategy/metric/slot specification.
Run it to produce a Python list of tuples, then paste into `_SEED_PHRASES`.

### Option C — Direct DB insert (runtime, not version-controlled)

```python
from database.phrase_db import PhraseDB
db = PhraseDB('data/chess_coach.db')
db._conn.execute("""
    INSERT INTO phrases (strategy, phase, metric, severity, fragment_type,
                         cause_tag, phrase_text, source, voice, priority)
    VALUES (?,?,?,?,?,?,?,?,?,?)
""", ('blitz', 'middlegame', 'king_exposure', 'high', 'evidence', '',
      'Three pieces aim at the weakened zone — the king cannot face them all.',
      'original', 'nimzowitsch', 7))
db._conn.commit()
```

---

## Phrase Quality Rules

1. **Never mention a player by name** — say "the master" or "classical practice"
2. **Nimzowitsch voice** — slightly formal, chess-as-philosophy, never casual
3. **Placeholders must resolve** — only use `{square}`, `{file}`, `{piece}`, `{side}`, `{target}`
4. **One thought per phrase** — diagnosis says what, evidence says why, plan says how, urgency says when
5. **Source every phrase** — cite the chapter of My System, or use `'original'` if authored fresh
6. **Priority 8-9** for specific cause_tag matches, **5-7** for general fallbacks

---

## Target Counts

| Strategy | Current | Target | Gap |
|----------|---------|--------|-----|
| blitz | 8 | 50 | 42 |
| flank | 10 | 60 | 50 |
| fortress | 7 | 45 | 38 |
| feint | 3 | 30 | 27 |
| general | 8 | 50 | 42 |
| tactic | 5 | 25 | 20 |
| **Total** | **41** | **260** | **219** |

The 300+ target from the spec accounts for the tactics table as well.
260 in `phrases` + 40+ in `tactics` clears the bar.
