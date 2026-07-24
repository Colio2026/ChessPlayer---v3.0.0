# Nimzo-Net Plumbing Plan
## Phase 4C → Phase 6: ML Classifier + RAG Coach Panel + Gating Network

**Last updated:** 2026-07-24  
**Current production:** Phase 4C checkpoint, NimzoNetEngine + ChessCoach (RAG not yet wired to panel)  
**Next milestone:** Phase 6A — ECO identification, RAG literary references, and tablebase indicators visible in coach panel (no retraining required)

---

## 1. Goal

Replace every rule-based extractor in `src/chess_coach/` with the trained
`ChessConceptClassifier` ("Nimzo-Net"). The narrator, phrase DB, and coach
panel UI survive unchanged. Only the signal-production layer is replaced.

**Coaching perspective:** always the side to move.

---

## 2. Concept Taxonomy (the 49 → 3 tiers)

Not all 49 classifier concepts are "strategies". The coach treats them
differently depending on their tier.

### Tier 1 — Grand Strategies
Multi-move plans. Shown in the coach panel as **"Recommended Grand Strategy"**
with a unique colour tag. These replace and extend the old 4-strategy set.

| Concept | Replaces / extends |
|---|---|
| `mating_attack` | blitz |
| `outpost` | flank (piece anchor) |
| `space_advantage` | flank (squeeze) |
| `blockade` | fortress |
| `prophylaxis` | fortress (prevention) |
| `passed_pawn` | — (new explicit strategy) |
| `pawn_storm` | — (new) |
| `pawn_majority` | — (new) |
| `initiative` | feint (tempo-based) |
| `development_lead` | — (opening-phase strategy) |
| `king_activity` | — (endgame strategy) |
| `piece_activity` | — (regroup/improve strategy) |

> `CoachOutput.strategy_primary` will be extended to accept any Tier 1 concept
> name. The old 4 names (blitz/flank/fortress/feint) become aliases and can be
> retired once phrase coverage is complete.

### Tier 2 — Positional Diagnostics
Structural observations about the position. Shown in plan_sentences as
supporting evidence or coaching context. Not shown as a "recommended strategy".

`weak_square`, `open_file`, `king_safety`, `bishop_pair`, `good_bishop`,
`bad_bishop`, `isolated_pawn`, `doubled_pawn`, `backward_pawn`, `pawn_chain`,
`pawn_island`, `rook_seventh`, `battery`

### Tier 3 — Tactics
Move-level pattern observations. Shown in `tactic_hints` (existing slot in
`CoachOutput`). The narrator already handles these in a separate section.

`pin`, `fork`, `skewer`, `x_ray`, `discovery`, `double_check`, `clearance`,
`deflection`, `overloading`, `zwischenzug`, `interference`, `back_rank`,
`trapped_piece`, `sacrifice`

### Tier 4 — Endgame Type
Identifies which endgame we are in. Used to select phase-appropriate phrases
and suppress irrelevant strategy suggestions.

`rook_endgame`, `pawn_endgame`, `bishop_endgame`, `knight_endgame`,
`queen_endgame`, `drawn_position`, `shouldering`, `opposition`, `zugzwang`,
`promotion`

---

## 3. Architecture: Current State (Phase 4C) and Target State (Phase 6A)

### 3A. Current state — what is and isn't wired

```
┌──────────────────────────────────────────────────────────────┐
│                    COACH PANEL (UI)                           │
│  Shows: strategy, concept list, tactic hints                  │
│  Missing: opening name, literary reference, tablebase WDL     │
└────────────────────────┬─────────────────────────────────────┘
                         │ CoachOutput
┌────────────────────────▼─────────────────────────────────────┐
│                      NARRATOR                                 │
│           coach/narrator.py   —   NO CHANGES                 │
└────────────────────────┬─────────────────────────────────────┘
                         │ MetricSignal[]  +  ResolverResult
┌────────────────────────▼─────────────────────────────────────┐
│              NIMZO-NET ENGINE                                 │
│       coach/nimzo_net_engine.py                              │
│   • ChessConceptClassifier (Phase 4C, 3.6M params)          │
│   • ConceptSignalAdapter → MetricSignal[]                    │
│   • strategy_primary from Tier 1 concepts                    │
└────────────────────────┬─────────────────────────────────────┘
                         │ concept probabilities (49 floats)
┌────────────────────────▼─────────────────────────────────────┐
│              CHESS COACH (RAG layer)   ← NOT REACHING PANEL  │
│       rag/coach.py                                           │
│   • Hysteresis filter (Schmitt trigger per concept)          │
│   • result["opening"] — ECO code + name  ← DISCARDED        │
│   • result["annotations"] — literary refs ← DISCARDED       │
└────────────────────────┬─────────────────────────────────────┘
                         │ (only concept list reaches engine)
┌────────────────────────▼─────────────────────────────────────┐
│              RAG RETRIEVER   ← RUNS BUT OUTPUT NOT SHOWN     │
│       rag/retriever.py                                       │
│   • ECO lookup via data/eco_db.json                          │
│   • FEN similarity + concept boost ranking                   │
│   • Returns annotations with source + quote                  │
└──────────────────────────────────────────────────────────────┘
```

The RAG retriever already runs and produces opening names and literary references. They are computed and then thrown away before reaching the panel. **Phase 6A closes this gap with UI wiring, not new ML.**

### 3B. Target state — Phase 6A (wiring only, no retraining)

```
┌──────────────────────────────────────────────────────────────┐
│                    COACH PANEL (UI)   [UPDATED]              │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Opening: B90 Sicilian, Najdorf (move 6)            │    │  ← NEW
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Theme: INITIATIVE (0.87)    Phase: middlegame               │
│  Supporting: PIECE_ACTIVITY (0.81)  SACRIFICE (0.73)         │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  "The initiative belongs to the player who controls │    │  ← NEW
│  │   the pace of central tension."                     │    │
│  │   — Nimzowitsch, My System                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Tactics: pin (0.71)  overloading (0.68)                     │
│                                                              │
│  [ Endgame: WIN (DTZ 14) ]    ← NEW (only shown ≤7 pieces)  │
└──────────────────────────────────────────────────────────────┘
```

### 3C. Target state — Phase 6B (after MoE retraining)

```
┌──────────────────────────────────────────────────────────────┐
│                    COACH PANEL (UI)                          │
└────────────────────────┬─────────────────────────────────────┘
                         │ CoachOutput (extended)
┌────────────────────────▼─────────────────────────────────────┐
│                      NARRATOR                                 │
└────────────────────────┬─────────────────────────────────────┘
                         │ ResolverResult + annotations + opening
┌────────────────────────▼─────────────────────────────────────┐
│              NIMZO-NET ENGINE (Phase 6B)                     │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              GATING NETWORK                         │   │
│   │   G(x, eco_emb) → [g_tac, g_str, g_pwn, g_eg, g_strat]│   │
│   │   ECO embedding conditions gate in opening phase   │   │
│   │   Tablebase prior boosts endgame gate (≤7 pieces)  │   │
│   └──────────┬──────────────────────────────────────────┘   │
│              │ gate weights                                  │
│   ┌──────────▼──────────────────────────────────────────┐   │
│   │  Expert 1: Tactical    Expert 4: Endgame            │   │
│   │  Expert 2: Structural  Expert 5: Strategic          │   │
│   │  Expert 3: Pawn                                     │   │
│   └──────────┬──────────────────────────────────────────┘   │
│              │ weighted concept logits (49)                  │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│              CHESS COACH (RAG + Tablebase + ECO)             │
│       rag/coach.py  [UPDATED]                               │
│   • Hysteresis filter                                        │
│   • ECO lookup → opening name + variation                    │
│   • Syzygy probe (≤7 pieces) → WDL/DTZ                      │
│   • Tablebase overrides: drawn_position, opposition,         │
│     zugzwang, shouldering set deterministically              │
│   • RAG retrieval → literary reference per primary concept   │
│   • All results forwarded to engine and panel                │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│              RAG RETRIEVER                                    │
│       rag/retriever.py  [UNCHANGED]                         │
│   • eco_db.json + rag_index.jsonl                           │
│   • Returns: eco, opening, annotations[]                     │
└──────────────────────────────────────────────────────────────┘

DELETED / REPLACED (from original Phase 4 plan — still pending):
  src/chess_coach/core/strategy_engine.py    ← replaced by nimzo_net_engine.py
  src/chess_coach/extractors/*.py            ← all deleted
  src/chess_coach/strategies/*.py            ← all deleted
```

---

## 4. Data Flow (per position change)

### Current (Phase 4C — RAG output not reaching panel)

```
User moves a piece
  → coach_panel.queue_analysis(fen, history_uci)
  → NimzoNetEngine.analyse(fen, history_uci)
      → ChessCoach.analyze(fen, history_uci)          [rag/coach.py]
          → ChessConceptClassifier.predict_concepts()
          → RAGRetriever.retrieve()                   ← result computed but lost
              → eco_db lookup  → "B90 Sicilian, Najdorf"  ← LOST
              → rag_index ranking → annotations[]          ← LOST
          → hysteresis filter
          → returns {concepts, opening, annotations}   ← opening + annotations LOST here
      → ConceptSignalAdapter.adapt(concepts, fen)     ← only concepts used
          → MetricSignal[]
      → _infer_strategy(tier1_signals)
      → narrator.assemble(resolver_result, signals, phrase_db)
          → CoachOutput
  → coach_panel renders CoachOutput                   ← no opening, no literary refs
```

### Target (Phase 6A — full signal reaches panel)

```
User moves a piece
  → coach_panel.queue_analysis(fen, history_uci)
  → NimzoNetEngine.analyse(fen, history_uci)
      → ChessCoach.analyze(fen, history_uci)
          → ChessConceptClassifier.predict_concepts()
          → [Phase 6B only] GatingNetwork(x, eco_emb) → expert weights
          → RAGRetriever.retrieve()
              → eco_db lookup  → eco_code, opening_name    ← NOW FORWARDED
              → rag_index ranking → annotations[]          ← NOW FORWARDED
          → [Phase 6A] Syzygy probe (if ≤7 pieces)
              → wdl, dtz                                   ← NEW
              → override drawn_position / opposition / zugzwang
          → hysteresis filter
          → returns {concepts, opening, annotations, tablebase}
      → ConceptSignalAdapter.adapt(concepts, fen)
      → _infer_strategy(tier1_signals)
      → narrator.assemble(resolver_result, signals, phrase_db)
          → CoachOutput  (extended: + opening + annotation + tablebase)
  → coach_panel renders:
      • Opening name (ECO + variation)                    ← NEW Phase 6A
      • Primary concept + gated supporting concepts
      • Literary reference (quote + source)               ← NEW Phase 6A
      • Tactic hints
      • Tablebase WDL/DTZ (endgame only)                 ← NEW Phase 6A
      • [Phase 6C] Recommended move + concept explanation ← future
```

---

## 5. Changes to `data_types.py`

```python
# Extend STRATEGIES to cover all Tier 1 concept names
STRATEGIES = (
    # legacy names kept for now
    'blitz', 'flank', 'fortress', 'feint',
    # new Tier 1 strategies
    'mating_attack', 'outpost', 'space_advantage', 'blockade',
    'prophylaxis', 'passed_pawn', 'pawn_storm', 'pawn_majority',
    'initiative', 'development_lead', 'king_activity', 'piece_activity',
    # fallback
    'general',
)
```

`CoachOutput.__post_init__` validation automatically picks up the extended list.

---

## 6. Strategy Inference Rules (inside NimzoNetEngine)

When multiple Tier 1 concepts fire, pick `strategy_primary` by:
1. Highest confidence Tier 1 concept wins
2. Tie-break by priority order:
   `mating_attack > passed_pawn > outpost > space_advantage > pawn_storm >
    pawn_majority > blockade > prophylaxis > initiative > development_lead >
    piece_activity > king_activity`
3. `strategy_secondary` = second highest Tier 1 if within 0.10 of primary
4. No Tier 1 fires → `strategy_primary = 'general'`

---

## 7. Phase Detection (inside NimzoNetEngine)

Derived from the FEN position directly — no ML needed:
- **Opening**: fullmove ≤ 10 AND most minor pieces still on back ranks
- **Endgame**: total material ≤ threshold (e.g. queens off + ≤ 12 points aside)
- **Middlegame**: everything else

This provides the `phase` field for MetricSignal and CoachOutput.

---

## 8. Phrase DB Rebuild Plan

### Schema change
`metric_name` column values change from old extractor names
(`king_exposure`, `outpost_occupation`, …) to concept names
(`king_safety`, `outpost`, …) matching the classifier vocabulary exactly.

### Coverage needed per concept
Every Tier 1 concept needs: `headline`, `diagnosis`, `evidence`, `plan`, `urgency`
Every Tier 2 concept needs: `evidence`, `plan`
Every Tier 3 concept needs: `tactic_hint`

### Source books to parse for new phrases
Current: My System (Nimzowitsch) — 120 phrases
Planned additions:
- Chess Fundamentals (Capablanca) — endgame technique, clarity
- Silman's Complete Endgame Course — endgame type phrases
- The Art of Attack in Chess (Vukovic) — mating attack, king safety phrases
- My Best Games (Tal) — sacrifice, initiative, mating attack flavour
- Pawn Structure Chess (Soltis) — all pawn structure diagnostics
- How to Reassess Your Chess (Silman) — imbalance / positional diagnostic language

Parsing approach: annotate a small PGN or text file per book with
`[concept=outpost]` tags before each extracted phrase, then run a simple
ingestion script to INSERT into the phrases table. Same structure as
existing seed data in `phrase_db.py`.

---

## 9. Build Order

| Step | Task | Files |
|------|------|-------|
| 1 | Extend `STRATEGIES` in `data_types.py` | `core/data_types.py` |
| 2 | Write `ConceptSignalAdapter` | `ml/concept_signal_adapter.py` |
| 3 | Write `NimzoNetEngine` (loads model, calls adapter, calls narrator) | `coach/nimzo_net_engine.py` |
| 4 | Wire `coach_panel.py` to use `NimzoNetEngine` instead of `StrategyEngine` | `ui/coach_panel.py` |
| 5 | Rebuild phrase DB schema + seed with concept-keyed phrases for Tier 1 | `database/phrase_db.py` |
| 6 | Add Tier 2 + Tier 3 phrase coverage | `database/phrase_db.py` |
| 7 | Parse additional books for richer phrase variants | new ingestion script |
| 8 | Delete old extractors + strategy detectors | many files |

Steps 1–4 can proceed with minimal phrase coverage (the narrator gracefully
falls back to `action_hint` when no phrase matches). Full phrase coverage
(steps 5–7) can be iterated while the model continues to improve.

---

## 10. What Stays Exactly As-Is

- `coach/narrator.py` — no changes
- `coach/plan_recommender.py` — no changes
- `database/pattern_matcher.py` — GM precedent lookup is independent
- `database/pgn_indexer.py` — game index is independent
- `core/data_types.py` — only `STRATEGIES` tuple extended
- The 4-slot narrator contract (diagnosis/evidence/plan/urgency) — unchanged

---

## 11. Phase 6A Build Steps — RAG Panel Wiring (no retraining)

These steps connect what already exists but isn't displayed. All changes are in
the engine and UI layers — the ML model, RAG retriever, and ECO db are untouched.

### Step 1: Extend CoachOutput to carry opening + annotation + tablebase

```python
# core/data_types.py — add three optional fields to CoachOutput
@dataclass
class CoachOutput:
    # ... existing fields ...
    opening_eco:   str  = ""     # e.g. "B90"
    opening_name:  str  = ""     # e.g. "Sicilian, Najdorf"
    rag_quote:     str  = ""     # top literary reference text
    rag_source:    str  = ""     # e.g. "My System — Nimzowitsch"
    tablebase_wdl: int  | None = None   # -2/-1/0/1/2
    tablebase_dtz: int  | None = None   # distance to zeroing
```

### Step 2: Forward RAG results through NimzoNetEngine

In `coach/nimzo_net_engine.py`, extract `result["opening"]` and
`result["annotations"]` from `ChessCoach.analyze()` and populate the new
`CoachOutput` fields before returning.

### Step 3: Add Syzygy probe to ChessCoach.analyze()

In `rag/coach.py`:
```python
import chess.syzygy
_SYZYGY_PATH = Path("data/syzygy")

def _probe_tablebase(board: chess.Board) -> tuple[int | None, int | None]:
    if board.occupied.bit_count() > 7:
        return None, None
    if not _SYZYGY_PATH.exists():
        return None, None
    try:
        with chess.syzygy.open_tablebase(str(_SYZYGY_PATH)) as tb:
            wdl = tb.probe_wdl(board)
            dtz = tb.probe_dtz(board)
            return wdl, dtz
    except Exception:
        return None, None
```

Tablebase results override ML probabilities for `drawn_position` (wdl==0 → 1.0),
and are added to the analyze() return dict alongside concepts and annotations.

### Step 4: Render opening name, literary reference, and tablebase in coach panel

In `ui/coach_panel.py`:
- Opening name widget: shows `output.opening_eco + " " + output.opening_name`; hidden when empty
- Literary reference widget: blockquote showing `output.rag_quote` with `— output.rag_source` attribution; hidden when empty
- Tablebase widget: shows "WIN (DTZ N)" / "DRAW" / "LOSS" in a coloured chip; visible only when `output.tablebase_wdl is not None`

### Step 5: Download Syzygy tables (manual)

3-4-5 piece Syzygy WDL + DTZ tables (~1 GB total). Place in `data/syzygy/`.
Free download from: https://syzygy-tables.info (or mirror).
6-piece optional (~18 GB). 7-piece not required.

---

## 12. Phase 6B Build Steps — Gating Network

See `docs/phase6_plan.md` §4 for full specification.
New files: `ml/classifier.py` (extended), `tools/build_eco_index.py`.
Training flag: `python -m src.chess_coach.ml.train --phase6`
