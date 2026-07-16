# Nimzo-Net Plumbing Plan
## Replacing the Rule-Based Chess Coach with the ML Classifier

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

## 3. Architecture: What Changes, What Stays

```
┌─────────────────────────────────────────────────────────────┐
│                      COACH PANEL (UI)                        │
│            coach_panel.py   —   NO CHANGES                  │
└────────────────────────┬────────────────────────────────────┘
                         │ CoachOutput
┌────────────────────────▼────────────────────────────────────┐
│                       NARRATOR                               │
│           coach/narrator.py   —   NO CHANGES                │
└────────────────────────┬────────────────────────────────────┘
                         │ MetricSignal[]  +  ResolverResult
┌────────────────────────▼────────────────────────────────────┐
│              NIMZO-NET ENGINE  [NEW]                         │
│       coach/nimzo_net_engine.py                             │
│   • loads ChessConceptClassifier once at startup            │
│   • calls predict_concepts(fen, history_uci)                │
│   • passes output to ConceptSignalAdapter                   │
│   • derives strategy_primary from Tier 1 concepts           │
│   • derives phase from FEN (piece count / move clock)       │
│   • returns ResolverResult ready for narrator               │
└────────────────────────┬────────────────────────────────────┘
                         │ [(concept, probability)]
┌────────────────────────▼────────────────────────────────────┐
│           CONCEPT SIGNAL ADAPTER  [NEW]                      │
│       ml/concept_signal_adapter.py                          │
│   • maps concept name → MetricSignal                        │
│   • maps probability → severity                             │
│     (≥0.85 critical, ≥0.70 high, ≥0.55 moderate, else mild)│
│   • sets side = side_to_move(fen)                           │
│   • sets phase = detect_phase(fen)                          │
│   • populates key_squares / key_pieces from board for       │
│     Tier 3 tactics (lightweight rule fallback for now)      │
└────────────────────────┬────────────────────────────────────┘
                         │ MetricSignal[]
┌────────────────────────▼────────────────────────────────────┐
│              PHRASE DB  [REBUILT]                            │
│       database/phrase_db.py  +  chess_coach.db              │
│   • metric_name column now matches concept names directly    │
│   • phrase coverage for all 49 concepts                     │
│   • more source books (not just My System)                  │
│   • Tier 1 phrases: diagnosis/evidence/plan/urgency/headline│
│   • Tier 2 phrases: evidence/plan only (supporting context) │
│   • Tier 3 phrases: tactic_hint slot only                   │
└─────────────────────────────────────────────────────────────┘

DELETED / REPLACED:
  src/chess_coach/core/strategy_engine.py    ← replaced by nimzo_net_engine.py
  src/chess_coach/core/conflict_resolver.py  ← absorbed into nimzo_net_engine.py
  src/chess_coach/extractors/king_safety.py  ← deleted
  src/chess_coach/extractors/material_balance.py ← deleted
  src/chess_coach/extractors/pawn_structure.py   ← deleted
  src/chess_coach/extractors/piece_mobility.py   ← deleted
  src/chess_coach/extractors/space_control.py    ← deleted
  src/chess_coach/extractors/tactic_scanner.py   ← deleted (Tier 3 handled by adapter)
  src/chess_coach/strategies/blitz_detector.py   ← deleted
  src/chess_coach/strategies/feint_detector.py   ← deleted
  src/chess_coach/strategies/flank_detector.py   ← deleted
  src/chess_coach/strategies/fortress_detector.py ← deleted
```

---

## 4. Data Flow (per position change)

```
User moves a piece
  → coach_panel.queue_analysis(fen, history_uci)
  → NimzoNetEngine.analyse(fen, history_uci)
      → ChessConceptClassifier.predict_concepts(fen, history_uci)
          → [(concept, prob), ...]   # above calibrated thresholds
      → ConceptSignalAdapter.adapt(concepts, fen)
          → MetricSignal[] (one per fired concept)
      → _infer_strategy(tier1_signals)
          → strategy_primary, strategy_secondary, confidence
      → _detect_phase(fen)
          → 'opening' | 'middlegame' | 'endgame'
      → narrator.assemble(resolver_result, signals, phrase_db)
          → CoachOutput
  → coach_panel renders CoachOutput
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
- `ui/coach_panel.py` — step 4 is a small wiring change, not a rewrite
- `database/pattern_matcher.py` — GM precedent lookup is independent
- `database/pgn_indexer.py` — game index is independent
- `core/data_types.py` — only `STRATEGIES` tuple extended
- `CoachOutput` field layout — unchanged
- The 4-slot narrator contract (diagnosis/evidence/plan/urgency) — unchanged
