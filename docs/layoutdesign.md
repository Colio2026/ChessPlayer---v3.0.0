# Coach Panel — Layout Design

**Status:** Agreed design. No implementation begins without sign-off on this doc.  
**Depends on:** Phase 6B checkpoint (`data/classifier_best.pt`, Macro F1 0.6785, MoEConceptClassifier)  
**Last updated:** 2026-07-25

### Resolved decisions

| # | Decision | Answer |
|---|---|---|
| Manual refresh | Coach does NOT auto-refresh on position change. A dedicated Refresh button in the toolbar triggers analysis on demand. | Manual — button in toolbar |
| "After" eval | Post-line concept detection runs on the position after **1 move only** (first PV move pushed). Label this clearly in the UI: "after move 1 of this line". Do not push the full PV. | 1 move, labelled |
| Weak squares source | Weak squares bubble pulls from the **current position** concept classification, not the PV lines. | Current position |
| Explanation structure | Two distinct labelled parts: (1) **Pattern** — action hint describing what's happening now; (2) **Precedent** — RAG annotation as a historical/literary reference. Not concatenated into one prose paragraph. | Two parts, distinct |

---

## 1. Design Goals

The panel should answer three questions for the player, in order:

1. **Where am I?** — Opening theory or endgame oracle (context bubble, top of panel)
2. **What does my position have?** — Weak squares I can target or must defend (positional layer)
3. **What do these Stockfish lines actually mean?** — Concept-labelled, RAG-explained per-line breakdown

The panel does **not** surface "best move" recommendations. It surfaces what each strong line *achieves or costs* in conceptual chess terms, so the player understands the plan rather than memorising the move.

---

## 2. Visual Layout

```
┌────────────────────────────────────────────────────────────┐
│  [ Coach ON ]  [ ♙ White ]  [ ↺ Refresh ]   [ Insert Note ]│
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  OPENING / ENDGAME BUBBLE                            │  │
│  │  (appears / disappears — see §3)                     │  │
│  │                                                      │  │
│  │  Opening:  B90 Sicilian, Najdorf  (move 6)           │  │
│  │  "In the Sicilian the initiative belongs to the      │  │
│  │   player who controls the pace of central tension."  │  │
│  │   — Nimzowitsch, My System                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  WEAK SQUARES BUBBLE  (§4)                           │  │
│  │                                                      │  │
│  │  White pressure points:  d5  ·  f5                   │  │
│  │  Black weaknesses:       d4  ·  e6                   │  │
│  │  "Plant a piece on d5 — no Black pawn can challenge  │  │
│  │   it. The d5 outpost anchors the entire plan."       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SF LINE BUBBLE  #1  (§5)                            │  │
│  │                                                      │  │
│  │  [ INITIATIVE ]  +1.43 cp                            │  │
│  │  1. Nd5  Nxd5  2. exd5  Nb8  3. d6                  │  │
│  │                                                      │  │
│  │  SF Eval Metrics                                     │  │
│  │  ┌──────────┬────────┬────────┐                      │  │
│  │  │ Term     │ Before │ After  │                      │  │
│  │  │ Mobility │ +0.12  │ +0.31  │                      │  │
│  │  │ Space    │ +0.08  │ +0.22  │                      │  │
│  │  │ King Saf │ -0.04  │ -0.04  │                      │  │
│  │  └──────────┴────────┴────────┘                      │  │
│  │                                                      │  │
│  │  This line converts the initiative into a lasting    │  │
│  │  structural advantage. Nd5 trades space-advantage    │  │
│  │  for an advanced passed pawn on d6 — a permanent     │  │
│  │  thorn Black cannot remove. The pawn restricts the   │  │
│  │  Black knight and ties a rook to passive defense.    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SF LINE BUBBLE  #2  (§5)                            │  │
│  │  [ SPACE ADVANTAGE ]  +0.87 cp                       │  │
│  │  ...                                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 3. Opening / Endgame Bubble

### 3.1 Visibility

| Phase | Bubble visible? | Content |
|---|---|---|
| **Opening** — ECO code still matched by `eco_db.json` | ✅ Yes | Opening name + variation + RAG commentary |
| **Middlegame** — no ECO match in history | ❌ Hidden | — |
| **Endgame (tablebase)** — piece count ≤ 7 AND Syzygy probe succeeds | ✅ Yes | Endgame type + RAG commentary + WDL/DTZ result |
| **Endgame (no tablebase)** | ❌ Hidden | — |

The bubble never shows both opening and endgame content simultaneously. Transition from opening → middlegame → endgame is driven by the ECO match disappearing and the tablebase probe succeeding.

### 3.2 Opening mode

**What it shows:**
- ECO code + full opening name + variation (e.g. `B90 Sicilian, Najdorf, English Attack`)
- Move depth at which the ECO was identified (e.g. `move 6`)
- RAG commentary: one or two sentences from `rag_index.jsonl` retrieved by ECO code match

**Data source:**
- `RAGRetriever.identify_opening(history_fens)` → `{eco, opening, variation, depth}`
- `RAGRetriever.retrieve(fen, history_fens=history_fens, eco_override=eco_code)` → top annotation

**Key constraint:** Bubble disappears the moment `identify_opening()` returns `None` (position has left known theory). It does NOT persist as a "last known opening" through the middlegame.

### 3.3 Endgame / tablebase mode

**What it shows:**
- Endgame type label (e.g. `Rook Endgame`, `King and Pawn vs King`)
- WDL result: `WIN (DTZ 14)` / `DRAW` / `LOSS`
- RAG commentary: retrieved by endgame-type concept (e.g. `rook_endgame`, `pawn_endgame`) from `rag_index.jsonl`

**Data source:**
- `chess.syzygy.open_tablebase("data/syzygy")` → `wdl`, `dtz`
- `MoEConceptClassifier.predict_concepts(fen)` → Tier 4 endgame concepts to identify endgame type
- `RAGRetriever.retrieve(fen, concepts=[endgame_type_concept])` → commentary

**WDL display:**
- `wdl == 2`: green chip `WIN (DTZ N)` — N = number of moves to decisive zeroing move
- `wdl == 0`: yellow chip `DRAW` — coach focuses on fortress / resistance / perpetual threat
- `wdl == -2`: red chip `LOSS` — coach focuses on resistance and delay tactics

---

## 4. Weak Squares Bubble

### 4.1 Visibility

Shown whenever the net fires `weak_square` above its calibrated threshold. Always shown in the middlegame when present. Hidden if `weak_square` confidence is below threshold.

### 4.2 What it shows

- **White's pressure points** (squares weak for Black = strong for White): up to 3 squares by name
- **Black's weaknesses** (squares weak for Black): same set expressed differently
- One sentence of commentary connecting the weak squares to the plan

**Key design note:** Weak for Black ≡ strong for White — the bubble labels the squares once, with polarity implied. No need to list both "white strong squares" AND "black weak squares" separately. The framing is always from the coached side's perspective: "You can occupy d5. Your opponent cannot drive you away."

### 4.3 Data source

- `MoEConceptClassifier.predict_concepts(fen)` → `weak_square` concept + probability
- `_extract_key_squares("weak_square", board, side)` from `concept_signal_adapter.py` → square names
- `RAGRetriever.retrieve(fen, concepts=["weak_square"])` → annotation text to drive the commentary sentence
- `_ACTION_HINTS["weak_square"]` as fallback if RAG returns nothing

### 4.4 Board overlay

The weak square names should also be forwarded to `CoachBoardWidget` via `weakness_squares_ready` signal (this already exists in the panel). The board overlay shows the squares visually. The bubble and the overlay are driven by the same data.

---

## 5. SF Line Bubbles

Two bubbles, one per Stockfish PV line (top 2 only). Each is a self-contained widget.

### 5.1 Contents of each bubble

**Header row:**
- **Strategic theme badge** — the primary Tier 1 concept label for this line's resulting position (e.g. `INITIATIVE`, `PASSED PAWN`, `PROPHYLAXIS`). If no Tier 1 concept fires, badge shows `POSITIONAL` (not `GENERAL` — never surface the internal fallback label to the user).
- **Centipawn score** — the SF eval for this line (e.g. `+1.43`)

**Moves row:**
- The PV moves in SAN notation (e.g. `1. Nd5  Nxd5  2. exd5  Nb8  3. d6`)

**Eval metrics table:**
- Small 3-column table: Term | Before | After
- "Before" = SF classical eval breakdown for the current position
- "After" = SF classical eval breakdown after the line plays out
- Terms to show (already in `sf_cache.npy`): Mobility, Space, King Safety, Threats, Passed Pawns
- **Delta column is optional** — show it only if space allows; the Before/After comparison is the primary signal

**Conceptual explanation — two distinct parts:**

*Part 1 — Pattern (what is happening now):*
- Driven by fired concept `_ACTION_HINTS` for the Tier 1 concept of this line
- Written from the current position's perspective: "This line activates the **initiative** — keep making threats to deny the opponent time to organise."
- Supporting Tier 2 concepts listed below the main hint if they fire (e.g. "Also: **open file** — seize it with a rook.")

*Part 2 — Precedent (historical / literary reference):*
- One RAG annotation excerpt from `rag_index.jsonl`, retrieved by the fired concept labels
- Presented as a blockquote with attribution: `"[annotation text]" — [source]`
- Clearly separate from Part 1 — not merged into the same paragraph
- Hidden if no relevant RAG record is found (no fallback placeholder text)

### 5.2 Data source per bubble

For each SF PV line:
1. Push `pv_uci[0]` only → get `post_move_fen` (one move ahead, NOT the full PV)
2. `MoEConceptClassifier.predict_concepts(post_move_fen, history_rich=history_rich)` → fired concepts for that one-move-ahead position
3. `infer_strategy(fired_concepts)` → strategic theme badge (Tier 1 winner)
4. `RAGRetriever.retrieve(post_move_fen, concepts=fired_concept_names, eco_override=eco_code)` → top annotation for the precedent block
5. Render Part 1 (Pattern) from Tier 1 + Tier 2 action_hints
6. Render Part 2 (Precedent) as a blockquote from the RAG annotation

**"After 1 move" label:** The UI must make this scope explicit. Suggested label: `"after 1 move"` shown in small text next to the theme badge or metrics table header. Manages user expectations — the concept assessment is not the full line.

**Important:** Concept detection runs on `post_move_fen` (one move deep), not the current position. Each line gets its own concept set — two lines can show different theme badges.

### 5.3 Theme badge colour map

Reuse the existing `_COLOURS` dict in `coach_panel.py`. All Tier 1 concepts already have colours defined. If the badge strategy is not in `_COLOURS`, use `#78909C` (neutral grey).

### 5.4 What is NOT in the line bubbles

- No "Load in Coach Board" per-concept link in the main explanation (keep the "Load in Coach Board" button in the header row only)
- No GM precedents inside the line bubble (GM precedents are a separate section if added later)
- No raw probability numbers visible to the user — concepts are surfaced by name, not by score

---

## 6. Component Visibility Summary

| Component | Shown when | Hidden when |
|---|---|---|
| Opening/Endgame Bubble | ECO match exists OR tablebase probe succeeds | Middlegame, no tablebase |
| Weak Squares Bubble | `weak_square` concept fires above threshold | Concept not active |
| SF Line Bubble #1 | Always (after first SF PV arrives) | Engine not running |
| SF Line Bubble #2 | Second PV available | Only 1 PV line |

---

## 7. Processing Flow (manual refresh, per Refresh button press)

```
User presses [ ↺ Refresh ]
  │
  ├─1─ RAGRetriever.identify_opening(history_fens)
  │        → eco_code, opening_name  (or None)
  │        → Opening/Endgame Bubble: SHOW with opening content (or HIDE)
  │
  ├─2─ syzygy.probe(board)   [only if piece_count ≤ 7]
  │        → wdl, dtz  (or None)
  │        → Opening/Endgame Bubble: SHOW with tablebase content (or remain hidden)
  │
  ├─3─ MoEConceptClassifier.predict_concepts(current_fen, history_rich, eco_code)
  │        → current_position_concepts
  │        → Extract weak_square squares + action_hint + RAG annotation
  │        → Render Weak Squares Bubble (or hide if weak_square not active)
  │
  ├─4─ [Read latest SF PV lines — already computed by engine; do not re-trigger SF]
  │
  └─5─ For each of top 2 SF PV lines:
        a. push pv_uci[0] only → post_move_fen  (1 move deep)
        b. MoEConceptClassifier.predict_concepts(post_move_fen, history_rich, eco_code)
             → fired_concepts_for_line
        c. infer_strategy(fired_concepts_for_line)
             → theme_badge
        d. RAGRetriever.retrieve(post_move_fen, concepts=fired_concept_names, eco_override=eco_code)
             → rag_annotation  (for Precedent block)
        e. Build Part 1 (Pattern): Tier 1 action_hint + supporting Tier 2 hints
        f. Build Part 2 (Precedent): RAG blockquote + attribution (or omit if empty)
        g. SF eval metrics: "Before" = current SF eval breakdown; "After" = post-move SF eval breakdown
        → Render SF Line Bubble #N  (labelled "after 1 move")
```

**Note on auto-refresh:** The debounce timer and automatic `_fire_analysis()` on position change are removed. Analysis only runs when the user presses Refresh. SF engine continues running in the background and its PV lines are always current — only the ML + RAG layer is gated behind the manual button.

---

## 8. Open Questions / Not Yet Resolved

| # | Question | Answer |
|---|---|---|
| 1 | ~~Which line's concept data drives the Weak Squares bubble?~~ | **Resolved: current position.** |
| 2 | How many sentences in Part 1 (Pattern)? | 1 main Tier 1 hint + up to 2 Tier 2 supporting hints. |
| 3 | ~~SF eval metrics "After" — separate SF call or reuse PV score?~~ | **Resolved: run SF classical eval on `post_move_fen`.** Full Before/After per-term breakdown in metrics table. Accept the latency — it's behind a manual Refresh. |
| 4 | ~~Opening bubble threshold — from move 1 or later?~~ | **Resolved: show for every ECO code past the starting position** (any move > 0 where `identify_opening()` returns a match). All ECO commentary is human-written. Starting position (move 0) is the only exclusion. |
| 5 | ~~Endgame type label — net or heuristic?~~ | **Resolved: both.** Tablebase (Syzygy) provides WDL/DTZ and confirms the endgame is deterministic. Net concepts (e.g. `rook_endgame`, `knight_endgame`) identify the *type*, which drives RAG retrieval of human commentary. Neither alone is sufficient — tablebase without net gives no RAG entry point; net without tablebase misses the objective evaluation. |

---

## 9. What Does NOT Change

- Coach toggle (ON/OFF) and side selector: unchanged
- "Insert Note" button: unchanged
- Board overlay for weak squares: unchanged (uses existing `weakness_squares_ready` signal)
- `CoachBoardWidget`: unchanged
- The `CoachOutput` data class: extended with new fields, not restructured

---

## 10. Implementation Order

This design is **not yet implemented**. Build order once this doc is signed off:

1. Extend `CoachOutput` in `data_types.py` — add `opening_eco`, `opening_name`, `rag_annotation`, `tablebase_wdl`, `tablebase_dtz`
2. Update `NimzoNetEngine.analyse()` — pass history_rich (from `board.move_stack`) and eco_code; wire Opening/Endgame bubble data through to `CoachOutput`
3. Add `_probe_tablebase()` to `rag/coach.py` — Syzygy probe with ≤7 piece guard
4. Add `analyse_line()` helper to `NimzoNetEngine` — takes `post_move_fen` + history, returns concept set + theme badge + RAG annotation
5. Update `_MultiAnalysisWorker` — call `analyse_line()` per PV instead of `analyse_from_pv()`
6. Rebuild `_render_line_card()` in `coach_panel.py` — new layout with theme badge, metrics table, explanation paragraph
7. Add Opening/Endgame Bubble widget to `_build_ui()`
8. Add Weak Squares Bubble widget to `_build_ui()` (separate from the board overlay it feeds)
9. Wire `_render_shared_sections()` to populate Weak Squares bubble
