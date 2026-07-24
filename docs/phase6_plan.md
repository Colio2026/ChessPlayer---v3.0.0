# Phase 6 — Gating Network, RAG Coach Panel, and Deterministic Oracles

**Status:** Planning  
**Depends on:** Phase 4C champion checkpoint (`data/classifier_best.pt`, Macro F1 0.6768)  
**Outcome target:** Coach panel that identifies the opening, presents literary references, and delivers gated concept explanations grounded in objective evaluation.

---

## 1. Core Insight

The game divides into three zones with fundamentally different tractability:

| Zone | Determinism | Right tool |
|---|---|---|
| Opening (≤ ~15 moves, ECO-identifiable) | High — known theory | ECO prior → bias gates |
| Endgame (≤ 7 pieces) | Near-perfect | Syzygy tablebase oracle |
| Middlegame | Low — combinatorial | ML gating network (MoE) |

**Do not use ML where perfect knowledge already exists.** ECO theory and tablebases cover the opening and endgame with far more precision than any ML model trained on annotated games. The gating network's job is to decide which regime the position is in and route accordingly — not to duplicate solved problems.

The ML classifier (Nimzo-Net) remains the core middlegame engine. The gating network wraps it, not replaces it.

---

## 2. Phase Breakdown

Phase 6 ships in three sub-phases so improvements reach the coach panel incrementally.

### Phase 6A — Coach Panel RAG + ECO Display (no retraining)
Wire the existing RAG retriever output into the coach panel UI. Uses Phase 4C weights as-is.  
**Deliverables:** Opening name shown in coach panel; literary reference displayed per concept; tablebase WDL indicator in endgame positions.

### Phase 6B — Mixture of Experts Gating Network (retraining)
Replace the single MLP head with a gating network + 5 expert heads. ECO embeddings condition the gate in the opening. Tablebase signals condition endgame expert weights.  
**Deliverables:** Retrained Phase 6B checkpoint. Per-concept F1 improvement target: bottom-quartile concepts (x_ray, interference, pawn_chain, mating_attack) → ≥ 0.55.

### Phase 6C — SF-Validated Move Connector
Post-classification layer: run SF on top-N candidate moves, score each via the gated concept vector, surface the move whose concept activation best explains the SF evaluation gain.  
**Deliverables:** Coach panel "Recommended Move" slot populated with concept-explained move.

---

## 3. Phase 6A — RAG + ECO + Tablebase Wiring

### 3.1 What the coach panel displays

```
┌─────────────────────────────────────────────────────┐
│  NIMZO COACH                                        │
│                                                     │
│  Opening:  B90 Sicilian, Najdorf (move 6)           │  ← NEW (ECO from retriever)
│                                                     │
│  Position theme:  INITIATIVE  (0.87)                │
│  Supporting:      PIECE_ACTIVITY (0.81)             │
│                   SACRIFICE (0.73)                  │
│                                                     │
│  "In the Sicilian, the initiative belongs to the    │  ← NEW (RAG literary reference)
│   player who controls the pace of central tension.  │
│   Do not allow the opponent to complete development  │
│   without cost."  — Nimzowitsch, My System          │
│                                                     │
│  Tactics spotted:  pin (0.71), overloading (0.68)   │
│                                                     │
│  [ Endgame: WDL +1 (winning) | DTZ 14 ]            │  ← NEW (tablebase, endgame only)
└─────────────────────────────────────────────────────┘
```

### 3.2 ECO identification

The RAG retriever (`rag/retriever.py`) already walks game history FENs against `data/eco_db.json` to identify the opening ECO code. This output is computed but **not passed to the coach panel**.

**Change:** `ChessCoach.analyze()` returns `result["opening"]` (already present in the dict). The coach panel must read and display `result["opening"]["eco"]` and `result["opening"]["name"]`. This is a UI wiring change only — no new ML work.

### 3.3 Literary reference in the panel

`ChessCoach.analyze()` returns `result["annotations"]` — a list of dicts with `annotation`, `source`, `concept`, `eco`. These are real human commentary excerpts from the RAG index.

**Change:** The coach panel renders the top-ranked annotation as a blockquote under the concept display. Format: `"{annotation}" — {source}`. Filter to only show annotations whose `concept` matches the primary fired concept.

This is a UI wiring change. The RAG retrieval already runs; the results are thrown away before reaching the UI.

### 3.4 Syzygy tablebase integration

When piece count ≤ 7 (no queens + few pieces), query Syzygy for WDL and DTZ.

```
data/syzygy/         ← directory for Syzygy WDL + DTZ files
                       3-4-5 piece: ~1 GB (free download)
                       6-piece: ~18 GB (optional)
                       7-piece: ~150 GB (optional, skip for now)
```

`python-chess` has built-in support:
```python
import chess.syzygy
with chess.syzygy.open_tablebase("data/syzygy") as tb:
    wdl = tb.probe_wdl(board)   # -2/-1/0/1/2
    dtz = tb.probe_dtz(board)   # distance to zeroing move
```

**Effect on coach output:**
- `wdl == 2` → display "Tablebase: WIN (DTZ {dtz})" — suppress speculative concepts
- `wdl == 0` → display "Tablebase: DRAW" — flag if model incorrectly fires `mating_attack`
- `wdl == -2` → display "Tablebase: LOSS" — coach focuses on resistance/delay

Concepts overridden by tablebase in endgame (bypass ML entirely for these):
- `drawn_position` → set probability 1.0 if wdl == 0
- `opposition`, `zugzwang`, `shouldering` → determined by DTZ trajectory scan (see §3.5)
- `promotion` → set probability 1.0 if pawn is on 7th rank AND wdl == 2

### 3.5 Endgame concept computation from tablebase

For K+P vs K and similar basic endings, compare WDL before and after king moves to detect zugzwang and opposition directly:

```python
def _probe_zugzwang(board, tb):
    wdl_now = tb.probe_wdl(board)
    board.push(chess.Move.null())   # pass move (null move)
    wdl_after = tb.probe_wdl(board)
    board.pop()
    # Zugzwang: side to move is worse off if they have to move
    return wdl_now < wdl_after
```

This is deterministic — no ML needed. The gating network will learn to down-weight ML classification and up-weight tablebase signals when piece count is low.

---

## 4. Phase 6B — Mixture of Experts Gating Network

### 4.1 Architecture overview

```
Input x (5004-dim: board + move + algo_v4 + v3 + sf)
    │
    ├──► Gating Network G(x, eco_emb)
    │        └── softmax → [g_tactical, g_structural, g_pawn, g_endgame, g_strategic]
    │
    ├──► Expert 1: TACTICAL head    → 15 concept logits
    ├──► Expert 2: STRUCTURAL head  → 8 concept logits
    ├──► Expert 3: PAWN head        → 9 concept logits
    ├──► Expert 4: ENDGAME head     → 11 concept logits
    └──► Expert 5: STRATEGIC head   → 6 concept logits

Final output = Σ g_i × Expert_i(x)    (weighted sum, reconstructed to 49 logits)
```

### 4.2 Expert concept partitioning

| Expert | Concepts | Count |
|---|---|---|
| **Tactical** | pin, fork, skewer, discovery, x_ray, double_check, clearance, deflection, overloading, zwischenzug, interference, back_rank, sacrifice, mating_attack, trapped_piece | 15 |
| **Structural** | outpost, blockade, bad_bishop, good_bishop, bishop_pair, piece_activity, battery, rook_seventh | 8 |
| **Pawn** | passed_pawn, promotion, isolated_pawn, backward_pawn, doubled_pawn, pawn_majority, pawn_chain, pawn_storm, pawn_island | 9 |
| **Endgame** | king_safety, king_activity, shouldering, opposition, zugzwang, rook_endgame, pawn_endgame, bishop_endgame, knight_endgame, queen_endgame, drawn_position | 11 |
| **Strategic** | weak_square, open_file, space_advantage, development_lead, initiative, prophylaxis | 6 |

### 4.3 Gating network specification

```python
class GatingNetwork(nn.Module):
    # Input: combined feature vector + ECO embedding
    # Output: 5 gate weights (softmax)

    def __init__(self, input_dim: int, eco_dim: int = 64, n_experts: int = 5):
        super().__init__()
        self.eco_proj = nn.Embedding(500, eco_dim)   # ECO codes A00-E99 ≈ 500 classes
        self.gate = nn.Sequential(
            nn.Linear(input_dim + eco_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_experts),
            # no softmax here — use F.softmax at inference for numerical stability
        )
```

**ECO embedding:** Each ECO code (A00–E99) is assigned an integer index. During opening phase (move ≤ 15), the ECO code identified by the retriever is embedded and concatenated to the input before the gate. Outside the opening, a neutral "no-ECO" token is used. The embedding learns that Sicilian openings bias the tactical gate, Queen's Gambit openings bias the structural/pawn gate, etc.

### 4.4 Expert head specification

Each expert is a lightweight MLP operating on the full input vector:

```python
class ExpertHead(nn.Module):
    def __init__(self, input_dim: int, n_concepts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_concepts),
        )
```

The shared spatial projection (`spatial_proj`, 3779→256) from Phase 4C is retained as a shared encoder — all experts receive the same projected representation. This reduces parameters and encourages the experts to specialize in routing, not in re-learning the same spatial features.

### 4.5 Load balancing loss

Without regularization, the gating network collapses — one expert dominates all inputs. Standard MoE load-balancing loss:

```python
# Auxiliary loss to encourage even expert utilization
def load_balance_loss(gate_weights):
    # gate_weights: [B, 5] after softmax
    mean_gate = gate_weights.mean(dim=0)          # [5] — average gate weight per expert
    target    = torch.ones(5) / 5                 # uniform target
    return F.mse_loss(mean_gate, target)

# Total loss = BCE(logits, labels) + 0.01 * load_balance_loss(gates)
```

### 4.6 Tablebase conditioning in the endgame gate

When piece count ≤ 7 and Syzygy is available, the endgame gate weight is boosted:
```python
if syzygy_available and board.occupied.bit_count() <= 7:
    gate_logits[:, ENDGAME_IDX] += 2.0    # strong prior toward endgame expert
```

This is a hard prior applied at inference time — the gate still learns, but in clear endgames the endgame expert dominates without needing to learn this from data.

### 4.7 Parameter count estimate

| Component | Params |
|---|---|
| Shared spatial_proj (3779→256) | ~967K |
| Shared GRU (144→256) | ~264K |
| Gating network (1737→256→5) + ECO emb | ~480K |
| 5 × Expert heads (1737→512→256→N) | ~5 × 920K = ~4.6M |
| **Total** | **~6.3M** |

Up from Phase 4C's 3.6M parameters. Increase is justified by specialization — each expert trains on a focused subset of concepts with cleaner gradients.

### 4.8 Training changes

All Phase 4C training data is reused. Only the model architecture changes.

```
New: python -m src.chess_coach.ml.train --phase6
     (adds --phase6 flag to train.py; triggers MoE model class)
```

Hyperparameter starting point:
- LR: 3e-4 (cosine schedule, same as 4C)
- Batch: 512
- Patience: 20
- Load balance loss weight: 0.01
- ECO conditioning: enabled from epoch 1

---

## 5. Phase 6C — SF-Validated Move Connector

After the gated classifier fires, the coach needs to recommend a move. This connector bridges classification → recommendation without training a move generator.

### 5.1 Algorithm

```
1. Get top-5 legal moves from SF (depth 8, fast — ~100ms)
2. For each candidate move:
   a. Advance board to post-move FEN
   b. Run gated classifier on post-move FEN
   c. Compute concept_delta = post_concepts - pre_concepts   (49-dim vector)
   d. SF_eval_delta = SF(post_move_FEN) - SF(current_FEN)   (centipawns)
3. Score each candidate: score = SF_eval_delta + Σ concept_delta[i] × concept_weight[i]
   (concept_weights are higher for Tier 1 concepts)
4. Best candidate = argmax(score)
5. Explanation = top-3 concepts with largest positive delta for best move
```

### 5.2 Coach panel output

> **Recommended:** Nd5  
> SF eval: +0.62 cp  
> This move strongly activates **outpost** (+0.41) and **piece_activity** (+0.28).  
> It weakens the opponent's **weak_square** control (-0.19).

The connector explains the move in the same concept language the classifier uses. The coach never recommends a move it cannot explain.

### 5.3 Alignment constraint

If SF's top move and the concept-scored top move differ: always play SF's top move, but explain it using the concept delta of SF's move rather than a different move. SF correctness is non-negotiable — the concept explanation follows the objectively best move.

---

## 6. Data Requirements

### 6A (no new data needed)
- `data/eco_db.json` — already built
- `data/rag_index.jsonl` — already built
- `data/syzygy/` — **new download required** (Syzygy 3-4-5 piece, ~1 GB)
- Phase 4C checkpoint — existing

### 6B (no new data, new architecture)
- `data/training_raw.jsonl` — existing 1.9M examples
- All existing caches — unchanged
- ECO code index — derived from `data/eco_db.json` at training time

### 6C (no new data)
- SF binary — existing
- Phase 6B checkpoint — new

---

## 7. Build Order

### Phase 6A (coach panel wiring — can start now)

| Step | Task | File(s) |
|---|---|---|
| 6A-1 | Display `result["opening"]` in coach panel | `ui/coach_panel.py` |
| 6A-2 | Display top RAG annotation in coach panel | `ui/coach_panel.py` |
| 6A-3 | Download Syzygy 3-4-5 piece tables to `data/syzygy/` | (manual) |
| 6A-4 | Add `_probe_tablebase()` to `rag/coach.py` | `rag/coach.py` |
| 6A-5 | Display WDL/DTZ in coach panel (endgame only) | `ui/coach_panel.py` |
| 6A-6 | Override drawn_position / opposition / zugzwang from tablebase | `rag/coach.py` |

### Phase 6B (MoE retraining)

| Step | Task | File(s) |
|---|---|---|
| 6B-1 | Add `GatingNetwork` + `ExpertHead` classes | `ml/classifier.py` |
| 6B-2 | Add `--phase6` flag to `train.py` | `ml/train.py` |
| 6B-3 | Build ECO code integer index from `eco_db.json` | `tools/build_eco_index.py` (new) |
| 6B-4 | Add ECO lookup to `ChessConceptDataset.__getitem__` | `ml/dataset.py` |
| 6B-5 | Train Phase 6B | `pipeline.ps1` (step 9 uncommented) |
| 6B-6 | Calibrate + eval + audit | `pipeline.ps1` steps 10-13 |

### Phase 6C (move connector)

| Step | Task | File(s) |
|---|---|---|
| 6C-1 | Add `recommend_move()` to `rag/coach.py` | `rag/coach.py` |
| 6C-2 | Display recommended move in coach panel | `ui/coach_panel.py` |

---

## 8. Success Criteria

| Phase | Metric | Target |
|---|---|---|
| 6A | Opening name visible in panel | ✓ visible on move 1 |
| 6A | Literary reference visible in panel | ✓ 1 quote per position |
| 6A | Tablebase WDL shown in K+P endgames | ✓ correct WDL on 5 test positions |
| 6B | Macro F1 vs Phase 4C | ≥ 0.6768 (must not regress) |
| 6B | Bottom-quartile concept F1 | ≥ 0.55 (vs 0.45-0.49 in 4C) |
| 6B | Gate utilization | No expert > 60% average weight |
| 6C | Move recommendation agrees with SF top move | ≥ 90% of positions |

---

## 9. Key Risks

| Risk | Mitigation |
|---|---|
| MoE gate collapses (one expert dominates) | Load balance loss + per-expert gradient monitoring |
| ECO conditioning hurts middlegame (ECO becomes stale after move 15) | Neutral "no-ECO" token after move 15 threshold |
| Syzygy probe adds latency to coach (called per position) | Cache last probe result; only probe when piece count changes |
| Expert heads overfit their concept subsets | Shared spatial encoder; cross-concept regularization via load-balance loss |
| Phase 6C recommendations contradict SF | Alignment constraint: SF move always wins; concept explanation follows it |
