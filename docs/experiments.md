# Experiment Log

This file is the single source of truth for what was trained, when, why, and what came out.

**How to update:** When you start a run, add a row to the Summary Table with a hypothesis and "in progress" status. When it finishes, fill in the result and outcome. Keep it current — a stale log is indistinguishable from no log.

---

## Current Champion

**Phase 6B (MoE)** — Macro F1 **0.6785** — calibrated on all 49 concepts, 190,037 test examples.  
Checkpoint: `data/classifier_best.pt` (epoch 20, val_loss=0.6272)  
Eval detail: `results/results0067_2026-07-25_1056_eval.txt`  
Inspect: `results/2026-07-25_2138_inspect.txt` | Hysteresis: `results/2026-07-25_2138_hysteresis.txt` | Audit: `results/2026-07-25_2138_audit.txt`

Previous champion: **Phase 4C** — Macro F1 **0.6768** (epoch 30) — `results/results0061_2026-07-24_0901_eval.txt`

---

## Summary Table

| Run | Date | Params | Dataset | Best Macro F1 | Best Epoch | Result file(s) | Outcome |
|---|---|---|---|---|---|---|---|
| Phase 1 (MLP baseline) | pre-2026-07-13 | ~1.2M | ~500K | ~0.30 | ? | (pre-logging) | Proof of concept. 0.30 is too low. |
| Phase 2 (MLP scaled) | pre-2026-07-13 | ~1.2M | ~1M | **0.51** | ? | (pre-logging) | Good for recall, poor precision. Plan: add spatial heuristics + GRU. |
| Phase 4A-α (lazy) | 2026-07-13 | 3,739,442 | 1,024,089 | 0.40 | ~9 | 0001–0013 | 13 short runs. Lazy encoding was the bottleneck — GPU waiting on CPU. Switched to mmap caches. |
| Phase 4A-β (mmap) | 2026-07-14–16 | 3,030,066 | 1,024,422 | ~0.47 | ~21 | 0014–0026 | Cache introduced. Algo dim expanded 1663 → 1811. Many hyperparameter probes. |
| Phase 4B | 2026-07-17 | 3,067,441 | 1,226,955 | **0.5614** | 68 | 0027–0031 | Champion. Patience-10 stopped early. Calibrated thresholds across 49 concepts. |
| Phase 5 (NNUE, mixed) | 2026-07-19 | 1,743,921 | 1,070,507 | 0.4701 | 33 | 0036 | Algo+NNUE caches both present — ambiguous which features dominated. Worse than 4B. |
| Phase 5C (NNUE-only) | 2026-07-20 | 1,748,017 | 1,591,442 | ~0.34 | early | 0039–0043 | Pure NNUE input. Stalled immediately. Task misalignment confirmed (see §Lessons). |
| Phase 5D (NNUE, small) | 2026-07-20 | 1,350,961 | 1,591,442 | ~0.35 | 25 | 0044–0045 | Reduced head size didn't help. NNUE retired as training input. |
| Phase 4B retrain | 2026-07-21 | 3,081,777 | 1,591,442 | 0.5057 | 37 | 0046–0052 | Same arch as champion, +30% more data. Actual result: 0.5057 — did not beat 4B. Parser had Lichess study folder alias bug; folder_concept injection blocked by comment-length gate. |
| Phase 4C prep | 2026-07-22 | N/A (code) | N/A | — | — | — | Detector expansion: 13 new binary detectors, B8 + B9 spatial maps. All 49 concepts now have detectors. v3: 59→82 dims. v4: 1811→3779 dims (+680 B8, +1288 B9). Parser fixed: _FOLDER_ALIASES added for 9 unrecognised Lichess study folder names; comment-length gate bypassed when folder_concept set. SF cache rebuilt with batch subprocess approach (eliminated OSError [Errno 22] on Windows). |
| Phase 4C | 2026-07-24 | 3,609,137 | 1,900,365 | **0.6768** | 30 | 0061, 0062 | **Champion (until Phase 6B).** B8+B9 spatial maps + all 49 v3 detectors + 1.9M dataset. Macro F1 0.6768. Calibrated eval run 0062. Top concepts: queen_endgame(0.952), knight_endgame(0.941), development_lead(0.938). |
| Phase 6B (MoE) | 2026-07-24–25 | 6,900,000 | 1,900,365 | **0.6785** | 20 | 0063–0067 | **Current champion.** MoE: 5 expert heads + GatingNetwork (501-class ECO embedding + 1740-dim combined → 256 → 5 logits). Macro F1 0.6785, val_loss=0.6272. Inspect/hysteresis/audit: results/2026-07-25_21xx_*.txt. |
| Phase 4C | 2026-07-24 | 3,609,137 | 1,900,365 | **0.6768** | 30 | 0061 | **New champion. +34% relative F1 improvement over 4B.** Macro F1 0.6768, Micro F1 0.6601. Top concepts: queen_endgame(0.952), knight_endgame(0.941), development_lead(0.938), space_advantage(0.919), rook_endgame(0.911). Bottom: pawn_island(0.457), mating_attack(0.457), x_ray(0.479). 0/1024 dead neurons in L1. Key insight: many apparent FPs are correct model predictions on unlabeled positions — label coverage gap, not model error. Hysteresis audit flagged initiative (64.6% fire rate at ACTIVATE=0.40) and x_ray (51.1% at 0.43) as needing threshold raises. weak_square ACTIVATE=0.90 almost never reached — needs lowering to ~0.60. |

---

## Run Detail

### Phase 1 — MLP Baseline
**Hypothesis:** A 3-layer MLP on handcrafted features can identify chess concepts well enough to build on.  
**Input:** ~1,188-dim static features (no move history, no spatial heuristics)  
**Result:** Macro F1 ~0.30. Confirms the problem is learnable but the baseline is clearly too weak.  
**Lesson:** Static features alone miss too much context. The model learns which concepts are common, not which patterns trigger them.

---

### Phase 2 — MLP Scaled
**Hypothesis:** A larger MLP with more data will close the gap.  
**Input:** Same feature space, more examples  
**Result:** Macro F1 ~0.51. Good recall on common concepts, but precision is poor — the model over-fires.  
**Lesson:** Scaling helps but hits a ceiling. Precision requires spatial pattern recognition that flat MLPs can't do well. Need move history for dynamics; need structured spatial features for geometry.

---

### Phase 4A-α — Lazy Encoding (13 runs, 2026-07-13)
**Hypothesis:** Phase 2 architecture upgraded to Phase 4 features (algo_v4 spatial heuristics + GRU).  
**Problem discovered:** Lazy per-example encoding on the data loader path — CPU was encoding each FEN during `__getitem__` which is called by 2 workers per batch. GPU sat idle between batches. ~200s/epoch at batch=128.  
**Result:** Macro F1 plateaued ~0.40 but 13 quick consecutive runs suggests debugging, not a deliberate training series.  
**Fix:** Pre-compute all features once → mmap `.npy` caches. Each `__getitem__` becomes a single array row lookup.

---

### Phase 4A-β — Cache Introduction (runs 0014–0026, 2026-07-14–16)
**Hypothesis:** With mmap caches, epoch time drops and we can train to convergence.  
**Key changes:** `algo_cache.npy` (initially 1663-dim, then 1811-dim after algo_v4 expansion), `v3_cache.npy`.  
**Result:** F1 improved toward ~0.47. Epoch time improved. Multiple hyperparameter probes (LR, batch, patience).  
**Lesson:** Algo dimension matters: 1663-dim (incomplete B7) vs. 1811-dim (with `king_safety_vec`) produced noticeably different early-epoch curves.

---

### Phase 4B — Champion (runs 0027–0031, 2026-07-17)
**Hypothesis:** With stable 1811-dim algo_v4 cache, cosine LR, patience=10, and 1.23M examples, the model should push past 0.50.  
**Architecture:** `[board(1001), move(128), algo_v4(1811→256 spatial_proj), v3(59), sf(14)] + GRU(256)` = 1714-dim combined → head  
**Result:** Macro F1 **0.5614** (calibrated). Best epoch 68. Stopped by patience-10 at epoch ~78.  
**Per-class range:** 0.999 (`knight_endgame`) → 0.308 (`mating_attack`).  
**Calibrated thresholds:** 0.60–0.95 for strong-signal concepts; 0.60–0.65 for weak-recall concepts.  
**Stopped by:** Early stopping (patience=10). Val loss was still trending up so the model may not be fully converged.  
**Decision:** Accepted as production model. Phase 5 NNUE experiments run next.

---

### Phase 5 (mixed, run 0036, 2026-07-19)
**Hypothesis:** NNUE Feature Transformer (2048-dim, SF16.1) replaces the 1811-dim spatial heuristics with superhuman learned perception.  
**Dataset:** 1,070,507 examples. Both `algo_cache.npy` AND `nnue_cache.npy` present in the data directory.  
**Ambiguity:** Dataset code loaded all caches present. It is unclear whether this run used NNUE-only or NNUE+algo in the x vector. The 0.4701 result is therefore not a clean signal.  
**Result:** Macro F1 0.4701 at epoch 33. Early stop at epoch 42.  
**Outcome:** Worse than Phase 4B 0.5614 regardless of interpretation. Proceed to clean NNUE-only test.

---

### Phase 5C — NNUE Only (runs 0039–0043, 2026-07-20)
**Hypothesis:** Clean NNUE-only test on the larger 1.59M example dataset. Remove algo_cache to force pure NNUE input.  
**Architecture:** `[nnue(2048), board_meta(13), move(128), sf(14), v3(59)] + GRU(256)` = 2518-dim combined  
**Result:** F1 stuck 0.33–0.34 across multiple runs. Does not improve past epoch 4.  
**Root cause confirmed:** NNUE Feature Transformer activations are organized around centipawn evaluation ("how good is this position"), not chess pattern identity ("what pattern is present"). The pre-training objective is orthogonal to our classification target.  
See: `docs/phase5_nnue_integration_plan.md` (marked ABANDONED).

---

### Phase 5D — Smaller Head (runs 0044–0045, 2026-07-20)
**Hypothesis:** NNUE representations are higher quality but maybe a smaller classifier head generalizes better.  
**Change:** Reduced head size, 1,350,961 total params.  
**Result:** F1 0.33–0.35. No improvement over Phase 5C.  
**Lesson:** The problem is not head capacity — it is that the NNUE input itself does not contain the signal we need. A better head cannot recover information that isn't in the features.

---

### Phase 4B Retrain on 1.59M (runs 0046–0050+, 2026-07-21)
**Hypothesis:** Champion Phase 4B architecture with +30% more training data should push Macro F1 past 0.58.  
**Changes from champion run:** Dataset 1.23M → 1.59M, patience 10 → 20, `num_workers` 2 → 4, SF cache and board cache now added.  
**Result:** Superseded by Phase 4C. This run was abandoned in favour of training the full B8+B9 architecture after the Phase 4C caches were rebuilt.

---

### Phase 4C — Detector Expansion Retrain (runs 0061–0062, 2026-07-24)
**Hypothesis:** Full B8+B9 spatial expansion (v4: 1811→3779 dims, v3: 59→82 dims, 13 new detectors) + 1.9M training examples will push Macro F1 above 0.63, with particular gains on previously geometry-starved concepts (discovery, isolated_pawn, interference, clearance, deflection).  
**Architecture:** 4C same head as 4B; algo_v4 expanded 1811→3779 (+B8 tactical maps +B9 pawn/mating/strategic maps); v3 expanded 59→82 (+13 new per-concept bits). Dataset grew 1.59M → 1.9M.  
**Training:** Patience 20, cosine LR over 100 epochs, BCEWithLogitsLoss + label smoothing ε=0.05. Best epoch: 30.  
**Result:** Macro F1 **0.6768**. Micro F1 0.6601. Top F1: queen_endgame(0.952), knight_endgame(0.941), development_lead(0.938), space_advantage(0.919), rook_endgame(0.911).  
**Bottom:** pawn_island(0.457), mating_attack(0.457), x_ray(0.479).  
**Lesson:** B8+B9 spatial maps delivered the expected gains on geometry-limited concepts. 0 dead neurons in L1. Hysteresis audit flagged initiative (64.6% fire rate at ACTIVATE=0.40) and x_ray (51.1% at 0.43) as needing threshold raises; weak_square ACTIVATE=0.90 almost never reached — needs lowering to ~0.60. The label coverage gap remains the main bottleneck for the bottom quartile: many apparent FPs are correct model predictions on unlabeled positions.  
**Decision:** Accepted as champion. Phase 6B MoE architecture next.

---

### Phase 6B — Mixture of Experts (runs 0063–0067, 2026-07-24–25)
**Hypothesis:** A MoE gating network conditioned on ECO opening code will improve routing of positional vs. tactical concepts and raise Macro F1 above 4C's 0.6768.  
**Architecture:** `MoEConceptClassifier` — 5 expert heads (Tactical/Structural/Pawn/Endgame/Strategic) + `GatingNetwork` (501-class ECO embedding + 1740-dim combined → 256 → 5 logits, softmax). Expert outputs are weighted sum. Total params: ~6.9M (vs. 3.6M for Phase 4C). Input features unchanged from Phase 4C (B8+B9, v3 82-dim).  
**Training:** Same 1.9M dataset as Phase 4C. Cosine LR, patience 20. Best epoch: 20. val_loss: 0.6272.  
**Result:** Macro F1 **0.6785**. New champion by a narrow margin (+0.0017 over 4C).  
**Eval detail:** `results/results0067_2026-07-25_1056_eval.txt`  
**Inspect:** `results/2026-07-25_2138_inspect.txt` | Hysteresis: `results/2026-07-25_2138_hysteresis.txt` | Audit: `results/2026-07-25_2138_audit.txt`  
**Lesson:** ECO conditioning in the gate provides meaningful signal for routing. The MoE improvement is modest (+0.0017) — the primary bottleneck remains label coverage for the bottom quartile, not model capacity. With 6.9M params vs. 3.6M, the architecture has more capacity than the data can fill. Next priority: improve data quality for the 10 weakest concepts before further architecture scaling.  
**Decision:** Accepted as champion. Phase 6 panel integration underway.

---

### Phase 4C Prep — Detector Expansion (2026-07-22, code change only)
**Motivation:** Audit revealed six major concepts (`x_ray`, `discovery`, `double_check`, `zugzwang`, `shouldering`, `mating_attack`) had zero algorithmic labels despite the v4 vector already carrying rich geometry for several of them (e.g. `_xray_vec` is 384 dims in B6). Training data was severely label-sparse for these concepts because `label_position()` never set them. Also, `pin`, `fork`, and `isolated_pawn` had binary-only signals in v3 but no spatial per-square maps in v4 — the net couldn't learn which specific squares were relevant.

**Changes (all in `tools/label_positions.py`):**
- **9 new binary detectors** added to `DETECTABLE_CONCEPTS`, `label_position()`, `_PER_COLOR_DETECTORS`:
  - Wave 1 (Phase 4C initial): `_has_x_ray()`, `_has_discovery()`, `_has_mating_attack()` per-color; `_has_double_check()`, `_has_zugzwang_heuristic()`, `_has_shouldering()` global
  - Wave 2 (Phase 4C extension): `_has_interference()`, `_has_initiative()`, `_has_prophylaxis_pos()`
  - Wave 3 (Phase 4C extension): `_has_sacrifice()` (piece under attack for less value + king-attack proxy), `_has_clearance()` (friendly non-slider blocks slider ray to enemy target), `_has_deflection()` (sole defender of enemy queen/rook that we attack), `_has_zwischenzug()` (losing exchange threatened + check available instead)
- **14 new spatial maps** added to v4 (B8 after B7, then B9):
  - `_pin_vec()` → 256 dims, `_fork_vec()` → 256 dims, `_isolated_pawn_vec()` → 128 dims, `_open_file_vec()` → 40 dims (B8)
  - `_pawn_chain_vec()` → 128 dims, `_pawn_island_vec()` → 130 dims, `_mating_pressure_vec()` → 128 dims (B9 wave 1)
  - `_interference_vec()` → 128 dims, `_initiative_vec()` → 130 dims, `_prophylaxis_vec()` → 130 dims (B9 wave 2)
  - `_sacrifice_vec()` → 130 dims, `_clearance_vec()` → 128 dims, `_deflection_vec()` → 128 dims, `_zwischenzug_vec()` → 128 dims (B9 wave 3)

**Dimension changes (all caches must be rebuilt):**
- `ALGO_FEATURE_SIZE` v3: 59 → 82 (36 per-color × 2 = 72, + 10 global)
- `ALGO_FEATURE_SIZE_V4` v4: 1811 → 3779 (+680 B8 +1288 B9 = +1968 total)
- `COMBINED_SIZE_V4B`: 1714 → 1737 (+23 from v3 expansion; spatial_proj stays 256)
- `SF_BREAK`: 3688 → 4990 (STATIC_SIZE_V4=4908, +ALGO_SIZE=82)

**Files touched:** `tools/label_positions.py`, `src/chess_coach/ml/board_encoder.py`, `src/chess_coach/ml/classifier.py`, `src/chess_coach/ml/dataset.py`, `src/chess_coach/ml/paths.py`, `tools/build_algo_cache.py`

**Next step:** Rebuild `algo_cache.npy` (3779-dim) and `v3_cache.npy` (82-dim), optionally re-scrape synonyms for x_ray and shouldering, then retrain from scratch as Phase 4C. All 49 concepts now have algorithmic detectors. The new spatial maps allow the network to localize which specific square is pinned / forked / an interference gap / a clearance blocker, not just whether the concept exists.

---

### Phase 6 — Panel Integration (2026-07-25 to 2026-07-28, code change only)
**Goal:** Integrate the Phase 6B MoE classifier fully into the live coach panel and fix accumulated bugs discovered during UI testing.

**Files changed:** `src/chess_coach/coach/nimzo_net_engine.py`, `src/chessplayer/ui/coach_panel.py`, `src/chessplayer/ui/coach_board.py`, `src/chessplayer/ui/main_window/window.py`

**Bug fixes:**

*Same RAG annotation on both SF line cards.* `_retrieve_line_rag()` passed the original `history_fens` (identical for both lines) to `_retrieve_one()`, so `identify_opening()` returned the same ECO pool for both. Fixed by building `pv_history` inside the depth-walk loop, appending each intermediate PV FEN before pushing, so each line traces its own ECO trajectory.

*Opening identifier showing wrong move depth.* `window.py` was passing `history=[]` to `queue_analysis()` — the actual UCI move list was computed but never forwarded. `queue_analysis()` now receives `list(prefix)` from `played_prefix_uci()` and rebuilds the board from the UCI history, ensuring `board.move_stack` is always complete regardless of how the position was constructed.

*Weak square dots never appearing.* Two-layer failure: `_extract_key_squares()` had no `weak_square` case (returned `[]` always), AND the entire computation was gated on `if 'weak_square' in concept_map`, so it never ran when the classifier didn't fire. Fixed by decoupling the overlay entirely from the classifier — `get_context()` now always computes weak squares via `_outpost_squares_bb` from `tools/label_positions.py`, the same bitboard logic used to generate training labels.

**New features:**

- Opening bubble now shows the full played SAN line (`opening_moves_san`) + theory continuations (`opening_continuations`) sourced from `eco_db.json` at `depth+1`.
- SF line cards: confidence badge (concept label + calibrated %), Before/After/Δ classical eval table (Term/Before/After/Δ columns), full PV SAN with no move cap.
- RAG minimum annotation length: `_retrieve_one()` fetches n=4 candidates, selects longest ≥200 chars; falls back to longest available if none reach the threshold.
- Weak square overlay: `get_context()` computes `weak_squares` (`_outpost_squares_bb(board, opponent_color) & ~occupied`, max 8) and `weak_squares_opp` (`_outpost_squares_bb(board, player_color) & ~occupied`, max 8) on every call. Emitted via `weakness_per_side_ready` signal → `CoachBoardWidget.set_per_side_weakness()` → QML `squareIndicators` context property. Orange `"weak"` dots = player's structural holes; yellow `"tactic"` dots = outpost targets in opponent's territory.
- Coach board rebuilt: full line display, PGN-editor dark style, legend in title bar, Back/Forward navigation.

**Lesson:** The concept classifier and the board overlay should never have been coupled. The classifier decides *whether* a concept is salient enough to raise to the user's attention; the board overlay should always show *where* the concept manifests geometrically, computed directly from the position. Keeping these separate means the dots are always correct even in positions where the classifier's confidence falls below the ACTIVATE threshold.

---

## Phase 5 Lessons

Task alignment is a prerequisite for transfer learning. Before adopting a pre-trained representation as input features, verify that the pre-training objective correlates with the target task. The NNUE was trained to minimize centipawn prediction error — the internal representations it learned are organized around "what makes this position good or bad," not "what pattern category does this position belong to."

`algo_feature_vector_v4` is explicitly designed around concept geometry: mobility counts, pawn structure metrics, piece coordination — things that directly correlate with concept labels. NNUE covered the same territory implicitly but organized its representation around a different signal entirely.

Future direction: NNUE's correct role is **post-classification gating** — after the classifier fires a concept, confirm that NNUE's evaluation matches what that concept should imply (e.g., `passed_pawn` should show a positive evaluation swing for the side with the passer). This is a filter on the coach output, not a training input.

---

## Evaluation Philosophy

### What macro F1 measures

Macro F1 is the unweighted average of per-class F1 scores. It answers: "across all concept categories equally, how often does the model fire the right labels and not fire the wrong ones?"

This is a necessary metric — it confirms the model can identify patterns at all — but it is not sufficient to confirm the model is a good coach.

### What macro F1 does not measure

**Coaching relevance.** A model that fires `passed_pawn` at probability 0.85 in 60% of middlegame positions has high recall for that concept but is useless as a coach — it's not telling the student anything specific to the position at hand.

**Annotation quality.** The RAG annotation retrieved for a concept may be technically correct (the concept exists) but pedagogically wrong (the concept isn't the most salient thing happening in this position, or the retrieved annotation doesn't match this specific flavor of the concept).

**Consistency.** A coach that fires `king_safety` on move 15 and drops it on move 16 with no piece trades in between is harder to learn from than one that holds a theme for several moves.

### What better evaluation looks like

1. **Calibrated F1 breakdown** (`python -m src.chess_coach.ml.evaluate --calibrate`) — per-class precision/recall/F1 on the test split with calibrated thresholds. The primary post-training metric. Run after every retrain.

2. **Hysteresis coverage check** (`python tools/survey_hysteresis.py`) — samples N positions and reports p50/p75/p90/p95/p99 probability percentiles per concept, plus the fraction that would fire at ACTIVATE=0.65 and remain at HOLD=0.40. Warns if any concept never reaches ACTIVATE or fires in >30% of positions. Run after calibration.

3. **False-positive / false-negative audit** (`python tools/audit_concepts.py`) — runs on the test split, ranks concepts by false-positive rate, and samples actual FEN positions for each misfiring concept. Outputs Lichess analysis links for manual review. Run on the bottom-5 concepts after every retrain to distinguish model errors from label noise.

4. **Game consistency check** (`python tools/check_consistency.py --uci "..."`) — replays a full game through `coach.analyze()` and flags concept transitions that happen on quiet moves (no capture, check, pawn push, or exchange). A high suspicion rate (>30% of transitions on quiet moves) indicates the Schmitt-trigger thresholds need adjustment.

5. **Human spot-checks** — open 10–20 of the Lichess URLs produced by the audit tool, assess: (a) is the concept actually present? (b) is the model's confidence proportional to how obvious the concept is? No tooling needed — this is the check that catches systematic label noise.

None of this requires a large annotated corpus. Even irregular spot-checks are far more informative than watching macro F1 go from 0.56 to 0.58 on a test set that may share distributional assumptions with training.

---

## Upcoming Experiments

| Priority | Hypothesis | Change | Metric to watch |
|---|---|---|---|
| 1 | Hysteresis thresholds need adjustment after Phase 6B | Raise ACTIVATE: initiative→0.75, x_ray→0.65, interference→0.65. Lower weak_square→0.60 (ACTIVATE; overlay now always-on independently) | Fire rate per concept in survey_hysteresis.py |
| 2 | Phase 7A: scraper rerun for data-starved concepts | Re-run Lichess scraper for x_ray (~19K), shouldering (~1.3K), interference (~18K) with synonym expansion | Per-concept F1 for bottom quartile ≥ 0.55 |
| 3 | Label coverage gaps inflate FP rate for bottom-quartile concepts | Hand-label 100–200 FP positions for x_ray, interference, mating_attack | True precision; manual Lichess spot-checks |
| 4 | Phase 6C: SF-validated move recommendation | Post-classification connector: SF top-N moves + concept delta scoring | SF agreement rate ≥ 90%; coach explanation plausible |
| 5 | Phase 7B: per-concept threshold tuning post-6B | Re-run evaluate.py --calibrate on 6B checkpoint; check if calibrated thresholds differ from 4C | Per-class F1 improvement on high-precision/low-recall concepts |
