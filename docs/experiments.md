# Experiment Log

This file is the single source of truth for what was trained, when, why, and what came out.

**How to update:** When you start a run, add a row to the Summary Table with a hypothesis and "in progress" status. When it finishes, fill in the result and outcome. Keep it current — a stale log is indistinguishable from no log.

---

## Current Champion

**Phase 4B** — Macro F1 **0.5614** — calibrated on all 49 concepts, 122,696 test examples.  
Checkpoint: `data/classifier_best.pt` (epoch 68, val_loss=0.7060)  
Eval detail: `results/results0031_2026-07-17_1840_eval.txt`

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
| Phase 4B retrain | 2026-07-21 | 3,081,777 | 1,591,442 | in progress | — | 0046–0050 | Same arch as champion, +30% more data. Hypothesis: ≥0.58. |

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
**Status:** In progress as of 2026-07-21.  
**Expected outcome:** F1 ≥ 0.58. The marginal data gain matters most for low-support concepts (`shouldering`, `double_check`, `x_ray`).

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

1. **Hysteresis coverage check** (`tools/survey_hysteresis.py`) — already implemented. Verifies that ACTIVATE_THRESHOLD (0.65) and HOLD_THRESHOLD (0.40) are well-calibrated against the actual probability distribution.

2. **Position spot-checks** — a chess-knowledgeable reviewer plays through 20–30 positions where the coach fires strong signals and evaluates: (a) is the concept present? (b) is the retrieved annotation appropriate? This is the real quality test. Even 20 annotated spot-checks per major architecture change would catch gross failures.

3. **Consistency check** — replay a full game through `coach.analyze()` and verify that concept transitions happen at move boundaries that make sense (after captures, pawn breaks, piece exchanges) rather than on consecutive quiet moves.

4. **False-positive audit** — identify the 5 highest-recall concepts (where the model fires most often) and manually examine 10 positions where they fired at p > 0.80 to estimate precision in the wild, independent of the test set.

None of this requires a large annotated corpus. Even irregular spot-checks are far more informative than watching macro F1 go from 0.56 to 0.58 on a test set that may share distributional assumptions with training.

---

## Upcoming Experiments

| Priority | Hypothesis | Change | Metric to watch |
|---|---|---|---|
| 1 | More data → better recall on weak concepts | Phase 4B retrain on 1.59M (in progress) | Per-class F1 for `shouldering`, `x_ray`, `double_check` |
| 2 | Patience 20 reaches the LR tail | Same as above | Whether best epoch appears after epoch 70 |
| 3 | Human spot-check of champion model | 20 positions, manual review | Qualitative: relevance, annotation match |
| 4 | SF-gated coach output | Post-classification: confirm concept vs. SF eval | Spot-check false-positive rate on `bishop_pair`, `clearance` |
