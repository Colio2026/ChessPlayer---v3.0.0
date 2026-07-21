# Phase 5 — NNUE Feature Extraction
## Leveraging Stockfish's Pretrained Neural Network as a Perception Layer

---

## ⛔ Status: ABANDONED — see commit 9b12f62

Three training runs (Phase 5, 5C, 5D) validated this plan against real data. All three stalled at Macro F1 ~0.33, versus 0.56 for Phase 4B without NNUE.

**Root cause:** NNUE Feature Transformer representations are organized around centipawn evaluation ("how good is this position"), not concept identity ("what pattern is present"). This is task misalignment — the pre-training labels don't correlate with our target labels. Section §7 of this document confidently claims `algo_feature_vector_v4` would be "retired" by NNUE. The opposite happened: algo_v4 *is* the primary signal source and NNUE is the one that was retired.

**What was learned:** Pre-trained representations must be evaluated for task alignment before being adopted as features. The hypothesis in §2 ("the NNUE has already done the hard work") was incorrect — NNUE's hard work was learning to evaluate, which is a different task from concept classification.

**Current direction:** Phase 4B with algo_v4 spatial features + GRU history + Schmitt-trigger hysteresis in the coach layer. NNUE's correct future role is post-classification gating (validate concept relevance given evaluation), not a training input.

This document is preserved for reference. Do not implement anything described here without re-validating the task alignment assumption first.

---

## 1. The Core Idea

Stockfish's NNUE is not one monolithic network — it has two distinct stages:

```
Stage 1 — Feature Transformer  (the part we adopt)
  Input:  ~40,000 sparse binary features (king-relative piece positions)
  Output: 1024 activations across two king perspectives (white + black)
  Weight: ~10MB, bundled with the SF binary as a .nnue file
  Trained: hundreds of billions of positions

Stage 2 — Eval Head  (we discard this)
  Input:  the 1024 activations
  Output: 1 centipawn score
```

We freeze Stage 1 permanently and replace Stage 2 with our own
53-class concept classifier. This is transfer learning from a
superhuman engine — the Feature Transformer becomes our perception layer.

This is fundamentally different from the 7 SF classical eval terms we
already cache. Those are SF's final *judgment*. The 1024 activations
are SF's internal *understanding* — the compressed representation it
built before it collapsed everything into a number.

---

## 2. Why This Changes Everything

Our hand-crafted 1811-dim algo_feature_vector_v4 is a set of heuristics
we designed. The NNUE Feature Transformer is what Stockfish discovered
by training on billions of positions. The difference:

| Concept | Our heuristic | NNUE equivalent |
|---|---|---|
| Piece activity | mobility count > 1.35× opponent | Learned from millions of "active piece" positions |
| King safety | 3 pawn shelter squares | Learned from millions of mating attack patterns |
| Space | pawn advancement ratio | Learned from millions of squeeze/fortress games |
| Passed pawn (dynamic) | static definition only | Learned candidate passer value from billions of games |
| Initiative | no signal | Tempo and forcing continuations implicit in activations |
| Zugzwang | ≤5 legal moves heuristic | Learned positions where passing move is catastrophic |

The NNUE has already done the hard work. Our job becomes
mapping its representation to human chess vocabulary.

---

## 3. Architecture Change

### Current (Phase 4-B)
```
x = [board(1001), move(128), algo_v4(1811), v3_summary(59), sf_classical(14)]
      = 3013 dims total

Spatial bottleneck:  algo_v4(1811) → Linear → ReLU → proj(256)
GRU output:          history_rich → 256
Combined:            1001+128+256+59+14+256 = 1714 dims into head
Head:                Linear(1714→1024) → BN → ReLU → Linear(1024→512) → BN → ReLU → Linear(512→53)
```

### Phase 5
```
x = [nnue(1024), board_meta(13), move(128), sf_classical(14), v3_summary(59)]
      = 1238 dims total

No spatial bottleneck — NNUE is already compressed
GRU output:          history_rich → 256
Combined:            1238 + 256 = 1494 dims into head
Head:                Linear(1494→512) → BN → ReLU → Linear(512→256) → BN → ReLU → Linear(256→53)
                     [smaller head — input quality is much higher]
```

**board_meta(13):** side to move(1) + castling rights(4) + en-passant file(8)
These aren't captured by king-relative NNUE encoding so they stay.

**algo_feature_vector_v4 (1811 dims) is fully retired.**
The NNUE covers everything it approximated, better.

**v3_summary (59 dims) stays** — these are binary concept bits that
directly match what labels measure. They're cheap to compute and
complement the NNUE's continuous representation.

**sf_classical (14 dims) stays** — explicit eval term signals for
Passed, King safety, Space, etc. Cross-validates the NNUE signal
for the concepts we care most about.

---

## 4. Expected Performance

### Macro F1 trajectory

| Phase | Architecture | Macro F1 |
|---|---|---|
| Phase 3 | MLP, 1188-dim | 0.4733 |
| Phase 4-B | MLP+GRU, 1811-dim spatial | 0.5614 |
| Phase 5 (target) | NNUE+GRU, frozen perception | **0.65–0.72** |

### Biggest expected gains

| Concept | Phase 4 | Phase 5 estimate | Reason |
|---|---|---|---|
| `piece_activity` | weak | strong | NNUE knows real piece coordination |
| `initiative` | very weak | moderate | Tempo encoded in activations |
| `space_advantage` | weak | strong | NNUE trained on squeeze patterns |
| `development_lead` | weak | moderate | NNUE knows undeveloped positions |
| `king_safety` | moderate | strong | NNUE trained on king attack patterns |
| `passed_pawn` (dynamic) | weak | strong | NNUE values candidate passers |
| `zugzwang` | very weak | moderate | NNUE most useful signal available |
| `fork`, `pin`, `skewer` | strong | unchanged | Our deterministic detectors already correct |

### Training time

- **Cache build:** one-time cost ~15–30 min (pure matrix multiplication on CPU)
  - `nnue_cache.npy`: 1.4M × 1024 float32 ≈ 5.4 GB
- **Per epoch:** similar or faster (smaller model, no bottleneck projection)
- **Epochs to convergence:** fewer — input quality is far higher so the
  model doesn't need to discover patterns from noisy spatial features
- **Net result:** shorter total training time for better results

---

## 5. NNUE File Format

SF ships a `.nnue` file alongside the binary (filename like `nn-xxxxxxxx.nnue`).
The format is documented in the SF source under `src/nnue/`.

### HalfKAv2 input encoding

For each position:
- White perspective: for each piece on the board, compute feature index
  as `(king_bucket × 641 + piece_type × 64 + square)`
- Black perspective: same, mirrored
- ~32–38 features are active per perspective for any real position (very sparse)
- Weight matrix: 40960 × 1024 (float16 in the file, cast to float32)

### Forward pass (one perspective)
```python
# Accumulate active feature rows
accumulator = bias.copy()
for feature_idx in active_features:
    accumulator += weights[feature_idx]

# Clipped ReLU activation (SF uses CReLU or SCReLU depending on version)
output = np.clip(accumulator, 0, 1)   # simplified — exact activation varies by SF version
```

Both perspectives are concatenated: `output = cat([white_view(512), black_view(512)])` = 1024 dims.

---

## 6. Implementation Plan

### Step 1 — Parse the .nnue file

New file: `tools/nnue_reader.py`

- Locate the `.nnue` file (auto-detect from SF binary path or `NNUE_PATH` env var)
- Parse the binary header (magic bytes, version, architecture type)
- Extract Feature Transformer weights and biases
- Handle both old (HalfKP) and current (HalfKAv2) architectures
- Validate by checking that a known FEN produces expected activations

Reference: `official-stockfish/nnue-pytorch` on GitHub has Python parsing code.

### Step 2 — Implement the feature transformer

In `tools/nnue_reader.py`:

```python
def nnue_feature_transformer(fen: str, weights, biases) -> np.ndarray:
    """
    Run the NNUE Feature Transformer for a single FEN.
    Returns shape (1024,) float32 — two 512-dim king perspectives concatenated.
    """
    board = chess.Board(fen)
    # ... HalfKAv2 feature indexing
    # ... accumulate active feature rows
    # ... apply activation
    # ... concatenate white + black perspectives
    return activations   # (1024,)
```

### Step 3 — Build the cache

New file: `tools/build_nnue_cache.py`

- Load NNUE weights once at startup
- Vectorize: process positions in batches of 1000 using NumPy broadcasting
- Read training_raw.jsonl (post-stripped, has `_ac` indices)
- Write `data/nnue_cache.npy` shape (N, 1024) at index `_ac`
- Target runtime: 15–30 minutes on CPU, faster on GPU via PyTorch

### Step 4 — Update board_encoder.py

```python
NNUE_SIZE        = 1024
BOARD_META_SIZE  = 13    # side_to_move(1) + castling(4) + ep_file(8)
# ALGO_SIZE_V4 retired — no longer in active path
STATIC_SIZE_V5   = NNUE_SIZE + BOARD_META_SIZE + MOVE_SIZE + SF_SIZE
                 # = 1024 + 13 + 128 + 14 = 1179
COMBINED_SIZE_V5 = STATIC_SIZE_V5 + ALGO_SIZE + GRU_HIDDEN
                 # = 1179 + 59 + 256 = 1494
```

Add `board_meta_tensor(fen)` — encodes only the 13 non-piece FEN features.

### Step 5 — Update dataset.py

- Add `_nnue_cache_path` / `_nnue_cache` lazy mmap (same pattern as algo/v3/sf)
- In `__getitem__`:
  ```python
  nnue_t     = nnue_cache[ac_idx]   # (1024,)
  board_meta = board_meta_tensor(fen)  # (13,)
  x = cat([nnue_t, board_meta, move_t, sf_t, v3_t])
  ```

### Step 6 — Update classifier.py

- Remove `spatial_proj` (the 1811→256 bottleneck — no longer needed)
- No split/slice needed — x is a single flat vector going straight to the head
- New Phase 5 head:
  ```python
  if phase5:
      input_size = COMBINED_SIZE_V5   # 1494
      hidden1    = 512
      hidden2    = 256
      dropout    = 0.40
      dropout2   = 0.20
  ```
- `predict_concepts()`: call `nnue_feature_transformer(fen)` for the 1024 dims

### Step 7 — Update retrain_and_reparse.ps1

Add step 3d before training:
```powershell
Step "Building NNUE feature cache"
python tools/build_nnue_cache.py
if (-not $?) { Write-Error "NNUE cache build failed"; exit 1 }
```

### Step 8 — Retrain with `--phase5` flag

```powershell
python -m src.chess_coach.ml.train --phase5
```

---

## 7. What Gets Retired

| Component | Status after Phase 5 |
|---|---|
| `algo_feature_vector_v4()` (1811 dims) | Retired — NNUE replaces it |
| `build_algo_cache.py` spatial phase | Retired — only v3 summary retained |
| `spatial_proj` bottleneck in classifier | Removed |
| `algo_cache.npy` (10 GB) | Can be deleted after transition |
| B1-B7 feature vectors (weak_sq, outpost, etc.) | Retired — NNUE covers these |

**What stays:**
- `algo_feature_vector()` — the 59-dim v3 summary bits (cheap, direct label match)
- `sf_cache.npy` — the 14 SF classical eval terms
- `v3_cache.npy` — rebuilt from the 59-dim function
- `label_positions.py` silver labelers — still used for training labels
- All parse/ingest pipeline tooling

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| .nnue file format changes between SF versions | Pin to a specific .nnue file; store hash |
| HalfKAv2 vs HalfKP architecture mismatch | Detect from file header; support both |
| 5.4 GB nnue_cache.npy disk requirement | One-time; can delete algo_cache.npy (10 GB) first — net saving |
| NNUE activations are scale-sensitive | Normalize to [0, 1] or apply BN before the head |
| Training data quality still limits ceiling | Phase 5 raises the ceiling; label quality (Phase 4 labeling improvements) fills it |

---

## 9. Success Criteria

- Macro F1 ≥ 0.65 on calibrated test set
- `piece_activity`, `initiative`, `space_advantage` F1 each ≥ 0.45
- `drawn_position`, `zugzwang` F1 each ≥ 0.35
- All previous PASS spot checks still pass
- Coach panel receives more accurate concept signals with fewer false positives
  (`clearance` and `bishop_pair` over-firing issues resolved by better representation)

---

## 10. Connection to the Full Vision

```
Layer 1 — Perception  (Phase 5)
  Frozen NNUE Feature Transformer
  + board meta + move + SF classical + v3 summary
  = the richest possible position representation

Layer 2 — Understanding  (current work, supercharged by Phase 5)
  Trained 53-class concept classifier
  = reliable chess concept detection

Layer 3 — Language  (future)
  Concept labels → Nimzowitsch-voice prose
  via phrase base (near term) + fine-tuned LLM (long term)
```

Phase 5 is the prerequisite that makes Layer 3 worth building.
Right now the concept labels are noisy enough that the phrase base
fires incorrectly too often. With NNUE backing the perception layer,
when `piece_activity` fires, it is real. When `passed_pawn` fires,
it is real. The language layer can be trusted.
