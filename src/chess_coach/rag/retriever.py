# retriever.py
# RAG retriever for chess coaching: ECO-keyed annotation lookup + FEN similarity ranking.
#
# Data dependencies (built by tools/build_rag_index.py):
#   data/eco_db.json        — normalized-FEN → {eco, opening, variation, depth}
#   data/rag_index.jsonl    — annotated positions with genuine human commentary
#
# Retrieval strategy (no embeddings required):
#   1. Identify opening ECO code by walking game history FENs against eco_db
#   2. Collect candidates: exact ECO match → same-prefix → same-letter group → all
#   3. Rank candidates by piece-placement similarity (Jaccard on piece-square pairs)
#   4. Boost records whose concept themes overlap the current position's detected concepts

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

_ECO_DB_PATH  = Path("data/eco_db.json")
_RAG_IDX_PATH = Path("data/rag_index.jsonl")


def _norm_fen(fen: str) -> str:
    """First 4 FEN fields only (strip halfmove / fullmove counters)."""
    return " ".join(fen.split()[:4])


def _fen_pieces(fen: str) -> frozenset:
    """Return frozenset of (piece_char, square_index) from FEN piece placement."""
    placement = fen.split()[0]
    pieces: list[tuple[str, int]] = []
    sq = 0
    for ch in placement:
        if ch == "/":
            continue
        elif ch.isdigit():
            sq += int(ch)
        else:
            pieces.append((ch, sq))
            sq += 1
    return frozenset(pieces)


def _fen_similarity(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity between two piece-placement sets."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class RAGRetriever:
    """Load-once, query-many RAG retriever for chess position commentary.

    Parameters
    ----------
    eco_db_path  : path to data/eco_db.json  (built by build_rag_index.py)
    rag_idx_path : path to data/rag_index.jsonl
    """

    def __init__(
        self,
        eco_db_path:  str | Path = _ECO_DB_PATH,
        rag_idx_path: str | Path = _RAG_IDX_PATH,
    ) -> None:
        eco_db_path  = Path(eco_db_path)
        rag_idx_path = Path(rag_idx_path)

        if not eco_db_path.exists():
            raise FileNotFoundError(
                f"{eco_db_path} not found — run: python tools/build_rag_index.py"
            )
        if not rag_idx_path.exists():
            raise FileNotFoundError(
                f"{rag_idx_path} not found — run: python tools/build_rag_index.py"
            )

        with open(eco_db_path, encoding="utf-8") as f:
            self._eco_db: dict[str, dict] = json.load(f)

        # Group records by ECO code for fast lookup
        self._by_eco: dict[str, list[dict]] = defaultdict(list)
        self._all: list[dict] = []

        with open(rag_idx_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    eco = ex.get("eco") or ""
                    self._by_eco[eco].append(ex)
                    self._all.append(ex)
                except Exception:
                    pass

        # Pre-cache piece sets for every record to avoid recomputation
        for ex in self._all:
            ex["_pieces"] = _fen_pieces(ex.get("fen", ""))

    # ── Opening identification ────────────────────────────────────────────────

    def identify_opening(self, history_fens: list[str]) -> dict | None:
        """Walk history FENs backwards; return the deepest ECO match found.

        Parameters
        ----------
        history_fens : list of FEN strings, earliest first, including the
                       current position as the last element.
        """
        best: dict | None = None
        for fen in history_fens:
            norm  = _norm_fen(fen)
            entry = self._eco_db.get(norm)
            if entry and (best is None or entry["depth"] > best["depth"]):
                best = entry
        return best

    # ── Main retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self,
        fen:           str,
        history_fens:  list[str]       | None = None,
        concepts:      list[str]       | None = None,
        eco_override:  str             | None = None,
        n:             int             = 5,
        phase_filter:  str             | None = None,
    ) -> list[dict]:
        """Return up to n annotated positions relevant to the given FEN.

        Parameters
        ----------
        fen           : query FEN (current position)
        history_fens  : ordered list of FENs from game start → current position.
                        Used to identify opening ECO via eco_db.
        concepts      : concept labels detected by Phase 5 classifier (used for boost)
        eco_override  : ECO code to use directly (skips history walk)
        n             : number of results to return
        phase_filter  : if set, only return records matching this phase
                        ('opening', 'middlegame', 'endgame')
        """
        # Determine ECO code for candidate selection
        eco_info: dict | None = None
        if eco_override:
            eco_info = {"eco": eco_override, "opening": "", "variation": ""}
        elif history_fens:
            eco_info = self.identify_opening(history_fens)

        eco_code = eco_info["eco"] if eco_info else ""

        # Collect candidates by widening ECO scope until we have enough
        candidates: list[dict] = []

        if eco_code:
            # Tier 1: exact ECO match
            candidates = list(self._by_eco.get(eco_code, []))

            # Tier 2: same opening group (e.g. E32 → all E3x)
            if len(candidates) < n * 4:
                prefix2 = eco_code[:2]
                for k, v in self._by_eco.items():
                    if k != eco_code and k.startswith(prefix2):
                        candidates.extend(v)

            # Tier 3: same ECO letter (e.g. E → all E codes)
            if len(candidates) < n * 2:
                prefix1 = eco_code[:1]
                for k, v in self._by_eco.items():
                    if not k.startswith(prefix2) and k.startswith(prefix1):
                        candidates.extend(v)

        if not candidates:
            candidates = self._all

        # Optional phase filter
        if phase_filter:
            filtered = [c for c in candidates if c.get("phase") == phase_filter]
            if filtered:
                candidates = filtered

        # Rank by piece-placement similarity
        query_pieces = _fen_pieces(fen)
        concept_set  = set(concepts) if concepts else set()

        def _score(record: dict) -> float:
            sim    = _fen_similarity(query_pieces, record["_pieces"])
            boost  = 0.15 * len(concept_set & set(record.get("themes", [])))
            return sim + boost

        candidates.sort(key=_score, reverse=True)

        # Strip the internal _pieces cache before returning
        results = []
        for rec in candidates[:n]:
            out = {k: v for k, v in rec.items() if k != "_pieces"}
            out["_opening"] = eco_info  # attach resolved opening info
            results.append(out)

        return results

    # ── Convenience ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Summary statistics for the loaded index."""
        return {
            "total_records":   len(self._all),
            "eco_groups":      len(self._by_eco),
            "eco_db_positions": len(self._eco_db),
        }
