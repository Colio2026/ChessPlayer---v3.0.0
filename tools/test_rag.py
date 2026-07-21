#!/usr/bin/env python3
"""Quick smoke test for the RAG retriever.

Tests ECO identification and annotation retrieval without loading the classifier.
Run from project root:
    python tools/test_rag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chess_coach.rag.retriever import RAGRetriever

import chess


def main() -> None:
    print("Loading RAG retriever ...")
    r = RAGRetriever()
    s = r.stats()
    print(f"  {s['total_records']:,} records  |  {s['eco_groups']} ECO groups  "
          f"|  {s['eco_db_positions']:,} ECO positions")

    # --- Test 1: opening identification via move replay ---
    print("\n--- Test 1: Nimzo-Indian E32 identification ---")
    board = chess.Board()
    moves = ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"]  # Nimzo-Indian
    history_fens = [board.fen()]
    for uci in moves:
        board.push_uci(uci)
        history_fens.append(board.fen())

    current_fen = history_fens[-1]
    opening = r.identify_opening(history_fens)
    print(f"  FEN: {current_fen}")
    print(f"  Identified: {opening}")

    # --- Test 2: retrieve annotations for a Nimzo position ---
    print("\n--- Test 2: retrieve commentary for Nimzo-Indian ---")
    results = r.retrieve(current_fen, history_fens=history_fens, n=3)
    for i, res in enumerate(results, 1):
        eco  = res.get("eco", "?")
        game = res.get("game", "?")
        move = res.get("move_san", "?")
        ann  = res.get("annotation", "")
        print(f"\n  [{i}] ECO {eco}  {game}  (move: {move})")
        print(f"      {ann[:200]}")

    # --- Test 3: Sicilian Alapin B22 ---
    print("\n--- Test 3: Sicilian Alapin B22 ---")
    board2 = chess.Board()
    moves2 = ["e2e4", "c7c5", "c2c3"]
    fens2  = [board2.fen()]
    for uci in moves2:
        board2.push_uci(uci)
        fens2.append(board2.fen())

    opening2 = r.identify_opening(fens2)
    print(f"  Identified: {opening2}")
    results2 = r.retrieve(fens2[-1], history_fens=fens2, n=2)
    for i, res in enumerate(results2, 1):
        ann = res.get("annotation", "")
        print(f"\n  [{i}] ECO {res.get('eco')}  {res.get('game')}")
        print(f"      {ann[:200]}")

    # --- Test 4: eco_override ---
    print("\n--- Test 4: direct ECO override (E32 Nimzo classical) ---")
    results3 = r.retrieve(current_fen, eco_override="E32", n=2)
    for i, res in enumerate(results3, 1):
        ann = res.get("annotation", "")
        print(f"\n  [{i}] ECO {res.get('eco')}  {res.get('game')}")
        print(f"      {ann[:250]}")


if __name__ == "__main__":
    main()
