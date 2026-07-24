#!/usr/bin/env python3
"""Game consistency check for the chess concept coach.

Replays a game through coach.analyze() move by move and reports:
  - When each concept activates and deactivates
  - Whether transitions happened on meaningful moves or "quiet" moves
  - A ply-by-ply timeline of the coach's concept state

The goal: verify that the Schmitt-trigger hysteresis is producing stable,
move-justified concept transitions rather than flickering on quiet moves.

What to look for
----------------
GOOD: Concept activates immediately after a capture that creates the structure
      supporting it (e.g., "isolated_pawn" fires after an exchange that isolates).

SUSPICIOUS: Concept activates or deactivates on a quiet pawn move or a piece
            shuffle that doesn't change the position's core structure.

VERY SUSPICIOUS: Concept alternates on consecutive quiet moves (flickering).
                 This suggests the hysteresis thresholds need adjustment for
                 this concept, or the model is boundary-sensitive at this threshold.

Usage
-----
    # Replay a game from UCI moves
    python tools/check_consistency.py --uci "e2e4 e7e5 g1f3 b8c6 f1b5"

    # Replay a specific game from a PGN file (first game by default)
    python tools/check_consistency.py --pgn-file data/mygame.pgn

    # Show only transitions, suppress unchanged-ply output
    python tools/check_consistency.py --uci "..." --transitions-only

    # Use raw model probabilities (no hysteresis) to see what the model actually thinks
    python tools/check_consistency.py --uci "..." --no-hysteresis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import chess.pgn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _classify_move(board: chess.Board, move: chess.Move) -> str:
    """Return a short tag describing the character of a move."""
    tags = []
    if board.is_capture(move):
        cap_type = board.piece_type_at(move.to_square)
        mover_type = board.piece_type_at(move.from_square)
        if cap_type and mover_type:
            # Rough exchange: lower-value captures higher-value
            VALUES = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 99}
            cap_val   = VALUES.get(cap_type,   0)
            mover_val = VALUES.get(mover_type, 0)
            if cap_val > mover_val:
                tags.append("WIN-CAP")
            elif cap_val == mover_val:
                tags.append("EXCHANGE")
            else:
                tags.append("SACRIFICE")
        else:
            tags.append("CAPTURE")
    if board.gives_check(move):
        tags.append("CHECK")
    piece = board.piece_type_at(move.from_square)
    if piece == chess.PAWN:
        from_rank = chess.square_rank(move.from_square)
        to_rank   = chess.square_rank(move.to_square)
        if abs(to_rank - from_rank) > 1 or to_rank in (0, 7):
            tags.append("PAWN-PUSH")
        else:
            tags.append("PAWN")
    if not tags:
        tags.append("quiet")
    return "+".join(tags)


def _uci_to_san(board: chess.Board, uci: str) -> str:
    try:
        return board.san(chess.Move.from_uci(uci))
    except Exception:
        return uci


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a game through the coach and report concept consistency."
    )
    parser.add_argument("--uci",          default="",
                        help="Space-separated UCI moves from starting position")
    parser.add_argument("--pgn-file",     default="",
                        help="PGN file to read a game from")
    parser.add_argument("--game-index",   type=int, default=0,
                        help="Which game in the PGN file (0-indexed, default: 0)")
    parser.add_argument("--start-fen",    default=chess.STARTING_FEN,
                        help="Starting FEN (default: standard start)")
    parser.add_argument("--max-ply",      type=int, default=80,
                        help="Maximum plies to replay (default: 80)")
    parser.add_argument("--checkpoint",   default="data/classifier_best.pt",
                        help="Checkpoint to load (default: data/classifier_best.pt)")
    parser.add_argument("--transitions-only", action="store_true",
                        help="Only print plies where the concept set changed")
    parser.add_argument("--no-hysteresis",    action="store_true",
                        help="Use calibrated thresholds only (no Schmitt-trigger)")
    parser.add_argument("--threshold",        type=float, default=None,
                        help="Override global threshold for --no-hysteresis mode")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}\nRun training first.")

    # --- Parse moves ---
    history_uci: list[str] = []

    if args.pgn_file:
        pgn_path = Path(args.pgn_file)
        if not pgn_path.exists():
            sys.exit(f"PGN file not found: {pgn_path}")
        with open(pgn_path, encoding="utf-8", errors="replace") as fh:
            game = None
            for i in range(args.game_index + 1):
                game = chess.pgn.read_game(fh)
                if game is None:
                    sys.exit(f"Game index {args.game_index} not found in {pgn_path} "
                             f"(file has fewer games).")
        for mv in game.mainline_moves():
            history_uci.append(mv.uci())
        print(f"Loaded game from {pgn_path.name} (index {args.game_index})")
        if game.headers.get("White"):
            print(f"  {game.headers.get('White', '?')} vs {game.headers.get('Black', '?')}  "
                  f"{game.headers.get('Result', '')}")
    elif args.uci:
        history_uci = args.uci.strip().split()
        print(f"Loaded {len(history_uci)} UCI moves from --uci")
    else:
        parser.print_help()
        sys.exit("\nProvide --uci or --pgn-file.")

    history_uci = history_uci[:args.max_ply]
    start_fen   = args.start_fen

    # --- Load coach ---
    from src.chess_coach.rag.coach import ChessCoach

    use_hyst = not args.no_hysteresis
    print(f"\nLoading coach from {args.checkpoint} (hysteresis={'ON' if use_hyst else 'OFF'}) ...")
    coach = ChessCoach(
        ckpt_path=args.checkpoint,
        use_hysteresis=use_hyst,
    )
    # Force-load model
    coach._ensure_loaded()
    coach.reset()

    # --- Replay ---
    board = chess.Board(start_fen)
    prev_concepts: set[str] = set()

    print(f"\nReplaying {len(history_uci)} plies ...\n")

    # Header
    print(f"{'Ply':>4}  {'Move':>7}  {'Event':>12}  Concepts  (+ activated  - dropped)")
    print("-" * 78)

    suspicious_count = 0
    total_transitions = 0

    for ply_idx, uci in enumerate(history_uci):
        # Classify move BEFORE pushing it
        try:
            move = chess.Move.from_uci(uci)
            san  = _uci_to_san(board, uci)
            tag  = _classify_move(board, move)
        except Exception:
            san = uci
            tag = "?"
        board.push(chess.Move.from_uci(uci))
        current_fen = board.fen()

        result = coach.analyze(
            fen         = current_fen,
            history_uci = history_uci[:ply_idx + 1],
            start_fen   = start_fen,
        )

        current_concepts = {name for name, _ in result["concepts"]}
        activated = current_concepts - prev_concepts
        dropped   = prev_concepts - current_concepts
        changed   = bool(activated or dropped)

        if changed:
            total_transitions += len(activated) + len(dropped)

        if args.transitions_only and not changed:
            prev_concepts = current_concepts
            continue

        # Format output line
        concepts_str = ", ".join(sorted(current_concepts)) if current_concepts else "(none)"
        side_char    = "w" if (ply_idx % 2 == 0) else "b"
        ply_label    = f"{(ply_idx // 2) + 1}{side_char}"

        is_suspicious = changed and tag == "quiet"
        flag = " (!)" if is_suspicious else ""

        print(f"{ply_label:>4}  {san:>7}  {tag:>12}  {concepts_str}{flag}")

        if activated:
            for c in sorted(activated):
                prob = next((p for n, p in result["concepts"] if n == c), 0.0)
                print(f"{'':>26}  + {c}  (p={prob:.3f})")
        if dropped:
            for c in sorted(dropped):
                print(f"{'':>26}  - {c}")

        if is_suspicious:
            suspicious_count += 1

        prev_concepts = current_concepts

    # --- Summary ---
    print("-" * 78)
    print(f"\nSummary")
    print(f"  Total plies              : {len(history_uci)}")
    print(f"  Total concept transitions: {total_transitions}")
    print(f"  Concepts active at end   : {len(prev_concepts)}")
    print(f"  Suspicious transitions   : {suspicious_count}  (concept changed on quiet move)")

    if suspicious_count > 0:
        frac = suspicious_count / max(total_transitions, 1)
        print(f"  Suspicion rate           : {frac:.0%} of transitions on quiet moves")
        if frac > 0.30:
            print("  (!)  High suspicion rate. Check hysteresis thresholds for the concepts above.")
        else:
            print("  OK  Most transitions correlated with board events.")
    else:
        print("  OK  No suspicious transitions detected.")

    print(f"\n  Final position (after ply {len(history_uci)}):")
    print(f"  FEN: {board.fen()}")
    print(f"  Active concepts: {', '.join(sorted(prev_concepts)) or '(none)'}")


if __name__ == "__main__":
    main()
