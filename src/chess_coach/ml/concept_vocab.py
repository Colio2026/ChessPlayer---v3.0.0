# concept_vocab.py
# Stable ordered list of all chess concept labels the classifier predicts.
# ORDER MATTERS — index position defines the output neuron for each concept.
# Add new concepts at the END only, or saved checkpoints become invalid.
#
# v3 vocab (50 concepts) — changes from v2 (44):
#   Removed : exchange_sacrifice (→ sacrifice), bishop_quality (→ bad_bishop + good_bishop),
#             pawn_weakness, endgame_technique, color_complex, square_control (→ piece_activity),
#             minority_attack (→ pawn_storm), tempo (→ initiative)
#   Added   : x_ray, double_check, clearance, promotion, shouldering,
#             bad_bishop, good_bishop,
#             rook_endgame, pawn_endgame, bishop_endgame, knight_endgame, queen_endgame,
#             drawn_position, initiative

CONCEPTS: list[str] = [
    # ── Tactical ──────────────────────────────────────────────────────────────
    "pin",
    "fork",
    "skewer",
    "discovery",
    "x_ray",
    "double_check",
    "clearance",
    "deflection",
    "overloading",
    "zwischenzug",
    "interference",
    "back_rank",
    "sacrifice",            # includes exchange sacrifice
    "mating_attack",
    "trapped_piece",
    # ── Piece concepts ────────────────────────────────────────────────────────
    "outpost",
    "blockade",
    "bad_bishop",
    "good_bishop",
    "bishop_pair",
    "piece_activity",       # includes square/center control
    "battery",
    "rook_seventh",
    # ── Pawn structure ────────────────────────────────────────────────────────
    "passed_pawn",
    "promotion",
    "isolated_pawn",
    "backward_pawn",
    "doubled_pawn",
    "pawn_majority",
    "pawn_chain",
    "pawn_storm",           # includes minority attack
    "pawn_island",
    # ── King & endgame ────────────────────────────────────────────────────────
    "king_safety",
    "king_activity",
    "shouldering",
    "opposition",
    "zugzwang",
    "rook_endgame",
    "pawn_endgame",
    "bishop_endgame",
    "knight_endgame",
    "queen_endgame",
    "drawn_position",
    # ── Positional / Strategic ────────────────────────────────────────────────
    "weak_square",
    "open_file",
    "space_advantage",
    "development_lead",
    "initiative",           # renamed from tempo; includes tempo/time-advantage language
    "prophylaxis",
    "attacking_chances",    # includes counterplay
]

NUM_CONCEPTS = len(CONCEPTS)
CONCEPT_TO_IDX = {c: i for i, c in enumerate(CONCEPTS)}
