# concept_vocab.py
# Stable ordered list of all chess concept labels the classifier predicts.
# ORDER MATTERS — index position defines the output neuron for each concept.
# Add new concepts at the END only, or saved checkpoints become invalid.

CONCEPTS: list[str] = [
    # Tactical
    "pin",
    "fork",
    "skewer",
    "discovered_attack",
    "deflection",
    "decoy",
    "overloading",
    "zwischenzug",
    "interference",
    "clearance",
    "back_rank",
    "sacrifice",
    "exchange_sacrifice",
    "combination",
    "mating_attack",
    "trapped_piece",
    # Piece concepts
    "outpost",
    "blockade",
    "bad_bishop",
    "good_bishop",
    "bishop_pair",
    "piece_activity",
    "overprotection",
    "battery",
    "rook_seventh",
    "rook_open_file",
    # Pawn structure
    "passed_pawn",
    "isolated_pawn",
    "backward_pawn",
    "doubled_pawn",
    "pawn_majority",
    "pawn_chain",
    "pawn_break",
    "pawn_storm",
    "pawn_weakness",
    "pawn_island",
    # King & squares
    "king_safety",
    "king_activity",
    "weak_square",
    "open_file",
    # Strategic
    "space_advantage",
    "initiative",
    "tempo",
    "zugzwang",
    "prophylaxis",
    "minority_attack",
    "simplification",
    "positional_sacrifice",
    "fortification",
    "coordination",
    "color_complex",
    "endgame_technique",
    "opposition",
    # Added in keyword pass 2
    "counterplay",
    "development_lead",
    "attacking_chances",
    "square_control",
]

NUM_CONCEPTS = len(CONCEPTS)
CONCEPT_TO_IDX = {c: i for i, c in enumerate(CONCEPTS)}
