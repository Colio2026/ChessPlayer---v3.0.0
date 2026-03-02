import sqlite3
from pathlib import Path
import chess.pgn

SCHEMA = '''
CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY AUTOINCREMENT,
    white TEXT,
    black TEXT,
    result TEXT,
    event TEXT,
    site TEXT,
    date TEXT,
    eco TEXT,
    opening TEXT,
    offset_bytes INTEGER
);
'''

def build_index(pgn_path: Path, db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        offset = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            headers = game.headers
            conn.execute(
                "INSERT INTO games (white, black, result, event, site, date, eco, opening, offset_bytes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    headers.get("White"),
                    headers.get("Black"),
                    headers.get("Result"),
                    headers.get("Event"),
                    headers.get("Site"),
                    headers.get("Date"),
                    headers.get("ECO"),
                    headers.get("Opening"),
                    offset
                )
            )
            offset = f.tell()

    conn.commit()
    conn.close()
