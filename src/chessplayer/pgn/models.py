from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class GameId:
    value: int

@dataclass
class GameMeta:
    game_id: int
    white: Optional[str]
    black: Optional[str]
    result: Optional[str]
    event: Optional[str]
    site: Optional[str]
    date: Optional[str]
    eco: Optional[str]
    opening: Optional[str]
    offset_bytes: int
