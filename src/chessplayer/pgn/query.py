from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal

Field = Literal["white","black","event","site","date","eco","opening","result"]
Op = Literal["contains","equals","starts_with"]

@dataclass(frozen=True)
class Clause:
    field: Field
    op: Op
    value: str

@dataclass(frozen=True)
class Query:
    clauses: tuple[Clause, ...] = ()

def compile_where(q: Query) -> tuple[str, list[Any]]:
    parts: list[str] = []
    params: list[Any] = []
    for c in q.clauses:
        col = c.field
        if c.op == "contains":
            parts.append(f"{col} LIKE ?"); params.append(f"%{c.value}%")
        elif c.op == "starts_with":
            parts.append(f"{col} LIKE ?"); params.append(f"{c.value}%")
        else:
            parts.append(f"{col} = ?"); params.append(c.value)
    return (" AND ".join(parts), params) if parts else ("1=1", [])

def default_multisort() -> list[tuple[str,str]]:
    return [("date","DESC"),("result","ASC"),("white","ASC")]
