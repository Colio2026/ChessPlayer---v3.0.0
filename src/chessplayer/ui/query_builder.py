from __future__ import annotations
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QLineEdit, QLabel
from pgn.query import Clause, Query

FIELDS = [
    ("White", "white"),
    ("Black", "black"),
    ("Event", "event"),
    ("Site", "site"),
    ("Date", "date"),
    ("ECO", "eco"),
    ("Opening", "opening"),
    ("Result", "result"),
]
OPS = [("contains","contains"),("equals","equals"),("starts with","starts_with")]

class _ClauseRow(QWidget):
    changed = Signal()
    removed = Signal(object)
    def __init__(self) -> None:
        super().__init__()
        l = QHBoxLayout(self); l.setContentsMargins(0,0,0,0)
        self.field = QComboBox(); [self.field.addItem(lbl, key) for lbl,key in FIELDS]
        self.op = QComboBox(); [self.op.addItem(lbl, key) for lbl,key in OPS]
        self.value = QLineEdit(); self.value.setPlaceholderText("value…")
        self.value.textChanged.connect(lambda _: self.changed.emit())
        self.field.currentIndexChanged.connect(lambda _: self.changed.emit())
        self.op.currentIndexChanged.connect(lambda _: self.changed.emit())
        rm = QPushButton("✕"); rm.setFixedWidth(34); rm.clicked.connect(lambda: self.removed.emit(self))
        l.addWidget(self.field); l.addWidget(self.op); l.addWidget(self.value, 1); l.addWidget(rm)
    def to_clause(self) -> Clause | None:
        v = self.value.text().strip()
        if not v: return None
        return Clause(field=self.field.currentData(), op=self.op.currentData(), value=v)

class QueryBuilder(QWidget):
    query_changed = Signal(object)
    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0)
        header = QHBoxLayout(); header.addWidget(QLabel("Query"))
        add_btn = QPushButton("Add clause"); header.addWidget(add_btn); header.addStretch(1)
        outer.addLayout(header)
        self._rows_layout = QVBoxLayout(); outer.addLayout(self._rows_layout)
        self._rows: list[_ClauseRow] = []
        add_btn.clicked.connect(self.add_clause)
        self.add_clause()
    def add_clause(self) -> None:
        r = _ClauseRow(); r.changed.connect(self._emit); r.removed.connect(self._remove_row)
        self._rows.append(r); self._rows_layout.addWidget(r); self._emit()
    def _remove_row(self, r: _ClauseRow) -> None:
        self._rows.remove(r); r.setParent(None); r.deleteLater(); self._emit()
    def current_query(self) -> Query:
        clauses = []
        for r in self._rows:
            c = r.to_clause()
            if c: clauses.append(c)
        return Query(clauses=tuple(clauses))
    def _emit(self) -> None:
        self.query_changed.emit(self.current_query())
