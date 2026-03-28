"""
coach_board.py
──────────────
Self-contained Coach Board widget.

Shows a read-only chess board displaying a "note line" — a coach-recommended
move sequence embedded in a PGN comment as a [%line] tag.

Features
--------
- Reuses the existing BoardView.qml unchanged (same context-property names)
- Own read-only board model; the main game board is never touched
- Move-label strip: SAN moves shown inline, current one highlighted cyan
- Back / Forward buttons to step through the sequence
- Weakness squares highlighted natively in QML via highlightSquares context property
- Revealed automatically when a note-line link is clicked; hidden when closed

Public API
----------
    widget.load_line(base_fen: str, uci_list: list[str], san_list: list[str])
        Load a new note line starting from base_fen.
    widget.set_weakness_squares(squares: list[str], colour: str)
        Highlight named squares (e.g. ["e4", "d5"]) on the board.
    widget.clear()
        Hide the widget and reset state.
"""

from __future__ import annotations

from pathlib import Path

import chess

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, Qt, QUrl, Signal, Slot
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ── Standalone board model (no PgnEditor dependency) ─────────────────────────

def _piece_code(piece: chess.Piece) -> str:
    color  = "W" if piece.color == chess.WHITE else "B"
    letter = {chess.PAWN:"P", chess.KNIGHT:"N", chess.BISHOP:"B",
               chess.ROOK:"R", chess.QUEEN:"Q",  chess.KING:"K"}
    return color + letter[piece.piece_type]


class _CoachBoardModel(QAbstractListModel):
    FileRole  = Qt.UserRole + 1
    RankRole  = Qt.UserRole + 2
    CodeRole  = Qt.UserRole + 3
    ImageRole = Qt.UserRole + 4

    def __init__(self, pieces_dir: Path) -> None:
        super().__init__()
        self._pieces_dir = pieces_dir
        self._flip       = False
        self._items: list[dict] = []

    def roleNames(self):
        return {
            self.FileRole:  b"file",
            self.RankRole:  b"rank",
            self.CodeRole:  b"code",
            self.ImageRole: b"image",
        }

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        item = self._items[index.row()]
        if role == self.FileRole:  return item["file"]
        if role == self.RankRole:  return item["rank"]
        if role == self.CodeRole:  return item["code"]
        if role == self.ImageRole: return item["image"]
        return None

    def load_board(self, board: chess.Board) -> None:
        items = []
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if not piece:
                continue
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            if self._flip:
                f, r = 7 - f, 7 - r
            code = _piece_code(piece)
            img  = (self._pieces_dir / f"{code}.png").resolve()
            items.append({"file": f, "rank": r, "code": code,
                          "image": img.as_uri()})
        self.beginResetModel()
        self._items = items
        self.endResetModel()


class _CoachBridge(QObject):
    """Read-only bridge — blocks all move attempts silently."""
    promotionRequested = Signal(str)   # never emitted but QML expects it

    @Slot(int, int, int, int)
    def attemptMove(self, *_):
        pass  # coach board is display-only

    @Slot(str)
    def choosePromotion(self, _):
        pass

    @Slot(bool)
    def setFlipped(self, flipped: bool):
        pass


# ── Coach Board Widget ────────────────────────────────────────────────────────

class CoachBoardWidget(QFrame):
    """
    A collapsible panel that shows a coach-recommended move line on a
    read-only board.  Sits below the game archive in the left panel.

    Weakness squares are highlighted natively inside QML via the
    'highlightSquares' context property — no overlay widget needed.

    Parameters
    ----------
    pieces_dir : Path to the piece image directory
    qml_path   : Path to BoardView.qml
    """

    closed = Signal()   # emitted when the user clicks ✕

    def __init__(self, pieces_dir: Path, qml_path: Path,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "CoachBoardWidget { border: 1px solid #3A5A3A; border-radius: 4px;"
            " background: #1A1A1A; }"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._pieces_dir = pieces_dir
        self._qml_path   = qml_path

        # State
        self._boards:   list[chess.Board] = []   # boards[0] = start, boards[N] = after N moves
        self._san_list: list[str]         = []
        self._idx       = 0                       # current board index

        # ── board model + bridge ──────────────────────────────────────────────
        self._model  = _CoachBoardModel(pieces_dir)
        self._bridge = _CoachBridge()

        # ── layout ────────────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Title bar
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_lbl = QLabel("♟  Coach Line")
        title_lbl.setStyleSheet("color:#4FC3F7; font-weight:bold;")
        title_row.addWidget(title_lbl)
        title_row.addStretch(1)
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setFlat(True)
        close_btn.setStyleSheet("color:#888; font-size:11px;")
        close_btn.clicked.connect(self._on_close)
        title_row.addWidget(close_btn)
        root.addLayout(title_row)

        # Move-label strip (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(30)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background:transparent; border:none;")
        self._move_strip_widget = QWidget()
        self._move_strip_layout = QHBoxLayout(self._move_strip_widget)
        self._move_strip_layout.setContentsMargins(0, 0, 0, 0)
        self._move_strip_layout.setSpacing(4)
        scroll.setWidget(self._move_strip_widget)
        root.addWidget(scroll)

        # QQuickWidget — board view with QML-native highlights
        self._board_view = QQuickWidget()
        self._board_view.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self._board_view.rootContext().setContextProperty("piecesModel",       self._model)
        self._board_view.rootContext().setContextProperty("bridge",            self._bridge)
        self._board_view.rootContext().setContextProperty("squareIndicators",  [])
        self._board_view.setSource(QUrl.fromLocalFile(str(qml_path)))
        self._board_view.setMinimumHeight(200)
        self._board_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._board_view, 1)

        # Back / Forward
        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        self._back_btn = QPushButton("◀  Back")
        self._back_btn.clicked.connect(self._on_back)
        self._fwd_btn  = QPushButton("Forward  ▶")
        self._fwd_btn.clicked.connect(self._on_forward)
        nav_row.addWidget(self._back_btn)
        nav_row.addStretch(1)
        nav_row.addWidget(self._fwd_btn)
        root.addLayout(nav_row)

    # ── public API ────────────────────────────────────────────────────────────

    def set_weakness_squares(
        self,
        squares: list[str],
        colour: str = "#FF5722",
    ) -> None:
        """
        Mark named squares as weak on the coach board (e.g. ["e4", "d5"]).
        Each square gets a top-left dot — type "weak".
        Call with an empty list to clear.
        More indicator types (strong, tactic, king_danger) can be added later
        via set_square_indicators() once the coach backend exposes them.
        """
        indicators = []
        for sq_name in squares:
            try:
                sq   = chess.parse_square(sq_name)
                indicators.append({
                    "file": chess.square_file(sq),
                    "rank": chess.square_rank(sq),
                    "type": "weak",
                })
            except Exception:
                pass
        self._set_indicators(indicators)

    def _set_indicators(self, indicators: list[dict]) -> None:
        """Push indicator list to QML. Context property updates are reactive."""
        self._board_view.rootContext().setContextProperty("squareIndicators", indicators)

    def load_line(
        self,
        base_fen: str,
        uci_list: list[str],
        san_list: list[str],
        start_idx: int = -1,
    ) -> None:
        """
        Load a note line and display it on the coach board.

        Parameters
        ----------
        base_fen  : FEN at the start of the line
        uci_list  : UCI moves in the line
        san_list  : SAN labels — same length as uci_list
        start_idx : which move to land on (0-based into uci_list).
                    -1 means the last move (full line).
        """
        # Build board states: boards[0]=base, boards[k]=after k moves
        start = chess.Board(base_fen)
        self._boards   = [start.copy()]
        self._san_list = list(san_list)

        board = start.copy()
        for uci in uci_list:
            try:
                board.push(chess.Move.from_uci(uci))
                self._boards.append(board.copy())
            except Exception:
                break

        # start_idx is 0-based into uci_list; boards[k] = after k moves
        if start_idx < 0:
            self._idx = len(self._boards) - 1
        else:
            # clamp to valid board range
            self._idx = min(start_idx + 1, len(self._boards) - 1)

        self._refresh_board()
        self._refresh_labels()

    def clear(self) -> None:
        self._boards   = []
        self._san_list = []
        self._idx      = 0
        self._set_indicators([])

    # ── nav ───────────────────────────────────────────────────────────────────

    def _on_back(self) -> None:
        if self._idx > 0:
            self._idx -= 1
            self._refresh_board()
            self._refresh_labels()

    def _on_forward(self) -> None:
        if self._idx < len(self._boards) - 1:
            self._idx += 1
            self._refresh_board()
            self._refresh_labels()

    def _on_close(self) -> None:
        self.clear()
        self.closed.emit()

    # ── internal ──────────────────────────────────────────────────────────────

    def _refresh_board(self) -> None:
        if not self._boards:
            return
        self._model.load_board(self._boards[self._idx])
        self._back_btn.setEnabled(self._idx > 0)
        self._fwd_btn.setEnabled(self._idx < len(self._boards) - 1)

    def _refresh_labels(self) -> None:
        # Clear old labels
        while self._move_strip_layout.count():
            item = self._move_strip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # "Start" label
        start_lbl = QLabel("Start")
        start_lbl.setStyleSheet(
            "color:#4FC3F7; font-weight:bold;" if self._idx == 0
            else "color:#666666;"
        )
        self._move_strip_layout.addWidget(start_lbl)

        for i, san in enumerate(self._san_list):
            sep = QLabel("→")
            sep.setStyleSheet("color:#444444;")
            self._move_strip_layout.addWidget(sep)

            lbl = QLabel(san)
            is_cur = (i + 1 == self._idx)
            lbl.setStyleSheet(
                "color:#4FC3F7; font-weight:bold; text-decoration:underline;"
                if is_cur else "color:#AAAAAA;"
            )
            self._move_strip_layout.addWidget(lbl)

        self._move_strip_layout.addStretch(1)
