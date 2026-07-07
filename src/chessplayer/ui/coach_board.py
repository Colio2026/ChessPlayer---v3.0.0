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
- Inline PGN-style move display (clickable, matching the PGN panel style)
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
    QSizePolicy,
    QTextBrowser,
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

    Move notation is displayed as inline PGN-style text (matching the PGN
    panel), with each move being a clickable link that jumps the board to
    that position.

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
        self._boards:          list[chess.Board] = []
        self._san_list:        list[str]         = []
        self._idx              = 0
        self._start_fullmove:  int               = 1
        self._start_turn:      bool              = chess.WHITE

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
        self._title_lbl = QLabel("♟  Coach Line")
        self._title_lbl.setStyleSheet("color:#4FC3F7; font-weight:bold; font-size:11px;")
        title_row.addWidget(self._title_lbl)
        title_row.addStretch(1)
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setFlat(True)
        close_btn.setStyleSheet("color:#888; font-size:11px;")
        close_btn.clicked.connect(self._on_close)
        title_row.addWidget(close_btn)
        root.addLayout(title_row)

        # Board view
        self._board_view = QQuickWidget()
        self._board_view.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self._board_view.rootContext().setContextProperty("piecesModel",       self._model)
        self._board_view.rootContext().setContextProperty("bridge",            self._bridge)
        self._board_view.rootContext().setContextProperty("squareIndicators",  [])
        self._board_view.setSource(QUrl.fromLocalFile(str(qml_path)))
        self._board_view.setMinimumHeight(200)
        self._board_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._board_view, 1)

        # Inline PGN-style move display (clickable links)
        self._move_display = QTextBrowser()
        self._move_display.setFixedHeight(72)
        self._move_display.setOpenLinks(False)
        self._move_display.anchorClicked.connect(self._on_anchor_clicked)
        self._move_display.setStyleSheet("""
            QTextBrowser {
                background:#0D0D1A; border:1px solid #1E1E3A; border-radius:3px;
                font-family:monospace; font-size:11px; color:#CFD8DC; padding:2px;
            }
            QScrollBar:vertical {
                width:5px; background:#0D0D1A;
            }
            QScrollBar::handle:vertical {
                background:#1E1E3A; border-radius:2px; min-height:20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
        """)
        root.addWidget(self._move_display)

        # Back / Forward navigation
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

    def set_title(self, title: str) -> None:
        """Update the header label (e.g. 'Coach Line — #1  +0.36')."""
        self._title_lbl.setText(f"♟  {title}")

    def set_weakness_squares(
        self,
        squares: list[str],
        colour: str = "#FF5722",
    ) -> None:
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
        self._board_view.rootContext().setContextProperty("squareIndicators", indicators)

    def load_line(
        self,
        base_fen: str,
        uci_list: list[str],
        san_list: list[str],
        start_idx: int = 0,
    ) -> None:
        """
        Load a note line and display it on the coach board.

        Parameters
        ----------
        base_fen  : FEN at the start of the line
        uci_list  : UCI moves in the line
        san_list  : SAN labels — same length as uci_list
        start_idx : which move to land on (0-based into uci_list).
                    0 (default) = starting position; use len(uci_list)-1 for end.
        """
        start = chess.Board(base_fen)
        self._start_fullmove = start.fullmove_number
        self._start_turn     = start.turn
        self._boards   = [start.copy()]
        self._san_list = list(san_list)

        board = start.copy()
        for uci in uci_list:
            try:
                board.push(chess.Move.from_uci(uci))
                self._boards.append(board.copy())
            except Exception:
                break

        # Clamp to valid board index (boards[0] = start, boards[k] = after k moves)
        self._idx = min(max(0, start_idx), len(self._boards) - 1)

        self._refresh_board()
        self._refresh_labels()

    def clear(self) -> None:
        self._boards          = []
        self._san_list        = []
        self._idx             = 0
        self._start_fullmove  = 1
        self._start_turn      = chess.WHITE
        self._move_display.clear()
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

    def _on_anchor_clicked(self, url) -> None:
        link = url.toString()
        if link.startswith('move:'):
            try:
                idx = int(link[5:])
                if 0 <= idx < len(self._boards):
                    self._idx = idx
                    self._refresh_board()
                    self._refresh_labels()
            except ValueError:
                pass

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
        """Build inline PGN-style HTML with each move as a clickable anchor."""
        parts: list[str] = []

        # "Start" link — navigates to the starting position
        if self._idx == 0:
            parts.append('<a name="cur"></a>')
            parts.append(
                '<a href="move:0" style="color:#4FC3F7;font-weight:bold;'
                'text-decoration:none;">Start</a>'
            )
        else:
            parts.append(
                '<a href="move:0" style="color:#455A64;text-decoration:none;">Start</a>'
            )
        parts.append('&nbsp;&nbsp;')

        move_num = self._start_fullmove
        turn     = self._start_turn

        for i, san in enumerate(self._san_list):
            board_idx = i + 1

            # Move number prefix
            if turn == chess.WHITE:
                parts.append(f'<span style="color:#546E7A;">{move_num}.&nbsp;</span>')
            elif i == 0:
                # Line starts on black's move
                parts.append(f'<span style="color:#546E7A;">{move_num}…&nbsp;</span>')

            # Escape SAN for HTML (bishop 'B', captures 'x', promotions '=Q', etc.)
            san_html = (
                san.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
            )

            if board_idx == self._idx:
                parts.append('<a name="cur"></a>')
                style = (
                    'background-color:#1A3A4A;color:#4FC3F7;font-weight:bold;'
                    'text-decoration:none;padding:0 2px;'
                )
            elif turn == chess.WHITE:
                style = 'color:#CFD8DC;text-decoration:none;'
            else:
                style = 'color:#90A4AE;text-decoration:none;'

            parts.append(f'<a href="move:{board_idx}" style="{style}">{san_html}</a>')

            if turn == chess.BLACK:
                parts.append('&nbsp;&nbsp;')
                move_num += 1
            else:
                parts.append('&nbsp;')

            turn = not turn

        html = (
            '<body style="margin:4px 4px;padding:0;font-family:monospace;'
            'font-size:11px;line-height:1.7;word-wrap:break-word;">'
            + ''.join(parts)
            + '</body>'
        )
        self._move_display.setHtml(html)

        # Scroll so the current move is visible
        if self._idx >= 0:
            self._move_display.scrollToAnchor("cur")
