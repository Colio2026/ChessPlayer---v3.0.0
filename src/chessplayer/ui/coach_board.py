"""
coach_board.py
──────────────
Coach Board widget — read-only board showing a SF recommended line.

Public API
----------
    widget.load_line(base_fen, uci_list, san_list, start_idx=0)
    widget.set_weakness_squares(squares, colour)     — legacy flat list
    widget.set_per_side_weakness(player_sq, opp_sq)  — per-side (orange / yellow)
    widget.set_title(title)
    widget.clear()

Indicator types (defined in BoardView.qml)
------------------------------------------
    "weak"        orange  top-left     — player's weak squares (defend)
    "tactic"      yellow  bottom-left  — opponent's weak squares (targets)
    "strong"      green   top-right    — outposts / strong squares
    "king_danger" purple  bottom-right — king danger
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


# ── Board model ───────────────────────────────────────────────────────────────

def _piece_code(piece: chess.Piece) -> str:
    color  = "W" if piece.color == chess.WHITE else "B"
    letter = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
               chess.ROOK: "R",  chess.QUEEN: "Q",  chess.KING: "K"}
    return color + letter[piece.piece_type]


class _CoachBoardModel(QAbstractListModel):
    FileRole  = Qt.UserRole + 1
    RankRole  = Qt.UserRole + 2
    CodeRole  = Qt.UserRole + 3
    ImageRole = Qt.UserRole + 4

    def __init__(self, pieces_dir: Path) -> None:
        super().__init__()
        self._pieces_dir = pieces_dir
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
            code = _piece_code(piece)
            img  = (self._pieces_dir / f"{code}.png").resolve()
            items.append({"file": f, "rank": r, "code": code, "image": img.as_uri()})
        self.beginResetModel()
        self._items = items
        self.endResetModel()


class _CoachBridge(QObject):
    promotionRequested = Signal(str)

    @Slot(int, int, int, int)
    def attemptMove(self, *_):
        pass

    @Slot(str)
    def choosePromotion(self, _):
        pass

    @Slot(bool)
    def setFlipped(self, flipped: bool):
        pass


# ── Widget ────────────────────────────────────────────────────────────────────

class CoachBoardWidget(QFrame):

    closed = Signal()

    # Shared stylesheet tokens matching the PGN editor
    _BG          = "#161616"
    _BG_BORDER   = "#232323"
    _BG_INNER    = "#0F0F0F"
    _TEXT        = "#C8C8C8"
    _TEXT_DIM    = "#5A5A5A"
    _TEXT_NUM    = "#363636"
    _ACCENT      = "#4DB8FF"
    _FONT        = "'Segoe UI','SF Pro Text',Arial,sans-serif"
    _MONO        = "'Cascadia Mono','Fira Mono','Consolas',monospace"

    def __init__(self, pieces_dir: Path, qml_path: Path,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(
            f"CoachBoardWidget {{ background:{self._BG};"
            f" border:1px solid {self._BG_BORDER}; border-radius:5px; }}"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._pieces_dir = pieces_dir
        self._qml_path   = qml_path

        self._boards:         list[chess.Board] = []
        self._san_list:       list[str]         = []
        self._idx             = 0
        self._start_fullmove: int               = 1
        self._start_turn:     bool              = chess.WHITE

        self._model  = _CoachBoardModel(pieces_dir)
        self._bridge = _CoachBridge()

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Title bar ─────────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        title_row.setSpacing(6)

        self._title_lbl = QLabel("Coach Line")
        self._title_lbl.setStyleSheet(
            f"color:{self._ACCENT}; font-weight:600; font-size:10pt;"
            f" font-family:{self._FONT};"
        )
        title_row.addWidget(self._title_lbl)

        title_row.addStretch(1)

        # Legend — inline in title bar
        legend_items = [
            ("#FF5722", "Weak"),
            ("#FFD54F", "Target"),
        ]
        for col, lbl in legend_items:
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{col}; font-size:9pt;")
            dot.setToolTip(lbl)
            title_row.addWidget(dot)
            txt = QLabel(lbl)
            txt.setStyleSheet(
                f"color:{self._TEXT_DIM}; font-size:8.5pt;"
                f" font-family:{self._FONT}; margin-right:6px;"
            )
            title_row.addWidget(txt)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setFlat(True)
        close_btn.setStyleSheet(f"color:{self._TEXT_DIM}; font-size:10px;")
        close_btn.clicked.connect(self._on_close)
        title_row.addWidget(close_btn)
        root.addLayout(title_row)

        # ── Board view ────────────────────────────────────────────────────────
        self._board_view = QQuickWidget()
        self._board_view.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self._board_view.rootContext().setContextProperty("piecesModel",      self._model)
        self._board_view.rootContext().setContextProperty("bridge",           self._bridge)
        self._board_view.rootContext().setContextProperty("squareIndicators", [])
        self._board_view.setSource(QUrl.fromLocalFile(str(qml_path)))
        self._board_view.setMinimumHeight(200)
        self._board_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._board_view, 1)

        # ── Move notation (styled like PGN editor) ────────────────────────────
        self._move_display = QTextBrowser()
        self._move_display.setMinimumHeight(56)
        self._move_display.setMaximumHeight(110)
        self._move_display.setOpenLinks(False)
        self._move_display.anchorClicked.connect(self._on_anchor_clicked)
        self._move_display.setStyleSheet(f"""
            QTextBrowser {{
                background:{self._BG_INNER};
                border:none;
                border-top:1px solid {self._BG_BORDER};
                font-family:{self._FONT};
                font-size:10.5pt;
                color:{self._TEXT};
                padding:4px 6px;
            }}
            QScrollBar:vertical {{
                width:4px; background:{self._BG_INNER};
            }}
            QScrollBar::handle:vertical {{
                background:#2A2A2A; border-radius:2px; min-height:16px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height:0; }}
        """)
        root.addWidget(self._move_display)

        # ── Navigation ────────────────────────────────────────────────────────
        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(4)

        _nav_style = (
            f"background:#1A1A1A; color:{self._TEXT_DIM}; font-size:9pt;"
            f" border:1px solid {self._BG_BORDER}; border-radius:3px; padding:3px 10px;"
            f" font-family:{self._FONT};"
        )
        self._back_btn = QPushButton("◀  Back")
        self._back_btn.setStyleSheet(_nav_style)
        self._back_btn.clicked.connect(self._on_back)

        self._fwd_btn  = QPushButton("Forward  ▶")
        self._fwd_btn.setStyleSheet(_nav_style)
        self._fwd_btn.clicked.connect(self._on_forward)

        nav_row.addWidget(self._back_btn)
        nav_row.addStretch(1)
        nav_row.addWidget(self._fwd_btn)
        root.addLayout(nav_row)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_title(self, title: str) -> None:
        self._title_lbl.setText(title or "Coach Line")

    def set_weakness_squares(
        self,
        squares: list[str],
        colour: str = "#FF5722",
    ) -> None:
        """Legacy: set all squares as 'weak' type (orange)."""
        indicators = self._squares_to_indicators(squares, "weak")
        self._set_indicators(indicators)

    def set_per_side_weakness(
        self,
        player_squares: list[str],
        opponent_squares: list[str],
    ) -> None:
        """Player's weak squares → orange dots. Opponent's weak squares → yellow dots."""
        indicators = (
            self._squares_to_indicators(player_squares, "weak") +
            self._squares_to_indicators(opponent_squares, "tactic")
        )
        self._set_indicators(indicators)

    def load_line(
        self,
        base_fen: str,
        uci_list: list[str],
        san_list: list[str],
        start_idx: int = 0,
    ) -> None:
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

    # ── Navigation ────────────────────────────────────────────────────────────

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

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _squares_to_indicators(squares: list[str], ind_type: str) -> list[dict]:
        out = []
        for sq_name in squares:
            try:
                sq = chess.parse_square(sq_name)
                out.append({
                    "file": chess.square_file(sq),
                    "rank": chess.square_rank(sq),
                    "type": ind_type,
                })
            except Exception:
                pass
        return out

    def _set_indicators(self, indicators: list[dict]) -> None:
        self._board_view.rootContext().setContextProperty("squareIndicators", indicators)

    def _refresh_board(self) -> None:
        if not self._boards:
            return
        self._model.load_board(self._boards[self._idx])
        self._back_btn.setEnabled(self._idx > 0)
        self._fwd_btn.setEnabled(self._idx < len(self._boards) - 1)

    def _refresh_labels(self) -> None:
        """Build PGN-style HTML matching the PGN editor's visual style."""
        BG   = self._BG
        ACT  = self._ACCENT
        CLR  = self._TEXT
        DIM  = "#5A5A5A"
        NUM  = "#363636"
        FONT = self._FONT

        parts: list[str] = []

        # "Start" entry
        if self._idx == 0:
            parts.append(
                f'<a name="cur"></a>'
                f'<a href="move:0" style="color:{ACT};font-weight:700;'
                f'text-decoration:none;">Start</a>'
            )
        else:
            parts.append(
                f'<a href="move:0" style="color:{DIM};text-decoration:none;">Start</a>'
            )
        parts.append('&nbsp;&nbsp;')

        move_num = self._start_fullmove
        turn     = self._start_turn

        for i, san in enumerate(self._san_list):
            board_idx = i + 1

            # Move number
            if turn == chess.WHITE:
                parts.append(
                    f'<span style="color:{NUM};font-size:8.5pt;">'
                    f'{move_num}.&thinsp;</span>'
                )
            elif i == 0:
                parts.append(
                    f'<span style="color:{NUM};font-size:8.5pt;">'
                    f'{move_num}…&thinsp;</span>'
                )

            san_html = (
                san.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
            )

            if board_idx == self._idx:
                parts.append('<a name="cur"></a>')
                style = (
                    f'background:{ACT}22;color:{ACT};font-weight:700;'
                    f'text-decoration:none;padding:0 3px;border-radius:2px;'
                )
            else:
                color = CLR if turn == chess.WHITE else "#90A4AE"
                style = f'color:{color};text-decoration:none;'

            parts.append(
                f'<a href="move:{board_idx}" style="{style}">{san_html}</a>'
            )

            if turn == chess.BLACK:
                parts.append('&ensp;')
                move_num += 1
            else:
                parts.append('&thinsp;')

            turn = not turn

        html = (
            f'<body style="margin:6px 6px;padding:0;'
            f'font-family:{FONT};font-size:10.5pt;line-height:1.9;'
            f'background:{BG};word-wrap:break-word;">'
            + ''.join(parts)
            + '</body>'
        )
        self._move_display.setHtml(html)
        if self._idx >= 0:
            self._move_display.scrollToAnchor("cur")
