from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from chessplayer.engine.uci_engine import BestMove


def _cp_to_ratio(cp: int) -> float:
    """
    Map centipawns to a white-advantage ratio in [0, 1].
    1.0 = white completely winning, 0.0 = black completely winning.
    Uses a sigmoid so ±300cp ≈ 70/30 and ±1000cp ≈ 98/2.
    """
    return 1.0 / (1.0 + math.exp(-0.004 * cp))


class EvalBar(QWidget):
    """
    Lichess-style vertical evaluation bar.

    White section fills from the bottom proportionally to advantage.
    Black section fills from the top.
    Score text (e.g. +1.4, M3, M-2) is drawn at the boundary.

    Call update_eval(best_move) on every analysis result.
    Call clear_eval() when engine is off.
    """

    _WHITE_COL  = QColor("#FFFFFF")
    _BLACK_COL  = QColor("#1A1A1A")
    _BORDER_COL = QColor("#555555")
    _TEXT_WHITE = QColor("#1A1A1A")   # text on white section
    _TEXT_BLACK = QColor("#FFFFFF")   # text on black section

    _BAR_WIDTH  = 28

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ratio: float       = 0.5    # 0=black, 1=white, 0.5=equal
        self._label: str         = "0.0"
        self._active: bool       = False

        self.setFixedWidth(self._BAR_WIDTH)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setToolTip("Engine evaluation")

    # ── public API ────────────────────────────────────────────────────────────

    def update_eval(self, result: BestMove | None, white_to_move: bool = True) -> None:
        """
        Update bar from a BestMove result.
        score_cp is from the side-to-move perspective — we flip for Black.
        """
        if result is None:
            self.clear_eval()
            return

        self._active = True

        if result.mate_in is not None:
            m = result.mate_in
            # Positive mate = side to move is mating
            if white_to_move:
                self._ratio = 1.0 if m > 0 else 0.0
                self._label = f"M{abs(m)}" if m > 0 else f"M-{abs(m)}"
            else:
                self._ratio = 0.0 if m > 0 else 1.0
                self._label = f"M-{abs(m)}" if m > 0 else f"M{abs(m)}"
        elif result.score_cp is not None:
            cp = result.score_cp if white_to_move else -result.score_cp
            self._ratio = _cp_to_ratio(cp)
            pawns       = cp / 100.0
            sign        = "+" if pawns >= 0 else ""
            self._label = f"{sign}{pawns:.1f}"
        else:
            self._ratio = 0.5
            self._label = "0.0"

        self.update()

    def clear_eval(self) -> None:
        """Reset to neutral (engine off or no result yet)."""
        self._active = False
        self._ratio  = 0.5
        self._label  = ""
        self.update()

    # ── paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        p   = QPainter(self)
        w   = self.width()
        h   = self.height()

        if not self._active:
            # Grey bar when engine is off
            p.fillRect(0, 0, w, h, QColor("#333333"))
            p.setPen(QPen(self._BORDER_COL))
            p.drawRect(0, 0, w - 1, h - 1)
            return

        # White ratio from bottom, black from top
        white_h = max(0, min(h, int(h * self._ratio)))
        black_h = h - white_h

        # Black section (top)
        if black_h > 0:
            p.fillRect(0, 0, w, black_h, self._BLACK_COL)

        # White section (bottom)
        if white_h > 0:
            p.fillRect(0, black_h, w, white_h, self._WHITE_COL)

        # Border
        p.setPen(QPen(self._BORDER_COL))
        p.drawRect(0, 0, w - 1, h - 1)

        # Score label — draw at the boundary, flip text colour per section
        if self._label:
            font = QFont("Consolas", 8)
            font.setBold(True)
            p.setFont(font)

            boundary_y = black_h
            label_h    = 16
            label_y    = max(2, min(h - label_h - 2, boundary_y - label_h // 2))

            # Shadow / contrast: draw on whichever section is larger
            # Always grey — readable on both white and black sections
            p.setPen(QPen(QColor("#888888")))
            p.drawText(
                QRect(0, label_y, w, label_h),
                Qt.AlignHCenter | Qt.AlignVCenter,
                self._label,
            )

        p.end()
