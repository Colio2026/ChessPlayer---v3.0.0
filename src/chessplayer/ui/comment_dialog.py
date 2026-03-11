

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class CommentDialog(QDialog):
    """
    Modal dialog for editing a PGN comment.

    Tip shown below the text area explains the (moves) syntax so the user
    knows how to embed navigatable lines.

    Parameters
    ----------
    title   : prompt label shown above the text area
    initial : pre-filled comment text (may be empty)
    """

    def __init__(
        self,
        parent: QWidget | None,
        title: str,
        initial: str,
        # editor and base_ply kept for API compatibility but no longer used here
        editor=None,
        base_ply: int = 0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Comment")
        self.setMinimumWidth(440)
        self.setMinimumHeight(260)

        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(10, 10, 10, 10)

        # Prompt
        root.addWidget(QLabel(title))

        # Text area
        self._edit = QTextEdit()
        self._edit.setPlainText(initial)
        self._edit.setAcceptRichText(False)
        self._edit.setMinimumHeight(140)
        root.addWidget(self._edit, 1)

        # Hint
        hint = QLabel(
            "<small style=\"color:#666666;\">"
            "Tip: wrap moves in parentheses to make them clickable — "
            "e.g. <i>Try (Nf3 Nc6 Bb5) here</i> or <i>(1.e4 e5)</i>"
            "</small>"
        )
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.RichText)
        root.addWidget(hint)

        # OK / Cancel
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        # Cursor at end
        cur = self._edit.textCursor()
        cur.movePosition(QTextCursor.End)
        self._edit.setTextCursor(cur)
        self._edit.setFocus()

    def text(self) -> str:
        """Return the edited comment text, stripped."""
        return self._edit.toPlainText().strip()
