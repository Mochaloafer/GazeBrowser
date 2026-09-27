"""Standalone assistant UI. Run from the repository root.

    .\\.venv\\Scripts\\python.exe assistant_ui/overlay.py

Reads mouse position for the eye animation.
Does not open the camera, move the mouse, or issue actions.
"""

import sys

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class EyeWidget(QWidget):
    """Decorative eye whose iris follows the current mouse direction."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 66)
        self.offset = QPointF()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_direction)
        self.timer.start(33)

    def update_direction(self):
        cursor = QCursor.pos()
        screen = QApplication.screenAt(cursor) or QApplication.primaryScreen()
        area = screen.geometry()

        # Convert cursor position to approximately -1 through +1.
        nx = 2 * (cursor.x() - area.left()) / max(area.width() - 1, 1) - 1
        ny = 2 * (cursor.y() - area.top()) / max(area.height() - 1, 1) - 1

        target = QPointF(nx * 12, ny * 7)

        # Gentle easing keeps the decorative iris movement smooth.
        self.offset += (target - self.offset) * 0.18
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # REPLACE: refine this eye artwork after the functional UI is approved.
        eye = QPainterPath()
        eye.moveTo(8, 33)
        eye.cubicTo(30, 5, 70, 5, 92, 33)
        eye.cubicTo(70, 61, 30, 61, 8, 33)

        painter.setPen(QPen(QColor("#A7F3D0"), 2.5))
        painter.setBrush(QColor(167, 243, 208, 16))
        painter.drawPath(eye)

        painter.save()
        painter.setClipPath(eye)

        center = QPointF(50, 33) + self.offset
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#A7F3D0"))
        painter.drawEllipse(center, 13, 13)

        painter.setBrush(QColor("#233B35"))
        painter.drawEllipse(center, 6, 6)

        painter.setBrush(QColor("#F0FFF8"))
        painter.drawEllipse(center + QPointF(-3, -4), 2.5, 2.5)
        painter.restore()


class AssistantOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.drag_offset = None

        self.setWindowTitle("GazeBrowser Assistant")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(330, 195)

        self.setStyleSheet("""
            QLabel {
                color: #F3F5F4;
                background: transparent;
                font-family: "Segoe UI";
            }
            QPushButton {
                color: #DFE7E3;
                background: rgba(255, 255, 255, 16);
                border: 1px solid rgba(255, 255, 255, 24);
                border-radius: 8px;
                padding: 6px 10px;
                font-family: "Segoe UI";
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 35);
            }
            QPushButton:disabled {
                color: #929B97;
                background: rgba(255, 255, 255, 7);
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 14)
        layout.setSpacing(8)

        header = QHBoxLayout()

        title = QLabel("GAZEBROWSER")
        title.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: #B9C7C0;"
        )
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        header.addWidget(title)
        header.addStretch()

        close_button = QPushButton("×")
        close_button.setFixedSize(28, 28)
        close_button.setToolTip("Close assistant UI only")
        close_button.clicked.connect(self.close)
        header.addWidget(close_button)
        layout.addLayout(header)

        body = QHBoxLayout()
        body.addWidget(EyeWidget())

        text = QVBoxLayout()
        text.setSpacing(4)

        self.status_label = QLabel("Assistant preview")
        self.status_label.setStyleSheet(
            "font-size: 17px; font-weight: 600;"
        )
        text.addWidget(self.status_label)

        self.detail_label = QLabel("Iris follows your cursor.\nTracker status not connected.")
        self.detail_label.setStyleSheet(
            "font-size: 11px; color: #C1CEC7;"
        )
        text.addWidget(self.detail_label)

        body.addLayout(text)
        layout.addLayout(body)

        footer = QHBoxLayout()

        mode = QLabel("●  UI PREVIEW")
        mode.setStyleSheet("color: #A7F3D0; font-size: 10px;")
        footer.addWidget(mode)
        footer.addStretch()

        keyboard_button = QPushButton("Keyboard")
        keyboard_button.setEnabled(False)
        keyboard_button.setToolTip("Coming in the next UI checkpoint")
        footer.addWidget(keyboard_button)
        layout.addLayout(footer)

        area = QApplication.primaryScreen().availableGeometry()
        self.move(area.left() + 24, area.top() + 24)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255, 45), 1))
        painter.setBrush(QColor(49, 55, 53, 220))
        painter.drawRoundedRect(
            QRectF(self.rect()).adjusted(1, 1, -1, -1),
            20,
            20,
        )

    def mousePressEvent(self, event):
        # Drag using the title area or empty background.
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_offset = (
                event.globalPosition().toPoint()
                - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event):
        if (
            self.drag_offset is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(event.globalPosition().toPoint() - self.drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.drag_offset = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AssistantOverlay()
    window.show()
    sys.exit(app.exec())