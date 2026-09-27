from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from assistant_ui.overlay import EyeWidget

# assistant window
class AssistantWindow(QWidget):
    quit_requested = Signal()
    pause_requested = Signal()
    mode_requested = Signal()

    def __init__(self):
        super().__init__()
        self.allow_close = False
        self.drag_offset = None

        self.setWindowTitle("GazeBrowser")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(390, 350)

        self.setStyleSheet("""
            QLabel {
                color: #EDF5F0;
                background: transparent;
                font: 12px "Segoe UI";
            }
            QPushButton {
                color: #A7F3D0;
                background: transparent;
                border: none;
                padding: 5px;
                font: 12px "Segoe UI";
            }
            QPushButton:hover { color: white; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 14)

        header = QHBoxLayout()
        title = QLabel("GAZEBROWSER")
        title.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.make_button("x", self.quit_requested.emit))
        layout.addLayout(header)

        eye_row = QHBoxLayout()
        eye_row.addWidget(EyeWidget())
        self.mode_label = QLabel("Starting…")
        self.mode_label.setStyleSheet(
            "color: #A7F3D0; font-size: 16px;"
        )
        eye_row.addWidget(self.mode_label)
        eye_row.addStretch()
        layout.addLayout(eye_row)

        self.transcript = QLabel("Speech preview appears here.")
        self.transcript.setTextFormat(Qt.TextFormat.PlainText)
        self.transcript.setWordWrap(True)
        self.transcript.setFixedHeight(72)
        layout.addWidget(self.transcript)

        self.notice = QLabel("Preparing devices…")
        self.notice.setTextFormat(Qt.TextFormat.PlainText)
        self.notice.setWordWrap(True)
        self.notice.setFixedHeight(34)
        self.notice.setStyleSheet(
            "color: #B8C6BE; font-size: 11px;"
        )
        layout.addWidget(self.notice)

        grid = QGridLayout()
        self.command_labels = []

        for index in range(6):
            label = QLabel()
            grid.addWidget(label, index // 2, index % 2)
            self.command_labels.append(label)

        layout.addLayout(grid)

        controls = QHBoxLayout()
        self.pause_button = self.make_button(
            "Pause · F8", self.pause_requested.emit
        )
        controls.addWidget(self.pause_button)
        controls.addStretch()
        controls.addWidget(
            self.make_button("Switch mode", self.mode_requested.emit)
        )
        controls.addWidget(QLabel("Exit · F10"))
        layout.addLayout(controls)

        area = QApplication.primaryScreen().availableGeometry()
        self.move(area.left() + 24, area.top() + 24)
        self.set_mode("command", False)

    def make_button(self, text, callback):
        button = QPushButton(text)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(callback)
        return button

    def set_mode(self, mode, paused):
        self.mode_label.setText("Paused" if paused else mode.title())
        if paused:
            labels = ["Resume", "Exit", "", "", "", ""]

        elif mode == "typing":
            labels = [
                "Assistant submit", "Assistant stop typing",
                "Assistant return", "Assistant delete",
                "Assistant clear field", "",
            ]

        else:
            labels = [
                "Click", "Open",
                "Search / Enter", "Return",
                "Stop / Exit", "Start typing",
            ]

        self.pause_button.setText(
            "Resume · F8" if paused else "Pause · F8"
        )

        for label, text in zip(self.command_labels, labels):
            label.setText(text)
            label.setStyleSheet("color: #D4E2D9;")

    def preview(self, text, final=False):
        self.transcript.setText(text[-300:] or "…")
        color = "#EDF5F0" if final else "#AABCB1"
        self.transcript.setStyleSheet(
            f"font-size: 16px; color: {color};"
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(49, 55, 53, 225))
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        painter.drawRoundedRect(
            QRectF(self.rect()).adjusted(1, 1, -1, -1), 20, 20
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_offset = (
                event.globalPosition().toPoint() - self.pos()
            )

    def mouseMoveEvent(self, event):
        if (
            self.drag_offset is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(
                event.globalPosition().toPoint() - self.drag_offset
            )

    def mouseReleaseEvent(self, event):
        self.drag_offset = None

    def closeEvent(self, event):
        if self.allow_close:
            event.accept()
        else:
            event.ignore()
            self.quit_requested.emit()
