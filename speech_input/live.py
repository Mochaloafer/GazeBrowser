import html
import sys

from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .dictation_worker import DictationWorker

class DictationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GazeBrowser — Live Dictation")
        self.resize(640, 340)

        self.previous_words = []
        self.final_text = ""
        self.closing = False

        self.worker = DictationWorker()
        self.worker.status.connect(self.show_status)
        self.worker.partial.connect(self.show_partial)
        self.worker.result.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.recording_finished)

        self.setStyleSheet("""
            QWidget {
                background: #252B29;
                color: #EDF5F0;
                font-family: "Segoe UI";
                font-size: 13px;
            }
            QTextBrowser {
                background: #303834;
                border: 1px solid #46554C;
                border-radius: 10px;
                padding: 12px;
                font-size: 22px;
            }
            QPushButton {
                background: #A7F3D0;
                color: #18372A;
                border: none;
                border-radius: 8px;
                padding: 10px 18px;
            }
            QPushButton:disabled {
                background: #404B45;
                color: #8E9D94;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self.status_label = QLabel("Ready. Start a recording to begin.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.preview = QTextBrowser()
        layout.addWidget(self.preview)

        self.detail_label = QLabel(
            "Gray words are provisional. All preview text can still change."
        )
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

        buttons = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_recording)
        buttons.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_recording)
        buttons.addWidget(self.stop_button)

        self.copy_button = QPushButton("Copy final text")
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self.copy_text)
        buttons.addWidget(self.copy_button)

        layout.addLayout(buttons)

    def start_recording(self):
        if self.worker.isRunning():
            return

        self.previous_words = []
        self.final_text = ""
        self.preview.clear()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.copy_button.setEnabled(False)

        self.status_label.setText("Starting…")
        self.worker.prepare()
        self.worker.start()

    def stop_recording(self):
        self.worker.request_stop()
        self.stop_button.setEnabled(False)
        self.status_label.setText(
            "Stopping — waiting for the current transcription…"
        )

    def show_status(self, message):
        self.status_label.setText(message)

    def show_partial(self, text, elapsed):
        words = text.split()

        # Find words shared by the previous and current result.
        shared = 0
        for old, new in zip(self.previous_words, words):
            if old != new:
                break
            shared += 1

        # Always leave the newest two words provisional.
        stable_count = min(shared, max(0, len(words) - 2))

        stable = html.escape(" ".join(words[:stable_count]))
        tentative = html.escape(" ".join(words[stable_count:]))

        self.preview.setHtml(
            f'<span style="color:#EDF5F0;">{stable}</span> '
            f'<span style="color:#9BACA1;">{tentative}</span>'
        )


        self.previous_words = words
        self.detail_label.setText(
            f"Last processing pass: {elapsed:.2f}s · "
            "Gray = provisional; white = repeated in successive results. "
            "Both may still be corrected."
        )

    def show_result(self, text, elapsed):
        self.final_text = text
        self.preview.setPlainText(text or "[No speech recognized]")
        self.status_label.setText("Recording finished. Review the text.")
        self.detail_label.setText(
            f"Final processing pass: {elapsed:.2f}s · "
            "Nothing has been typed into another application."
        )
        self.copy_button.setEnabled(bool(text))

    def show_error(self, message):
        self.status_label.setText("Could not complete recording.")
        self.preview.setPlainText(message)
        self.copy_button.setEnabled(False)

    def recording_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        if self.closing:
            self.close()

    def copy_text(self):
        QApplication.clipboard().setText(self.final_text)
        self.status_label.setText("Copied. Paste into your chosen text field.")

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.closing = True
            self.stop_recording()
            self.status_label.setText(
                "Finishing the current model operation before closing…"
            )
            event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DictationWindow()
    window.show()
    sys.exit(app.exec())