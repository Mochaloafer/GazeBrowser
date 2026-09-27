"""Remember the user's consent to microphone use by this application."""
# reset: python -c "from assistant_app.audio_permission import reset_microphone_consent; reset_microphone_consent()"

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox


def settings():
    return QSettings("GazeBrowser", "Nova")


def ensure_microphone_consent():
    preferences = settings()

    if preferences.value(
        "microphone/allowed", False, type=bool
    ):
        return True

    dialog = QMessageBox()
    dialog.setWindowTitle("Nova — Microphone access")
    dialog.setIcon(QMessageBox.Icon.Question)
    dialog.setText("Allow Nova to use your microphone?")
    dialog.setInformativeText(
        "Nova listens for commands and dictation while the app is open.\n\n"
        "Audio is processed locally. This implementation does not save "
        "audio recordings.\n\n"
        "Pausing actions keeps the microphone active so you can say "
        "“resume.” Exit the app to stop listening.\n\n"
        "Choosing Allow remembers your consent for future launches."
    )

    dialog.setStandardButtons(
        QMessageBox.StandardButton.Yes
        | QMessageBox.StandardButton.No
    )
    dialog.setDefaultButton(QMessageBox.StandardButton.No)

    dialog.button(QMessageBox.StandardButton.Yes).setText("Allow")
    dialog.button(QMessageBox.StandardButton.No).setText("Not now")

    allowed = (
        dialog.exec() == QMessageBox.StandardButton.Yes
    )

    if allowed:
        preferences.setValue("microphone/allowed", True)
        preferences.sync()

    return allowed


def reset_microphone_consent():
    preferences = settings()
    preferences.remove("microphone/allowed")
    preferences.sync()