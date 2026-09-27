import ctypes
import math
import time

import pyautogui
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

from .desktop import DesktopActions, foreground_window
from .transcript_parser import parse_transcript

class AssistantController(QObject):
    def __init__(self, window, speech, focus_worker, gaze=None):
        super().__init__()

        self.window = window
        self.speech = speech
        self.focus_worker = focus_worker
        self.gaze = gaze
        self.actions = DesktopActions()

        self.mode = "command"
        self.override = "command"
        self.paused = False
        self.closing = False
        self.focus = None

        # Increment whenever focus or mode changes.
        # Results from an older state must not trigger actions.
        self.epoch = 0
        self.phrases = {}
        self.held_phrase = None
        self.held_position = None
        self.hold_deadline = 0.0

        self.gaze_valid = False
        self.last_gaze_time = 0.0

        self.width, self.height = pyautogui.size()
        self.x, self.y = pyautogui.position()
        self.last_x, self.last_y = self.x, self.y

        # Speech notifications.
        speech.began.connect(self.begin_phrase)
        speech.partial.connect(self.partial)
        speech.final.connect(self.final)
        speech.status.connect(window.notice.setText)
        speech.failed.connect(self.failure)
        speech.ended.connect(self.release_cursor)

        # Focus notifications.
        focus_worker.updated.connect(self.focus_changed)

        # Gaze is optional when running with --no-gaze.
        if gaze is not None:
            gaze.sample.connect(self.gaze_sample)
            gaze.failed.connect(self.failure)

        # Overlay controls.
        window.quit_requested.connect(self.shutdown)
        window.pause_requested.connect(self.toggle_pause)
        window.mode_requested.connect(self.toggle_mode)

        # Global F8/F10 polling and graceful shutdown checks.
        self.key_states = {}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(50)

    def workers(self):
        return [
            worker
            for worker in (
                self.speech,
                self.focus_worker,
                self.gaze,
            )
            if worker is not None
        ]

    def tick(self):
        if (self.held_phrase is not None and time.monotonic() >= self.hold_deadline):
            expired_phrase = self.held_phrase

            # Prevent a late result from executing after timeout.
            self.phrases.pop(expired_phrase, None)
            self.release_cursor()

            self.window.notice.setText(
                "Target hold timed out — repeat the command"
            )
        for key, action in (
            (0x77, self.toggle_pause),  # F8
            (0x79, self.shutdown),      # F10
        ):
            down = bool(
                ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000
            )

            if down and not self.key_states.get(key, False):
                action()

            self.key_states[key] = down

        if self.closing and not any(
            worker.isRunning() for worker in self.workers()
        ):
            self.timer.stop()
            self.window.allow_close = True
            self.window.close()
            QApplication.instance().quit()

    def focus_changed(self, data):
        if self.closing or data["own"]:
            return

        if self.focus is None:
            # Record initial focus without switching into typing mode.
            self.epoch += 1

        elif data["key"] != self.focus["key"]:
            self.epoch += 1
            self.override = None

        self.focus = data

        mode = self.override or (
            "typing" if data["editable"] else "command"
        )

        if mode != self.mode:
            self.mode = mode
            self.epoch += 1

        self.window.set_mode(self.mode, self.paused)

        self.speech.set_mode(
            "command" if self.paused else self.mode
        )

    def set_mode(self, mode):
        self.mode = mode
        self.override = mode
        self.epoch += 1
        self.window.set_mode(mode, self.paused)
        self.speech.set_mode(
            "command" if self.paused else self.mode
        )

    def toggle_mode(self):
        if self.closing or self.paused:
            return

        if self.focus and self.focus["password"]:
            self.window.notice.setText(
                "Typing is disabled in password fields"
            )
            return

        self.set_mode(
            "typing" if self.mode == "command" else "command"
        )

        self.window.notice.setText(
            "Manual mode lasts until focus changes"
        )

    def toggle_pause(self):
        if self.closing:
            return

        self.release_cursor()
        self.paused = not self.paused
        self.epoch += 1

        self.window.set_mode(self.mode, self.paused)
        self.window.notice.setText(
            "Say resume or press F8" if self.paused else "Resumed"
        )

        self.speech.set_mode(
            "command" if self.paused else self.mode
        )

    def hold_cursor(self, phrase_id):
        """Remember the target and suspend gaze-driven movement."""
        self.release_cursor()

        self.held_phrase = phrase_id
        self.held_position = tuple(pyautogui.position())
        self.hold_deadline = time.monotonic() + 10.0

        self.window.notice.setText(
            "Target held — recognizing command…"
        )

    def release_cursor(self, phrase_id=None):
        """Release only the matching phrase, or release unconditionally."""
        if (
            phrase_id is not None
            and phrase_id != self.held_phrase
        ):
            return

        self.held_phrase = None
        self.held_position = None
        self.hold_deadline = 0.0

    def begin_phrase(self, phrase_id):
        if self.closing:
            return

        self.phrases = {
            key: value
            for key, value in self.phrases.items()
            if key > phrase_id - 8
        }

        self.phrases[phrase_id] = (
            self.epoch,
            "command" if self.paused else self.mode,
            dict(self.focus) if self.focus else None,
        )

        if self.mode == "command" and not self.paused:
            self.hold_cursor(phrase_id)

    def partial(self, phrase_id, text):
        context = self.phrases.get(phrase_id)

        if (
            not self.closing
            and context is not None
            and context[0] == self.epoch
        ):
            # Partial guesses are displayed but never executed.
            self.window.preview(text)

    def target_unchanged(self, context):
        original = context[2]
        current = self.focus

        return bool(
            original
            and current
            and original["key"] == current["key"]
            and current["window"] == foreground_window()
            and time.monotonic() - current["time"] < 1.2
            and not current["password"]
            and not current["own"]
        )

    def final(self, phrase_id, text, last_voice_time):
        try:
            self.process_final(
                phrase_id, text, last_voice_time
            )
        finally:
            self.release_cursor(phrase_id)

    def process_final(self, phrase_id, text, last_voice_time):
        context = self.phrases.pop(phrase_id, None)

        if self.closing or context is None:
            return

        if context[0] != self.epoch:
            self.window.notice.setText(
                "Focus or mode changed — repeat the phrase"
            )
            return

        if time.monotonic() - last_voice_time > 5.0:
            self.window.notice.setText(
                "Recognition too late — try tiny.en"
            )
            return

        self.window.preview(text, final=True)

        # Determine the action before logging or executing it.
        action, prefix = parse_transcript(text, context[1])

        if context[1] == "command":
            print(f"[Command] Heard {text!r} -> {action}")

        if self.paused and action not in ("resume", "exit"):
            self.window.notice.setText(
                "Paused — say resume or exit"
            )
            return

        try:
            if action == "exit":
                self.shutdown()
                return

            if action in ("pause", "resume"):
                should_pause = action == "pause"

                if should_pause != self.paused:
                    self.toggle_pause()
                return

            if action in ("ignore", "unclear"):
                self.window.notice.setText(
                    "No command matched"
                    if action == "ignore"
                    else "Unclear assistant phrase — repeat it"
                )
                return

            if not self.target_unchanged(context):
                self.window.notice.setText(
                    "Target changed or unavailable — repeat the phrase"
                )
                return

            if action == "typing":
                self.set_mode("typing")

            elif action in ("click", "open"):
                if (
                    self.held_phrase != phrase_id
                    or self.held_position is None
                ):
                    self.window.notice.setText(
                        "Target hold expired — repeat the command"
                    )
                    return

                if self.gaze is not None and (
                    not self.gaze_valid
                    or time.monotonic() - self.last_gaze_time > 0.6
                ):
                    self.window.notice.setText(
                        "No fresh gaze — action ignored"
                    )
                    return

                x, y = self.held_position

                if action == "click":
                    self.actions.click_at(x, y)
                else:
                    self.actions.open_at(x, y)

                self.override = None

            elif action == "enter":
                self.actions.enter()

            elif action == "back":
                # Discard this phrase's uninserted prefix.
                self.actions.back()
                self.set_mode("command")

            elif action == "clear_field":
                if context[1] != "typing":
                    return

                self.actions.clear_field()

            elif action == "delete":
                if context[1] != "typing":
                    self.window.notice.setText(
                        "Delete is available in typing mode"
                    )
                    return

                if prefix:
                    # Remove the last word before inserting the draft.
                    words = prefix.rstrip().rsplit(None, 1)
                    remaining = words[0] if len(words) == 2 else ""
                    self.actions.insert(remaining)
                else:
                    # No pending draft: delete from the focused field.
                    self.actions.delete_word()

            elif action in ("dictate", "search", "finish_typing"):
                self.actions.insert(prefix)

                if action == "search":
                    self.actions.enter()
                    self.set_mode("command")

                elif action == "finish_typing":
                    self.set_mode("command")

            else:
                self.window.notice.setText(
                    f"Unsupported action: {action}"
                )
                return

            # Show interpretation after a successful action.
            if context[1] == "command":
                self.window.notice.setText(
                    f'Heard “{text}” → '
                    f'{action.replace("_", " ").title()}'
                )
            else:
                self.window.notice.setText(
                    "Done: " + action.replace("_", " ")
                )

        except pyautogui.FailSafeException:
            if not self.paused:
                self.toggle_pause()

            self.window.notice.setText(
                "Mouse fail-safe triggered. Move away from corner."
            )

        except Exception as error:
            self.window.notice.setText(
                f"Action failed: {error}"
            )

    def gaze_sample(self, x, y, valid, timestamp):
        self.last_gaze_time = timestamp
        self.gaze_valid = (
            valid and math.isfinite(x) and math.isfinite(y)
        )

        if (
            self.closing
            or self.paused
            or self.held_phrase is not None
            or not self.gaze_valid
            or time.monotonic() - timestamp > 0.3
        ):
            return

        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))

        self.x += (x - self.x) * 0.30
        self.y += (y - self.y) * 0.30

        distance = (
            (self.x - self.last_x) ** 2
            + (self.y - self.last_y) ** 2
        ) ** 0.5

        if distance > 20:
            try:
                self.actions.move(self.x, self.y)
                self.last_x, self.last_y = self.x, self.y

            except pyautogui.FailSafeException:
                self.toggle_pause()

            except Exception as error:
                self.failure(f"Mouse: {error}")

    def failure(self, message):
        self.release_cursor()
        self.gaze_valid = False

        if not self.paused:
            self.toggle_pause()

        self.window.notice.setText(
            message + " — restart after fixing"
        )

    def shutdown(self):
        if self.closing:
            return

        self.closing = True
        self.release_cursor()
        self.epoch += 1

        self.window.notice.setText(
            "Closing devices; waiting for current processing…"
        )

        for worker in self.workers():
            worker.stop()