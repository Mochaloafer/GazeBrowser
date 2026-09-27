import ctypes
import os
import threading
import time

import pyautogui
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication

def foreground_window():
    function = ctypes.windll.user32.GetForegroundWindow
    function.restype = ctypes.c_void_p
    return int(function() or 0)

"""
FocusWorker checks the focused control
DesktopActions sends mouse/keyboard input
"""

class FocusWorker(QThread):
    updated = Signal(object)

    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
            import uiautomation as auto

            with auto.UIAutomationInitializerInThread():
                while not self.stop_event.is_set():
                    window = foreground_window()

                    data = {
                        "key": (window, ()),
                        "window": window,
                        "editable": False,
                        "password": False,
                        "own": False,
                        "time": time.monotonic(),
                    }

                    try:
                        control = auto.GetFocusedControl()

                        if control is not None:
                            data["own"] = (
                                control.ProcessId == os.getpid()
                            )
                            data["password"] = bool(control.IsPassword)
                            data["key"] = (
                                window,
                                tuple(control.GetRuntimeId()),
                            )

                            data["editable"] = is_editable(control)

                            signature = (
                                data["key"],
                                control.ControlTypeName,
                                data["editable"],
                            )

                            if signature != getattr(self, "_last_signature", None):
                                print(
                                    "[Focus]",
                                    control.ControlTypeName,
                                    "| editable:",
                                    data["editable"],
                                )
                                self._last_signature = signature

                    except Exception as error:
                        message = str(error)

                        if message != getattr(self, "_last_focus_error", None):
                            print("[Focus error]", message)
                            self._last_focus_error = message

                    if foreground_window() == window:
                        self.updated.emit(data)

                    self.stop_event.wait(0.3)

        except Exception:
            # Manual mode is still available if UI Automation fails.
            while not self.stop_event.is_set():
                window = foreground_window()
                self.updated.emit({
                    "key": (window, ()),
                    "window": window,
                    "editable": False,
                    "password": False,
                    "own": False,
                    "time": time.monotonic(),
                })
                self.stop_event.wait(0.3)


class DesktopActions:
    def __init__(self):
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05

    def click(self):
        pyautogui.click()

    def click_at(self, x, y):
        pyautogui.click(x=x, y=y)

    def open_at(self, x, y):
        pyautogui.doubleClick(
            x=x,
            y=y,
            interval=0.15,
        )

    def open_target(self):
        """Double-click at the current pointer position."""
        pyautogui.doubleClick(interval=0.15)

    def enter(self):
        pyautogui.press("enter")

    def back(self):
        pyautogui.hotkey("alt", "left")

    def delete_word(self):
        pyautogui.hotkey("ctrl", "backspace")

    def insert(self, text):
        text = text.strip()

        if not text:
            return

        QApplication.clipboard().setText(text)
        pyautogui.hotkey("ctrl", "v")

        # Send the separator separately from the clipboard text.
        pyautogui.press("space")

    def move(self, x, y):
        pyautogui.moveTo(int(x), int(y), _pause=False)

    def clear_field(self):
        pyautogui.hotkey("ctrl", "a")
        pyautogui.press("backspace")


def is_editable(control):
    """Recognize writable text controls without reading their text."""
    if control.IsPassword or not control.IsEnabled:
        return False

    if control.ControlTypeName not in (
        "EditControl",
        "DocumentControl",
        "ComboBoxControl",
    ):
        return False

    try:
        pattern = control.GetValuePattern()

        if pattern is not None:
            return not pattern.IsReadOnly
    except Exception:
        pass

    try:
        pattern = control.GetTextPattern()

        if pattern is not None:
            # Microsoft UI Automation: IsReadOnly text attribute.
            IS_READ_ONLY_ATTRIBUTE = 40015

            read_only = pattern.DocumentRange.GetAttributeValue(
                IS_READ_ONLY_ATTRIBUTE
            )

            # Unsupported/mixed attribute values are not proof
            # that a field is writable.
            if isinstance(read_only, (bool, int)):
                if read_only in (0, 1):
                    return not bool(read_only)
    except Exception:
        pass

    return False