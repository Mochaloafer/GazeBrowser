"""Run the teammate's gaze loop; this is the only gaze mouse-movement owner."""

import threading
import time
from collections import deque
from contextlib import nullcontext

import cv2
import numpy as np
import pyautogui
from pynput import mouse
from PySide6.QtCore import QThread, Signal

from . import teammate_gaze as cfg


class GazeWorker(QThread):
    sample = Signal(float, float, bool, float)
    failed = Signal(str)
    pause_requested = Signal()
    quit_requested = Signal()

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.stop_event = threading.Event()
        self._input_lock = threading.RLock()
        self._paused = False
        self._held = False
        self._generation = 0
        self._last_manual_move = 0.0

    def action_guard(self):
        """Serialize speech actions with gaze movement and blink clicks."""
        return self._input_lock

    def _blocked(self):
        return self._paused or self._held or self.stop_event.is_set()

    def set_paused(self, paused):
        with self._input_lock:
            self._paused = bool(paused)
            self._generation += 1

    def hold_cursor(self):
        """Stop automatic input before capturing the speech target."""
        with self._input_lock:
            self._held = True
            self._generation += 1
            return tuple(pyautogui.position())

    def release_cursor(self):
        with self._input_lock:
            if self._held:
                self._held = False
                self._generation += 1

    def stop(self):
        with self._input_lock:
            self.stop_event.set()
            self._generation += 1

    def _on_move(self, x, y, injected=False):
        if not injected:
            self._last_manual_move = time.time()

    def _on_click(self, x, y, button, pressed, injected=False):
        if pressed and not injected and button == mouse.Button.left:
            self._last_manual_move = time.time()

    @staticmethod
    def _physical_mouse_events(msg, data):
        # Ignore injected events in our listener, not in the operating system.
        # This also supports pynput versions without the injected callback arg.
        return not bool(data.flags & 0x01)  # LLMHF_INJECTED

    def run(self):
        try:
            automation = (
                cfg.uia.UIAutomationInitializerInThread()
                if cfg.uia is not None else nullcontext()
            )
            with automation:
                self._track()
        except pyautogui.FailSafeException:
            self.failed.emit("Gaze stopped: mouse fail-safe triggered; restart")
        except Exception as error:
            self.failed.emit(f"Gaze: {error}")
        finally:
            self.stop_event.set()
            self.sample.emit(0.0, 0.0, False, time.monotonic())
            try:
                self.tracker.close()
            except Exception as error:
                print(f"[Gaze cleanup] {error}")

    def _track(self):
        corrector, pipeline, accepted_ids = cfg.load_correction()
        screen_w, screen_h = pyautogui.size()
        camera = None
        listener = None
        preview_open = False
        history = deque()
        cursor_x, cursor_y = map(float, pyautogui.position())
        locked = None
        candidate = None
        candidate_since = 0.0
        last_query = 0.0
        blink_start = None
        snapping = cfg.uia is not None
        message, message_until = "", 0.0
        seen_generation = self._generation

        def flash(text):
            nonlocal message, message_until
            message, message_until = text, time.time() + 1.5

        def handle_blink_click(blink_began):
            nonlocal cursor_x, cursor_y
            snap_t = blink_began - cfg.PRE_BLINK_LOOKBACK
            before = [h for h in history if h[0] <= snap_t]
            if not before:
                pyautogui.click(int(cursor_x), int(cursor_y), _pause=False)
                return
            _, _, _, cx, cy, rect = before[-1]
            if rect is None:
                pyautogui.click(int(cx), int(cy), _pause=False)
                cursor_x, cursor_y = cx, cy
                flash("CLICK")
                return
            tx, ty = cfg.center(rect)
            pyautogui.click(tx, ty, _pause=False)
            cursor_x, cursor_y = float(tx), float(ty)
            flash("CLICK")

        try:
            camera = cv2.VideoCapture(cfg.CAMERA_INDEX)
            if not camera.isOpened():
                raise RuntimeError("Camera could not be opened. Close the standalone main.py.")
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cv2.namedWindow(cfg.WINDOW, cv2.WINDOW_NORMAL)
            preview_open = True
            cv2.resizeWindow(cfg.WINDOW, 320, 240)
            cv2.setWindowProperty(cfg.WINDOW, cv2.WND_PROP_TOPMOST, 1)
            listener = mouse.Listener(
                on_move=self._on_move,
                on_click=self._on_click,
                win32_event_filter=self._physical_mouse_events,
            )
            listener.start()

            while not self.stop_event.is_set():
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames")
                now = time.time()
                features, blink = self.tracker.extract_features(frame)
                manual = now - self._last_manual_move < cfg.MANUAL_HOLD
                raw = (
                    self.tracker.predict(np.array([features]))[0]
                    if features is not None and not blink else None
                )
                filtered = pipeline.step(raw, now)
                valid = filtered is not None and features is not None and not blink
                if valid:
                    valid = bool(np.isfinite(filtered).all())
                self.sample.emit(
                    float(filtered[0]) if valid else 0.0,
                    float(filtered[1]) if valid else 0.0,
                    valid, time.monotonic(),
                )
                if filtered is not None and not np.isfinite(filtered).all():
                    raise RuntimeError("Gaze pipeline returned non-finite coordinates")

                with self._input_lock:
                    generation = self._generation
                    blocked = self._blocked()
                    if generation != seen_generation:
                        # Do not replay a blink or old cursor position after a hold.
                        seen_generation = generation
                        blink_start = None
                        history.clear()
                        locked = candidate = None
                        cursor_x, cursor_y = map(float, pyautogui.position())

                if features is None:
                    status = "no face"
                    blink_start = None
                elif blink:
                    status = "eyes closed"
                    if blocked:
                        blink_start = None
                    elif blink_start is None:
                        blink_start = now
                else:
                    status = "TOUCHPAD" if manual else ("LOCKED" if locked else "tracking")
                    if blink_start is not None:
                        duration = now - blink_start
                        with self._input_lock:
                            if (
                                not self._blocked()
                                and generation == self._generation
                                and not manual
                                and cfg.CLICK_BLINK_MIN <= duration <= cfg.CLICK_BLINK_MAX
                            ):
                                handle_blink_click(blink_start)
                        blink_start = None

                    if filtered is not None:
                        sx, sy = filtered
                        gx, gy = corrector.correct(sx, sy)
                        if not np.isfinite([gx, gy]).all():
                            raise RuntimeError("Gaze corrector returned non-finite coordinates")
                        gx = cfg.clamp(gx, cfg.EDGE_MARGIN, screen_w - 1 - cfg.EDGE_MARGIN)
                        gy = cfg.clamp(gy, cfg.EDGE_MARGIN, screen_h - 1 - cfg.EDGE_MARGIN)

                        if locked and not cfg.inside(locked, gx, gy, cfg.STICKY_PAD):
                            locked = None
                        if candidate and not cfg.inside(candidate, gx, gy):
                            candidate = None
                        if snapping and locked is None and now - last_query >= cfg.QUERY_INTERVAL:
                            last_query = now
                            rect = cfg.find_target(gx, gy)
                            if rect is None:
                                candidate = None
                            elif rect != candidate:
                                candidate, candidate_since = rect, now
                            elif now - candidate_since >= cfg.LOCK_DWELL:
                                locked, candidate = rect, None

                        if locked:
                            tx, ty = cfg.center(locked)
                            speed, should_move = cfg.GLIDE, True
                        elif candidate:
                            tx, ty = gx, gy
                            speed, should_move = cfg.SLOW_GLIDE, True
                        else:
                            tx, ty = gx, gy
                            speed = cfg.GLIDE
                            should_move = np.hypot(tx - cursor_x, ty - cursor_y) > cfg.DEAD_ZONE

                        with self._input_lock:
                            # Recheck after UI Automation, which may take time.
                            manual = time.time() - self._last_manual_move < cfg.MANUAL_HOLD
                            if manual or self._blocked():
                                cursor_x, cursor_y = map(float, pyautogui.position())
                            elif generation == self._generation and should_move:
                                cursor_x += (tx - cursor_x) * speed
                                cursor_y += (ty - cursor_y) * speed
                                pyautogui.moveTo(int(cursor_x), int(cursor_y), _pause=False)
                            if not self._blocked() and generation == self._generation:
                                history.append((now, sx, sy, cursor_x, cursor_y, locked))
                                while history and now - history[0][0] > 3.0:
                                    history.popleft()

                preview = cv2.flip(frame, 1)
                with self._input_lock:
                    paused, held = self._paused, self._held
                lines = [
                    "PAUSED" if paused else ("TARGET HELD" if held else status),
                    f"accepted: {len(accepted_ids)}  snap: {'on' if snapping else 'off'}",
                ]
                if now < message_until:
                    lines.append(message)
                for i, text in enumerate(lines):
                    cv2.putText(
                        preview, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255) if paused else (0, 255, 0), 2,
                    )
                cv2.imshow(cfg.WINDOW, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.stop()
                    self.quit_requested.emit()
                    break
                elif key == ord("p"):
                    self.pause_requested.emit()
                elif key == ord("s") and cfg.uia is not None:
                    snapping, locked, candidate = not snapping, None, None
                if cv2.getWindowProperty(cfg.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    self.stop()
                    self.quit_requested.emit()
                    break
        finally:
            if listener is not None:
                listener.stop()
                listener.join(timeout=1.0)
            if camera is not None:
                camera.release()
            if preview_open:
                cv2.destroyAllWindows()
            print("Gaze closed. Accepted lessons unchanged; manage sessions with learning_manager.py.")
