import csv
import json
import queue
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyautogui
from eyetrax import GazeEstimator
from eyetrax.calibration import run_dense_grid_calibration
from eyetrax.filters import KalmanEMASmoother, make_kalman
from pynput import mouse  # pip install pynput

try:
    import uiautomation as uia  # Windows only: pip install uiautomation
except ImportError:
    uia = None
    print("uiautomation not installed -> button snapping disabled")


# ================= Settings =================
MODEL_FILE = Path("gaze_model.pkl")
LEARN_FILE = Path("gaze_corrections.json")
LOG_FILE = Path("gaze_log.csv")
RECALIBRATE = False
CAMERA_INDEX = 0
GRID_ROWS, GRID_COLS = 5, 5
WINDOW = "Eye Tracker"

# Smoothing
EMA_ALPHA = 0.8
MEDIAN_WINDOW = 7
DEAD_ZONE = 40
GLIDE = 0.25
SLOW_GLIDE = 0.08
EDGE_MARGIN = 5

# Touchpad takeover
MANUAL_HOLD = 1.0            # s the eyes stay hands-off after you touch the touchpad

# Button snapping
CLICKABLE_TYPES = {
    "ButtonControl", "HyperlinkControl", "MenuItemControl", "TabItemControl",
    "ListItemControl", "CheckBoxControl", "RadioButtonControl", "ComboBoxControl",
    "EditControl", "SplitButtonControl", "TreeItemControl",
}
MIN_TARGET = 8
MAX_TARGET_W, MAX_TARGET_H = 600, 200
LOCK_DWELL = 0.15
STICKY_PAD = 40
QUERY_INTERVAL = 0.08

# Blink click
CLICK_BLINK_MIN = 0.35
CLICK_BLINK_MAX = 1.5
SETTLE_TIME = 0.15
PRE_BLINK_LOOKBACK = 0.15

# Learning
LEARN_FROM_TOUCHPAD = True   # real clicks = "this is where I meant"
LEARN_FROM_BLINK = True      # blink clicks on a locked button
LEARN_WINDOW = 0.3           # s of gaze before the click that gets averaged
MIN_LEARN_FRAMES = 3         # need at least this many gaze readings in that window
LEARN_SIGMA = 200
LEARN_PRIOR = 1.0
LEARN_MAX_SHIFT = 400        # px; bigger = probably not looking at what you clicked
LEARN_MAX_SAMPLES = 400

pyautogui.PAUSE = 0


# ================= Learning system =================
class GazeCorrector:
    """Learns how far off the tracker is in each part of the screen, from your clicks."""

    def __init__(self, path):
        self.path = path
        self.samples = []  # each: [raw_x, raw_y, error_x, error_y]
        if path.exists():
            try:
                self.samples = json.loads(path.read_text())
            except Exception:
                self.samples = []

    def correct(self, x, y):
        if not self.samples:
            return x, y
        s = np.asarray(self.samples, dtype=float)
        dist2 = (s[:, 0] - x) ** 2 + (s[:, 1] - y) ** 2
        w = np.exp(-dist2 / (2 * LEARN_SIGMA ** 2))
        denom = w.sum() + LEARN_PRIOR
        return x + (w * s[:, 2]).sum() / denom, y + (w * s[:, 3]).sum() / denom

    def learn(self, raw, target):
        ex, ey = target[0] - raw[0], target[1] - raw[1]
        if np.hypot(ex, ey) > LEARN_MAX_SHIFT:
            return False, "error too big"
        self.samples.append([float(raw[0]), float(raw[1]), float(ex), float(ey)])
        self.samples = self.samples[-LEARN_MAX_SAMPLES:]
        self.save()
        return True, "ok"

    def undo(self):
        if self.samples:
            self.samples.pop()
            self.save()

    def clear(self):
        self.samples = []
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.samples))


# ================= Debug log =================
new_log = not LOG_FILE.exists()
log_fh = LOG_FILE.open("a", newline="")
log = csv.writer(log_fh)
if new_log:
    log.writerow(["time", "source", "raw_x", "raw_y", "corrected_x", "corrected_y",
                  "target_x", "target_y", "raw_error_px", "corrected_error_px",
                  "frames", "learned", "reason"])


def learn_from_click(source, click_t, target):
    """Compare where the tracker thought you looked with where you actually clicked."""
    tx, ty = target
    window = [h for h in history if click_t - LEARN_WINDOW <= h[0] <= click_t]

    if len(window) < MIN_LEARN_FRAMES:
        rx = ry = cx = cy = raw_err = cor_err = ""
        learned, reason = False, "no gaze data (eyes closed / no face?)"
    else:
        rx = float(np.mean([h[1] for h in window]))
        ry = float(np.mean([h[2] for h in window]))
        cx, cy = corrector.correct(rx, ry)  # what the system predicted BEFORE this lesson
        raw_err = round(float(np.hypot(tx - rx, ty - ry)))
        cor_err = round(float(np.hypot(tx - cx, ty - cy)))
        learned, reason = corrector.learn((rx, ry), (tx, ty))
        rx, ry, cx, cy = round(rx), round(ry), round(cx), round(cy)

    log.writerow([f"{click_t:.2f}", source, rx, ry, cx, cy, tx, ty,
                  raw_err, cor_err, len(window), learned, reason])
    log_fh.flush()
    print(f"[{source}] target=({tx},{ty}) raw_err={raw_err}px "
          f"corrected_err={cor_err}px learned={learned} ({reason})")
    flash("learned" if learned else f"not learned: {reason}")


# ================= Button detection =================
def find_target(x, y):
    if uia is None:
        return None
    try:
        ctrl = uia.ControlFromPoint(int(x), int(y))
        for _ in range(4):
            if ctrl is None:
                return None
            if ctrl.ControlTypeName in CLICKABLE_TYPES:
                r = ctrl.BoundingRectangle
                w, h = r.width(), r.height()
                if MIN_TARGET <= w <= MAX_TARGET_W and MIN_TARGET <= h <= MAX_TARGET_H:
                    return (r.left, r.top, r.right, r.bottom)
                return None
            ctrl = ctrl.GetParentControl()
    except Exception:
        pass
    return None


def inside(rect, x, y, pad=0):
    left, top, right, bottom = rect
    return left - pad <= x < right + pad and top - pad <= y < bottom + pad


def center(rect):
    return (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


# ================= Touchpad listener (runs in its own thread) =================
touch_clicks = queue.Queue()
last_manual_move = 0.0


def on_move(x, y, injected=False):
    global last_manual_move
    if not injected:  # injected = moved by our own pyautogui calls -> ignore
        last_manual_move = time.time()


def on_click(x, y, button, pressed, injected=False):
    global last_manual_move
    if pressed and not injected and button == mouse.Button.left:
        last_manual_move = time.time()
        touch_clicks.put((time.time(), int(x), int(y)))


# ================= Calibration =================
tracker = GazeEstimator(model_name="ridge")
corrector = GazeCorrector(LEARN_FILE)

if MODEL_FILE.exists() and not RECALIBRATE:
    tracker.load_model(MODEL_FILE)
    print(f"Loaded calibration, {len(corrector.samples)} learned corrections")
else:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if MODEL_FILE.exists():
        MODEL_FILE.rename(f"gaze_model_backup_{stamp}.pkl")
    if LEARN_FILE.exists():
        LEARN_FILE.rename(f"gaze_corrections_backup_{stamp}.json")
    run_dense_grid_calibration(tracker, rows=GRID_ROWS, cols=GRID_COLS,
                               camera_index=CAMERA_INDEX)
    tracker.save_model(MODEL_FILE)
    corrector.clear()
    print(f"New calibration saved, old one backed up as gaze_model_backup_{stamp}.pkl")

smoother = KalmanEMASmoother(make_kalman(process_var=5.0), ema_alpha=EMA_ALPHA)
smoother.tune(tracker, camera_index=CAMERA_INDEX)


# ================= Setup =================
screen_w, screen_h = pyautogui.size()
camera = cv2.VideoCapture(CAMERA_INDEX)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 320, 240)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)

listener = mouse.Listener(on_move=on_move, on_click=on_click)
listener.start()

raw_hist = deque(maxlen=MEDIAN_WINDOW)
history = deque()        # (time, raw_x, raw_y, cursor_x, cursor_y, locked_rect)
cursor_x, cursor_y = map(float, pyautogui.position())
locked = None
candidate = None
candidate_since = 0.0
last_query = 0.0
blink_start = None
reopened_at = 0.0
paused = False
snapping = uia is not None
message, message_until = "", 0.0


def flash(text):
    global message, message_until
    message, message_until = text, time.time() + 1.5


def handle_blink_click(blink_began):
    global cursor_x, cursor_y
    snap_t = blink_began - PRE_BLINK_LOOKBACK
    before = [h for h in history if h[0] <= snap_t]
    if not before:
        pyautogui.click(int(cursor_x), int(cursor_y))
        return

    _, _, _, cx, cy, rect = before[-1]
    if rect is None:
        pyautogui.click(int(cx), int(cy))
        cursor_x, cursor_y = cx, cy
        flash("CLICK")
        return

    tx, ty = center(rect)
    pyautogui.click(tx, ty)
    cursor_x, cursor_y = float(tx), float(ty)
    if LEARN_FROM_BLINK:
        learn_from_click("blink", snap_t, (tx, ty))


# ================= Main loop =================
try:
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        now = time.time()
        features, blink = tracker.extract_features(frame)
        manual = now - last_manual_move < MANUAL_HOLD

        if features is None:
            status = "no face"
            blink_start = None

        elif blink:
            status = "eyes closed"
            if blink_start is None:
                blink_start = now

        else:
            status = "TOUCHPAD" if manual else ("LOCKED" if locked else "tracking")

            if blink_start is not None:
                duration = now - blink_start
                if not paused and not manual and CLICK_BLINK_MIN <= duration <= CLICK_BLINK_MAX:
                    handle_blink_click(blink_start)
                blink_start = None
                reopened_at = now

            if now - reopened_at >= SETTLE_TIME:
                x, y = tracker.predict(np.array([features]))[0]
                raw_hist.append((x, y))
                mx, my = np.median(raw_hist, axis=0)
                sx, sy = smoother.step(int(mx), int(my))

                gx, gy = corrector.correct(sx, sy)
                gx = clamp(gx, EDGE_MARGIN, screen_w - 1 - EDGE_MARGIN)
                gy = clamp(gy, EDGE_MARGIN, screen_h - 1 - EDGE_MARGIN)

                if locked and not inside(locked, gx, gy, STICKY_PAD):
                    locked = None
                if candidate and not inside(candidate, gx, gy):
                    candidate = None
                if snapping and locked is None and now - last_query >= QUERY_INTERVAL:
                    last_query = now
                    rect = find_target(gx, gy)
                    if rect is None:
                        candidate = None
                    elif rect != candidate:
                        candidate, candidate_since = rect, now
                    elif now - candidate_since >= LOCK_DWELL:
                        locked, candidate = rect, None

                if locked:
                    tx, ty = center(locked)
                    speed, should_move = GLIDE, True
                elif candidate:
                    tx, ty = gx, gy
                    speed, should_move = SLOW_GLIDE, True
                else:
                    tx, ty = gx, gy
                    speed = GLIDE
                    should_move = np.hypot(tx - cursor_x, ty - cursor_y) > DEAD_ZONE

                if manual:
                    # You're driving: keep our position in sync, don't fight you
                    cursor_x, cursor_y = map(float, pyautogui.position())
                elif not paused and should_move:
                    cursor_x += (tx - cursor_x) * speed
                    cursor_y += (ty - cursor_y) * speed
                    pyautogui.moveTo(int(cursor_x), int(cursor_y))

                # Always record gaze, even while you use the touchpad (needed for learning)
                history.append((now, sx, sy, cursor_x, cursor_y, locked))
                while history and now - history[0][0] > 3.0:
                    history.popleft()

        # Learn from real touchpad clicks
        while not touch_clicks.empty():
            click_t, click_x, click_y = touch_clicks.get()
            if LEARN_FROM_TOUCHPAD:
                learn_from_click("touchpad", click_t, (click_x, click_y))

        # Preview window
        preview = cv2.flip(frame, 1)
        lines = [
            "PAUSED" if paused else status,
            f"learned: {len(corrector.samples)}  snap: {'on' if snapping else 'off'}",
        ]
        if now < message_until:
            lines.append(message)
        for i, text in enumerate(lines):
            cv2.putText(preview, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255) if paused else (0, 255, 0), 2)
        cv2.imshow(WINDOW, preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("s") and uia is not None:
            snapping, locked, candidate = not snapping, None, None
        elif key == ord("u"):
            corrector.undo()
            flash("undid last lesson")
        elif key == ord("c"):
            corrector.clear()
            flash("cleared all lessons")

        # X button: the window stops being visible once it's closed
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    print("Stopped with Ctrl+C")
except pyautogui.FailSafeException:
    print("Stopped by PyAutoGUI fail-safe (mouse in a screen corner)")
finally:
    listener.stop()
    camera.release()
    cv2.destroyAllWindows()
    tracker.close()
    log_fh.close()
    print(f"Closed. {len(corrector.samples)} lessons saved to {LEARN_FILE}, log in {LOG_FILE}")