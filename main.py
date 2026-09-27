import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyautogui
from eyetrax import GazeEstimator
from eyetrax.calibration import run_dense_grid_calibration
from eyetrax.filters import KalmanEMASmoother, make_kalman


# ---------------- Settings ----------------
MODEL_FILE = Path("gaze_model.pkl")
RECALIBRATE = True        # set True (or delete gaze_model.pkl) to redo calibration
CAMERA_INDEX = 0

GRID_ROWS, GRID_COLS = 5, 5  # more dots = better coverage, longer calibration
EMA_ALPHA = 0.5              # 0 = no extra smoothing, closer to 1 = smoother but laggier
DEAD_ZONE = 15               # px, ignore tiny movements of the smoothed cursor
EDGE_MARGIN = 5              # px, stay off the exact corners (PyAutoGUI fail-safe)

CLICK_BLINK_MIN = 0.35       # s, eyes closed at least this long = click
CLICK_BLINK_MAX = 1.5        # s, closed longer than this = just resting eyes, no click
SETTLE_TIME = 0.15           # s, ignore gaze right after eyes reopen (it's jumpy)
PRE_BLINK_LOOKBACK = 0.15    # s, click where you looked just BEFORE the lids started closing

pyautogui.PAUSE = 0          # default is a 0.1 s sleep after EVERY pyautogui call


# ---------------- Calibration ----------------
tracker = GazeEstimator(model_name="ridge")  # also try "tiny_mlp" or "svr"

if MODEL_FILE.exists() and not RECALIBRATE:
    tracker.load_model(MODEL_FILE)
    print(f"Loaded calibration from {MODEL_FILE}")
else:
    run_dense_grid_calibration(
        tracker, rows=GRID_ROWS, cols=GRID_COLS, camera_index=CAMERA_INDEX
    )
    tracker.save_model(MODEL_FILE)
    print(f"Saved calibration to {MODEL_FILE}")

# Kalman filter + EMA smoothing; tune() shows 3 dots to measure your jitter
smoother = KalmanEMASmoother(make_kalman(), ema_alpha=EMA_ALPHA)
smoother.tune(tracker, camera_index=CAMERA_INDEX)


# ---------------- Setup ----------------
screen_w, screen_h = pyautogui.size()
camera = cv2.VideoCapture(CAMERA_INDEX)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # fewer stale frames = less lag

cv2.namedWindow("Eye Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Eye Tracker", 320, 240)
cv2.setWindowProperty("Eye Tracker", cv2.WND_PROP_TOPMOST, 1)


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def position_before(history, t, fallback):
    """Most recent cursor position recorded at or before time t."""
    pos = fallback
    for ht, hx, hy in history:
        if ht > t:
            break
        pos = (hx, hy)
    return pos


history = deque()                      # (time, x, y) of recent smoothed positions
cursor_x, cursor_y = screen_w // 2, screen_h // 2
blink_start = None
reopened_at = 0.0
paused = False


# ---------------- Main loop ----------------
while True:
    ok, frame = camera.read()
    if not ok:
        break

    now = time.time()
    features, blink = tracker.extract_features(frame)

    if features is None:
        status = "no face"
        blink_start = None

    elif blink:
        status = "eyes closed"
        if blink_start is None:
            blink_start = now

    else:
        status = "tracking"

        # Eyes just reopened: was that a deliberate long blink?
        if blink_start is not None:
            duration = now - blink_start
            if not paused and CLICK_BLINK_MIN <= duration <= CLICK_BLINK_MAX:
                cx, cy = position_before(
                    history, blink_start - PRE_BLINK_LOOKBACK, (cursor_x, cursor_y)
                )
                pyautogui.click(cx, cy)
                cursor_x, cursor_y = cx, cy
                status = "CLICK"
            blink_start = None
            reopened_at = now

        # Move the cursor (skip the jumpy frames right after a blink)
        if now - reopened_at >= SETTLE_TIME:
            x, y = tracker.predict(np.array([features]))[0]
            sx, sy = smoother.step(int(x), int(y))
            sx = clamp(sx, EDGE_MARGIN, screen_w - 1 - EDGE_MARGIN)
            sy = clamp(sy, EDGE_MARGIN, screen_h - 1 - EDGE_MARGIN)

            history.append((now, sx, sy))
            while history and now - history[0][0] > 2.0:
                history.popleft()

            if not paused and np.hypot(sx - cursor_x, sy - cursor_y) > DEAD_ZONE:
                pyautogui.moveTo(sx, sy)
                cursor_x, cursor_y = sx, sy

    # Preview (mirrored only for display; the tracker sees the raw frame)
    preview = cv2.flip(frame, 1)
    label = "PAUSED (p)" if paused else status
    cv2.putText(preview, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255) if paused else (0, 255, 0), 2)
    cv2.imshow("Eye Tracker", preview)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("p"):
        paused = not paused


# ---------------- Cleanup ----------------
camera.release()
cv2.destroyAllWindows()
tracker.close()