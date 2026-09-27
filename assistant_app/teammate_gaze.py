"""Teammate gaze settings, target detection and dense-grid calibration.

Extracted from the supplied main.py. Keep gaze_pipeline.py, lesson_store.py,
and their own dependencies/data from the same teammate revision.
"""
import shutil
import time
from collections import deque
import cv2
import numpy as np
import pyautogui
from eyetrax import GazeEstimator
from eyetrax.calibration import run_dense_grid_calibration
from gaze_pipeline import MODEL_FILE, Pipeline, Corrector, context, load_tracker
from lesson_store import Store
from sklearn.utils.validation import check_is_fitted
from pynput import mouse
try:
    import uiautomation as uia
except ImportError:
    uia = None
    print('uiautomation not installed -> button snapping disabled')
RECALIBRATE = True
CAMERA_INDEX = 0
GRID_ROWS, GRID_COLS = (5, 5)
WINDOW = 'Eye Tracker'
DEAD_ZONE = 40
GLIDE = 0.25
SLOW_GLIDE = 0.08
EDGE_MARGIN = 5
MANUAL_HOLD = 1.0
CLICKABLE_TYPES = {'ButtonControl', 'HyperlinkControl', 'MenuItemControl', 'TabItemControl', 'ListItemControl', 'CheckBoxControl', 'RadioButtonControl', 'ComboBoxControl', 'EditControl', 'SplitButtonControl', 'TreeItemControl'}
MIN_TARGET = 8
MAX_TARGET_W, MAX_TARGET_H = (600, 200)
LOCK_DWELL = 0.15
STICKY_PAD = 40
QUERY_INTERVAL = 0.08
CLICK_BLINK_MIN = 0.35
CLICK_BLINK_MAX = 1.5
PRE_BLINK_LOOKBACK = 0.15

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
                w, h = (r.width(), r.height())
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
    return ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def calibrate_tracker():
    if MODEL_FILE.exists() and (not RECALIBRATE):
        tracker = load_tracker(MODEL_FILE)
    else:
        tracker = GazeEstimator(model_name='ridge')
        temporary = MODEL_FILE.with_suffix('.pending.pkl')
        try:
            run_dense_grid_calibration(tracker, rows=GRID_ROWS, cols=GRID_COLS, camera_index=CAMERA_INDEX)
            check_is_fitted(tracker.model.scaler)
            tracker.predict(np.zeros((1, tracker.model.scaler.n_features_in_)))
            tracker.save_model(temporary)
            if MODEL_FILE.exists():
                stamp = time.strftime('%Y%m%d_%H%M%S') + f'_{time.time_ns()}'
                shutil.copy2(MODEL_FILE, MODEL_FILE.with_name(f'gaze_model_backup_{stamp}.pkl'))
            temporary.replace(MODEL_FILE)
        except Exception:
            tracker.close()
            raise
        finally:
            temporary.unlink(missing_ok=True)
    return tracker

def load_correction():
    store = Store()
    ctx = context(MODEL_FILE, pyautogui.size())
    manifest = store.manifest()
    samples, accepted_ids, skipped_ids = store.active(ctx, manifest)
    corrector = Corrector(samples)
    pipeline = Pipeline()
    print(f'Loaded {len(accepted_ids)} accepted sessions ({len(samples)} fixations).')
    if skipped_ids:
        print(f'Skipped {len(skipped_ids)} incompatible sessions; revalidate after calibration/settings changes.')
    return (corrector, pipeline, accepted_ids)
