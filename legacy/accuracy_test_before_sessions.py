"""Archived accuracy tester; use ../accuracy_test.py for current evaluations.

Usage:
    python accuracy_test.py
    python accuracy_test.py my_model.pkl
    python accuracy_test.py my_model.pkl my_corrections.json

Each camera sample is evaluated twice:
    baseline  = prediction from the calibrated EyeTrax model
    corrected = the same prediction after gaze_corrections.json is applied

Each completed test is appended to accuracy_history.csv, so later runs show
how accuracy changes as the number of learned lessons grows. Esc aborts.
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui
from eyetrax import GazeEstimator


# ================= Files and test settings =================
MODEL_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("gaze_model.pkl")
LEARN_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("gaze_corrections.json")
HISTORY_FILE = Path("accuracy_history.csv")
CAMERA_INDEX = 0
SETTLE = 0.8      # seconds to let the eyes land on the dot (not recorded)
RECORD = 1.2      # seconds of gaze recorded per dot
WINDOW = "Accuracy Test"

# Keep these synchronized with the learning settings in main.py.
LEARN_SIGMA = 200
LEARN_PRIOR = 1.0

RAW_COLOR = (0, 0, 255)          # red (OpenCV uses BGR)
CORRECTED_COLOR = (255, 255, 0)  # cyan
TARGET_COLOR = (0, 255, 0)       # green


# ================= Learned correction =================
def load_lessons(path):
    """Load [raw_x, raw_y, error_x, error_y] lessons, ignoring invalid files."""
    if not path.exists():
        return np.empty((0, 4), dtype=float)
    try:
        lessons = np.asarray(json.loads(path.read_text()), dtype=float)
        if lessons.size == 0:
            return np.empty((0, 4), dtype=float)
        if lessons.ndim != 2 or lessons.shape[1] != 4:
            raise ValueError("expected rows of [raw_x, raw_y, error_x, error_y]")
        return lessons
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Warning: could not load {path}: {exc}")
        return np.empty((0, 4), dtype=float)


def apply_correction(predictions, lessons):
    """Apply the same local weighted correction used by GazeCorrector in main.py."""
    predictions = np.asarray(predictions, dtype=float)
    if len(lessons) == 0:
        return predictions.copy()

    positions = lessons[:, :2]
    offsets = lessons[:, 2:]
    dist2 = ((predictions[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2)
    weights = np.exp(-dist2 / (2 * LEARN_SIGMA ** 2))
    correction = (weights @ offsets) / (
        weights.sum(axis=1, keepdims=True) + LEARN_PRIOR
    )
    return predictions + correction


def median_or_nan(values):
    return float(np.median(values)) if values else float("nan")


def fmt_px(value):
    return "-" if np.isnan(value) else f"{value:.0f}px"


def is_edge_target(index, target, screen_center):
    """The first nine targets form the edge grid; its center counts as inner."""
    return index < 9 and target != screen_center


# ================= Persistent run history =================
HISTORY_FIELDS = [
    "timestamp", "model", "model_modified", "lessons", "targets",
    "raw_median_px", "corrected_median_px", "improvement_percent",
    "raw_edges_px", "corrected_edges_px", "raw_inner_px", "corrected_inner_px",
]


def append_history(row):
    new_file = not HISTORY_FILE.exists()
    with HISTORY_FILE.open("a", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=HISTORY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def read_history_for_current_model():
    """Exclude runs from older calibrations, even if they used the same filename."""
    if not HISTORY_FILE.exists():
        return []
    model_modified = str(MODEL_FILE.stat().st_mtime_ns)
    with HISTORY_FILE.open(newline="") as file_handle:
        return [
            row for row in csv.DictReader(file_handle)
            if row["model"] == str(MODEL_FILE)
            and row["model_modified"] == model_modified
        ]


def draw_history(history):
    """Draw raw and corrected median error across runs for this calibration."""
    width, height = 1200, 650
    left, right, top, bottom = 100, 50, 80, 100
    graph = np.full((height, width, 3), 35, np.uint8)
    if not history:
        return graph

    raw = np.array([float(row["raw_median_px"]) for row in history])
    corrected = np.array([float(row["corrected_median_px"]) for row in history])
    lesson_counts = [row["lessons"] for row in history]
    y_max = max(100.0, np.ceil(max(raw.max(), corrected.max()) / 100) * 100)

    cv2.putText(graph, "Accuracy across learning runs", (left, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)
    cv2.putText(graph, "red = baseline   cyan = learned correction", (left + 500, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    cv2.line(graph, (left, top), (left, height - bottom), (180, 180, 180), 2)
    cv2.line(graph, (left, height - bottom), (width - right, height - bottom),
             (180, 180, 180), 2)
    for fraction in np.linspace(0, 1, 6):
        y = int(height - bottom - fraction * (height - top - bottom))
        value = int(fraction * y_max)
        cv2.line(graph, (left, y), (width - right, y), (65, 65, 65), 1)
        cv2.putText(graph, str(value), (35, y + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (190, 190, 190), 1)

    count = len(history)
    if count > 1:
        x_positions = np.linspace(left + 20, width - right - 20, count)
    else:
        x_positions = np.array([width // 2])

    def make_points(values):
        return [
            (int(x), int(height - bottom - value / y_max * (height - top - bottom)))
            for x, value in zip(x_positions, values)
        ]

    raw_points = make_points(raw)
    corrected_points = make_points(corrected)
    for points, color in ((raw_points, RAW_COLOR),
                          (corrected_points, CORRECTED_COLOR)):
        for start, end in zip(points, points[1:]):
            cv2.line(graph, start, end, color, 3)
        for point in points:
            cv2.circle(graph, point, 6, color, -1)

    # Keep labels readable after the history grows, while retaining every point.
    label_every = max(1, int(np.ceil(count / 10)))
    for index, (x, lesson_count) in enumerate(zip(x_positions, lesson_counts)):
        if index % label_every == 0 or index == count - 1:
            cv2.putText(graph, lesson_count, (int(x) - 10, height - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(graph, "learned lessons at time of test",
                (width // 2 - 145, height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)
    cv2.putText(graph, "median error (px)", (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return graph


# ================= Capture gaze samples =================
if not MODEL_FILE.exists():
    sys.exit(f"Model not found: {MODEL_FILE}")

lessons = load_lessons(LEARN_FILE)
screen_width, screen_height = pyautogui.size()
screen_center = (screen_width // 2, screen_height // 2)

# Nine points near the edges/corners plus four inner points.
fractions = [(x, y) for y in (0.05, 0.5, 0.95) for x in (0.05, 0.5, 0.95)]
fractions += [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]
targets = [
    (int(x_fraction * screen_width), int(y_fraction * screen_height))
    for x_fraction, y_fraction in fractions
]

tracker = GazeEstimator()
tracker.load_model(MODEL_FILE)
camera = cv2.VideoCapture(CAMERA_INDEX)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

canvas = np.full((screen_height, screen_width, 3), 40, np.uint8)
intro = f"Testing baseline + {len(lessons)} learned corrections. Press any key to start."
cv2.putText(canvas, intro, (max(40, screen_width // 2 - 480), screen_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
cv2.imshow(WINDOW, canvas)
if cv2.waitKey(0) == 27:
    camera.release()
    tracker.close()
    cv2.destroyAllWindows()
    sys.exit("Aborted")

results = []  # each entry is (target, list of raw model predictions)
aborted = False
for target_x, target_y in targets:
    predictions = []
    start = time.time()
    while time.time() - start < SETTLE + RECORD:
        ok, frame = camera.read()
        if not ok:
            continue
        elapsed = time.time() - start
        features, blink = tracker.extract_features(frame)
        if elapsed > SETTLE and features is not None and not blink:
            predictions.append(tracker.predict(np.array([features]))[0])

        canvas = np.full((screen_height, screen_width, 3), 40, np.uint8)
        radius = 18 if elapsed < SETTLE else 10
        cv2.circle(canvas, (target_x, target_y), radius, (0, 255, 255), -1)
        cv2.circle(canvas, (target_x, target_y), 3, (0, 0, 0), -1)
        cv2.imshow(WINDOW, canvas)
        if cv2.waitKey(1) == 27:
            aborted = True
            break
    if aborted:
        break
    results.append(((target_x, target_y), predictions))

camera.release()
tracker.close()
if aborted:
    cv2.destroyAllWindows()
    sys.exit("Aborted")


# ================= Compare baseline and learned correction =================
canvas = np.full((screen_height, screen_width, 3), 40, np.uint8)
raw_errors, corrected_errors = [], []
raw_edges, corrected_edges = [], []
raw_inner, corrected_inner = [], []

print(f"\nResults for {MODEL_FILE} with {len(lessons)} learned lessons")
print(f"{'target':>14} {'raw':>8} {'corrected':>10} {'change':>9} "
      f"{'raw jit':>9} {'cor jit':>9} {'samples':>8}")

for index, ((target_x, target_y), predictions) in enumerate(results):
    target = (target_x, target_y)
    cv2.circle(canvas, target, 10, TARGET_COLOR, -1)
    if len(predictions) < 5:
        print(f"({target_x:4},{target_y:4})    no data (blinking / face lost)")
        continue

    raw = np.asarray(predictions, dtype=float)
    corrected = apply_correction(raw, lessons)
    raw_mean = raw.mean(axis=0)
    corrected_mean = corrected.mean(axis=0)
    raw_error = float(np.linalg.norm(raw_mean - target))
    corrected_error = float(np.linalg.norm(corrected_mean - target))
    raw_jitter = float(np.mean(np.linalg.norm(raw - raw_mean, axis=1)))
    corrected_jitter = float(
        np.mean(np.linalg.norm(corrected - corrected_mean, axis=1))
    )
    change = (1 - corrected_error / raw_error) * 100 if raw_error else 0.0

    raw_errors.append(raw_error)
    corrected_errors.append(corrected_error)
    if is_edge_target(index, target, screen_center):
        raw_edges.append(raw_error)
        corrected_edges.append(corrected_error)
    else:
        raw_inner.append(raw_error)
        corrected_inner.append(corrected_error)

    raw_point = tuple(
        np.clip(raw_mean, (0, 0), (screen_width - 1, screen_height - 1)).astype(int)
    )
    corrected_point = tuple(
        np.clip(corrected_mean, (0, 0),
                (screen_width - 1, screen_height - 1)).astype(int)
    )
    cv2.line(canvas, target, raw_point, RAW_COLOR, 2)
    cv2.circle(canvas, raw_point, int(max(raw_jitter, 5)), RAW_COLOR, 2)
    cv2.line(canvas, target, corrected_point, CORRECTED_COLOR, 2)
    cv2.circle(canvas, corrected_point, int(max(corrected_jitter, 5)),
               CORRECTED_COLOR, 2)
    label_y = target_y + 25 if target_y < screen_height // 2 else target_y - 15
    cv2.putText(canvas, f"R{raw_error:.0f} C{corrected_error:.0f}",
                (target_x + 15, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)
    print(f"({target_x:4},{target_y:4}) {raw_error:7.0f}px "
          f"{corrected_error:9.0f}px {change:8.0f}% {raw_jitter:8.0f}px "
          f"{corrected_jitter:8.0f}px {len(predictions):8}")

if not raw_errors:
    cv2.destroyAllWindows()
    sys.exit("No targets had enough valid gaze samples")

raw_median = median_or_nan(raw_errors)
corrected_median = median_or_nan(corrected_errors)
improvement = (1 - corrected_median / raw_median) * 100 if raw_median else 0.0
raw_edge_median = median_or_nan(raw_edges)
corrected_edge_median = median_or_nan(corrected_edges)
raw_inner_median = median_or_nan(raw_inner)
corrected_inner_median = median_or_nan(corrected_inner)

print("\nSUMMARY (same gaze samples)")
print(f"  baseline:  median {fmt_px(raw_median)} | "
      f"edges {fmt_px(raw_edge_median)} | inner {fmt_px(raw_inner_median)}")
print(f"  corrected: median {fmt_px(corrected_median)} | "
      f"edges {fmt_px(corrected_edge_median)} | inner {fmt_px(corrected_inner_median)}")
print(f"  overall change: {improvement:+.0f}%  (positive means learning helped)")

summary_y = screen_height // 2 - 80
cv2.rectangle(canvas, (25, summary_y - 45),
              (min(screen_width - 25, 1120), summary_y + 95), (20, 20, 20), -1)
cv2.putText(canvas,
            f"BASELINE  median {fmt_px(raw_median)} | edge {fmt_px(raw_edge_median)} | inner {fmt_px(raw_inner_median)}",
            (40, summary_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, RAW_COLOR, 2)
cv2.putText(canvas,
            f"LEARNED   median {fmt_px(corrected_median)} | edge {fmt_px(corrected_edge_median)} | inner {fmt_px(corrected_inner_median)}",
            (40, summary_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            CORRECTED_COLOR, 2)
cv2.putText(canvas, f"change {improvement:+.0f}% using {len(lessons)} lessons",
            (40, summary_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (240, 240, 240), 2)
cv2.putText(canvas, "green target | red baseline | cyan learned | any key closes",
            (40, screen_height // 2 + 80), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (210, 210, 210), 2)


# ================= Save this run and the cross-run learning curve =================
stamp = time.strftime("%Y%m%d_%H%M%S")
comparison_file = f"accuracy_comparison_{MODEL_FILE.stem}_{stamp}.png"
cv2.imwrite(comparison_file, canvas)

append_history({
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model": str(MODEL_FILE),
    "model_modified": str(MODEL_FILE.stat().st_mtime_ns),
    "lessons": len(lessons),
    "targets": len(raw_errors),
    "raw_median_px": f"{raw_median:.2f}",
    "corrected_median_px": f"{corrected_median:.2f}",
    "improvement_percent": f"{improvement:.2f}",
    "raw_edges_px": f"{raw_edge_median:.2f}",
    "corrected_edges_px": f"{corrected_edge_median:.2f}",
    "raw_inner_px": f"{raw_inner_median:.2f}",
    "corrected_inner_px": f"{corrected_inner_median:.2f}",
})

history = read_history_for_current_model()
history_canvas = draw_history(history)
history_image = "accuracy_learning_curve.png"
cv2.imwrite(history_image, history_canvas)

print(f"\nSaved comparison: {comparison_file}")
print(f"Saved run history: {HISTORY_FILE}")
print(f"Saved learning curve: {history_image}")
print("\nRUN HISTORY FOR THIS CALIBRATION")
print(f"{'run':>4} {'lessons':>8} {'baseline':>10} "
      f"{'corrected':>10} {'change':>9}")
for index, row in enumerate(history, start=1):
    print(f"{index:4} {row['lessons']:>8} {float(row['raw_median_px']):9.0f}px "
          f"{float(row['corrected_median_px']):9.0f}px "
          f"{float(row['improvement_percent']):+8.0f}%")

cv2.imshow(WINDOW, canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
