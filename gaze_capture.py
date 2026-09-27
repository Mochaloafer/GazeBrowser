"""Controlled fixation capture. Saves coordinates only, never camera images."""
import random
import time
import numpy as np
from gaze_pipeline import Pipeline, load_tracker


def targets(screen, training):
    fractions = ([0.08, 0.29, 0.5, 0.71, 0.92] if training else [0.05, 0.35, 0.65, 0.95])
    points = [[int(x * screen[0]), int(y * screen[1])] for y in fractions for x in fractions]
    random.shuffle(points)
    return points


def capture(model, screen, training=False, camera_index=0, settle=1.5, record=1.5):
    if not np.isfinite([settle, record]).all() or settle < 0 or record <= 0:
        raise ValueError("Settle must be nonnegative and record must be positive seconds")
    import cv2
    tracker = load_tracker(model)
    camera = None
    window = "Lesson training" if training else "Three-way accuracy test"
    pipeline = Pipeline()
    points = targets(screen, training)
    rows = []
    try:
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise ValueError("Camera could not be opened")
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        width, height = screen
        def show(point=None, message=""):
            canvas = np.full((height, width, 3), 35, np.uint8)
            if point is not None:
                cv2.circle(canvas, tuple(point), 12, (0, 255, 255), -1)
                cv2.circle(canvas, tuple(point), 3, (0, 0, 0), -1)
            cv2.putText(canvas, message, (30, height // 2 if point is None else height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1)
            cv2.imshow(window, canvas)
        show(message="Look at each dot until it moves. Any key starts; Esc cancels.")
        if cv2.waitKey(0) == 27:
            raise ValueError("Capture cancelled; no session saved")
        origin = time.monotonic()
        for index, point in enumerate(points):
            show(point, f"{index + 1}/{len(points)} - Keep looking at the dot. Esc cancels.")
            if cv2.waitKey(1) == 27:
                raise ValueError("Capture cancelled; no session saved")
            start = time.monotonic()
            while time.monotonic() - start < settle + record:
                ok, frame = camera.read()
                now = time.monotonic()
                if not ok:
                    raise ValueError("Camera stopped delivering frames; no session saved")
                features, blink = tracker.extract_features(frame)
                raw = None
                status = "no_face" if features is None else "blink" if blink else "valid"
                if features is not None and not blink:
                    prediction = tracker.predict(np.array([features]))[0]
                    if np.isfinite(prediction).all():
                        raw = prediction.tolist()
                    else:
                        status = "nonfinite"
                filtered = pipeline.step(raw, now - origin)
                rows.append(dict(t=now-origin, target=index, raw=raw,
                                 scored=now-start >= settle, status=status,
                                 filtered=None if filtered is None else filtered.tolist()))
                if cv2.waitKey(1) == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    raise ValueError("Capture cancelled; no session saved")
        return dict(kind="training" if training else "evaluation", targets=points, frames=rows,
                    timing=dict(settle=settle, record=record))
    finally:
        if camera is not None:
            camera.release()
        tracker.close()
        cv2.destroyAllWindows()


def replay(recording):
    pipeline = Pipeline()
    result = []
    for frame in recording["frames"]:
        value = pipeline.step(frame["raw"], frame["t"])
        if frame["scored"]:
            result.append((frame["target"], value))
    return result


def lessons_from(recording):
    samples, quality = [], []
    replayed = replay(recording)
    for index, target in enumerate(recording["targets"]):
        frames = [v for i, v in replayed if i == index]
        valid = [v for v in frames if v is not None]
        coverage = len(valid) / max(len(frames), 1)
        jitter = None
        offset_px = None
        reasons = []
        if len(valid) < 15:
            reasons.append("too_few_samples")
        if coverage < 0.8:
            reasons.append("low_tracking_coverage")
        if valid:
            values = np.array(valid)
            center = np.median(values, axis=0)
            jitter = float(np.percentile(np.linalg.norm(values-center, axis=1), 90))
            offset = np.asarray(target)-center
            offset_px = float(np.linalg.norm(offset))
            if jitter > 60:
                reasons.append("unstable_gaze")
            if offset_px > 400:
                reasons.append("offset_too_large")
            if not reasons:
                samples.append([*center.tolist(), *offset.tolist()])
        scored = [f for f in recording["frames"] if f["target"] == index and f["scored"]]
        statuses = {key: sum(f.get("status") == key for f in scored)
                    for key in ("valid", "no_face", "blink", "nonfinite")}
        quality.append(dict(target=target, samples=len(valid), coverage=coverage,
                            jitter_p90=jitter, offset_px=offset_px, reasons=reasons,
                            capture_statuses=statuses, accepted=not reasons))
    return samples, quality


def print_quality(quality):
    from collections import Counter
    print("\nFixation diagnostics (jitter limit 60px; offset limit 400px):")
    print("target          samples  coverage  jitter90  offset  result")
    def fmt(value):
        return "n/a" if value is None else f"{value:.0f}px"
    for item in quality:
        result = ", ".join(item["reasons"]) or "accepted"
        print(f"{str(tuple(item['target'])):<16} {item['samples']:>6} "
              f"{item['coverage']:>8.0%} {fmt(item['jitter_p90']):>9} "
              f"{fmt(item['offset_px']):>7}  {result}")
    counts = Counter(reason for item in quality for reason in item["reasons"])
    print("Rejections:", dict(counts))
    statuses = Counter()
    for item in quality:
        statuses.update(item["capture_statuses"])
    print("Scored camera frames:", dict(statuses))
    if counts["too_few_samples"]:
        print("Too few samples: try --record 3; close other camera/CPU-heavy apps.")
    if counts["low_tracking_coverage"]:
        print("Low coverage: inspect blink/no_face counts; keep your face visible with steady lighting.")
    if counts["unstable_gaze"]:
        print("Unstable gaze: try --settle 2; maintain your calibrated posture and fixate on the dot.")
    if counts["offset_too_large"]:
        print("Large offsets: verify screen/posture match calibration; a fresh base calibration may be needed.")
