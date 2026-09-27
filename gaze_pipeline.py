"""Shared deterministic preprocessing and correction for app, training and tests."""
from collections import deque
from pathlib import Path
import hashlib
import importlib.metadata

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "gaze_model.pkl"
SETTINGS = dict(version=1, median_window=7, ema_alpha=0.8, process_var=5.0,
                measurement_var=0.2, settle_after_loss=0.15, reset_gap=0.5,
                correction_sigma=200.0, correction_prior=1.0)


def context(model_path, screen):
    return dict(model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                screen=list(map(int, screen)), pipeline=SETTINGS.copy(),
                eyetrax=importlib.metadata.version("eyetrax"))


def load_tracker(path):
    from eyetrax import GazeEstimator
    from sklearn.utils.validation import check_is_fitted
    tracker = GazeEstimator()
    try:
        tracker.load_model(path)
        check_is_fitted(tracker.model.scaler)
        tracker.predict(np.zeros((1, tracker.model.scaler.n_features_in_)))
    except Exception:
        tracker.close()
        raise
    return tracker


class Pipeline:
    def __init__(self):
        self.reset()

    def reset(self):
        from eyetrax.filters import KalmanEMASmoother, make_kalman
        self.history = deque(maxlen=SETTINGS["median_window"])
        self.smoother = KalmanEMASmoother(make_kalman(
            process_var=SETTINGS["process_var"],
            measurement_var=SETTINGS["measurement_var"]),
            ema_alpha=SETTINGS["ema_alpha"])
        self.last_valid = None
        self.ready_at = None

    def step(self, prediction, timestamp):
        if prediction is None or not np.isfinite(prediction).all():
            self.ready_at = None
            return None
        if self.last_valid is not None and timestamp - self.last_valid > SETTINGS["reset_gap"]:
            self.reset()
        self.last_valid = timestamp
        if self.ready_at is None:
            self.ready_at = timestamp + SETTINGS["settle_after_loss"]
        if timestamp < self.ready_at:
            return None
        self.history.append(prediction)
        x, y = np.median(self.history, axis=0)
        return np.asarray(self.smoother.step(int(x), int(y)), dtype=float)


class Corrector:
    def __init__(self, samples=()):
        self.samples = np.asarray(samples, dtype=float).reshape(-1, 4)
        if not np.isfinite(self.samples).all():
            raise ValueError("Lessons contain non-finite coordinates")

    def apply(self, predictions):
        values = np.asarray(predictions, dtype=float).reshape(-1, 2)
        if not len(self.samples):
            return values.copy()
        dist2 = ((values[:, None] - self.samples[None, :, :2]) ** 2).sum(axis=2)
        weights = np.exp(-dist2 / (2 * SETTINGS["correction_sigma"] ** 2))
        return values + (weights @ self.samples[:, 2:]) / (
            weights.sum(axis=1, keepdims=True) + SETTINGS["correction_prior"])

    def correct(self, x, y):
        return self.apply([[x, y]])[0]
