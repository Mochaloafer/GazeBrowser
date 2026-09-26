"""Synchronous gaze sources. Run one provider at a time on a single display."""

from collections.abc import Iterator
from dataclasses import dataclass
import math
import sys
import time
from typing import Literal, Protocol


@dataclass(frozen=True)
class GazeSample:
    """Screen pixels: top-left origin, x right, y down; no clipping/smoothing.

    timestamp is time.monotonic() in seconds, not wall-clock time.
    Invalid events never contain coordinates, including previous coordinates.
    """

    timestamp: float
    x: float | None
    y: float | None
    valid: bool
    event: Literal["gaze", "blink", "tracking_loss"]

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be finite monotonic seconds")
        if self.valid:
            if (self.event != "gaze" or self.x is None or self.y is None
                    or not math.isfinite(self.x) or not math.isfinite(self.y)):
                raise ValueError("valid gaze requires finite screen coordinates")
        elif (self.event not in ("blink", "tracking_loss")
              or self.x is not None or self.y is not None):
            raise ValueError("invalid events require x=None and y=None")


class GazeProvider(Protocol):
    def samples(self) -> Iterator[GazeSample]:
        """Yield samples synchronously; close the iterator when stopping early."""
        ...


class EyeTraxProvider:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index

    def samples(self) -> Iterator[GazeSample]:
        # Lazy imports keep the mouse mode independent of the tracker runtime.
        import cv2
        from eyetrax import GazeEstimator, run_9_point_calibration

        tracker = GazeEstimator()
        camera = None
        try:
            # Preserve the supplied calibration; it owns its capture internally.
            run_9_point_calibration(tracker, camera_index=self.camera_index)
            camera = cv2.VideoCapture(self.camera_index)
            if not camera.isOpened():
                yield GazeSample(time.monotonic(), None, None, False, "tracking_loss")
                return
            while True:
                success, frame = camera.read()
                timestamp = time.monotonic()
                if not success:
                    yield GazeSample(timestamp, None, None, False, "tracking_loss")
                    return
                features, blink = tracker.extract_features(frame)
                if blink:
                    yield GazeSample(timestamp, None, None, False, "blink")
                elif features is None:
                    yield GazeSample(timestamp, None, None, False, "tracking_loss")
                else:
                    x, y = tracker.predict([features])[0]
                    x, y = float(x), float(y)
                    if math.isfinite(x) and math.isfinite(y):
                        yield GazeSample(timestamp, x, y, True, "gaze")
                    else:
                        yield GazeSample(timestamp, None, None, False, "tracking_loss")
        finally:
            try:
                if camera is not None:
                    camera.release()
            finally:
                try:
                    cv2.destroyAllWindows()
                finally:
                    tracker.close()


class MouseProvider:
    """Read the Windows physical cursor position without moving it."""

    def __init__(self, hz: float = 30.0) -> None:
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError("hz must be finite and positive")
        self.hz = hz

    def samples(self) -> Iterator[GazeSample]:
        if sys.platform != "win32":
            raise RuntimeError("The mouse mock currently supports Windows only")
        import ctypes
        from ctypes import wintypes

        get_cursor = ctypes.WinDLL("user32", use_last_error=True).GetPhysicalCursorPos
        get_cursor.argtypes = [ctypes.POINTER(wintypes.POINT)]
        get_cursor.restype = wintypes.BOOL
        point = wintypes.POINT()
        while True:
            success = get_cursor(ctypes.byref(point))
            timestamp = time.monotonic()
            if success:
                yield GazeSample(timestamp, float(point.x), float(point.y), True, "gaze")
            else:
                yield GazeSample(timestamp, None, None, False, "tracking_loss")
            time.sleep(1.0 / self.hz)
