import threading
import time

import cv2
from PySide6.QtCore import QThread, Signal

# handles mouse movement following the gaze (smoothing on controller)
class GazeWorker(QThread):
    sample = Signal(float, float, bool, float)
    failed = Signal(str)

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        camera = cv2.VideoCapture(0)

        try:
            if not camera.isOpened():
                raise RuntimeError(
                    "Cannot open camera. Close the old main.py."
                )

            while not self.stop_event.is_set():
                success, frame = camera.read()

                if not success:
                    raise RuntimeError("Camera stopped returning frames.")

                features, blink = self.tracker.extract_features(frame)
                timestamp = time.monotonic()

                if features is not None and not blink:
                    x, y = self.tracker.predict([features])[0]
                    self.sample.emit(
                        float(x), float(y), True, timestamp
                    )
                else:
                    self.sample.emit(
                        0.0, 0.0, False, timestamp
                    )

        except Exception as error:
            self.failed.emit(f"Gaze: {error}")

        finally:
            camera.release()