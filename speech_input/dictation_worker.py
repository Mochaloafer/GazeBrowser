import threading
import time

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QThread, Signal

from .engine import SpeechEngine

SAMPLE_RATE = 16000
UPDATE_INTERVAL = 0.65
MAX_SECONDS = 30
MODEL_NAME = "base.en"
MIC_DEVICE = None

class DictationWorker(QThread):
    status = Signal(str)
    partial = Signal(str, float)
    result = Signal(str, float)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.engine = None
        self.stop_requested = threading.Event()

    def prepare(self):
        self.stop_requested.clear()

    def request_stop(self):
        self.stop_requested.set()

    def run(self):
        chunks = []
        sample_count = 0
        maximum_samples = SAMPLE_RATE * MAX_SECONDS
        audio_lock = threading.Lock()
        recording_problem = threading.Event()

        def receive_audio(indata, frames, timing, status): # microphone
            nonlocal sample_count

            if status:
                recording_problem.set()

            with audio_lock:
                remaining = maximum_samples - sample_count

                if remaining > 0:
                    chunk = indata[:remaining, 0].copy()
                    chunks.append(chunk)
                    sample_count += len(chunk)

                if sample_count >= maximum_samples:
                    self.stop_requested.set()

        def snapshot():
            with audio_lock:
                current_chunks = list(chunks)

            if not current_chunks:
                return np.empty(0, dtype=np.float32)

            return np.concatenate(current_chunks)

        try:
            if self.engine is None:
                self.status.emit(
                    f"Loading {MODEL_NAME} — first run may download the model…"
                )
                self.engine = SpeechEngine(MODEL_NAME)

            if self.stop_requested.is_set():
                return

            sd.check_input_settings(
                device=MIC_DEVICE,
                channels=1,
                dtype="float32",
                samplerate=SAMPLE_RATE,
            )

            last_decoded_samples = 0
            previous_text = ""
            previous_elapsed = 0.0

            with sd.InputStream(
                device=MIC_DEVICE,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=receive_audio,
            ):
                self.status.emit(
                    "Listening — speak now. Click Stop when finished."
                )
                next_decode = time.monotonic() + UPDATE_INTERVAL

                while not self.stop_requested.is_set():
                    if recording_problem.is_set():
                        raise RuntimeError(
                            "The microphone reported dropped audio. "
                            "Stop other audio programs and try again."
                        )

                    now = time.monotonic()

                    if now < next_decode:
                        self.stop_requested.wait(
                            min(0.05, next_decode - now)
                        )
                        continue

                    audio = snapshot()

                    if audio.size < SAMPLE_RATE // 2:
                        next_decode = now + UPDATE_INTERVAL
                        continue

                    started = time.monotonic()
                    text, elapsed = self.engine.transcribe(audio)

                    last_decoded_samples = audio.size
                    previous_text = text
                    previous_elapsed = elapsed

                    self.partial.emit(text, elapsed)

                    # No queue of obsolete transcription jobs:
                    # the next pass takes a fresh snapshot.
                    next_decode = max(
                        started + UPDATE_INTERVAL,
                        time.monotonic(),
                    )

            # The microphone is now closed. Include audio received while
            # the previous transcription was running.
            if recording_problem.is_set():
                raise RuntimeError(
                    "Recording contained dropped audio. Please retry."
                )

            audio = snapshot()

            if audio.size == 0:
                self.result.emit("", 0.0)
                return

            self.status.emit("Finalizing the complete recording…")

            if audio.size != last_decoded_samples:
                previous_text, previous_elapsed = (
                    self.engine.transcribe(audio)
                )

            self.result.emit(previous_text, previous_elapsed)

        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")