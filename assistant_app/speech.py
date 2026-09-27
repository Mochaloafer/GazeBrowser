import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad
from PySide6.QtCore import QThread, Signal

from speech_input.engine import SpeechEngine


SAMPLE_RATE = 16000
FRAME_MS = 20
UPDATE_INTERVAL = 0.65
END_SILENCE = 1.4
MAX_PHRASE_SECONDS = 20
MIC_DEVICE = None

def audio_rms(frame):
    samples = frame.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples * samples)))

class SpeechWorker(QThread):
    status = Signal(str)
    began = Signal(int)
    partial = Signal(int, str)
    final = Signal(int, str, float)
    failed = Signal(str)
    ended = Signal(int)

    def __init__(self, model_name="base.en"):
        super().__init__()
        self.model_name = model_name
        self.stop_event = threading.Event()
        self.mode = "command"

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
            self.status.emit("Loading speech model…")
            engine = SpeechEngine(self.model_name)

            if not self.stop_event.is_set():
                self.listen(engine)

        except Exception as error:
            self.failed.emit(f"Speech: {error}")

    def set_mode(self, mode):
        self.mode = mode

    def listen(self, engine):
        audio_queue = queue.Queue(maxsize=250)
        audio_problem = threading.Event()
        block_size = SAMPLE_RATE * FRAME_MS // 1000
        phrase_mode = "command"

        def callback(indata, frames, timing, status):
            if status:
                audio_problem.set()

            try:
                audio_queue.put_nowait(
                    (indata.copy(), time.monotonic())
                )
            except queue.Full:
                audio_problem.set()

        vad = webrtcvad.Vad(2)
        pre_roll = deque(maxlen=15)
        recent_voice = deque(maxlen=5)

        chunks = []
        active = False
        silent_frames = 0
        voiced_frames = 0
        phrase_id = 0
        last_preview = 0.0
        last_voice_time = 0.0

        silence_limit = round(END_SILENCE * 1000 / FRAME_MS)
        phrase_limit = round(MAX_PHRASE_SECONDS * 1000 / FRAME_MS)

        def transcribe():
            audio = (
                np.concatenate(chunks)
                .reshape(-1)
                .astype(np.float32)
                / 32768.0
            )
            text, _ = engine.transcribe(audio)
            return text

        with sd.InputStream(
            device=MIC_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=block_size,
            callback=callback,
        ):
            self.status.emit(
                "Stay quiet for one second — measuring microphone noise"
            )

            noise_levels = []
            calibration_frames = 1000 // FRAME_MS

            for _ in range(calibration_frames):
                if self.stop_event.is_set():
                    return

                try:
                    frame, _ = audio_queue.get(timeout=2.0)
                except queue.Empty:
                    raise RuntimeError(
                        "No microphone audio received. Check MIC_DEVICE."
                    )

                noise_levels.append(audio_rms(frame))

            noise_floor = float(np.percentile(noise_levels, 90))

            # Starting values to test on your microphone.
            speech_threshold = max(0.002, noise_floor * 3.0)

            print(
                f"[Microphone] noise={noise_floor:.5f}, "
                f"speech threshold={speech_threshold:.5f}"
            )

            self.status.emit("Listening — pause after each command")

            while not self.stop_event.is_set():
                if audio_problem.is_set():
                    raise RuntimeError(
                        "Audio dropped or processing fell behind. "
                        "Try tiny.en and restart."
                    )

                try:
                    frame, captured_at = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                loud_enough = audio_rms(frame) >= speech_threshold

                is_speech = (
                    loud_enough
                    and vad.is_speech(frame.tobytes(), SAMPLE_RATE)
                )

                if not active:
                    pre_roll.append(frame)
                    recent_voice.append(is_speech)

                    if sum(recent_voice) < 3:
                        continue

                    phrase_id += 1
                    active = True
                    
                    # Keep one endpoint setting throughout this phrase.
                    phrase_mode = self.mode

                    silence_seconds = (
                        0.6 if phrase_mode == "command" else 1.4
                    )

                    silence_limit = round(
                        silence_seconds * 1000 / FRAME_MS
                    )

                    chunks = list(pre_roll)
                    pre_roll.clear()
                    recent_voice.clear()

                    silent_frames = 0
                    voiced_frames = 3
                    last_voice_time = captured_at
                    last_preview = time.monotonic()

                    self.began.emit(phrase_id)

                else:
                    chunks.append(frame)

                    if is_speech:
                        voiced_frames += 1
                        silent_frames = 0
                        last_voice_time = captured_at
                    else:
                        silent_frames += 1

                phrase_finished = (
                    silent_frames >= silence_limit
                    or len(chunks) >= phrase_limit
                )

                if phrase_finished:
                    try:
                        if voiced_frames >= 5:
                            text = transcribe()

                            if not self.stop_event.is_set():
                                self.final.emit(
                                    phrase_id,
                                    text,
                                    last_voice_time,
                                )
                    finally:
                        self.ended.emit(phrase_id)

                    active = False
                    chunks = []
                    continue

                should_preview = (
                    phrase_mode == "typing"
                    and audio_queue.empty()
                    and voiced_frames >= 5
                    and time.monotonic() - last_preview
                    >= UPDATE_INTERVAL
                )

                if should_preview:
                    text = transcribe()

                    if not self.stop_event.is_set():
                        self.partial.emit(phrase_id, text)

                    last_preview = time.monotonic()