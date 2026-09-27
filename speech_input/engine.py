from time import perf_counter # measures for duration

import numpy as np # for audio
from faster_whisper import WhisperModel # loads the speech model

# speech recognition engine
class SpeechEngine:
    def __init__(self, model_name="base.en"):
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
        )

    def transcribe(self, audio):
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim != 1:
            raise ValueError("Expected one-dimensional mono audio.")

        if audio.size == 0:
            return "", 0.0

        if not np.isfinite(audio).all():
            raise ValueError("invalid values")

        started = perf_counter()

        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        text = " ".join(
            segment.text.strip() for segment in segments
        ).strip()

        return text, perf_counter() - started