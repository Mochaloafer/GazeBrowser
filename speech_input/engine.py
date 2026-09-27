from time import perf_counter

import numpy as np
from faster_whisper import WhisperModel


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
            raise ValueError("Audio contains invalid values.")

        # Never ask the model to interpret digital silence.
        if not np.any(audio):
            return "", 0.0

        started = perf_counter()

        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,

            # Use one decoding temperature instead of retrying
            # at progressively higher temperatures.
            temperature=0.0,

            vad_filter=True,
            condition_on_previous_text=False,

            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        accepted = []

        for segment in segments:
            # Conservative starting filters. They can also reject
            # genuine quiet or unclear speech, so test your commands.
            if segment.no_speech_prob > 0.6:
                continue

            if segment.avg_logprob < -1.0:
                continue

            if segment.compression_ratio > 2.4:
                continue

            text = segment.text.strip()

            if text:
                accepted.append(text)

        return " ".join(accepted), perf_counter() - started