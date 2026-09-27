"""Standalone microphone test. Does not control the desktop."""

import argparse

import numpy as np
import sounddevice as sd

from engine import SpeechEngine


SAMPLE_RATE = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if not 1 <= args.seconds <= 30:
        parser.error("--seconds must be between 1 and 30")

    sd.check_input_settings(
        device=args.device,
        channels=1,
        dtype="float32",
        samplerate=SAMPLE_RATE,
    )

    print(f"Loading {args.model}...")
    print("First launch downloads the model; internet is needed.")
    engine = SpeechEngine(args.model)
    print("Ready. Recordings stay in memory.")

    while True:
        command = input("\nEnter to record, or q then Enter to quit: ")
        if command.strip().lower() == "q":
            break

        print(f"Speak now — recording for {args.seconds:g} seconds.")

        audio = sd.rec(
            frames=int(args.seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=args.device,
        )
        status = sd.wait()

        if status:
            print(f"Recording problem: {status}. Try again.")
            continue

        audio = audio[:, 0].copy()

        if np.max(np.abs(audio)) >= 0.99:
            print("Audio may be clipping; lower microphone gain.")

        print("Transcribing...")
        text, elapsed = engine.transcribe(audio)

        print(f"Text: {text or '[No speech recognized]'}")
        print(f"Processing time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sd.stop()