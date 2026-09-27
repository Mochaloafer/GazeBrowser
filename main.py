"""Windows entry point: python -m assistant_app from the repository root."""

import argparse
import sys


def run():
    if sys.platform != "win32":
        raise SystemExit("This integration is for Windows.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gaze", action="store_true")
    parser.add_argument("--model", default="base.en")
    args = parser.parse_args()

    from PySide6.QtWidgets import QApplication
    from .audio_permission import ensure_microphone_consent

    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    if not ensure_microphone_consent():
        print("Microphone consent not granted. Exiting.")
        return 0

    # Import before calibration so a missing assistant dependency fails early.
    from .controller import AssistantController
    from .desktop import FocusWorker
    from .speech import SpeechWorker
    from .window import AssistantWindow

    tracker = None
    gaze = None
    if not args.no_gaze:
        import cv2
        # These import the teammate's real gaze_pipeline and lesson_store.
        # Missing modules must be restored from their branch, not substituted.
        from .teammate_gaze import calibrate_tracker
        from .gaze import GazeWorker

        print("Running teammate's dense-grid calibration. Assistant opens afterward.")
        try:
            tracker = calibrate_tracker()
        finally:
            cv2.destroyAllWindows()

    workers = []
    gaze_started = False
    try:
        window = AssistantWindow()
        speech = SpeechWorker(args.model)
        speech.failed.connect(lambda message: print(f"[Speech error] {message}"))
        focus = FocusWorker()
        gaze = GazeWorker(tracker) if tracker is not None else None
        controller = AssistantController(window, speech, focus, gaze)
        workers = controller.workers()
        window.show()
        focus.start()
        speech.start()
        if gaze is not None:
            gaze.start()
            gaze_started = True
        return app.exec()
    finally:
        # Also cover unexpected event-loop exit. Never destroy a running QThread.
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.wait()
        # A started GazeWorker closes its tracker in its own finally block.
        if tracker is not None and not gaze_started:
            tracker.close()


if __name__ == "__main__":
    sys.exit(run())
