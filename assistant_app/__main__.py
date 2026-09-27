import argparse
import sys


def run():
    if sys.platform != "win32":
        raise SystemExit("This integration is for Windows.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gaze", action="store_true")
    parser.add_argument("--model", default="base.en")
    args = parser.parse_args()

    tracker = None

    if not args.no_gaze:
        import cv2
        from eyetrax import GazeEstimator, run_9_point_calibration

        print("Calibrate first. The assistant opens afterward.")
        tracker = GazeEstimator()

        try:
            run_9_point_calibration(tracker)
        finally:
            cv2.destroyAllWindows()

    from PySide6.QtWidgets import QApplication

    from .controller import AssistantController
    from .desktop import FocusWorker
    from .gaze import GazeWorker
    from .speech import SpeechWorker
    from .window import AssistantWindow

    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)

    window = AssistantWindow()
    speech = SpeechWorker(args.model)
    focus = FocusWorker()
    gaze = GazeWorker(tracker) if tracker is not None else None

    controller = AssistantController(window, speech, focus, gaze)

    window.show()
    focus.start()
    speech.start()

    if gaze is not None:
        gaze.start()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run())