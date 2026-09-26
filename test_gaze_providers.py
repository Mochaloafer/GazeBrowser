"""Hardware-free checks for event semantics and camera lifetime."""

from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import Mock, patch

import gaze_cli
from gaze_providers import EyeTraxProvider, GazeSample, MouseProvider


class GazeTests(unittest.TestCase):
    def setUp(self):
        self.camera = Mock()
        self.camera.isOpened.return_value = True
        self.cv2 = Mock()
        self.cv2.VideoCapture.return_value = self.camera
        self.tracker = Mock()
        self.eyetrax = Mock()
        self.eyetrax.GazeEstimator.return_value = self.tracker
        self.modules = patch.dict("sys.modules", {"cv2": self.cv2, "eyetrax": self.eyetrax})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_events_and_calibration_before_capture(self):
        self.eyetrax.run_9_point_calibration.side_effect = (
            lambda *a, **kw: self.cv2.VideoCapture.assert_not_called()
        )
        self.camera.read.side_effect = [(True, "frame")] * 4 + [(False, None)]
        self.tracker.extract_features.side_effect = [
            ([1], False), ([2], True), (None, False), ([3], False),
        ]
        self.tracker.predict.side_effect = [[(10.5, 20.5)], [(float("nan"), 1)]]
        samples = list(EyeTraxProvider(2).samples())
        self.assertEqual([s.event for s in samples],
                         ["gaze", "blink", "tracking_loss", "tracking_loss", "tracking_loss"])
        self.assertEqual((samples[0].x, samples[0].y), (10.5, 20.5))
        for sample in samples[1:]:
            self.assertFalse(sample.valid)
            self.assertIsNone(sample.x)
            self.assertIsNone(sample.y)
        self.assertEqual([s.timestamp for s in samples], sorted(s.timestamp for s in samples))
        self.assertEqual(self.tracker.predict.call_count, 2)
        self.tracker.predict.assert_any_call([[1]])
        self.eyetrax.run_9_point_calibration.assert_called_once_with(self.tracker, camera_index=2)
        self.cv2.VideoCapture.assert_called_once_with(2)
        self.camera.release.assert_called_once()
        self.tracker.close.assert_called_once()

    def test_early_close_releases_camera(self):
        self.camera.read.return_value = (True, "frame")
        self.tracker.extract_features.return_value = (None, False)
        stream = EyeTraxProvider().samples()
        next(stream)
        stream.close()
        self.camera.release.assert_called_once()
        self.cv2.destroyAllWindows.assert_called_once()

    def test_exception_and_interrupt_release_camera(self):
        for error in (RuntimeError("capture failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.camera.release.reset_mock()
                self.camera.read.side_effect = error
                with self.assertRaises(type(error)):
                    next(EyeTraxProvider().samples())
                self.camera.release.assert_called_once()

    def test_failed_open_emits_loss_and_releases(self):
        self.camera.isOpened.return_value = False
        samples = list(EyeTraxProvider().samples())
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].event, "tracking_loss")
        self.assertIsNone(samples[0].x)
        self.camera.read.assert_not_called()
        self.camera.release.assert_called_once()

    def test_contract_rejects_invalid_coordinates(self):
        for values in [(1, 2, 3, False, "blink"),
                       (1, None, None, True, "gaze"),
                       (1, float("inf"), 3, True, "gaze")]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                GazeSample(*values)

    def test_cli_sample_limit_closes_provider(self):
        closed = []

        def samples():
            try:
                while True:
                    yield GazeSample(1, 10, 20, True, "gaze")
            finally:
                closed.append(True)

        output = io.StringIO()
        with patch.object(gaze_cli, "MouseProvider") as provider, \
                patch("sys.argv", ["gaze_cli.py", "mock", "--count", "2"]), \
                redirect_stdout(output):
            provider.return_value.samples.return_value = samples()
            self.assertEqual(gaze_cli.main(), 0)
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        self.assertTrue(all(json.loads(line)["valid"] for line in output.getvalue().splitlines()))
        self.assertEqual(closed, [True])
        self.cv2.VideoCapture.assert_not_called()

    def test_mock_rate_validation(self):
        for hz in (0, -1, float("nan"), float("inf")):
            with self.subTest(hz=hz), self.assertRaises(ValueError):
                MouseProvider(hz)


if __name__ == "__main__":
    unittest.main()
