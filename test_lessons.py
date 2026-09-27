"""Offline checks: no webcam, windows, or mouse control."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from gaze_pipeline import Pipeline, Corrector
from gaze_capture import replay, lessons_from, targets
from accuracy_test import metrics, evaluate
from lesson_store import Store, write, read


class WorkflowTests(unittest.TestCase):
    def test_rejection_diagnostics_separate_causes(self):
        from gaze_capture import lessons_from
        recording = dict(targets=[[100, 100]], frames=[])
        cases = [([None] * 20, {"too_few_samples", "low_tracking_coverage"}),
                 ([np.array([100., 100.])] * 5, {"too_few_samples"}),
                 ([np.array([700., 100.])] * 20, {"offset_too_large"}),
                 ([np.array([100. + (-1)**i * 100, 100.]) for i in range(20)], {"unstable_gaze"})]
        for values, expected in cases:
            with patch("gaze_capture.replay", return_value=[(0, v) for v in values]):
                samples, quality = lessons_from(recording)
            self.assertEqual(samples, [])
            self.assertEqual(set(quality[0]["reasons"]), expected)

    def test_replay_report_image_and_acceptance_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "lessons")
            ctx = {"screen": [640, 480], "model": "test"}
            name = store.save("sessions", dict(context=ctx, samples=[[100, 100, 20, 0]]))
            recording = dict(kind="evaluation", context=ctx, targets=[[120, 100]],
                             frames=[dict(t=i/30, raw=[100, 100], target=0, scored=i>=45)
                                     for i in range(90)])
            path = Path(directory) / "evaluation.json"
            write(path, recording)
            with patch("accuracy_test.Store", return_value=store), patch("accuracy_test.context", return_value=ctx):
                report_id = evaluate(name, recording_path=path)
            report = read(store.path("reports", report_id))
            self.assertTrue(report["complete"])
            self.assertTrue(store.path("reports", report_id).with_suffix(".png").exists())
            self.assertLess(report["results"]["candidate"]["median_target_error"],
                            report["results"]["current"]["median_target_error"])
            store.decide(name, "accept", ctx, report_id)
            self.assertEqual(store.active(ctx)[1], [name])

    def test_promotion_staleness_compatibility_and_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            ctx = {"calibration": "one"}
            candidate = store.save("sessions", dict(context=ctx, samples=[[100, 100, 10, 0]]))
            report = store.save("reports", dict(candidate=candidate, context=ctx,
                                                active_version="empty", complete=True))
            with self.assertRaises(ValueError):
                store.decide(candidate, "accept", {"calibration": "two"}, report)
            store.decide(candidate, "accept", ctx, report)
            self.assertEqual(store.active(ctx)[1], [candidate])
            self.assertEqual(store.active({"calibration": "two"})[2], [candidate])
            store.rollback()
            self.assertEqual(store.manifest()["accepted"], [])
            with self.assertRaises(ValueError):
                store.decide(candidate, "accept", ctx, report)
            store.decide(candidate, "reject", ctx)
            self.assertIn(candidate, store.manifest()["rejected"])
            self.assertTrue(store.path("sessions", candidate).exists())

    def test_incomplete_report_cannot_promote(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            name = store.save("sessions", dict(context={}, samples=[[0, 0, 1, 1]]))
            report = store.save("reports", dict(candidate=name, context={}, active_version="empty", complete=False))
            with self.assertRaises(ValueError):
                store.decide(name, "accept", {}, report)
            self.assertEqual(store.manifest()["version"], "empty")
            with self.assertRaises(ValueError):
                store.lesson("../outside")

    def test_recorded_replay_matches_live_pipeline(self):
        pipeline = Pipeline()
        frames, live = [], []
        for i in range(120):
            raw = None if 30 <= i <= 55 else [300+i, 200-i]
            t = i / 30
            value = pipeline.step(raw, t)
            live.append(value)
            frames.append(dict(t=t, raw=raw, scored=True, target=0))
        replayed = replay(dict(frames=frames))
        for expected, (_, actual) in zip(live, replayed):
            if expected is None:
                self.assertIsNone(actual)
            else:
                np.testing.assert_array_equal(expected, actual)

    def test_three_way_metrics_and_large_error_visibility(self):
        recording = dict(context={"screen": [1000, 1000]}, targets=[[100, 100]])
        filtered = [(0, np.array([150., 100.])) for _ in range(20)]
        base = metrics(recording, filtered, Corrector())
        current = metrics(recording, filtered, Corrector([[150, 100, -40, 0]]))
        candidate = metrics(recording, filtered, Corrector([[150, 100, -40, 0], [150, 100, -50, 0]]))
        self.assertEqual(base["median_target_error"], 50)
        self.assertLess(candidate["median_target_error"], current["median_target_error"])
        self.assertLess(current["median_target_error"], base["median_target_error"])
        unstable = [(0, np.array([100 + (-1)**i * 300, 100])) for i in range(20)]
        result = metrics(recording, unstable, Corrector())
        self.assertEqual(result["median_target_error"], 0)
        self.assertEqual(result["frame_p95"], 300)
        self.assertEqual(result["hit_rate_100px"], 0)
        self.assertIsNone(metrics(recording, [(0, None)], Corrector())["median_target_error"])

    def test_training_quality_and_heldout_positions(self):
        screen = [1920, 1080]
        self.assertFalse(set(map(tuple, targets(screen, True))) & set(map(tuple, targets(screen, False))))
        frames = [dict(t=i/30, target=0, raw=[100, 100], scored=i >= 45) for i in range(90)]
        samples, quality = lessons_from(dict(targets=[[120, 100]], frames=frames))
        self.assertEqual(len(samples), 1)
        self.assertTrue(quality[0]["accepted"])
        for frame in frames:
            frame["raw"] = None
        self.assertEqual(lessons_from(dict(targets=[[120, 100]], frames=frames))[0], [])


if __name__ == "__main__":
    unittest.main()
