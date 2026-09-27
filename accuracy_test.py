"""Compare baseline, accepted lessons, and accepted + candidate on identical samples."""
import argparse
import numpy as np
from gaze_pipeline import MODEL_FILE, Corrector, context
from gaze_capture import capture, replay
from lesson_store import Store, read


def metrics(recording, filtered, corrector):
    per_target, all_errors, jitters = [], [], []
    valid_count = sum(v is not None for _, v in filtered)
    width, height = recording["context"]["screen"]
    for index, target in enumerate(recording["targets"]):
        values = [v for i, v in filtered if i == index and v is not None]
        if len(values) < 5:
            per_target.append(dict(target=target, samples=len(values), error=None))
            continue
        corrected = corrector.apply(values)
        mean = corrected.mean(axis=0)
        errors = np.linalg.norm(corrected-np.asarray(target), axis=1)
        jitter = float(np.mean(np.linalg.norm(corrected-mean, axis=1)))
        edge = (target[0] <= width * .1 or target[0] >= width * .9
                or target[1] <= height * .1 or target[1] >= height * .9)
        per_target.append(dict(target=target, samples=len(values), mean=mean.tolist(),
                               error=float(np.linalg.norm(mean-target)), jitter=jitter, edge=edge))
        all_errors.extend(errors.tolist())
        jitters.append(jitter)
    def median(values):
        return float(np.median(values)) if values else None
    return dict(targets=per_target,
                median_target_error=median([t["error"] for t in per_target if t["error"] is not None]),
                edge_error=median([t["error"] for t in per_target if t.get("edge")]),
                inner_error=median([t["error"] for t in per_target if t.get("edge") is False]),
                frame_p95=float(np.percentile(all_errors, 95)) if all_errors else None,
                hit_rate_100px=float(np.mean(np.asarray(all_errors) <= 100)*100) if all_errors else None,
                mean_jitter=float(np.mean(jitters)) if jitters else None,
                availability=valid_count/max(len(filtered), 1)*100)


def evaluate(candidate=None, model=MODEL_FILE, recording_path=None, camera=0):
    import pyautogui
    store = Store()
    ctx = context(model, pyautogui.size())
    manifest = store.manifest()
    active, active_ids, skipped = store.active(ctx, manifest)
    additional = []
    if candidate:
        lesson = store.lesson(candidate)
        if lesson["context"] != ctx:
            raise ValueError("Candidate is incompatible with this calibration, screen or pipeline")
        if candidate in manifest["accepted"]:
            raise ValueError("Candidate is already active; choose an unaccepted session")
        additional = lesson["samples"]
    if recording_path:
        recording = read(recording_path)
        if recording.get("kind") != "evaluation" or recording.get("context") != ctx:
            raise ValueError("Use an independent evaluation recording with matching context")
    else:
        recording = dict(capture(model, ctx["screen"], camera_index=camera), context=ctx)
        name = store.save("recordings", recording)
        recording_path = store.path("recordings", name)
    filtered = replay(recording)
    variants = {"baseline": Corrector(), "current": Corrector(active),
                "candidate": Corrector(active + additional)}
    results = {name: metrics(recording, filtered, corrector) for name, corrector in variants.items()}
    complete = all(t["error"] is not None for t in results["baseline"]["targets"])
    report = dict(context=ctx, candidate=candidate, active_version=manifest["version"],
                  active_lessons=active_ids, skipped_lessons=skipped,
                  recording=str(recording_path), complete=complete, results=results)
    report_id = store.save("reports", report)
    print("\nIdentical filtered samples; lower pixel errors are better.")
    print(f"{'Variant':<14} {'Median':>9} {'Edges':>9} {'Inner':>9} {'P95/frame':>10} {'Jitter':>9} {'Hit<=100%':>10}")
    def fmt(value):
        return "n/a" if value is None else f"{value:.1f}"
    for name, result in results.items():
        print(f"{name:<14}" + "".join(f"{fmt(result[key]):>10}" for key in
              ["median_target_error", "edge_error", "inner_error", "frame_p95", "mean_jitter", "hit_rate_100px"]))
    print(f"Tracking availability: {results['baseline']['availability']:.1f}%")
    if not candidate:
        print("No candidate selected: candidate equals current. Pass a session ID to test new lessons.")
    old = results["current"]["median_target_error"]
    new = results["candidate"]["median_target_error"]
    if old is not None and new is not None:
        print(f"Candidate median change from current: {new-old:+.1f}px (negative is better)")
    print("Decision: manual review; no automatic threshold or learning curve.")
    if not complete:
        print("INCONCLUSIVE: missing targets. Repeat the test before accepting.")
    if skipped:
        print(f"Skipped {len(skipped)} incompatible accepted sessions.")
    print(f"Report ID: {report_id}\nReport: {store.path('reports', report_id)}")
    draw_report(report, store.path("reports", report_id).with_suffix(".png"))
    return report_id


def draw_report(report, path):
    import cv2
    width, height = report["context"]["screen"]
    canvas = np.full((height, width, 3), 35, np.uint8)
    colors = {"baseline": (0, 0, 255), "current": (255, 200, 0), "candidate": (255, 0, 255)}
    for name, result in report["results"].items():
        for entry in result["targets"]:
            target = tuple(entry["target"])
            cv2.circle(canvas, target, 7, (0, 255, 0), -1)
            if entry["error"] is None:
                continue
            point = tuple(np.clip(entry["mean"], (0, 0), (width-1, height-1)).astype(int))
            cv2.line(canvas, target, point, colors[name], 1)
            cv2.circle(canvas, point, min(int(max(entry["jitter"], 4)), max(width, height)), colors[name], 1)
    cv2.rectangle(canvas, (10, height//2-65), (min(width-10, 1100), height//2+65), (15, 15, 15), -1)
    for index, (name, result) in enumerate(report["results"].items()):
        error = result["median_target_error"]
        label = "n/a" if error is None else f"{error:.1f}px"
        cv2.putText(canvas, f"{name}: median target error {label}", (25, height//2-35+index*35),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, colors[name], 2)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not save comparison image: {path}")
    print(f"Comparison image: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?", help="Unaccepted lesson session ID")
    parser.add_argument("--model", default=str(MODEL_FILE))
    parser.add_argument("--recording", help="Replay a saved independent evaluation JSON")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    try:
        evaluate(args.candidate, args.model, args.recording, args.camera)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
