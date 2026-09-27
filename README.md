## Development environment

- Python: 3.14.2
- eyetrax: 0.4.0
- opencv-python: 4.11.0.86
- OS: Windows

## Lesson workflow

Run these commands from GazeBrowser with its virtual environment activated.
Close main.py before training or testing so only one process uses the camera.

```powershell
python learning_manager.py train
python learning_manager.py evaluate SESSION_ID
python learning_manager.py accept SESSION_ID --report REPORT_ID
python main.py
```

Replace SESSION_ID and REPORT_ID with the identifiers printed by the tools.
Training records 25 targets, three seconds each. Keep looking at the dot until
it moves; do not click. The first 1.5 seconds let your gaze settle. Esc cancels
without saving a partial session. Camera images are never saved.

Only stable fixations become teaching examples: at least 15 valid samples,
80% availability, 90th-percentile deviation at most 60 pixels, and an offset
no greater than 400 pixels. At least five usable targets are required to save
a candidate. These are sample-quality checks, not improvement/acceptance
thresholds. Per-target quality details and raw coordinates remain in the session.
Every completed run prints rejection reasons, sample counts, coverage, jitter,
offset, and blink/face tracking counts. Runs with fewer than five usable targets
are saved under lessons/diagnostics for debugging, without creating a candidate.
For more collection time, use `python learning_manager.py train --settle 2 --record 3`.
This extends each target to five seconds without relaxing the quality checks.

Evaluation records 16 separate target positions and replays exactly the same
filtered samples for all three variants:

- **Baseline:** calibrated EyeTrax model plus shared smoothing, without lessons.
- **Current:** baseline plus compatible accepted lessons.
- **Candidate:** current plus the specified pending lesson session.

Initially current equals baseline because the accepted set starts empty.
New training never changes the running app. Acceptance is an explicit command;
there is no learning curve, automatic promotion, or improvement threshold.
Review median target error, edge/inner error, frame P95 error, mean jitter,
hit rate within 100 pixels, and tracking availability before accepting.
Lower pixel errors are better; higher hit rate and availability are better.
Missing targets make a report inconclusive and prevent acceptance. Small
differences should be checked with fresh repeated evaluation sessions.

The test saves a three-color comparison PNG, a JSON report with per-target
metrics, and the evaluation coordinates. Circles show mean radial jitter;
they are not confidence intervals. Drawing endpoints are clipped to the screen,
but metrics use unclipped coordinates. Evaluation measures the gaze estimate
before cursor glide, clamping and button snapping, not final clicking success.

```powershell
python accuracy_test.py SESSION_ID
python accuracy_test.py
python learning_manager.py list
python learning_manager.py reject SESSION_ID
python learning_manager.py rollback
python learning_manager.py evaluate SESSION_ID --recording lessons/recordings/RECORDING_ID.json
```

With no candidate, accuracy_test.py still displays three results but explicitly
marks candidate as identical to current. Replaying a saved evaluation is useful
for debugging; use fresh recordings periodically to avoid selecting every new
candidate against the same data. Training recordings cannot be used as evaluation
recordings. Rejected sessions remain available for inspection and reevaluation.

Acceptance requires a complete report for the exact current manifest version,
candidate, calibration hash, screen dimensions, and pipeline configuration.
Changing the accepted set invalidates older acceptance reports. Restart main.py
after accepting, rejecting, or rolling back; it snapshots accepted lessons at startup.
Rollback restores the previous manifest contents as a new revision.

Use `--model PATH` before the learning_manager subcommand to select another
calibration, or `accuracy_test.py SESSION_ID --model PATH`. Use `--camera N`
after train/evaluate to select a different camera. Default paths are relative to
the scripts, so the current working directory cannot silently change the model.

## Architecture and storage

```text
main.py                 Application: accepted lessons, gaze movement, blink click
gaze_pipeline.py        Shared median + Kalman/EMA filtering and local correction
gaze_capture.py         Controlled capture, deterministic replay, sample quality
lesson_store.py         Lesson persistence, compatible loading, atomic manifests
learning_manager.py     Training and explicit acceptance/rejection/rollback
accuracy_test.py        Three-way evaluation and comparison images
lessons/
  active.json           Accepted and rejected session IDs; initially empty
  sessions/             Immutable candidate batches and their training recordings
  recordings/           Independent evaluation captures
  reports/              Evaluation metrics and comparison PNGs
  revisions/            Acceptance history and previous manifests
legacy/                 Previous scripts and old correction data (never loaded)
```

Normal mouse and blink clicks do not create lessons. Main retains touchpad
takeover, blink clicking, snapping, pause (`p`), snapping toggle (`s`), and quit
(`q`). The old in-app undo/clear lesson keys are replaced by manager commands.

The pipeline uses fixed shared filter settings; the former per-launch interactive
smoother tuning has been removed so training, testing, and use agree. This changes
smoothing compared with a previously tuned run. All three tools share that change.
Accepted offsets are never transferred automatically to a new calibration or
incompatible pipeline. Train fresh sessions after such changes.

Set RECALIBRATE=True in main.py for a new base calibration, then return it to False.
The old calibration is backed up only after a fitted replacement is ready; cancelled
calibration cannot overwrite it with an unfitted model. Existing model backups,
logs, and accuracy images are preserved. Old correction files are archived and do
not populate the new lesson store.

Offline checks (no camera or mouse movement):

```powershell
python -m unittest test_lessons -v
```
