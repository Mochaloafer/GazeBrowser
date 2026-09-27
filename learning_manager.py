"""Train, evaluate, accept, reject and roll back controlled lesson sessions."""
import argparse
from gaze_pipeline import MODEL_FILE, context
from lesson_store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL_FILE))
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--camera", type=int, default=0)
    train.add_argument("--settle", type=float, default=1.5, help="Unscored seconds per target")
    train.add_argument("--record", type=float, default=1.5, help="Scored seconds per target")
    test = commands.add_parser("evaluate")
    test.add_argument("session")
    test.add_argument("--recording")
    test.add_argument("--camera", type=int, default=0)
    accept = commands.add_parser("accept")
    accept.add_argument("session")
    accept.add_argument("--report", required=True)
    reject = commands.add_parser("reject")
    reject.add_argument("session")
    commands.add_parser("list")
    commands.add_parser("rollback")
    args = parser.parse_args()
    store = Store()
    try:
        if args.command == "list":
            manifest = store.manifest()
            print(f"Active revision: {manifest['version']}")
            for path in sorted((store.root / "sessions").glob("*.json")):
                name = path.stem
                status = "accepted" if name in manifest["accepted"] else "rejected" if name in manifest["rejected"] else "pending"
                lesson = store.lesson(name)
                print(f"{name}  {status}  {len(lesson['samples'])} fixations")
            return
        if args.command == "rollback":
            print("Restored accepted set:", store.rollback())
            return
        if args.command == "evaluate":
            from accuracy_test import evaluate
            evaluate(args.session, args.model, args.recording, args.camera)
            return
        import pyautogui
        ctx = context(args.model, pyautogui.size())
        if args.command == "train":
            from gaze_capture import capture, lessons_from, print_quality
            recording = capture(args.model, ctx["screen"], training=True, camera_index=args.camera,
                                settle=args.settle, record=args.record)
            samples, quality = lessons_from(recording)
            print_quality(quality)
            if len(samples) < 5:
                name = store.save("diagnostics", dict(context=ctx, quality=quality, recording=recording))
                raise ValueError(f"Only {len(samples)} stable fixations; need at least five. "
                                 f"No candidate created. Diagnostics saved: {store.path('diagnostics', name)}")
            name = store.save("sessions", dict(context=ctx, samples=samples, quality=quality, recording=recording))
            print(f"Candidate: {name}\nStable fixations: {len(samples)}/{len(quality)}")
            print(f"Next: python learning_manager.py evaluate {name}")
        else:
            version = store.decide(args.session, args.command, ctx, getattr(args, "report", None))
            print(f"Saved {version}. Restart main.py to load the updated accepted set.")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
