"""Print samples from exactly one provider as JSON lines."""

import argparse
from contextlib import closing, redirect_stdout
from dataclasses import asdict
import json
import sys

from gaze_providers import EyeTraxProvider, GazeProvider, MouseProvider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("real", "mock"))
    parser.add_argument("--camera", type=int, default=0, help="real mode camera index (default: 0)")
    parser.add_argument("--hz", type=float, default=30.0, help="mock mode sample rate (default: 30)")
    parser.add_argument("--count", type=int, default=0, help="stop after N samples; 0 runs until Ctrl+C")
    args = parser.parse_args()
    if args.count < 0:
        parser.error("--count must be nonnegative")
    if args.camera < 0:
        parser.error("--camera must be nonnegative")

    try:
        provider: GazeProvider = (EyeTraxProvider(args.camera) if args.mode == "real"
                                  else MouseProvider(args.hz))
        with closing(provider.samples()) as samples:
            count = 0
            while args.count == 0 or count < args.count:
                # EyeTrax diagnostics belong on stderr, leaving stdout as JSONL.
                with redirect_stdout(sys.stderr):
                    sample = next(samples, None)
                if sample is None:
                    break
                print(json.dumps(asdict(sample), allow_nan=False), flush=True)
                count += 1
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"gaze: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
