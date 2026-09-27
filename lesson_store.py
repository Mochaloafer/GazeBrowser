"""Immutable lessons and reports; atomic, versioned acceptance manifests."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parent / "lessons"


def identifier(prefix):
    return prefix + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


class Store:
    def __init__(self, root=ROOT):
        self.root = Path(root)

    def path(self, kind, name):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Invalid lesson/report identifier")
        return self.root / kind / (name + ".json")

    def manifest(self):
        path = self.root / "active.json"
        return read(path) if path.exists() else dict(version="empty", accepted=[], rejected=[])

    def lesson(self, name):
        return read(self.path("sessions", name))

    def save(self, kind, data):
        name = identifier(kind.rstrip("s"))
        data = dict(data, id=name)
        write(self.path(kind, name), data)
        return name

    def active(self, ctx, manifest=None):
        manifest = self.manifest() if manifest is None else manifest
        samples, compatible, skipped = [], [], []
        for name in manifest["accepted"]:
            lesson = self.lesson(name)
            if lesson["context"] == ctx:
                samples.extend(lesson["samples"])
                compatible.append(name)
            else:
                skipped.append(name)
        return samples, compatible, skipped

    @contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".mutation.lock"
        try:
            handle = lock.open("x")
        except FileExistsError:
            raise ValueError("Another lesson update is running (.mutation.lock exists)")
        try:
            with handle:
                yield
        finally:
            lock.unlink()

    def commit(self, old, accepted, rejected, action):
        # Caller holds the mutation lock. History is saved before changing the pointer.
        version = identifier("revision")
        updated = dict(version=version, accepted=accepted, rejected=rejected,
                       previous=old, action=action)
        write(self.path("revisions", version), updated)
        write(self.root / "active.json", updated)
        return version

    def decide(self, name, decision, ctx, report_id=None):
        lesson = self.lesson(name)
        with self.locked():
            current = self.manifest()
            if decision == "accept":
                if lesson["context"] != ctx:
                    raise ValueError("Candidate belongs to a different calibration, screen or pipeline")
                if name in current["accepted"]:
                    raise ValueError("Lesson is already accepted")
                if not report_id:
                    raise ValueError("Acceptance requires --report from a completed evaluation")
                report = read(self.path("reports", report_id))
                if (report["candidate"] != name or report["context"] != ctx
                        or report["active_version"] != current["version"]):
                    raise ValueError("Evaluation is stale or belongs to another candidate; evaluate again")
                if not report["complete"]:
                    raise ValueError("Evaluation has missing targets; repeat evaluation before accepting")
            accepted = [x for x in current["accepted"] if x != name]
            rejected = [x for x in current["rejected"] if x != name]
            (accepted if decision == "accept" else rejected).append(name)
            return self.commit(current, accepted, rejected, dict(decision=decision, lesson=name, report=report_id))

    def rollback(self):
        with self.locked():
            current = self.manifest()
            if "previous" not in current:
                raise ValueError("No previous accepted set to restore")
            previous = current["previous"]
            return self.commit(current, previous["accepted"], previous["rejected"], "rollback")
