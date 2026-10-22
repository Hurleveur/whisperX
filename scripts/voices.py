#!/usr/bin/env python3
"""Name the speakers of a diarized transcript from voices enrolled earlier.

whisperX with `--speaker_embeddings` stores one voice embedding per diarization
label in the transcript .json. Enrolling keeps a label's embedding under a person's
name; matching compares a new transcript's labels against every enrolled person
and prints the `--speakers` map meeting.sh takes.

    python scripts/voices.py enroll transcripts/rec.json SPEAKER_01 Bob
    python scripts/voices.py match transcripts/rec.json   # SPEAKER_00=Alice,...
    python scripts/voices.py list
    python scripts/voices.py --selftest

A voice embedding identifies a person: the store stays out of git, in $VOICES_DIR
(default ~/.local/share/whisperx/voices), one <name>.json per person.
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

# Cosine similarity a label needs to an enrolled voice before it gets the name.
# Measured on two recordings of one meeting (2026-10-21, same room and phone): the same
# person scored 0.97-0.99 across them, the other person 0.39-0.44.
# shortcut: one pair of recordings, one room; re-measure with `match` when a voice
# recorded elsewhere goes unnamed or lands on a stranger, and set VOICE_THRESHOLD.
DEFAULT_THRESHOLD = 0.7


def voices_dir() -> Path:
    return Path(os.environ.get("VOICES_DIR") or "~/.local/share/whisperx/voices").expanduser()


def threshold() -> float:
    return float(os.environ.get("VOICE_THRESHOLD") or DEFAULT_THRESHOLD)


def embeddings(transcript: Path) -> dict[str, list[float]]:
    """The transcript's {label: embedding}; empty when whisperX ran without the flag."""
    return json.loads(transcript.read_text()).get("speaker_embeddings") or {}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def load(store: Path) -> dict[str, list[list[float]]]:
    """{name: [embedding, ...]}, one sample per enrolled recording."""
    if not store.is_dir():
        return {}
    return {f.stem: json.loads(f.read_text())["samples"] for f in sorted(store.glob("*.json"))}


def enroll(transcript: Path, label: str, name: str, store: Path) -> int:
    emb = embeddings(transcript)
    if label not in emb:
        raise SystemExit(f"{label} has no embedding in {transcript} (labels: {', '.join(emb) or 'none'})")
    if not name or "/" in name or name.startswith("."):
        raise SystemExit(f"not a usable name: {name!r}")
    store.mkdir(parents=True, exist_ok=True)
    store.chmod(0o700)
    samples = load(store).get(name, []) + [emb[label]]
    path = store / f"{name}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"samples": samples}))
    tmp.chmod(0o600)
    tmp.replace(path)
    return len(samples)


def match(emb: dict[str, list[float]], known: dict[str, list[list[float]]], floor: float) -> dict[str, str]:
    """{label: name}, best pairs first, each label and each name used once.

    A label is scored against a person by its best sample, so a voice enrolled from
    two rooms matches either. Below `floor` the label stays unnamed: a wrong name in
    the notes is worse than SPEAKER_02.
    """
    scores = sorted(
        ((max(cosine(vec, s) for s in samples), label, name)
         for label, vec in emb.items() for name, samples in known.items() if samples),
        reverse=True,
    )
    named: dict[str, str] = {}
    for score, label, name in scores:
        if score >= floor and label not in named and name not in named.values():
            named[label] = name
    return named


def main(argv: list[str]) -> int:
    store = voices_dir()
    if argv[:1] == ["enroll"] and len(argv) == 4:
        n = enroll(Path(argv[1]), argv[2], argv[3], store)
        print(f"{argv[3]}: {n} sample(s) in {store}", file=sys.stderr)
        return 0
    if argv[:1] == ["match"] and len(argv) == 2:
        emb, known = embeddings(Path(argv[1])), load(store)
        for label, vec in emb.items():
            best = {name: round(max(cosine(vec, s) for s in samples), 3) for name, samples in known.items()}
            print(f"  {label}: {best}", file=sys.stderr)
        print(",".join(f"{k}={v}" for k, v in sorted(match(emb, known, threshold()).items())))
        return 0
    if argv[:1] == ["list"]:
        for name, samples in load(store).items():
            print(f"{name}: {len(samples)} sample(s)")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


def selftest() -> None:
    assert abs(cosine([1, 0], [1, 0]) - 1) < 1e-9 and cosine([1, 0], [0, 1]) == 0
    assert cosine([0, 0], [1, 0]) == 0.0
    known = {"Alice": [[1.0, 0, 0]], "Bob": [[0.0, 1, 0], [0.0, 0, 1]]}
    # Bob matches on his second sample; the stranger stays unnamed.
    emb = {"S0": [0.9, 0.1, 0], "S1": [0, 0.1, 0.9], "S2": [-1, 0, 0]}
    assert match(emb, known, 0.5) == {"S0": "Alice", "S1": "Bob"}
    # Two labels close to one person: only the closer one gets the name.
    assert match({"S0": [1.0, 0, 0], "S1": [0.8, 0.2, 0]}, {"Alice": [[1.0, 0, 0]]}, 0.5) == {"S0": "Alice"}
    assert match({}, known, 0.5) == {} and match(emb, {}, 0.5) == {}
    tmp = Path(tempfile.mkdtemp())
    t = tmp / "rec.json"
    t.write_text(json.dumps({"segments": [], "speaker_embeddings": {"SPEAKER_00": [1, 0, 0]}}))
    store = tmp / "voices"
    assert enroll(t, "SPEAKER_00", "Alice", store) == 1
    assert enroll(t, "SPEAKER_00", "Alice", store) == 2
    assert load(store) == {"Alice": [[1, 0, 0], [1, 0, 0]]}
    assert oct((store / "Alice.json").stat().st_mode & 0o777) == "0o600"
    for bad in [("SPEAKER_09", "Bob"), ("SPEAKER_00", "../x")]:
        try:
            enroll(t, *bad, store)
            raise AssertionError(f"enrolled {bad}")
        except SystemExit:
            pass
    t.write_text(json.dumps({"segments": []}))
    assert embeddings(t) == {}
    print("ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else sys.exit(main(sys.argv[1:]))
