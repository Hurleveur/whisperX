#!/usr/bin/env python3
"""Turbo, large-v3 and the Tiiny on the same stretch of a recording, minute by minute, with the
seconds each one took. Run on a machine with a CUDA GPU (the home laptop), with the whisperX
venv's python; the Tiiny is asked through the owner door (~/.config/hai/door).

All three hear the same clips, cut as the live captions and tiiny_transcribe.py cut them, and
Whisper holds to --language as the live captions do. whisperX in meeting.sh hears the whole file
in 30 s windows instead, so its text differs a little; the comparison between models holds.

    .venv/bin/python scripts/compare_asr.py <audio> <start_s> <seconds> [--language nl] [--out FILE]
    python3 scripts/compare_asr.py --selftest

Writes the comparison to transcripts/compare-<name>-<start>s.md unless --out says otherwise.
"""

import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import live_translate as live  # noqa: E402
import tiiny_transcribe  # noqa: E402

WINDOW_S = 60
WHISPERS = ("large-v3-turbo", "large-v3")


def windows(starts: list[float], width: float = WINDOW_S) -> list[list[int]]:
    """Clip indexes per window of `width` seconds, by where each clip starts; empty windows stay."""
    out: list[list[int]] = [[] for _ in range(int(max(starts, default=-1) // width) + 1)]
    for i, start in enumerate(starts):
        out[int(start // width)].append(i)
    return out


def slice_pcm(audio: str, start: float, seconds: float) -> bytes:
    return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-t", str(seconds),
                           "-i", audio, "-ac", "1", "-ar", str(live.RATE), "-f", "s16le", "-"],
                          capture_output=True, check=True).stdout


def run(name: str, write, pcms: list[bytes], language: str) -> list[tuple[str, float]]:
    out = []
    for pcm in pcms:
        began = time.monotonic()
        try:
            text = write(pcm, language)
        except (RuntimeError, OSError) as e:
            text = f"[{e}]"
        out.append((text, time.monotonic() - began))
    print(f"{name}: {sum(s for _, s in out):.0f} s for {len(pcms)} clips", file=sys.stderr, flush=True)
    return out


def main(audio: str, start: float, seconds: float, language: str, out: Path) -> int:
    if not live.cuda_gpus():
        raise SystemExit("no CUDA GPU here: run it on the home laptop with the whisperX venv's python")
    live.DOOR = live.DOOR_FILE.read_text().strip().rstrip("/")
    parts = tiiny_transcribe.clips(slice_pcm(audio, start, seconds))
    if not parts:
        raise SystemExit(f"no speech between {start:.0f} and {start + seconds:.0f} s (past the end?)")
    pcms = [pcm for _, _, pcm in parts]
    heard = {}
    for size in WHISPERS:  # one model on the card at a time, beside the box's own
        write = live.gpu_transcriber(language, size)
        heard[size] = run(size, write, pcms, language)
        del write
        gc.collect()
    # the Tiiny ignores the language; keep_heard drops the sentences it hears in another one
    heard["Tiiny"] = [(live.keep_heard(t), s) for t, s in run("Tiiny", live.transcribe, pcms, "")]

    lines = [f"# {Path(audio).name}, {start:.0f} to {start + seconds:.0f} s, language {language}", ""]
    lines += [f"- {name}: {sum(s for _, s in rows):.0f} s in all" for name, rows in heard.items()]
    for w, idx in enumerate(windows([s for s, _, _ in parts])):
        at = start + w * WINDOW_S
        lines += ["", f"## {int(at // 60)}:{int(at % 60):02d} to {int((at + WINDOW_S) // 60)}:{int((at + WINDOW_S) % 60):02d}"]
        for name, rows in heard.items():
            text = " ".join(rows[i][0] for i in idx if rows[i][0])
            lines += ["", f"**{name}** ({sum(rows[i][1] for i in idx):.1f} s): {text or '(nothing)'}"]
    report = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"Written: {out}", file=sys.stderr)
    return 0


def selftest() -> None:
    assert windows([0.0, 12.5, 59.9, 60.0, 130.2]) == [[0, 1, 2], [3], [4]]
    assert windows([65.0]) == [[], [0]], "an empty first minute keeps its place"
    assert windows([]) == []
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        sys.exit(0)
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("start", type=float)
    p.add_argument("seconds", type=float)
    p.add_argument("--language", default="nl")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    repo = Path(__file__).resolve().parent.parent
    sys.exit(main(a.audio, a.start, a.seconds, a.language,
                  a.out or repo / "transcripts" / f"compare-{Path(a.audio).stem}-{a.start:.0f}s.md"))
