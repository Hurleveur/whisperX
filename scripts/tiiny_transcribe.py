#!/usr/bin/env python3
"""A recording -> transcripts/<name>.json and .txt in whisperX's shape, written by the home Tiiny.

This laptop's CPU is always busy, so meeting.sh hands the transcription to the Tiiny's Qwen3-ASR
through the HAI OS owner door instead of running whisperX here. ffmpeg decodes the file; the
clips are cut as the live captions cut them (scripts/live_translate.py, 12 s at most, at the
quietest moment) and sent one at a time. There are no speaker labels: the Tiiny has no
diarization (to find, 2026-11-09).

Each clip's text is kept in <name>.tiiny.jsonl as it arrives, so a run stopped halfway (box
unplugged, laptop asleep) resumes where it was. A clip the box refuses three times stops the
run with exit code 1; the next run starts from that clip.

    python3 scripts/tiiny_transcribe.py <audio> <out-dir>
    python3 scripts/tiiny_transcribe.py --selftest
"""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import live_translate as live  # noqa: E402

TRIES, WAIT_S = 3, 10


def pcm_of(audio: Path) -> bytes:
    # shortcut: the whole recording in memory (115 MB an hour); stream it if a recording runs past a day
    return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(audio), "-ac", "1", "-ar",
                           str(live.RATE), "-f", "s16le", "-"], capture_output=True, check=True).stdout


def clips(pcm: bytes) -> list[tuple[float, float, bytes]]:
    """(start s, end s, pcm) per clip. The speech threshold is 1.3x the 5th-percentile 100 ms of
    the whole file: a file can be read twice, unlike the live mic."""
    step = live.FRAME * 2
    frames = [pcm[i:i + step] for i in range(0, len(pcm), step)]
    levels = sorted(live.rms(f) for f in frames)
    floor = levels[len(levels) // 20] if levels else 0.0
    chunker = live.Chunker(threshold=max(floor * 1.3, 150), longest=120)
    out = []
    for i, frame in enumerate(frames):
        if clip := chunker.feed(frame):
            end = (i + 1 - len(chunker.frames)) / 10  # frames left in the chunker are after the cut
            out.append((end - len(clip) / step / 10, end, clip))
    # the file's last words, still waiting for a pause that never comes
    if sum(x > chunker.threshold for x in chunker.levels) >= chunker.shortest:
        out.append((len(frames) / 10 - len(chunker.frames) / 10, len(frames) / 10, b"".join(chunker.frames)))
    return out


def transcribe(audio: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    done_file = out_dir / f"{audio.stem}.tiiny.jsonl"
    done = {}
    if done_file.is_file():
        done = {row["i"]: row for row in map(json.loads, done_file.read_text().splitlines())}
    parts = clips(pcm_of(audio))
    print(f"{audio.name}: {len(parts)} clips, {len(done)} already written", file=sys.stderr, flush=True)
    with done_file.open("a") as log:
        for i, (start, end, pcm) in enumerate(parts):
            if i in done:
                continue
            text = ""
            for attempt in range(TRIES):
                try:
                    text = live.transcribe(pcm)
                    break
                except (RuntimeError, OSError) as e:
                    print(f"clip {i}: {e}", file=sys.stderr, flush=True)
                    if attempt == TRIES - 1:
                        raise SystemExit(f"the Tiiny refused clip {i} {TRIES} times; rerun to resume")
                    time.sleep(WAIT_S)
            done[i] = {"i": i, "start": round(start, 1), "end": round(end, 1), "text": text}
            log.write(json.dumps(done[i], ensure_ascii=False) + "\n")
            log.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(parts)}", file=sys.stderr, flush=True)
    # a clip the box heard in another script is noise ("呃。"); other Latin languages are kept,
    # since a meeting may mix French or English into the Dutch
    segments = [{"start": r["start"], "end": r["end"], "text": " " + r["text"]}
                for _, r in sorted(done.items()) if r["text"] and not live.NON_LATIN.search(r["text"])]
    json_path = out_dir / f"{audio.stem}.json"
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"segments": segments, "asr": "tiiny Qwen3-ASR"}, ensure_ascii=False))
    (out_dir / f"{audio.stem}.txt").write_text("".join(s["text"].strip() + "\n" for s in segments))
    tmp.replace(json_path)  # the .json last: meeting.sh treats it as the mark of a finished transcript
    done_file.unlink()
    return json_path


def selftest() -> None:
    import array
    loud = array.array("h", [3000] * live.FRAME).tobytes()
    quiet = array.array("h", [10] * live.FRAME).tobytes()
    parts = clips(quiet * 30 + loud * 20 + quiet * 10 + loud * 150)
    assert [(s, e) for s, e, _ in parts[:1]] == [(3.0, 5.7)], f"times follow the audio: {parts[:1]}"
    assert all(e - s <= 12 for s, e, _ in parts), "a long stretch is cut at 12 s at most"
    assert len(parts) == 3 and parts[-1][1] == 21.0, f"the file's tail is kept: {[(s, e) for s, e, _ in parts]}"
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif len(sys.argv) == 3:
        try:
            live.DOOR = live.DOOR_FILE.read_text().strip().rstrip("/")
        except FileNotFoundError:
            sys.exit(f"No door: write the owner door's URL into {live.DOOR_FILE}")
        print(transcribe(Path(sys.argv[1]), Path(sys.argv[2])))
    else:
        sys.exit(__doc__)
